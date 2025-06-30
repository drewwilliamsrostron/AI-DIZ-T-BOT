"""Tkinter dashboard for training progress and live prices."""

# ruff: noqa: F403, F405
import artibot.globals as G
import numpy as np
import datetime
import time
import os
import threading
import pandas as pd
import logging
from .metrics import nuclear_key_condition
from .live_risk import update_auto_pause

tk = None  # imported lazily
ttk = None
matplotlib = None
plt = None
FigureCanvasTkAgg = None


from typing import Callable  # noqa: F401,E402

GUI_INSTANCE = None

try:
    if tk is None:  # noqa: SIM105
        import tkinter as tk
        from tkinter import ttk
except Exception:  # pragma: no cover - headless import may fail
    pass


def build_scrollable(master: "tk.Widget") -> "ttk.Frame":
    class _Scrollable(ttk.Frame):
        """A reusable scrollable container that exposes `.inner`."""

        def __init__(self, master: tk.Widget, **kw):
            super().__init__(master, **kw)
            cv = tk.Canvas(self, highlightthickness=0)
            self._canvas = cv
            vsb = ttk.Scrollbar(self, orient="vertical", command=cv.yview)
            hsb = ttk.Scrollbar(self, orient="horizontal", command=cv.xview)
            cv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

            vsb.pack(side="right", fill="y")
            hsb.pack(side="bottom", fill="x")
            cv.pack(side="left", fill="both", expand=True)

            self.inner = ttk.Frame(cv)
            self._win = cv.create_window((0, 0), window=self.inner, anchor="nw")

            def _update(_evt=None):
                cv.configure(scrollregion=cv.bbox("all"))

            self.inner.bind("<Configure>", _update)

            def _on_wheel(evt):
                cv.yview_scroll(int(-1 * (evt.delta / 120)), "units")

            cv.bind_all("<MouseWheel>", _on_wheel, add="+")

            def _resize(event):
                cv.itemconfig(self._win, width=event.width)

            cv.bind("<Configure>", _resize)

    return _Scrollable(master).inner


def redraw_everything() -> None:
    if GUI_INSTANCE is not None:
        GUI_INSTANCE.update_dashboard()


###############################################################################
# Tkinter GUI
###############################################################################


def format_trade_details(trades, limit=50):
    """Return a compact table string for the most recent trades."""
    if not trades:
        return "No Trade Details"
    df = pd.DataFrame(trades)
    if df.empty:
        return "No Trade Details"
    if df["entry_time"].max() > 1_000_000_000_000:
        df["entry_time"] //= 1000
    if df["exit_time"].max() > 1_000_000_000_000:
        df["exit_time"] //= 1000
    df["Entry"] = pd.to_datetime(df["entry_time"], unit="s").dt.strftime("%Y-%m-%d")
    df["Exit"] = pd.to_datetime(df["exit_time"], unit="s").dt.strftime("%Y-%m-%d")
    df["ReturnPct"] = df["return"] * 100
    cols = ["Entry", "Exit", "side", "entry_price", "exit_price", "ReturnPct"]
    out_df = df.loc[:, cols].tail(limit)
    return out_df.to_string(index=False, float_format=lambda x: f"{x:.2f}")


def should_enable_live_trading() -> bool:
    """Return ``True`` when validation and live metrics meet risk limits."""
    sharpe = G.global_holdout_sharpe
    dd = G.global_holdout_max_drawdown
    pnl = G.live_equity - G.start_equity
    trades = G.live_trade_count
    try:  # avoid circular import during tests
        from .bot_app import CONFIG

        min_pnl = float(CONFIG.get("NK_MIN_PNL", 0.0))
        min_trades = int(CONFIG.get("NK_MIN_TRADES", 0))
    except Exception:  # pragma: no cover - CONFIG may not exist
        min_pnl = 0.0
        min_trades = 0

    return sharpe >= 1.0 and dd >= -0.30 and pnl >= min_pnl and trades >= min_trades


def select_weight_file(use_prev: bool = True) -> str | None:
    """Return the weight file path based on user selection."""
    from tkinter import messagebox, filedialog

    if use_prev and messagebox.askyesno("Load Weights", "Use best.pt?"):
        return "best.pt"
    return (
        filedialog.askopenfilename(
            title="Select weight file", filetypes=[("PyTorch", "*.pth")]
        )
        or None
    )


def ask_use_prev_weights(default: bool = True, tk_module=None) -> bool:
    """Return ``True`` when the user opts to load the last weights."""
    if tk_module is None:
        import tkinter as tk_module  # lazy import for tests
        from tkinter import messagebox
    else:
        messagebox = tk_module.messagebox

    root = tk_module.Tk()
    root.withdraw()
    try:
        result = messagebox.askyesno("Load Weights", "Use previous best weights?")
    finally:
        root.destroy()
    if result is None:
        return default
    return bool(result)


def startup_options_dialog(
    defaults: dict[str, object] | None = None,
    tk_module=None,
) -> dict[str, object]:
    """Return user-selected startup options via a single Tk dialog."""
    if tk_module is None:
        import tkinter as tk_module  # lazy import for tests

    defaults = defaults or {}
    result: dict[str, object] = {}
    try:
        root = tk_module.Tk()
    except Exception as exc:  # pragma: no cover - headless env
        logging.warning("Tk unavailable: %s; using defaults", exc)
        return {
            "skip_sentiment": bool(defaults.get("skip_sentiment", False)),
            "use_live": bool(defaults.get("use_live", False)),
            "use_prev_weights": bool(defaults.get("use_prev_weights", True)),
            "threads": int(defaults.get("threads", os.cpu_count() or 1)),
        }

    root.title("Startup Options")
    skip_var = tk_module.BooleanVar(value=bool(defaults.get("skip_sentiment", False)))
    live_var = tk_module.BooleanVar(value=bool(defaults.get("use_live", False)))
    weights_var = tk_module.BooleanVar(
        value=bool(defaults.get("use_prev_weights", True))
    )
    threads_max = os.cpu_count() or 1
    threads_var = tk_module.IntVar(value=int(defaults.get("threads", threads_max)))

    tk_module.Checkbutton(
        root, text="Skip GDELT sentiment download", variable=skip_var
    ).pack(anchor="w")
    tk_module.Checkbutton(root, text="Enable LIVE trading", variable=live_var).pack(
        anchor="w"
    )
    tk_module.Checkbutton(
        root, text="Load previous weights", variable=weights_var
    ).pack(anchor="w")
    tk_module.Label(root, text="CPU threads:").pack(anchor="w")
    tk_module.Spinbox(
        root,
        from_=1,
        to=threads_max,
        textvariable=threads_var,
        width=5,
    ).pack(anchor="w")

    def cont() -> None:
        result["skip_sentiment"] = skip_var.get()
        result["use_live"] = live_var.get()
        result["use_prev_weights"] = weights_var.get()
        result["threads"] = threads_var.get()
        root.quit()
        root.destroy()

    tk_module.Button(root, text="Continue", command=cont).pack(pady=5)
    root.mainloop()
    return result


def _fetch_position(exchange):
    """Return (side, size, entry) for the BTCUSD swap position."""
    try:
        if hasattr(exchange, "fetch_position_risk"):
            pos = exchange.fetch_position_risk("BTCUSD")
            sz = pos.get("size", 0)
            side = "LONG" if sz > 0 else ("SHORT" if sz < 0 else "NONE")
            return (side, abs(sz), pos.get("entryPrice", 0))
        allpos = exchange.fetch_positions()
        for p in allpos:
            if p["symbol"] == "BTCUSD" and p["info"].get("type") == "swap":
                sz = p["contracts"]
                side = "LONG" if sz > 0 else ("SHORT" if sz < 0 else "NONE")
                return (side, abs(sz), p["entryPrice"])
    except Exception as e:  # pragma: no cover - network errors
        logging.error("Position fetch failed: %s", e)
    return ("NONE", 0.0, 0.0)


class TradingGUI:
    def __init__(self, root, ensemble, weights_path: str | None = None, connector=None):
        global tk, ttk, matplotlib, plt, FigureCanvasTkAgg
        if tk is None:
            import tkinter as tk
            from tkinter import ttk
        if matplotlib is None:
            import matplotlib

            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            plt.ioff()
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        # fallbacks for heavily stubbed modules during tests
        if not hasattr(tk, "BOTH"):
            for name in [
                "BOTH",
                "LEFT",
                "RIGHT",
                "Y",
                "X",
                "END",
                "W",
                "CENTER",
                "DISABLED",
                "NORMAL",
            ]:
                setattr(tk, name, name.lower())

        class _Dummy:
            def __init__(self, *a, **k):
                self.root = k.get("master")
                self.attrs = {"state": "normal"}

            def pack(self, *a, **k):
                pass

            def grid(self, *a, **k):
                pass

            def config(self, **kw):
                self.attrs.update(kw)

            def configure(self, **kw):
                pass

            def create_window(self, *a, **k):
                pass

            def bbox(self, *a):
                return (0, 0, 100, 100)

            def yview(self, *a, **k):
                pass

            def yview_moveto(self, *a, **k):
                pass

            def set(self, *a, **k):
                pass

            def add(self, *a, **k):
                pass

            def heading(self, *a, **k):
                pass

            def column(self, *a, **k):
                pass

            def insert(self, *a, **k):
                pass

            def delete(self, *a, **k):
                pass

            def get_children(self):
                return []

            def bind(self, *a, **k):
                pass

            def columnconfigure(self, *a, **k):
                pass

            def rowconfigure(self, *a, **k):
                pass

            def winfo_width(self):
                return int(getattr(self.root, "width", 100) * 0.5 * G.UI_SCALE)

        if not hasattr(ttk, "Frame"):
            for n in [
                "Frame",
                "Label",
                "LabelFrame",
                "Button",
                "Progressbar",
                "Notebook",
                "Scrollbar",
                "Treeview",
                "Checkbutton",
            ]:
                setattr(ttk, n, _Dummy)
        if not hasattr(tk, "Canvas"):
            tk.Canvas = _Dummy

        self.root = root
        self.ensemble = ensemble
        self.weights_path = weights_path
        self.connector = connector
        self.close_requested = False
        # Track dataset lengths to detect chart updates
        self._last_loss_len = len(G.global_training_loss)
        self._last_val_len = len(G.global_validation_loss)
        self._last_equity_len = len(G.global_equity_curve)
        self._last_trade_details_len = len(G.global_trade_details)
        self._last_price_len = len(G.global_phemex_data)
        self._last_backtest_len = len(G.global_backtest_profit)
        scale = max(1.0, min(1.5, root.winfo_fpixels("1i") / 72))
        root.tk.call("tk", "scaling", scale)
        G.UI_SCALE = scale
        self.root.title("Complex AI Trading w/ Robust Backtest + Live Phemex")
        if hasattr(ttk, "Style"):
            style = ttk.Style()
            try:
                style.theme_use("clam")
            except Exception:
                pass

        # ------------------------------------------------------------------
        # overall layout
        # ------------------------------------------------------------------
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=0)

        self.main_frame = ttk.Frame(root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        self.sidebar = ttk.Frame(root)
        self.sidebar.grid(row=0, column=1, sticky="ns")

        self.footer = ttk.Frame(root)
        self.footer.grid(row=1, column=0, columnspan=2, sticky="ew")

        self._build_notebook()
        self._build_sidebar()
        self._build_footer()

        for child in self.main_frame.winfo_children():
            child.columnconfigure(0, weight=1)
            child.rowconfigure(0, weight=1)

        w_func = getattr(root, "winfo_screenwidth", lambda: 1280)
        h_func = getattr(root, "winfo_screenheight", lambda: 720)
        w, h = int(w_func()), int(h_func())
        w = max(800, int(w * 0.7))
        h = max(600, int(h * 0.7))
        root.minsize(w, h)
        try:  # some stubs lack geometry()
            root.geometry(f"{w}x{h}")
        except Exception:  # pragma: no cover - not critical during tests
            pass

        w = max(800, int(w * 0.7))
        h = max(600, int(h * 0.7))
        root.minsize(w, h)
        try:  # some stubs lack geometry()
            root.geometry(f"{w}x{h}")
        except Exception:  # pragma: no cover - not critical during tests
            pass

        self.update_interval = 100
        self.after_id = None
        self.root.after_idle(self.update_dashboard)
        self.root.after(10000, self.refresh_stats)

        global GUI_INSTANCE
        GUI_INSTANCE = self

    def _animate_attention(
        self, X: np.ndarray, Y: np.ndarray, new_data: np.ndarray
    ) -> None:
        """Animate the attention surface from the previous values."""
        if G.global_last_attention is None or len(G.global_last_attention) == 0:
            self.ax_attention.text(0.5, 0.5, "No attention data", ha="center")
            return
        if self._last_attention is None or self._last_attention.shape != new_data.shape:
            self._last_attention = new_data.copy()
        diff = (new_data - self._last_attention) / float(self.anim_steps)
        current = self._last_attention.copy()
        for _ in range(self.anim_steps):
            current += diff
            if self._surf is not None:
                try:
                    self._surf.remove()
                except NotImplementedError:
                    self.ax_attention.cla()
            self._surf = self.ax_attention.plot_surface(X, Y, current, cmap="viridis")
            self.canvas_train.draw()
            self.canvas_train.flush_events()
        self._last_attention = new_data.copy()

    def _loss_comment(self) -> str:
        """Return a simple English description of training vs validation loss."""
        train = list(G.global_training_loss)
        val = [v for v in G.global_validation_loss if v is not None]
        if not train or not val:
            return "Waiting for validation data..."
        comment = []
        if val[-1] > train[-1]:
            comment.append(
                "orange above blue - validation loss higher, possible overfitting"
            )
        else:
            comment.append(
                "blue above orange - training loss higher, model still learning"
            )
        if len(train) >= 2 and len(val) >= 2:
            down_train = train[-1] < train[-2]
            down_val = val[-1] < val[-2]
            if down_train and down_val:
                comment.append("both lines going down - good")
            elif not down_train and not down_val:
                comment.append("both lines going up - might need tuning")
        return ". ".join(comment)

    def _log_chart_updates(self) -> None:
        """Output a short message whenever chart data changes."""
        if (
            len(G.global_training_loss) != self._last_loss_len
            or len(G.global_validation_loss) != self._last_val_len
        ):
            print(
                f"[GUI] Loss updated at epoch {G.epoch_count}"
                f" (train={len(G.global_training_loss)}, val={len(G.global_validation_loss)})"
            )
            self._last_loss_len = len(G.global_training_loss)
            self._last_val_len = len(G.global_validation_loss)

        if len(G.global_equity_curve) != self._last_equity_len:
            start = G.global_equity_curve[0][0] if G.global_equity_curve else 0
            end = G.global_equity_curve[-1][0] if G.global_equity_curve else 0
            s_str = datetime.datetime.fromtimestamp(start).strftime("%Y-%m-%d")
            e_str = datetime.datetime.fromtimestamp(end).strftime("%Y-%m-%d")
            print(f"[GUI] Equity data updated: start {s_str} end {e_str}")
            self._last_equity_len = len(G.global_equity_curve)

        if len(G.global_trade_details) != self._last_trade_details_len:
            print(f"[GUI] Trade details updated: {len(G.global_trade_details)} trades")
            self._last_trade_details_len = len(G.global_trade_details)

        if len(G.global_phemex_data) != self._last_price_len:
            start_ms = G.global_phemex_data[0][0] if G.global_phemex_data else 0
            end_ms = G.global_phemex_data[-1][0] if G.global_phemex_data else 0
            s_str = datetime.datetime.fromtimestamp(start_ms / 1000).strftime(
                "%Y-%m-%d"
            )
            e_str = datetime.datetime.fromtimestamp(end_ms / 1000).strftime("%Y-%m-%d")
            print(f"[GUI] Price data updated: start {s_str} end {e_str}")
            self._last_price_len = len(G.global_phemex_data)

        if len(G.global_backtest_profit) != self._last_backtest_len:
            print(
                f"[GUI] Backtest profit updated: {len(G.global_backtest_profit)} points"
            )
            self._last_backtest_len = len(G.global_backtest_profit)

    def update_dashboard(self):
        """Refresh all dashboard widgets from shared state."""
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        # console output when chart data changes
        self._log_chart_updates()
        self.ax_loss.clear()
        self.ax_loss.set_title("Training vs. Validation Loss")
        n = min(len(G.global_training_loss), len(G.global_validation_loss))
        train = G.global_training_loss[:n]
        val = G.global_validation_loss[:n]
        x1 = range(1, n + 1)
        self.ax_loss.plot(x1, train, color="blue", marker="o", label="Train")
        val_filtered = [(i + 1, v) for i, v in enumerate(val) if v is not None]
        if val_filtered:
            xv, yv = zip(*val_filtered)
            self.ax_loss.plot(xv, yv, color="orange", marker="x", label="Val")
        self.ax_loss.legend()
        self.loss_comment_label.config(text=self._loss_comment())

        self.ax_equity_train.clear()
        self.ax_equity_train.set_title("Equity: Current (red) vs Best (green)")
        # draw baseline at 0% for quick reference
        self.ax_equity_train.axhline(
            0.0,
            color="black",
            linestyle="-",
            linewidth=1,
        )
        try:
            valid_eq = [
                (t, b)
                for (t, b) in G.global_equity_curve
                if isinstance(t, (int, float, np.integer, np.floating))
            ]
            if valid_eq:
                ts_, bs_ = zip(*valid_eq)
                ts_dt = [datetime.datetime.fromtimestamp(t_) for t_ in ts_]
                self.ax_equity_train.plot(
                    ts_dt, bs_, color="red", marker=".", label="Current"
                )
            if G.global_best_equity_curve:
                best_eq = [
                    (t, b)
                    for (t, b) in G.global_best_equity_curve
                    if isinstance(t, (int, float, np.integer, np.floating))
                ]
                if best_eq:
                    t2, b2 = zip(*best_eq)
                    t2dt = [datetime.datetime.fromtimestamp(x) for x in t2]
                    self.ax_equity_train.plot(
                        t2dt, b2, color="green", marker=".", label="Best"
                    )
            handles, labels = self.ax_equity_train.get_legend_handles_labels()
            if handles and any(labels):
                self.ax_equity_train.legend(handles, labels)
        except Exception:
            pass

        # --------------------------------------------------------------
        # plot cumulative trades over time
        # --------------------------------------------------------------
        self.ax_trades_time.clear()
        self.ax_trades_time.set_title("Trades Over Time")
        try:
            if G.global_trade_details:
                ts = [t.get("entry_time", 0) for t in G.global_trade_details]
                ts = [t // 1000 if t > 1_000_000_000_000 else t for t in ts]
                ts_dt = [datetime.datetime.fromtimestamp(t) for t in ts]
                counts = list(range(1, len(ts_dt) + 1))
                self.ax_trades_time.step(ts_dt, counts, where="post")
        except Exception:
            pass
        self.canvas_train.draw()

        self.ax_live.clear()
        self.ax_live.set_title("Phemex Live Price (1h)")
        try:
            times, closes = [], []
            for bar in G.global_phemex_data:
                if len(bar) >= 5 and bar[0] > 0:
                    t_ = bar[0]
                    c_ = bar[4]
                    times.append(datetime.datetime.fromtimestamp(t_ / 1000))
                    closes.append(c_)
            if times and closes:
                self.ax_live.plot(times, closes, marker="o")
        except Exception:
            pass
        self.canvas_live.draw()

        self.ax_net_profit.clear()
        self.ax_net_profit.set_title("Net Profit (%)")
        if G.global_backtest_profit:
            x2 = range(1, len(G.global_backtest_profit) + 1)
            self.ax_net_profit.plot(
                x2, G.global_backtest_profit, marker="o", color="green"
            )
        self.canvas_backtest.draw()

        # ---------- Activity Timeline ----------
        self.ax_tl.clear()
        depth = G.timeline_depth
        idx = G.timeline_index
        xs = np.arange(depth)
        inds = np.roll(G.timeline_ind_on, -idx, axis=0)
        trade = np.roll(G.timeline_trades, -idx)
        names = ["EMA", "SMA", "RSI", "KIJ", "TEN", "DISP"]
        for k in range(inds.shape[1]):
            self.ax_tl.plot(xs, k + inds[:, k] * 0.8, lw=0.8, label=names[k])
        self.ax_tl.plot(xs, 6 + trade * 0.8, lw=1.1, c="black", label="Trade ON")
        self.ax_tl.set_ylim(-0.5, 7.5)
        self.ax_tl.set_yticks([])
        self.ax_tl.set_xlabel("Bars")
        self.ax_tl.set_title("Indicators ON/OFF & Trade State")
        if idx > 0:
            self.ax_tl.legend(ncol=4, fontsize=6, framealpha=0.3)
        self.canvas_tl.draw()

        self.ax_attention.clear()
        self.ax_attention.set_title("Live Attention Weights")
        if G.global_last_attention is not None:
            data = np.array(G.global_last_attention)
            avg = data.mean(axis=0) if data.ndim == 3 else data
            if avg.ndim >= 2:
                x_ = np.arange(avg.shape[0])
                y_ = np.arange(avg.shape[1])
                X, Y = np.meshgrid(x_, y_)
                self._animate_attention(X, Y, avg)
            else:
                x_ = np.arange(len(avg))
                self.ax_attention.plot(x_, avg, marker="o", color="purple")
                self._last_attention = None
        elif G.global_attention_weights_history:
            x_ = list(range(1, len(G.global_attention_weights_history) + 1))
            self.ax_attention.plot(
                x_, G.global_attention_weights_history, marker="o", color="purple"
            )
            self._last_attention = None
        self.canvas_train.draw()

        for row in self.trade_tree.get_children():
            self.trade_tree.delete(row)
        if G.global_trade_details:
            recent = G.global_trade_details[-50:]
            for tr in recent:
                dt = datetime.datetime.fromtimestamp(tr.get("entry_time", 0))
                date = dt.strftime("%Y-%m-%d")
                values = (
                    date,
                    tr.get("side", ""),
                    tr.get("size", 0),
                    f"{tr.get('entry_price', 0):.2f}",
                    f"{tr.get('exit_price', 0):.2f}",
                    f"{tr.get('return', 0) * 100:.2f}",
                )
                self.trade_tree.insert("", tk.END, values=values)
        else:
            self.trade_tree.insert("", tk.END, values=("No data", "", "", "", "", ""))

        ypos = self.yearly_perf_text.yview()
        self.yearly_perf_text.delete("1.0", tk.END)
        if G.global_best_yearly_stats_table:
            self.yearly_perf_text.insert(tk.END, G.global_best_yearly_stats_table)
        else:
            self.yearly_perf_text.insert(tk.END, "No yearly data")
        self.yearly_perf_text.yview_moveto(ypos[0])

        mpos = self.monthly_perf_text.yview()
        self.monthly_perf_text.delete("1.0", tk.END)
        if G.global_best_monthly_stats_table:
            self.monthly_perf_text.insert(tk.END, G.global_best_monthly_stats_table)
        else:
            self.monthly_perf_text.insert(tk.END, "No monthly data")
        self.monthly_perf_text.yview_moveto(mpos[0])

        pred_str = G.global_current_prediction if G.global_current_prediction else "N/A"
        conf = G.global_ai_confidence if G.global_ai_confidence else 0.0
        steps = G.epoch_count

        color_map = {"BUY": "green", "SELL": "red", "HOLD": "black"}
        pred_color = color_map.get(pred_str.upper(), "black")
        if G.global_sharpe > 2 or G.global_max_drawdown < -0.20:
            pred_color = "purple"
        self.pred_label.config(text=f"AI Prediction: {pred_str}", foreground=pred_color)

        price = 0.0
        if G.global_phemex_data and len(G.global_phemex_data[-1]) >= 5:
            price = float(G.global_phemex_data[-1][4])
        self.price_label.config(text=f"Live Price: {price:.2f}")
        self.conf_label.config(text=f"Confidence: {conf:.2f}")
        self.epoch_label.config(text=f"Training Steps: {steps}")
        bal = G.global_account_stats.get("total", {}).get("USDT", 0.0)
        self.balance_label.config(text=f"USDT Balance: {bal:.2f}")
        self.label_long["text"] = f"{G.gross_long_usd:.0f}"
        self.label_short["text"] = f"{G.gross_short_usd:.0f}"
        if G.global_position_side:
            pos = f"{G.global_position_side} {G.global_position_size:.4f}"
        else:
            pos = "None"
        self.position_label.config(text=f"Position: {pos}")

        current_lr = self.ensemble.optimizers[0].param_groups[0]["lr"]
        self.lr_label.config(text=f"LR: {current_lr:.2e}")
        hp = self.ensemble.hp
        ind = self.ensemble.indicator_hparams
        info = (
            "Indicators:\n"
            f" ATR(p={ind.atr_period}) [{'✓' if ind.use_atr else '✗'}]\n"
            f" VORTEX(p={ind.vortex_period}) [{'✓' if ind.use_vortex else '✗'}]\n"
            f" CMF(p={ind.cmf_period}) [{'✓' if ind.use_cmf else '✗'}]\n"
            f" RSI(p={ind.rsi_period}) [{'✓' if ind.use_rsi else '✗'}]\n"
            f" SMA(p={ind.sma_period}) [{'✓' if ind.use_sma else '✗'}]\n"
            f" MACD({ind.macd_fast}/{ind.macd_slow}/{ind.macd_signal}) [{'✓' if ind.use_macd else '✗'}]\n"
            f" EMA(p={ind.ema_period}) [{'✓' if ind.use_ema else '✗'}]\n"
            f" DONCHIAN(p={ind.donchian_period}) [{'✓' if ind.use_donchian else '✗'}]\n"
            f" KIJUN(p={ind.kijun_period}) [{'✓' if ind.use_kijun else '✗'}]\n"
            f" TENKAN(p={ind.tenkan_period}) [{'✓' if ind.use_tenkan else '✗'}]\n"
            f" DISP({ind.displacement}) [{'✓' if ind.use_displacement else '✗'}]\n"
            f" ICHIMOKU [{'✓' if hp.use_ichimoku else '✗'}]\n"
            f" SL Mult: {hp.sl}\n"
            f" TP Mult: {hp.tp}\n"
            f" Long %:  {hp.long_frac * 100:.1f}\n"
            f" Short %: {hp.short_frac * 100:.1f}"
        )
        self.indicator_label.config(text=info)

        # show latest contextual features
        try:
            import artibot.feature_store as _fs

            ts = int(time.time()) // 3600 * 3600
            sent = _fs.news_sentiment(ts)
            macro = _fs.macro_surprise(ts)
            rvol = _fs.realised_vol(ts)
            ctx = f"Sent:{sent:+.2f}  Macro:{macro:+.2f}  RV:{rvol:.3f}"
            self.context_row.config(text=f"Context: {ctx}")
        except Exception:  # pragma: no cover - UI update best effort
            pass
        self.best_lr_label.config(
            text=f"Best LR: {G.global_best_lr if G.global_best_lr else 'N/A'}"
        )
        self.best_wd_label.config(
            text=f"Weight Decay: {G.global_best_wd if G.global_best_wd else 'N/A'}"
        )

        if hasattr(self, "current_sharpe_label"):
            self.current_sharpe_label.config(text=f"Sharpe: {G.global_sharpe:.2f}")
        if hasattr(self, "current_drawdown_label"):
            self.current_drawdown_label.config(
                text=f"Max DD: {G.global_max_drawdown:.3f}"
            )
        if hasattr(self, "current_netprofit_label"):
            self.current_netprofit_label.config(text=f"Net Pct: {G.global_net_pct:.2f}")
        if hasattr(self, "current_trades_label"):
            self.current_trades_label.config(text=f"Trades: {G.global_num_trades}")
        if hasattr(self, "current_inactivity_label"):
            if G.global_inactivity_penalty is not None:
                self.current_inactivity_label.config(
                    text=f"Inact: {G.global_inactivity_penalty:.2f}"
                )
            else:
                self.current_inactivity_label.config(text="Inactivity Penalty: N/A")
        if hasattr(self, "current_composite_label"):
            if G.global_composite_reward is not None:
                self.current_composite_label.config(
                    text=f"Comp: {G.global_composite_reward:.2f}"
                )
            else:
                self.current_composite_label.config(text="Current Composite: N/A")
        if hasattr(self, "current_days_profit_label"):
            if G.global_days_in_profit is not None:
                self.current_days_profit_label.config(
                    text=f"Days in Profit: {G.global_days_in_profit:.2f}"
                )
            else:
                self.current_days_profit_label.config(
                    text="Current Days in Profit: N/A"
                )
        if hasattr(self, "current_winrate_label"):
            self.current_winrate_label.config(text=f"Win Rate: {G.global_win_rate:.2f}")
        if hasattr(self, "current_profit_factor_label"):
            self.current_profit_factor_label.config(
                text=f"Profit Factor: {G.global_profit_factor:.2f}"
            )
        if hasattr(self, "current_avg_win_label"):
            self.current_avg_win_label.config(text=f"Avg Win: {G.global_avg_win:.3f}")
        if hasattr(self, "current_avg_loss_label"):
            self.current_avg_loss_label.config(
                text=f"Avg Loss: {G.global_avg_loss:.3f}"
            )

        if hasattr(self, "best_sharpe_label"):
            self.best_sharpe_label.config(
                text=f"Best Sharpe: {G.global_best_sharpe:.2f}"
            )
        if hasattr(self, "best_drawdown_label"):
            self.best_drawdown_label.config(
                text=f"Best Max DD: {G.global_best_drawdown:.3f}"
            )
        if hasattr(self, "best_netprofit_label"):
            self.best_netprofit_label.config(
                text=f"Best Net Pct: {G.global_best_net_pct:.2f}"
            )
        if hasattr(self, "best_trades_label"):
            self.best_trades_label.config(
                text=f"Best Trades: {G.global_best_num_trades}"
            )
        if hasattr(self, "best_inactivity_label"):
            if G.global_best_inactivity_penalty is not None:
                self.best_inactivity_label.config(
                    text=f"Best Inact: {G.global_best_inactivity_penalty:.2f}"
                )
            else:
                self.best_inactivity_label.config(text="Best Inactivity Penalty: N/A")
        if hasattr(self, "best_composite_label"):
            if G.global_best_composite_reward is not None:
                self.best_composite_label.config(
                    text=f"Best Comp: {G.global_best_composite_reward:.2f}"
                )
            else:
                self.best_composite_label.config(text="Best Composite: N/A")
        if hasattr(self, "best_days_profit_label"):
            if G.global_best_days_in_profit is not None:
                self.best_days_profit_label.config(
                    text=f"Best Days in Profit: {G.global_best_days_in_profit:.2f}"
                )
            else:
                self.best_days_profit_label.config(text="Best Days in Profit: N/A")
        if hasattr(self, "best_winrate_label"):
            self.best_winrate_label.config(
                text=f"Best Win Rate: {G.global_best_win_rate:.2f}"
            )
        if hasattr(self, "best_profit_factor_label"):
            self.best_profit_factor_label.config(
                text=f"Best Profit Factor: {G.global_best_profit_factor:.2f}"
            )
        if hasattr(self, "best_avg_win_label"):
            self.best_avg_win_label.config(
                text=f"Best Avg Win: {G.global_best_avg_win:.3f}"
            )
        if hasattr(self, "best_avg_loss_label"):
            self.best_avg_loss_label.config(
                text=f"Best Avg Loss: {G.global_best_avg_loss:.3f}"
            )

        aipos = self.ai_output_text.yview()
        self.ai_output_text.delete("1.0", tk.END)
        self.ai_output_text.insert(tk.END, G.global_ai_adjustments)
        self.ai_output_text.yview_moveto(aipos[0])

        log_lines = G.global_ai_adjustments_log.strip().splitlines()
        if len(log_lines) > self._log_lines:
            for line in log_lines[self._log_lines :]:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                self.ai_log_list.insert(tk.END, f"{ts} | {line}")
                self.ai_log_list.yview_moveto(1.0)
            self._log_lines = len(log_lines)

        # update status line
        primary, secondary = G.get_status_full()
        nk_state = "ARMED" if G.nuke_armed else "SAFE"
        self.status_var.set(f"{primary} | NK {nk_state}\n{secondary}")
        self.progress["value"] = G.global_progress_pct

        # manage trading buttons
        try:
            from .bot_app import CONFIG

            min_pnl = float(CONFIG.get("NK_MIN_PNL", 0.0))
            min_trades = int(CONFIG.get("NK_MIN_TRADES", 0))
        except Exception:  # pragma: no cover - CONFIG may not exist
            min_pnl = 0.0
            min_trades = 0

        live_pnl = G.live_equity - G.start_equity
        trade_cnt = G.live_trade_count

        if (
            should_enable_live_trading()
            and not G.live_trading_enabled
            and live_pnl >= min_pnl
            and trade_cnt >= min_trades
        ):
            self.nuclear_button.config(state=tk.NORMAL)
        else:
            self.nuclear_button.config(state=tk.DISABLED)
        if G.live_trading_enabled:
            self.nuclear_button.config(text="Live Trading ON")

        if G.global_validation_summary:
            sharpe = G.global_validation_summary.get("mean_sharpe", 0.0)
            enabled = G.nuclear_key_enabled
            self.validation_label.config(text=f"Val Sharpe: {sharpe:.2f} NK: {enabled}")

        # evaluate nuclear key and auto-pause rules
        allowed = nuclear_key_condition(
            G.global_sharpe, G.global_max_drawdown, G.global_profit_factor
        )
        if allowed and live_pnl >= min_pnl and trade_cnt >= min_trades:
            self.nuclear_button.config(state=tk.NORMAL)
        else:
            self.nuclear_button.config(state=tk.DISABLED)

        update_auto_pause(G.global_sharpe, G.global_max_drawdown)

        self.after_id = self.root.after(self.update_interval, self.update_dashboard)

    def update_position(self, side: str, size: float, entry: float) -> None:
        self.label_side["text"] = side
        self.label_size["text"] = f"{size:.0f}"
        self.label_entry["text"] = f"{entry:.2f}"
        if side == "NONE":
            self.btn_close["state"] = "disabled"
            self.btn_buy["state"] = "normal"
            self.btn_sell["state"] = "normal"
        else:
            self.btn_close["state"] = "normal"
            self.btn_buy["state"] = "disabled"
            self.btn_sell["state"] = "disabled"

    def refresh_stats(self) -> None:
        if not self.connector:
            return
        side, sz, entry = _fetch_position(self.connector.exchange)
        self.update_position(side, sz, entry)
        self.root.after(10000, self.refresh_stats)

    def log_trade(self, msg: str) -> None:
        """Append a trade message to the AI log list."""
        logging.info(msg)
        if hasattr(self, "ai_log_list"):
            try:
                self.ai_log_list.insert(tk.END, msg)
                self.ai_log_list.yview_moveto(1.0)
            except Exception:
                pass

    def on_test_buy(self) -> None:
        """Handle Test BUY button press."""
        logging.info("BUTTON Test BUY clicked")
        self.on_test_trade("buy")

    def on_test_sell(self) -> None:
        """Handle Test SELL button press."""
        logging.info("BUTTON Test SELL clicked")
        self.on_test_trade("sell")

    def on_test_trade(self, side: str) -> None:
        """Submit a single-contract order using the latest known price."""

        print(f"[UI] Test {side.upper()} clicked")
        try:
            if G.global_phemex_data and G.global_phemex_data[-1][4] > 0:
                price = G.global_phemex_data[-1][4]
            else:
                bars = self.connector.fetch_latest_bars(limit=1)
                price = bars[-1][4] if bars else 0.0

            order = self.connector.create_order(side, 1, price)
            print(f"[UI] Test order placed: {order}")
            self.log_trade(f"[TEST] {order}")

            close_side = "sell" if side == "buy" else "buy"

            def auto_close() -> None:
                try:
                    if G.global_phemex_data and G.global_phemex_data[-1][4] > 0:
                        close_price = G.global_phemex_data[-1][4]
                    else:
                        bars = self.connector.fetch_latest_bars(limit=1)
                        close_price = bars[-1][4] if bars else price

                    close_order = self.connector.create_order(
                        close_side, 1, close_price
                    )
                    print(f"[UI] Test close order: {close_order}")
                    self.log_trade(f"[TEST-CLOSE] {close_order}")
                except Exception as e:  # pragma: no cover - network errors
                    print(f"[UI] Test close failed: {e}")
                    self.log_trade(f"[TEST-CLOSE-ERROR] {e}")

            self.root.after(10000, auto_close)
        except Exception as e:  # pragma: no cover - network errors
            print(f"[UI] Test trade failed: {e}")
            self.log_trade(f"[TEST-ERROR] {e}")

    def enable_live_trading(self):
        """Activate live trading after user confirmation."""
        G.live_trading_enabled = True
        logging.info("LIVE_TRADING_ENABLED")
        G.set_status("Live trading enabled", "Use caution")
        self.nuclear_button.config(state=tk.DISABLED)

    def close_trade(self):
        """Signal the trading loop to close the current position."""
        logging.info("BUTTON Close Active Trade clicked")
        self.close_requested = True

    def edit_trade(self):
        """Popup dialog to adjust SL/TP multipliers."""
        logging.info("BUTTON Edit Trade clicked")
        win = tk.Toplevel(self.root)
        win.title("Edit Trade")
        ttk.Label(win, text="SL Multiplier:").grid(row=0, column=0, padx=5, pady=5)
        sl_var = tk.DoubleVar(value=G.global_SL_multiplier)
        ttk.Entry(win, textvariable=sl_var).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(win, text="TP Multiplier:").grid(row=1, column=0, padx=5, pady=5)
        tp_var = tk.DoubleVar(value=G.global_TP_multiplier)
        ttk.Entry(win, textvariable=tp_var).grid(row=1, column=1, padx=5, pady=5)

        def apply():
            G.update_trade_params(sl_var.get(), tp_var.get())
            win.destroy()

        ttk.Button(win, text="Apply", command=apply).grid(
            row=2, column=0, columnspan=2, pady=5
        )

    def adjust_cpu_limit(self) -> None:
        """Popup dialog to change the number of CPU threads."""

        logging.info("BUTTON CPU Limit clicked")
        win = tk.Toplevel(self.root)
        win.title("CPU Limit")
        ttk.Label(win, text="Threads:").grid(row=0, column=0, padx=5, pady=5)
        cpu_var = tk.IntVar(value=G.cpu_limit)
        ttk.Spinbox(
            win,
            from_=1,
            to=os.cpu_count() or 1,
            textvariable=cpu_var,
            width=5,
        ).grid(row=0, column=1, padx=5, pady=5)

        def apply() -> None:
            G.set_cpu_limit(cpu_var.get())
            win.destroy()

        ttk.Button(win, text="Apply", command=apply).grid(
            row=1, column=0, columnspan=2, pady=5
        )

    def on_toggle_force_nk(self) -> None:
        """Update ``G.nuke_armed`` from the checkbox state."""
        G.nuke_armed = bool(self.force_nk_var.get())

    def update_composite_flags(self) -> None:
        """Sync composite term toggles with ``artibot.globals``."""
        G.use_net_term = bool(self.use_net_var.get())
        G.use_sharpe_term = bool(self.use_sharpe_var.get())
        G.use_drawdown_term = bool(self.use_dd_var.get())
        G.use_trade_term = bool(self.use_trade_var.get())
        G.use_profit_days_term = bool(self.use_days_var.get())

    def manual_validate(self) -> None:
        """Run validation in a worker thread and update UI."""

        self.validation_label.config(text="Validating...")
        self.validate_button.config(state="disabled")

        def _run() -> None:
            from .bot_app import CONFIG
            from .validation import validate_and_gate

            csv_path = CONFIG.get("CSV_PATH", "")
            if not os.path.isabs(csv_path):
                here = os.path.dirname(os.path.abspath(__file__))
                csv_path = os.path.join(here, "..", csv_path)
            csv_path = os.path.abspath(os.path.expanduser(csv_path))
            try:
                G.set_status("Manual validation", "running")
                validate_and_gate(csv_path, CONFIG)
            except Exception as exc:  # pragma: no cover - validation errors
                logging.error("Manual validation failed: %s", exc)
            finally:
                G.set_status("Manual validation done", "")
                G.global_progress_pct = 0
                self.root.after(0, lambda: self.validate_button.config(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def toggle_bot(self) -> None:
        """Pause or resume all worker threads."""
        running = G.is_bot_running()
        G.set_bot_running(not running)
        new_text = "Pause Bot" if not running else "Resume Bot"
        self.run_button.config(text=new_text)

    def _build_notebook(self) -> None:
        """Create notebook with all matplotlib figures."""
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.frame_train = build_scrollable(self.notebook)
        self.notebook.add(
            self.frame_train.master.master, text="MAIN - TRAINING VS VALIDATION"
        )
        self.fig_train = plt.figure(figsize=(8, 6), constrained_layout=True)
        self.fig_train.set_constrained_layout(True)
        self.ax_loss = self.fig_train.add_subplot(4, 1, 1)
        self.ax_attention = self.fig_train.add_subplot(4, 1, 2, projection="3d")
        self.ax_equity_train = self.fig_train.add_subplot(4, 1, 3)
        self.ax_trades_time = self.fig_train.add_subplot(4, 1, 4)
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=self.frame_train)
        self.canvas_train.get_tk_widget().pack(
            fill=tk.BOTH, expand=True, padx=10, pady=10
        )
        self.loss_comment_label = ttk.Label(
            self.frame_train,
            text="",
            font=("Helvetica", 9),
            justify=tk.LEFT,
            wraplength=400,
        )
        self.loss_comment_label.pack(fill=tk.X, padx=5, pady=5)
        self.help_box = ttk.Label(
            self.frame_train,
            text=(
                "\N{INFORMATION SOURCE}  Chart legend:\n"
                "\u2022  BLUE line = training loss\n"
                "\u2022  ORANGE line = validation loss\n"
                "   \u25b8  Blue below orange \u2192 model still learning\n"
                "   \u25b8  Orange sharply below blue \u2192 possible over-fitting\n"
                "\u2022  Equity chart: red = current run, green = best run\n"
                "\u2022  Trades-over-time rising steeply \u279c more activity\n"
                "\u2022  Attention surface peaks show which past bars the model focuses on"
            ),
            justify=tk.LEFT,
            font=("Helvetica", 9),
            foreground="#444",
            anchor="w",
        )
        self.help_box.pack(side=tk.LEFT, anchor="sw", padx=5, pady=(0, 5))
        self.attention_info = ttk.Label(
            self.frame_train,
            text=(
                "This 3D surface shows which past price bars the model focuses on.\n"
                "Higher peaks mean more attention. Updated live."
            ),
            font=("Helvetica", 9),
            justify=tk.LEFT,
            wraplength=400,
        )
        self.attention_info.pack(fill=tk.X, padx=5, pady=5)

        self.frame_live = build_scrollable(self.notebook)
        self.notebook.add(self.frame_live.master.master, text="Phemex Live Price")
        self.fig_live, self.ax_live = plt.subplots(
            figsize=(8, 6), constrained_layout=True
        )
        self.fig_live.set_constrained_layout(True)
        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=self.frame_live)
        self.canvas_live.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.frame_backtest = build_scrollable(self.notebook)
        self.notebook.add(self.frame_backtest.master.master, text="Backtest Results")
        self.fig_backtest, self.ax_net_profit = plt.subplots(
            figsize=(8, 6), constrained_layout=True
        )
        self.fig_backtest.set_constrained_layout(True)
        self.canvas_backtest = FigureCanvasTkAgg(
            self.fig_backtest, master=self.frame_backtest
        )
        self.canvas_backtest.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._last_attention: np.ndarray | None = None
        self._surf = None
        self.anim_steps = 10

        self.frame_trades = build_scrollable(self.notebook)
        self.notebook.add(self.frame_trades.master.master, text="Trade Details")
        cols = ("Date", "Side", "Size", "Entry", "Exit", "PnL")
        self.trade_tree = ttk.Treeview(
            self.frame_trades, columns=cols, show="headings", height=10
        )
        for c in cols:
            self.trade_tree.heading(c, text=c)
            self.trade_tree.column(c, anchor=tk.CENTER)
        trade_scroll = ttk.Scrollbar(
            self.frame_trades, orient="vertical", command=self.trade_tree.yview
        )
        self.trade_tree.configure(yscrollcommand=trade_scroll.set)
        self.trade_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trade_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.frame_yearly_perf = build_scrollable(self.notebook)
        self.notebook.add(
            self.frame_yearly_perf.master.master, text="Best Strategy Yearly Perf"
        )
        self.yearly_perf_text = tk.Text(self.frame_yearly_perf, width=50, height=20)
        yearly_scroll = ttk.Scrollbar(
            self.frame_yearly_perf,
            orient="vertical",
            command=self.yearly_perf_text.yview,
        )
        self.yearly_perf_text.configure(yscrollcommand=yearly_scroll.set)
        self.yearly_perf_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yearly_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.frame_monthly_perf = build_scrollable(self.notebook)
        self.notebook.add(
            self.frame_monthly_perf.master.master, text="Best Strategy Monthly Results"
        )
        self.monthly_perf_text = tk.Text(self.frame_monthly_perf, width=50, height=20)
        monthly_scroll = ttk.Scrollbar(
            self.frame_monthly_perf,
            orient="vertical",
            command=self.monthly_perf_text.yview,
        )
        self.monthly_perf_text.configure(yscrollcommand=monthly_scroll.set)
        self.monthly_perf_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        monthly_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.frame_timeline = build_scrollable(self.notebook)
        self.notebook.add(self.frame_timeline.master.master, text="Activity Timeline")
        self.fig_tl, self.ax_tl = plt.subplots(figsize=(8, 6), constrained_layout=True)
        self.fig_tl.set_constrained_layout(True)
        self.canvas_tl = FigureCanvasTkAgg(self.fig_tl, master=self.frame_timeline)
        self.canvas_tl.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_sidebar(self) -> None:
        """Create sidebar widgets and AI log panels."""
        self.info_frame = ttk.LabelFrame(self.sidebar, text="Performance")
        self.info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.pred_label = ttk.Label(
            self.info_frame, text="AI Prediction: N/A", font=("Helvetica", 12)
        )
        self.pred_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.price_label = ttk.Label(
            self.info_frame, text="Live Price: N/A", font=("Helvetica", 12)
        )
        self.price_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        self.conf_label = ttk.Label(
            self.info_frame, text="Confidence: N/A", font=("Helvetica", 12)
        )
        self.conf_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)

        self.epoch_label = ttk.Label(
            self.info_frame, text="Training Steps: 0", font=("Helvetica", 12)
        )
        self.epoch_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)

        self.balance_label = ttk.Label(
            self.info_frame, text="USDT Balance: N/A", font=("Helvetica", 12)
        )
        self.balance_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.position_label = ttk.Label(
            self.info_frame, text="Position: None", font=("Helvetica", 12)
        )
        self.position_label.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

        self.current_hyper_label = ttk.Label(
            self.info_frame,
            text="Current Hyperparameters:",
            font=("Helvetica", 12, "underline"),
        )
        self.current_hyper_label.grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.lr_label = ttk.Label(
            self.info_frame, text="LR: N/A", font=("Helvetica", 12)
        )
        self.lr_label.grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)

        self.indicator_label = ttk.Label(
            self.info_frame,
            text="",
            font=("Helvetica", 12),
            justify=tk.LEFT,
        )
        self.indicator_label.grid(
            row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5
        )

        self.context_row = ttk.Label(
            self.info_frame, text="Context: N/A", font=("Helvetica", 11, "italic")
        )
        self.context_row.grid(
            row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2
        )

        # Current performance metrics
        self.current_stats_frame = ttk.LabelFrame(self.info_frame, text="Current Stats")
        self.current_stats_frame.grid(
            row=8, column=0, columnspan=2, sticky="ew", padx=5, pady=5
        )
        self.current_sharpe_label = ttk.Label(
            self.current_stats_frame, text="Sharpe: N/A"
        )
        self.current_sharpe_label.grid(row=0, column=0, sticky=tk.W)
        self.current_drawdown_label = ttk.Label(
            self.current_stats_frame, text="Max DD: N/A"
        )
        self.current_drawdown_label.grid(row=0, column=1, sticky=tk.W)
        self.current_netprofit_label = ttk.Label(
            self.current_stats_frame, text="Net Pct: N/A"
        )
        self.current_netprofit_label.grid(row=1, column=0, sticky=tk.W)
        self.current_trades_label = ttk.Label(
            self.current_stats_frame, text="Trades: N/A"
        )
        self.current_trades_label.grid(row=1, column=1, sticky=tk.W)
        self.current_days_profit_label = ttk.Label(
            self.current_stats_frame, text="Days in Profit: N/A"
        )
        self.current_days_profit_label.grid(row=2, column=0, sticky=tk.W)
        self.current_winrate_label = ttk.Label(
            self.current_stats_frame, text="Win Rate: N/A"
        )
        self.current_winrate_label.grid(row=2, column=1, sticky=tk.W)
        self.current_profit_factor_label = ttk.Label(
            self.current_stats_frame, text="Profit Factor: N/A"
        )
        self.current_profit_factor_label.grid(row=3, column=0, sticky=tk.W)
        self.current_avg_win_label = ttk.Label(
            self.current_stats_frame, text="Avg Win: N/A"
        )
        self.current_avg_win_label.grid(row=3, column=1, sticky=tk.W)
        self.current_avg_loss_label = ttk.Label(
            self.current_stats_frame, text="Avg Loss: N/A"
        )
        self.current_avg_loss_label.grid(row=4, column=0, sticky=tk.W)
        self.current_inactivity_label = ttk.Label(
            self.current_stats_frame, text="Inact: N/A"
        )
        self.current_inactivity_label.grid(row=4, column=1, sticky=tk.W)
        self.current_composite_label = ttk.Label(
            self.current_stats_frame, text="Comp: N/A"
        )
        self.current_composite_label.grid(row=5, column=0, columnspan=2, sticky=tk.W)

        # Best performance metrics
        self.best_stats_frame = ttk.LabelFrame(self.info_frame, text="Best Stats")
        self.best_stats_frame.grid(
            row=9, column=0, columnspan=2, sticky="ew", padx=5, pady=5
        )
        self.best_sharpe_label = ttk.Label(
            self.best_stats_frame, text="Best Sharpe: N/A"
        )
        self.best_sharpe_label.grid(row=0, column=0, sticky=tk.W)
        self.best_drawdown_label = ttk.Label(
            self.best_stats_frame, text="Best Max DD: N/A"
        )
        self.best_drawdown_label.grid(row=0, column=1, sticky=tk.W)
        self.best_netprofit_label = ttk.Label(
            self.best_stats_frame, text="Best Net Pct: N/A"
        )
        self.best_netprofit_label.grid(row=1, column=0, sticky=tk.W)
        self.best_trades_label = ttk.Label(
            self.best_stats_frame, text="Best Trades: N/A"
        )
        self.best_trades_label.grid(row=1, column=1, sticky=tk.W)
        self.best_days_profit_label = ttk.Label(
            self.best_stats_frame, text="Best Days in Profit: N/A"
        )
        self.best_days_profit_label.grid(row=2, column=0, sticky=tk.W)
        self.best_winrate_label = ttk.Label(
            self.best_stats_frame, text="Best Win Rate: N/A"
        )
        self.best_winrate_label.grid(row=2, column=1, sticky=tk.W)
        self.best_profit_factor_label = ttk.Label(
            self.best_stats_frame, text="Best Profit Factor: N/A"
        )
        self.best_profit_factor_label.grid(row=3, column=0, sticky=tk.W)
        self.best_avg_win_label = ttk.Label(
            self.best_stats_frame, text="Best Avg Win: N/A"
        )
        self.best_avg_win_label.grid(row=3, column=1, sticky=tk.W)
        self.best_avg_loss_label = ttk.Label(
            self.best_stats_frame, text="Best Avg Loss: N/A"
        )
        self.best_avg_loss_label.grid(row=4, column=0, sticky=tk.W)
        self.best_inactivity_label = ttk.Label(
            self.best_stats_frame, text="Best Inact: N/A"
        )
        self.best_inactivity_label.grid(row=4, column=1, sticky=tk.W)
        self.best_composite_label = ttk.Label(
            self.best_stats_frame, text="Best Comp: N/A"
        )
        self.best_composite_label.grid(row=5, column=0, columnspan=2, sticky=tk.W)

        self.best_hyper_label = ttk.Label(
            self.info_frame,
            text="Best Hyperparameters:",
            font=("Helvetica", 12, "underline"),
            foreground="darkgreen",
        )
        self.best_hyper_label.grid(row=4, column=1, sticky=tk.W, padx=5, pady=5)
        self.best_lr_label = ttk.Label(
            self.info_frame,
            text="Best LR: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_lr_label.grid(row=5, column=1, sticky=tk.W, padx=5, pady=5)

        self.best_wd_label = ttk.Label(
            self.info_frame,
            text="Weight Decay: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_wd_label.grid(row=6, column=1, sticky=tk.W, padx=5, pady=5)

        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(
            self.info_frame,
            textvariable=self.status_var,
            font=("Helvetica", 10, "italic"),
            justify=tk.LEFT,
        )
        self.status_label.grid(
            row=20, column=0, sticky=tk.W, padx=5, pady=5, columnspan=2
        )

        self.controls_frame = ttk.Frame(self.info_frame)
        self.controls_frame.grid(row=21, column=0, columnspan=2, pady=5)
        self.nuclear_button = ttk.Button(
            self.controls_frame,
            text="Nuclear Key",
            command=self.enable_live_trading,
            state=tk.DISABLED,
        )
        self.nuclear_button.pack(side=tk.LEFT, padx=5)
        self.btn_buy = ttk.Button(
            self.controls_frame,
            text="Test BUY",
            command=self.on_test_buy,
        )
        self.btn_buy.pack(side=tk.LEFT, padx=5)
        self.btn_sell = ttk.Button(
            self.controls_frame,
            text="Test SELL",
            command=self.on_test_sell,
        )
        self.btn_sell.pack(side=tk.LEFT, padx=5)
        self.btn_close = ttk.Button(
            self.controls_frame,
            text="Close Active Trade",
            command=self.close_trade,
            state="disabled",
        )
        self.btn_close.pack(side=tk.LEFT, padx=5)
        self.edit_button = ttk.Button(
            self.controls_frame,
            text="Edit Trade",
            command=self.edit_trade,
        )
        self.edit_button.pack(side=tk.LEFT, padx=5)
        self.validate_button = ttk.Button(
            self.controls_frame,
            text="Manual Validate",
            command=self.manual_validate,
        )
        self.validate_button.pack(side=tk.LEFT, padx=5)
        self.run_button = ttk.Button(
            self.controls_frame,
            text="Pause Bot",
            command=self.toggle_bot,
        )
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.cpu_button = ttk.Button(
            self.controls_frame,
            text="CPU Limit",
            command=self.adjust_cpu_limit,
        )
        self.cpu_button.pack(side=tk.LEFT, padx=5)
        self.force_nk_var = tk.BooleanVar(value=False)
        self.force_nk_chk = ttk.Checkbutton(
            self.controls_frame,
            text="Bypass NK",
            variable=self.force_nk_var,
            command=self.on_toggle_force_nk,
        )
        self.force_nk_chk.pack(side=tk.LEFT, padx=5)
        self.on_toggle_force_nk()

        self.validation_label = ttk.Label(
            self.info_frame, text="Validation: N/A", font=("Helvetica", 12)
        )
        self.validation_label.grid(
            row=22, column=0, sticky=tk.W, padx=5, pady=5, columnspan=2
        )

        self.pos_frame = ttk.LabelFrame(self.info_frame, text="Current Position")
        self.pos_frame.grid(row=23, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Label(self.pos_frame, text="Side:").grid(row=0, column=0, sticky=tk.W)
        self.label_side = ttk.Label(self.pos_frame, text="NONE")
        self.label_side.grid(row=0, column=1, sticky=tk.W)
        ttk.Label(self.pos_frame, text="Size:").grid(row=1, column=0, sticky=tk.W)
        self.label_size = ttk.Label(self.pos_frame, text="0")
        self.label_size.grid(row=1, column=1, sticky=tk.W)
        ttk.Label(self.pos_frame, text="Entry:").grid(row=2, column=0, sticky=tk.W)
        self.label_entry = ttk.Label(self.pos_frame, text="0.0 USDT")
        self.label_entry.grid(row=2, column=1, sticky=tk.W)
        ttk.Label(self.pos_frame, text="Long Exposure:").grid(
            row=3, column=0, sticky=tk.W
        )
        self.label_long = ttk.Label(self.pos_frame, text="0 USD")
        self.label_long.grid(row=3, column=1, sticky=tk.W)
        ttk.Label(self.pos_frame, text="Short Exposure:").grid(
            row=4, column=0, sticky=tk.W
        )
        self.label_short = ttk.Label(self.pos_frame, text="0 USD")
        self.label_short.grid(row=4, column=1, sticky=tk.W)

        self.comp_frame = ttk.LabelFrame(self.info_frame, text="Composite Terms")
        self.comp_frame.grid(
            row=24, column=0, columnspan=2, sticky="ew", padx=5, pady=5
        )
        self.use_net_var = tk.BooleanVar(value=G.use_net_term)
        self.use_sharpe_var = tk.BooleanVar(value=G.use_sharpe_term)
        self.use_dd_var = tk.BooleanVar(value=G.use_drawdown_term)
        self.use_trade_var = tk.BooleanVar(value=G.use_trade_term)
        self.use_days_var = tk.BooleanVar(value=G.use_profit_days_term)
        ttk.Checkbutton(
            self.comp_frame,
            text="Net%",
            variable=self.use_net_var,
            command=self.update_composite_flags,
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(
            self.comp_frame,
            text="Sharpe",
            variable=self.use_sharpe_var,
            command=self.update_composite_flags,
        ).grid(row=0, column=1, sticky=tk.W)
        ttk.Checkbutton(
            self.comp_frame,
            text="Drawdown",
            variable=self.use_dd_var,
            command=self.update_composite_flags,
        ).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(
            self.comp_frame,
            text="Trades",
            variable=self.use_trade_var,
            command=self.update_composite_flags,
        ).grid(row=1, column=1, sticky=tk.W)
        ttk.Checkbutton(
            self.comp_frame,
            text="Profit Days",
            variable=self.use_days_var,
            command=self.update_composite_flags,
        ).grid(row=2, column=0, sticky=tk.W)
        self.update_composite_flags()

        self.frame_ai = ttk.Frame(self.sidebar)
        self.frame_ai.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ai_output_label = ttk.Label(
            self.frame_ai, text="Latest AI Adjustments:", font=("Helvetica", 12, "bold")
        )
        self.ai_output_label.pack(anchor="n")
        self.ai_output_text = tk.Text(self.frame_ai, width=40, height=10, wrap="word")
        ai_scroll = ttk.Scrollbar(
            self.frame_ai, orient="vertical", command=self.ai_output_text.yview
        )
        self.ai_output_text.configure(yscrollcommand=ai_scroll.set)
        self.ai_output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ai_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.frame_ai_log = ttk.Frame(self.sidebar)
        self.frame_ai_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ai_log_label = ttk.Label(
            self.frame_ai_log,
            text="AI Adjustments Log:",
            font=("Helvetica", 12, "bold"),
        )
        self.ai_log_label.pack(anchor="n")
        self.ai_log_list = tk.Listbox(self.frame_ai_log, width=40, height=10)
        self.ai_log_list.pack(fill=tk.BOTH, expand=True)
        self._log_lines = 0

    def _build_footer(self) -> None:
        """Create footer widgets like status bar and progress."""
        disclaimer_text = "NOT INVESTMENT ADVICE! Demo only."
        self.disclaimer_label = ttk.Label(
            self.footer,
            text=disclaimer_text,
            font=("Helvetica", 9, "italic"),
            foreground="darkred",
        )
        self.disclaimer_label.pack(side=tk.LEFT, padx=5)

        self.weights_label = ttk.Label(
            self.footer,
            text=f"Weights: {os.path.basename(self.weights_path) if self.weights_path else 'N/A'}",
            font=("Helvetica", 9, "italic"),
        )
        self.weights_label.pack(side=tk.RIGHT, padx=5)

        self.progress = ttk.Progressbar(
            self.footer, mode="determinate", length=150, maximum=100
        )
        self.progress.pack(side=tk.LEFT, padx=5)
