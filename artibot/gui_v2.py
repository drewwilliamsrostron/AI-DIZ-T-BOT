"""Modernized Artibot GUI using ttk widgets and dark mode."""

from __future__ import annotations

import datetime as _dt
import logging
import os
import threading
from typing import Optional

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import artibot.globals as G

from .live_risk import update_auto_pause
from .metrics import nuclear_key_condition

if hasattr(plt, "style"):
    plt.style.use("dark_background")

GUI_INSTANCE: Optional["TradingGUI"] = None


# ---------------------------------------------------------------------------
# Helper dialogs and utilities
# ---------------------------------------------------------------------------


def format_trade_details(trades: list[dict], limit: int = 50) -> str:
    """Return a compact table string for the most recent trades."""
    if not trades:
        return "No Trade Details"

    if pd is None:
        lines = []
        for tr in trades[-limit:]:
            lines.append(
                f"{tr.get('entry_time', 0)} {tr.get('side','')} {tr.get('return',0):.2f}"
            )
        return "\n".join(lines) if lines else "No Trade Details"

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
    except Exception:
        min_pnl = 0.0
        min_trades = 0

    return sharpe >= 1.0 and dd >= -0.30 and pnl >= min_pnl and trades >= min_trades


def select_weight_file(use_prev: bool = True) -> str | None:
    """Return the weight file path based on user selection."""
    from tkinter import filedialog, messagebox

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
        import tkinter as tk_module
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
    defaults: dict[str, object] | None = None, tk_module=None
) -> dict[str, object]:
    """Return user-selected startup options via a simple Tk dialog."""
    if tk_module is None:
        import tkinter as tk_module

    defaults = defaults or {}
    result: dict[str, object] = {}
    try:
        root = tk_module.Tk()
    except Exception as exc:  # pragma: no cover
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
    risk_var = tk_module.BooleanVar(value=bool(defaults.get("risk_filter", True)))
    net_var = tk_module.BooleanVar(value=bool(defaults.get("use_net_term", True)))
    sharpe_var = tk_module.BooleanVar(value=bool(defaults.get("use_sharpe_term", True)))
    dd_var = tk_module.BooleanVar(value=bool(defaults.get("use_drawdown_term", True)))
    trade_var = tk_module.BooleanVar(value=bool(defaults.get("use_trade_term", True)))
    days_var = tk_module.BooleanVar(
        value=bool(defaults.get("use_profit_days_term", True))
    )

    tk_module.Checkbutton(
        root, text="Skip GDELT sentiment download", variable=skip_var
    ).pack(anchor="w")
    tk_module.Checkbutton(root, text="Enable LIVE trading", variable=live_var).pack(
        anchor="w"
    )
    tk_module.Checkbutton(
        root, text="Load previous weights", variable=weights_var
    ).pack(anchor="w")
    tk_module.Checkbutton(root, text="Enable risk filter", variable=risk_var).pack(
        anchor="w"
    )
    tk_module.Label(root, text="Reward terms:").pack(anchor="w")
    tk_module.Checkbutton(root, text="Net%", variable=net_var).pack(anchor="w")
    tk_module.Checkbutton(root, text="Sharpe", variable=sharpe_var).pack(anchor="w")
    tk_module.Checkbutton(root, text="Drawdown", variable=dd_var).pack(anchor="w")
    tk_module.Checkbutton(root, text="Trades", variable=trade_var).pack(anchor="w")
    tk_module.Checkbutton(root, text="Profit Days", variable=days_var).pack(anchor="w")
    tk_module.Label(root, text="CPU threads:").pack(anchor="w")
    tk_module.Spinbox(
        root, from_=1, to=threads_max, textvariable=threads_var, width=5
    ).pack(anchor="w")

    def cont() -> None:
        result["skip_sentiment"] = skip_var.get()
        result["use_live"] = live_var.get()
        result["use_prev_weights"] = weights_var.get()
        result["threads"] = threads_var.get()

        result["risk_filter"] = risk_var.get()
        result["use_net_term"] = net_var.get()
        result["use_sharpe_term"] = sharpe_var.get()
        result["use_drawdown_term"] = dd_var.get()
        result["use_trade_term"] = trade_var.get()
        result["use_profit_days_term"] = days_var.get()

        root.quit()
        root.destroy()

    tk_module.Button(root, text="Continue", command=cont).pack(pady=5)
    root.mainloop()
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
    except Exception as e:  # pragma: no cover
        logging.error("Position fetch failed: %s", e)
    return ("NONE", 0.0, 0.0)


# ---------------------------------------------------------------------------
# Main GUI class
# ---------------------------------------------------------------------------


class TradingGUI:
    """Tkinter dashboard showing training progress and live stats."""

    def __init__(
        self,
        root: tk.Tk,
        ensemble: Optional[object] = None,
        weights_path: str | None = None,
        connector=None,
        *,
        dev: bool = False,
    ) -> None:
        global GUI_INSTANCE
        GUI_INSTANCE = self

        self.root = root
        self.ensemble = ensemble
        self.weights_path = weights_path
        self.connector = connector
        self.dev = dev

        # scale according to DPI
        scale = max(1.0, min(1.5, root.winfo_fpixels("1i") / 96))
        root.tk.call("tk", "scaling", scale)
        G.UI_SCALE = scale

        root.title("Artibot Dashboard")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#222")
        style.configure("TLabel", background="#222", foreground="white")
        style.configure("TLabelframe", background="#222", foreground="white")
        style.configure("TLabelframe.Label", background="#222", foreground="white")
        root.configure(bg="#222")

        self._init_layout()
        self._last_attention: Optional[np.ndarray] = None
        self.after_id: Optional[str] = None
        self.update_interval = 1000
        self.root.after_idle(self.update_dashboard)
        # poll for refresh events triggered by worker threads
        self.root.after(100, self._poll_gui_event)
        self.root.after(10000, self.refresh_stats)

    def set_ensemble(self, ensemble: object, weights_path: str | None = None) -> None:
        """Attach ``ensemble`` to the GUI and update labels."""
        self.ensemble = ensemble
        if weights_path is not None:
            self.weights_path = weights_path
            basename = os.path.basename(weights_path)
            self.weights_label.config(text=f"Weights: {basename}")

    # ------------------------------------------------------------------
    # Layout builders
    # ------------------------------------------------------------------
    def _init_layout(self) -> None:
        """Create widgets and figures."""
        root = self.root
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)
        root.rowconfigure(1, weight=0)

        self.main = ttk.Frame(root)
        self.main.grid(row=0, column=0, sticky="nsew")
        self.sidebar = ttk.Frame(root)
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        self.footer = ttk.Frame(root)
        self.footer.grid(row=1, column=0, columnspan=2, sticky="ew")

        self._build_notebook()
        self._build_sidebar()
        self._build_footer()

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self.main)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Training page
        self.frame_train = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_train, text="Training")

        self.fig_train, axs = plt.subplots(
            2, 2, figsize=(8, 6), constrained_layout=True
        )

        self.ax_loss = axs[0, 0]
        self.ax_equity = axs[0, 1]
        self.ax_attention = axs[1, 0]
        self.ax_trades = axs[1, 1]
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=self.frame_train)
        self.canvas_train.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.loss_comment_label = ttk.Label(self.frame_train, text="")
        self.loss_comment_label.pack(anchor="w", padx=5, pady=2)

        # Live price
        self.frame_live = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_live, text="Live")

        self.fig_live, self.ax_live = plt.subplots(
            figsize=(8, 4), constrained_layout=True
        )

        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=self.frame_live)
        self.canvas_live.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Backtest page
        self.frame_back = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_back, text="Backtest")

        self.fig_back, self.ax_net = plt.subplots(
            figsize=(8, 4), constrained_layout=True
        )

        self.canvas_back = FigureCanvasTkAgg(self.fig_back, master=self.frame_back)
        self.canvas_back.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Timeline page
        self.frame_tl = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_tl, text="Timeline")
        self.fig_tl, self.ax_tl = plt.subplots(figsize=(8, 4), constrained_layout=True)
        self.canvas_tl = FigureCanvasTkAgg(self.fig_tl, master=self.frame_tl)
        self.canvas_tl.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Trade details
        self.frame_trades = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_trades, text="Trades")
        cols = ("Date", "Side", "Size", "Entry", "Exit", "PnL")

        self.trade_tree = ttk.Treeview(
            self.frame_trades, columns=cols, show="headings", height=10
        )
        for c in cols:
            self.trade_tree.heading(c, text=c)
            self.trade_tree.column(c, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(
            self.frame_trades, orient="vertical", command=self.trade_tree.yview
        )

        self.trade_tree.configure(yscrollcommand=vsb.set)
        self.trade_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Yearly page
        self.frame_yearly = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_yearly, text="Yearly")
        self.yearly_text = tk.Text(self.frame_yearly, width=50, height=20)

        yscroll = ttk.Scrollbar(
            self.frame_yearly, orient="vertical", command=self.yearly_text.yview
        )

        self.yearly_text.configure(yscrollcommand=yscroll.set)
        self.yearly_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Monthly page
        self.frame_monthly = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_monthly, text="Monthly")
        self.monthly_text = tk.Text(self.frame_monthly, width=50, height=20)

        mscroll = ttk.Scrollbar(
            self.frame_monthly, orient="vertical", command=self.monthly_text.yview
        )

        self.monthly_text.configure(yscrollcommand=mscroll.set)
        self.monthly_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        mscroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_sidebar(self) -> None:
        self.info = ttk.LabelFrame(self.sidebar, text="Performance")
        self.info.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.pred_label = ttk.Label(self.info, text="AI Prediction: N/A")
        self.pred_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.price_label = ttk.Label(self.info, text="Live Price: N/A")
        self.price_label.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        self.conf_label = ttk.Label(self.info, text="Confidence: N/A")
        self.conf_label.grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.epoch_label = ttk.Label(self.info, text="Training Steps: 0")
        self.epoch_label.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self.balance_label = ttk.Label(self.info, text="USDT Balance: N/A")
        self.balance_label.grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.position_label = ttk.Label(self.info, text="Position: None")
        self.position_label.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        self.best_lr_label = ttk.Label(self.info, text="Best LR: N/A")
        self.best_lr_label.grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.best_wd_label = ttk.Label(self.info, text="Weight Decay: N/A")
        self.best_wd_label.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        self.current_stats = ttk.LabelFrame(self.info, text="Current Stats")
        self.current_stats.grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5
        )
        self.current_sharpe_label = ttk.Label(self.current_stats, text="Sharpe: N/A")
        self.current_sharpe_label.grid(row=0, column=0, sticky="w")
        self.current_drawdown_label = ttk.Label(self.current_stats, text="Max DD: N/A")
        self.current_drawdown_label.grid(row=0, column=1, sticky="w")
        self.current_netprofit_label = ttk.Label(
            self.current_stats, text="Net Pct: N/A"
        )
        self.current_netprofit_label.grid(row=1, column=0, sticky="w")
        self.current_trades_label = ttk.Label(self.current_stats, text="Trades: N/A")
        self.current_trades_label.grid(row=1, column=1, sticky="w")
        self.current_days_profit_label = ttk.Label(
            self.current_stats, text="Days in Profit: N/A"
        )
        self.current_days_profit_label.grid(row=2, column=0, sticky="w")
        self.current_winrate_label = ttk.Label(self.current_stats, text="Win Rate: N/A")
        self.current_winrate_label.grid(row=2, column=1, sticky="w")
        self.current_profit_factor_label = ttk.Label(
            self.current_stats, text="Profit Factor: N/A"
        )
        self.current_profit_factor_label.grid(row=3, column=0, sticky="w")
        self.current_avg_win_label = ttk.Label(self.current_stats, text="Avg Win: N/A")
        self.current_avg_win_label.grid(row=3, column=1, sticky="w")
        self.current_avg_loss_label = ttk.Label(
            self.current_stats, text="Avg Loss: N/A"
        )
        self.current_avg_loss_label.grid(row=4, column=0, sticky="w")
        self.current_inactivity_label = ttk.Label(self.current_stats, text="Inact: N/A")
        self.current_inactivity_label.grid(row=4, column=1, sticky="w")
        self.current_composite_label = ttk.Label(self.current_stats, text="Comp: N/A")
        self.current_composite_label.grid(row=5, column=0, columnspan=2, sticky="w")

        self.best_stats = ttk.LabelFrame(self.info, text="Best Stats")
        self.best_stats.grid(row=5, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.best_drawdown_label = ttk.Label(self.best_stats, text="Best Max DD: N/A")
        self.best_drawdown_label.grid(row=0, column=0, sticky="w")
        self.best_netprofit_label = ttk.Label(self.best_stats, text="Best Net Pct: N/A")
        self.best_netprofit_label.grid(row=1, column=0, sticky="w")
        self.best_trades_label = ttk.Label(self.best_stats, text="Best Trades: N/A")
        self.best_trades_label.grid(row=1, column=1, sticky="w")
        self.best_days_profit_label = ttk.Label(
            self.best_stats, text="Best Days in Profit: N/A"
        )
        self.best_days_profit_label.grid(row=2, column=0, sticky="w")
        self.best_winrate_label = ttk.Label(self.best_stats, text="Best Win Rate: N/A")
        self.best_winrate_label.grid(row=2, column=1, sticky="w")
        self.best_profit_factor_label = ttk.Label(
            self.best_stats, text="Best Profit Factor: N/A"
        )
        self.best_profit_factor_label.grid(row=3, column=0, sticky="w")
        self.best_avg_win_label = ttk.Label(self.best_stats, text="Best Avg Win: N/A")
        self.best_avg_win_label.grid(row=3, column=1, sticky="w")
        self.best_avg_loss_label = ttk.Label(self.best_stats, text="Best Avg Loss: N/A")
        self.best_avg_loss_label.grid(row=4, column=0, sticky="w")
        self.best_inactivity_label = ttk.Label(self.best_stats, text="Best Inact: N/A")
        self.best_inactivity_label.grid(row=4, column=1, sticky="w")
        self.best_composite_label = ttk.Label(self.best_stats, text="Best Comp: N/A")
        self.best_composite_label.grid(row=5, column=0, columnspan=2, sticky="w")

        self.validation_label = ttk.Label(self.info, text="Validation: N/A")
        self.validation_label.grid(
            row=6, column=0, columnspan=2, sticky="w", padx=5, pady=2
        )

        self.pos_frame = ttk.LabelFrame(self.info, text="Current Position")
        self.pos_frame.grid(row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        ttk.Label(self.pos_frame, text="Side:").grid(row=0, column=0, sticky="w")
        self.label_side = ttk.Label(self.pos_frame, text="NONE")
        self.label_side.grid(row=0, column=1, sticky="w")
        ttk.Label(self.pos_frame, text="Size:").grid(row=1, column=0, sticky="w")
        self.label_size = ttk.Label(self.pos_frame, text="0")
        self.label_size.grid(row=1, column=1, sticky="w")
        ttk.Label(self.pos_frame, text="Entry:").grid(row=2, column=0, sticky="w")
        self.label_entry = ttk.Label(self.pos_frame, text="0.0")
        self.label_entry.grid(row=2, column=1, sticky="w")
        ttk.Label(self.pos_frame, text="Long Exposure:").grid(
            row=3, column=0, sticky="w"
        )
        self.label_long = ttk.Label(self.pos_frame, text="0")
        self.label_long.grid(row=3, column=1, sticky="w")
        ttk.Label(self.pos_frame, text="Short Exposure:").grid(
            row=4, column=0, sticky="w"
        )
        self.label_short = ttk.Label(self.pos_frame, text="0")
        self.label_short.grid(row=4, column=1, sticky="w")

        self.comp_frame = ttk.LabelFrame(self.info, text="Composite Terms")
        self.comp_frame.grid(row=8, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
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
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            self.comp_frame,
            text="Sharpe",
            variable=self.use_sharpe_var,
            command=self.update_composite_flags,
        ).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(
            self.comp_frame,
            text="Drawdown",
            variable=self.use_dd_var,
            command=self.update_composite_flags,
        ).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(
            self.comp_frame,
            text="Trades",
            variable=self.use_trade_var,
            command=self.update_composite_flags,
        ).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(
            self.comp_frame,
            text="Profit Days",
            variable=self.use_days_var,
            command=self.update_composite_flags,
        ).grid(row=2, column=0, sticky="w")
        self.update_composite_flags()

        self.ai_frame = ttk.Frame(self.sidebar)
        self.ai_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(self.ai_frame, text="Latest AI Adjustments:").pack(anchor="nw")
        self.ai_output_text = tk.Text(self.ai_frame, width=40, height=8, wrap="word")
        ai_scroll = ttk.Scrollbar(
            self.ai_frame, orient="vertical", command=self.ai_output_text.yview
        )
        self.ai_output_text.configure(yscrollcommand=ai_scroll.set)
        self.ai_output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ai_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.ai_log_frame = ttk.Frame(self.sidebar)
        self.ai_log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(self.ai_log_frame, text="AI Adjustments Log:").pack(anchor="nw")
        self.ai_log_list = tk.Listbox(self.ai_log_frame, width=40, height=8)
        self.ai_log_list.pack(fill=tk.BOTH, expand=True)
        self._log_lines = 0

        # Buttons
        self.controls = ttk.Frame(self.sidebar)
        self.controls.pack(fill=tk.X, padx=5, pady=5)
        self.nuclear_button = ttk.Button(
            self.controls, text="Enable Live Trading", command=self.enable_live_trading
        )
        self.nuclear_button.pack(side=tk.LEFT, padx=2)
        self.btn_buy = ttk.Button(
            self.controls, text="Test BUY", command=self.on_test_buy
        )
        self.btn_buy.pack(side=tk.LEFT, padx=2)
        self.btn_sell = ttk.Button(
            self.controls, text="Test SELL", command=self.on_test_sell
        )
        self.btn_sell.pack(side=tk.LEFT, padx=2)
        self.btn_close = ttk.Button(
            self.controls,
            text="Close Trade",
            command=self.close_trade,
            state="disabled",
        )
        self.btn_close.pack(side=tk.LEFT, padx=2)
        self.edit_button = ttk.Button(
            self.controls, text="Edit Trade", command=self.edit_trade
        )
        self.edit_button.pack(side=tk.LEFT, padx=2)
        self.validate_button = ttk.Button(
            self.controls, text="Manual Validate", command=self.manual_validate
        )
        self.validate_button.pack(side=tk.LEFT, padx=2)
        self.run_button = ttk.Button(
            self.controls, text="Pause Bot", command=self.toggle_bot
        )
        self.run_button.pack(side=tk.LEFT, padx=2)
        self.cpu_button = ttk.Button(
            self.controls, text="CPU Limit", command=self.adjust_cpu_limit
        )
        self.cpu_button.pack(side=tk.LEFT, padx=2)
        if self.dev:
            self.force_nk_var = tk.BooleanVar(value=False)
            self.force_nk_chk = ttk.Checkbutton(
                self.controls,
                text="Bypass NK",
                variable=self.force_nk_var,
                command=self.on_toggle_force_nk,
            )
            self.force_nk_chk.pack(side=tk.LEFT, padx=2)

    def _build_footer(self) -> None:
        disclaimer = ttk.Label(
            self.footer, text="NOT INVESTMENT ADVICE!", foreground="orange"
        )

        disclaimer.pack(side=tk.LEFT, padx=5)
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self.footer, textvariable=self.status_var)
        status.pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(
            self.footer, mode="determinate", maximum=100, length=150
        )

        self.progress.pack(side=tk.RIGHT, padx=5)
        self.weights_label = ttk.Label(
            self.footer,
            text=f"Weights: {os.path.basename(self.weights_path) if self.weights_path else 'N/A'}",
        )
        self.weights_label.pack(side=tk.RIGHT, padx=5)

    # ------------------------------------------------------------------
    # Data refresh utilities
    # ------------------------------------------------------------------
    def _loss_comment(self) -> str:
        train = list(G.global_training_loss)
        val = [v for v in G.global_validation_loss if v is not None]
        if not train or not val:
            return "Waiting for validation data..."
        if val[-1] > train[-1]:
            return "validation loss above training - watch for overfitting"
        return "training loss above validation - model learning"

    def _poll_gui_event(self) -> None:
        """Update dashboard when the worker event is set."""
        if G.gui_event.is_set():
            G.gui_event.clear()
            self.root.after_idle(self.update_dashboard)
        self.root.after(100, self._poll_gui_event)

    def update_dashboard(self) -> None:  # noqa: C901 - full dashboard update
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        # Loss and equity
        self.ax_loss.clear()
        n = max(len(G.global_training_loss), len(G.global_validation_loss))
        x = range(1, n + 1)

        train_vals = list(G.global_training_loss)
        if pd is not None and train_vals:
            train_vals = pd.Series(train_vals).ewm(span=10).mean().tolist()
        train_vals = train_vals[:n]
        self.ax_loss.plot(x[: len(train_vals)], train_vals, label="Train", marker="o")

        val_vals = [np.nan if v is None else v for v in G.global_validation_loss]
        val_vals = val_vals[:n]
        if pd is not None and any(not np.isnan(v) for v in val_vals):
            val_vals = pd.Series(val_vals).ewm(span=10).mean().tolist()
        if any(not np.isnan(v) for v in val_vals):
            self.ax_loss.plot(x[: len(val_vals)], val_vals, label="Val", marker="x")

        all_losses = [v for v in train_vals if not np.isnan(v)] + [
            v for v in val_vals if not np.isnan(v)
        ]
        if all_losses:
            self.ax_loss.set_ylim(min(all_losses), max(all_losses))
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
        self.ax_loss.set_title("Loss")
        handles, labels = self.ax_loss.get_legend_handles_labels()
        if labels:
            self.ax_loss.legend(handles, labels)
        self.loss_comment_label.config(text=self._loss_comment())

        self.ax_equity.clear()
        self.ax_equity.set_title("Equity")
        eq = G.global_equity_curve
        if eq:
            ts, bal = zip(*eq)

            ts_dt = [_dt.datetime.fromtimestamp(t) for t in ts]
            self.ax_equity.plot(ts_dt, bal, color="red", label="Current")
        if G.global_best_equity_curve:
            ts, bal = zip(*G.global_best_equity_curve)
            ts_dt = [_dt.datetime.fromtimestamp(t) for t in ts]

            self.ax_equity.plot(ts_dt, bal, color="green", label="Best")
        handles, labels = self.ax_equity.get_legend_handles_labels()
        if labels:
            self.ax_equity.legend(handles, labels)

        # Attention
        self.ax_attention.clear()
        if G.global_last_attention is not None:
            data = np.array(G.global_last_attention)
            avg = data.mean(axis=0) if data.ndim == 3 else data
            self.ax_attention.imshow(avg, aspect="auto", origin="lower")
        elif G.global_attention_weights_history:
            x = range(1, len(G.global_attention_weights_history) + 1)
            self.ax_attention.plot(x, G.global_attention_weights_history)
        self.ax_attention.set_title("Attention")

        # Trades over time
        self.ax_trades.clear()
        if G.global_trade_details:
            ts = [t.get("entry_time", 0) for t in G.global_trade_details]
            ts = [t // 1000 if t > 1_000_000_000_000 else t for t in ts]
            ts_dt = [_dt.datetime.fromtimestamp(t) for t in ts]
            counts = list(range(1, len(ts_dt) + 1))
            self.ax_trades.step(ts_dt, counts, where="post")
        self.ax_trades.set_title("Trades")

        self.canvas_train.draw()

        # Live price
        self.ax_live.clear()
        if G.global_phemex_data:

            times = [
                _dt.datetime.fromtimestamp(b[0] / 1000)
                for b in G.global_phemex_data
                if b
            ]

            closes = [b[4] for b in G.global_phemex_data if b]
            self.ax_live.plot(times, closes, marker="o")
        self.ax_live.set_title("Live Price")
        self.canvas_live.draw()

        # Backtest profit
        self.ax_net.clear()
        if G.global_backtest_profit:
            x = range(1, len(G.global_backtest_profit) + 1)
            self.ax_net.plot(x, G.global_backtest_profit, marker="o", color="green")
        self.ax_net.set_title("Net Profit (%)")
        self.canvas_back.draw()

        # Timeline
        self.ax_tl.clear()
        depth = G.timeline_depth
        idx = G.timeline_index
        xs = np.arange(depth)
        inds = np.roll(G.timeline_ind_on, -idx, axis=0)
        trade = np.roll(G.timeline_trades, -idx)
        for k, name in enumerate(["EMA", "SMA", "RSI", "KIJ", "TEN", "DISP"]):
            self.ax_tl.plot(xs, k + inds[:, k] * 0.8, lw=0.8, label=name)
        self.ax_tl.plot(xs, 6 + trade * 0.8, lw=1.1, c="white", label="Trade ON")
        self.ax_tl.set_ylim(-0.5, 7.5)
        self.ax_tl.set_yticks([])
        if idx > 0:
            handles, labels = self.ax_tl.get_legend_handles_labels()
            if labels:
                self.ax_tl.legend(handles, labels, ncol=4, fontsize=6, framealpha=0.3)
        self.ax_tl.set_title("Indicators / Trade")
        self.canvas_tl.draw()

        # Trade table
        for row in self.trade_tree.get_children():
            self.trade_tree.delete(row)
        if G.global_trade_details:
            recent = G.global_trade_details[-50:]
            for tr in recent:
                dt = _dt.datetime.fromtimestamp(tr.get("entry_time", 0))
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

        # Text pages
        self.yearly_text.delete("1.0", tk.END)
        table = G.global_best_yearly_stats_table or G.global_yearly_stats_table
        if table:
            self.yearly_text.insert(tk.END, table)
        else:
            self.yearly_text.insert(tk.END, "No yearly data")
        self.monthly_text.delete("1.0", tk.END)
        table_m = G.global_best_monthly_stats_table or G.global_monthly_stats_table
        if table_m:
            self.monthly_text.insert(tk.END, table_m)
        else:
            self.monthly_text.insert(tk.END, "No monthly data")

        # Stats labels
        pred_str = G.global_current_prediction or "N/A"
        color_map = {"BUY": "lightgreen", "SELL": "red", "HOLD": "white"}

        self.pred_label.config(
            text=f"AI Prediction: {pred_str}",
            foreground=color_map.get(pred_str.upper(), "white"),
        )

        price = 0.0
        if G.global_phemex_data and len(G.global_phemex_data[-1]) >= 5:
            price = float(G.global_phemex_data[-1][4])
        self.price_label.config(text=f"Live Price: {price:.2f}")
        conf = G.global_ai_confidence or 0.0
        self.conf_label.config(text=f"Confidence: {conf:.2f}")
        self.epoch_label.config(text=f"Training Steps: {G.epoch_count}")
        bal = G.global_account_stats.get("total", {}).get("USDT", 0.0)
        self.balance_label.config(text=f"USDT Balance: {bal:.2f}")
        if G.global_position_side:
            pos = f"{G.global_position_side} {G.global_position_size:.4f}"
        else:
            pos = "None"
        self.position_label.config(text=f"Position: {pos}")

        if self.ensemble is not None:
            current_lr = self.ensemble.optimizers[0].param_groups[0]["lr"]
        else:
            current_lr = 0.0

        self.best_lr_label.config(
            text=f"Best LR: {G.global_best_lr if G.global_best_lr else current_lr:.2e}"
        )
        self.best_wd_label.config(
            text=f"Weight Decay: {G.global_best_wd if G.global_best_wd else 'N/A'}"
        )

        if G.global_validation_summary:
            sharpe = G.global_validation_summary.get("mean_sharpe", 0.0)
            enabled = G.nuclear_key_enabled
            self.validation_label.config(text=f"Val Sharpe: {sharpe:.2f} NK: {enabled}")

        self.current_sharpe_label.config(text=f"Sharpe: {G.global_sharpe:.2f}")
        self.current_drawdown_label.config(text=f"Max DD: {G.global_max_drawdown:.3f}")
        self.current_netprofit_label.config(text=f"Net Pct: {G.global_net_pct:.2f}")
        self.current_trades_label.config(text=f"Trades: {G.global_num_trades}")
        if G.global_inactivity_penalty is not None:
            self.current_inactivity_label.config(
                text=f"Inact: {G.global_inactivity_penalty:.2f}"
            )
        else:
            self.current_inactivity_label.config(text="Inactivity Penalty: N/A")
        if G.global_composite_reward is not None:
            self.current_composite_label.config(
                text=f"Comp: {G.global_composite_reward:.2f}"
            )
        else:
            self.current_composite_label.config(text="Current Composite: N/A")
        if G.global_days_in_profit is not None:
            self.current_days_profit_label.config(
                text=f"Days in Profit: {G.global_days_in_profit:.2f}"
            )
        else:
            self.current_days_profit_label.config(text="Current Days in Profit: N/A")
        self.current_winrate_label.config(text=f"Win Rate: {G.global_win_rate:.2f}")
        self.current_profit_factor_label.config(
            text=f"Profit Factor: {G.global_profit_factor:.2f}"
        )
        self.current_avg_win_label.config(text=f"Avg Win: {G.global_avg_win:.3f}")
        self.current_avg_loss_label.config(text=f"Avg Loss: {G.global_avg_loss:.3f}")

        self.best_drawdown_label.config(
            text=f"Best Max DD: {G.global_best_drawdown:.3f}"
        )
        self.best_netprofit_label.config(
            text=f"Best Net Pct: {G.global_best_net_pct:.2f}"
        )
        self.best_trades_label.config(text=f"Best Trades: {G.global_best_num_trades}")
        if G.global_best_inactivity_penalty is not None:
            self.best_inactivity_label.config(
                text=f"Best Inact: {G.global_best_inactivity_penalty:.2f}"
            )
        else:
            self.best_inactivity_label.config(text="Best Inactivity Penalty: N/A")
        if G.global_best_composite_reward is not None:
            self.best_composite_label.config(
                text=f"Best Comp: {G.global_best_composite_reward:.2f}"
            )
        else:
            self.best_composite_label.config(text="Best Composite: N/A")
        if G.global_best_days_in_profit is not None:
            self.best_days_profit_label.config(
                text=f"Best Days in Profit: {G.global_best_days_in_profit:.2f}"
            )
        else:
            self.best_days_profit_label.config(text="Best Days in Profit: N/A")
        self.best_winrate_label.config(
            text=f"Best Win Rate: {G.global_best_win_rate:.2f}"
        )
        self.best_profit_factor_label.config(
            text=f"Best Profit Factor: {G.global_best_profit_factor:.2f}"
        )
        self.best_avg_win_label.config(
            text=f"Best Avg Win: {G.global_best_avg_win:.3f}"
        )
        self.best_avg_loss_label.config(
            text=f"Best Avg Loss: {G.global_best_avg_loss:.3f}"
        )

        primary, secondary = G.get_status_full()
        nk_state = "ARMED" if G.nuke_armed else "SAFE"
        self.status_var.set(f"{primary} | NK {nk_state} \n{secondary}")
        self.progress["value"] = G.global_progress_pct

        _ = G.live_equity - G.start_equity
        _ = G.live_trade_count

        if should_enable_live_trading() and not G.live_trading_enabled:
            self.nuclear_button.config(state=tk.NORMAL)
        else:
            self.nuclear_button.config(state=tk.DISABLED)
        if G.live_trading_enabled:
            self.nuclear_button.config(text="Live Trading ON")

        allowed = nuclear_key_condition(
            G.global_sharpe, G.global_max_drawdown, G.global_profit_factor
        )

        if not allowed or not should_enable_live_trading():
            self.nuclear_button.config(state=tk.DISABLED)

        update_auto_pause(G.global_sharpe, G.global_max_drawdown)

        self.after_id = self.root.after(self.update_interval, self.update_dashboard)

    # ------------------------------------------------------------------
    # External API methods
    # ------------------------------------------------------------------
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
        logging.info(msg)

        if hasattr(self, "ai_log_list"):
            try:
                self.ai_log_list.insert(tk.END, msg)
                self.ai_log_list.yview_moveto(1.0)
            except Exception:
                pass

    def on_test_buy(self) -> None:
        self.on_test_trade("buy")

    def on_test_sell(self) -> None:
        self.on_test_trade("sell")

    def on_test_trade(self, side: str) -> None:
        if not self.connector:
            return
        try:
            if G.global_phemex_data and G.global_phemex_data[-1][4] > 0:
                price = G.global_phemex_data[-1][4]
            else:
                bars = self.connector.fetch_latest_bars(limit=1)
                price = bars[-1][4] if bars else 0.0
            order = self.connector.create_order(side, 1, price)
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

                    self.log_trade(f"[TEST-CLOSE] {close_order}")
                except Exception as e:  # pragma: no cover
                    self.log_trade(f"[TEST-CLOSE-ERROR] {e}")

            self.root.after(10000, auto_close)
        except Exception as e:  # pragma: no cover
            self.log_trade(f"[TEST-ERROR] {e}")

    def enable_live_trading(self) -> None:
        G.live_trading_enabled = True
        logging.info("LIVE_TRADING_ENABLED")
        G.set_status("Live trading enabled", "Use caution")
        self.nuclear_button.config(state=tk.DISABLED)

    def close_trade(self) -> None:
        logging.info("BUTTON Close Active Trade clicked")
        self.close_requested = True

    def edit_trade(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Edit Trade")
        ttk.Label(win, text="SL Multiplier:").grid(row=0, column=0, padx=5, pady=5)
        sl_var = tk.DoubleVar(value=G.global_SL_multiplier)
        ttk.Entry(win, textvariable=sl_var).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(win, text="TP Multiplier:").grid(row=1, column=0, padx=5, pady=5)
        tp_var = tk.DoubleVar(value=G.global_TP_multiplier)
        ttk.Entry(win, textvariable=tp_var).grid(row=1, column=1, padx=5, pady=5)

        def apply() -> None:
            G.update_trade_params(sl_var.get(), tp_var.get())
            win.destroy()

        ttk.Button(win, text="Apply", command=apply).grid(
            row=2, column=0, columnspan=2, pady=5
        )

    def adjust_cpu_limit(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("CPU Limit")
        ttk.Label(win, text="Threads:").grid(row=0, column=0, padx=5, pady=5)
        cpu_var = tk.IntVar(value=G.cpu_limit)
        ttk.Spinbox(
            win, from_=1, to=os.cpu_count() or 1, textvariable=cpu_var, width=5
        ).grid(row=0, column=1, padx=5, pady=5)

        def apply() -> None:
            G.set_cpu_limit(cpu_var.get())
            win.destroy()

        ttk.Button(win, text="Apply", command=apply).grid(
            row=1, column=0, columnspan=2, pady=5
        )

    def manual_validate(self) -> None:
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
            finally:
                G.set_status("Manual validation done", "")
                G.global_progress_pct = 0
                self.root.after(0, lambda: self.validate_button.config(state="normal"))

        threading.Thread(target=_run, daemon=True).start()

    def toggle_bot(self) -> None:
        running = G.is_bot_running()
        G.set_bot_running(not running)
        new_text = "Pause Bot" if not running else "Resume Bot"
        self.run_button.config(text=new_text)

    def update_composite_flags(self) -> None:

        G.use_net_term = bool(self.use_net_var.get())
        G.use_sharpe_term = bool(self.use_sharpe_var.get())
        G.use_drawdown_term = bool(self.use_dd_var.get())
        G.use_trade_term = bool(self.use_trade_var.get())
        G.use_profit_days_term = bool(self.use_days_var.get())

    def on_toggle_force_nk(self) -> None:
        G.nuke_armed = bool(self.force_nk_var.get())


# ---------------------------------------------------------------------------
# Testing stub
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import types

    G.global_training_loss = [1.0, 0.8, 0.6]
    G.global_validation_loss = [1.2, 0.9, 0.7]

    G.global_equity_curve = [
        [_dt.datetime.now().timestamp() - 3600, 0.0],
        [_dt.datetime.now().timestamp(), 1.0],
    ]

    G.global_best_equity_curve = G.global_equity_curve
    G.timeline_ind_on[:] = 0
    G.timeline_trades[:] = 0

    ens = types.SimpleNamespace(
        optimizers=[types.SimpleNamespace(param_groups=[{"lr": 1e-3}])]
    )

    root = tk.Tk()
    gui = TradingGUI(root, ens)
    root.mainloop()
