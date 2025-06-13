"""Tkinter dashboard for training progress and live prices."""

# ruff: noqa: F403, F405
import artibot.globals as G
import numpy as np
import datetime
import os
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import logging
from .metrics import nuclear_key_condition
from .live_risk import update_auto_pause


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


def ask_use_prev_weights(default: bool = True, tk_module=tk) -> bool:
    """Return ``True`` when the user opts to load the last weights."""
    root = tk_module.Tk()
    root.withdraw()
    try:
        result = tk_module.messagebox.askyesno(
            "Load Weights", "Use previous best weights?"
        )
    finally:
        root.destroy()
    if result is None:
        return default
    return bool(result)


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
        self.root = root
        self.ensemble = ensemble
        self.weights_path = weights_path
        self.connector = connector
        self.close_requested = False
        self.root.title("Complex AI Trading w/ Robust Backtest + Live Phemex")

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

        # ------------------------------------------------------------------
        # notebook with charts
        # ------------------------------------------------------------------
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.frame_train = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_train, text="Training vs. Validation")
        self.fig_train, (self.ax_loss, self.ax_equity_train) = plt.subplots(
            2, 1, figsize=(5, 6)
        )
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=self.frame_train)
        self.canvas_train.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.frame_live = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_live, text="Phemex Live Price")
        self.fig_live, self.ax_live = plt.subplots(figsize=(5, 3))
        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=self.frame_live)
        self.canvas_live.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.frame_backtest = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_backtest, text="Backtest Results")
        self.fig_backtest, self.ax_net_profit = plt.subplots(figsize=(5, 3))
        self.canvas_backtest = FigureCanvasTkAgg(
            self.fig_backtest, master=self.frame_backtest
        )
        self.canvas_backtest.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.frame_details = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_details, text="Attention Weights")
        self.fig_details, self.ax_details = plt.subplots(
            figsize=(5, 3), subplot_kw={"projection": "3d"}
        )
        self.canvas_details = FigureCanvasTkAgg(
            self.fig_details, master=self.frame_details
        )
        self.canvas_details.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        # previous attention weights for smooth animations
        self._last_attention: np.ndarray | None = None
        self._surf = None
        self.anim_steps = 10

        self.frame_trades = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_trades, text="Trade Details")
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

        self.frame_yearly_perf = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_yearly_perf, text="Best Strategy Yearly Perf")
        self.yearly_perf_text = tk.Text(self.frame_yearly_perf, width=50, height=20)
        self.yearly_perf_text.pack(fill=tk.BOTH, expand=True)

        self.frame_monthly_perf = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_monthly_perf, text="Best Strategy Monthly Results")
        self.monthly_perf_text = tk.Text(self.frame_monthly_perf, width=50, height=20)
        self.monthly_perf_text.pack(fill=tk.BOTH, expand=True)

        self.info_frame = ttk.LabelFrame(self.sidebar, text="Performance")
        self.info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
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

        self.current_sharpe_label = ttk.Label(
            self.info_frame, text="Sharpe: N/A", font=("Helvetica", 12)
        )
        self.current_sharpe_label.grid(row=9, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_drawdown_label = ttk.Label(
            self.info_frame, text="Max DD: N/A", font=("Helvetica", 12)
        )
        self.current_drawdown_label.grid(row=10, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_netprofit_label = ttk.Label(
            self.info_frame, text="Net Profit (%): N/A", font=("Helvetica", 12)
        )
        self.current_netprofit_label.grid(row=11, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_trades_label = ttk.Label(
            self.info_frame, text="Trades: N/A", font=("Helvetica", 12)
        )
        self.current_trades_label.grid(row=12, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_inactivity_label = ttk.Label(
            self.info_frame, text="Inactivity Penalty: N/A", font=("Helvetica", 12)
        )
        self.current_inactivity_label.grid(
            row=13, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.current_composite_label = ttk.Label(
            self.info_frame, text="Composite: N/A", font=("Helvetica", 12)
        )
        self.current_composite_label.grid(row=14, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_days_profit_label = ttk.Label(
            self.info_frame, text="Days in Profit: N/A", font=("Helvetica", 12)
        )
        self.current_days_profit_label.grid(
            row=15, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.current_winrate_label = ttk.Label(
            self.info_frame, text="Win Rate: N/A", font=("Helvetica", 12)
        )
        self.current_winrate_label.grid(row=16, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_profit_factor_label = ttk.Label(
            self.info_frame, text="Profit Factor: N/A", font=("Helvetica", 12)
        )
        self.current_profit_factor_label.grid(
            row=17, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.current_avg_win_label = ttk.Label(
            self.info_frame, text="Avg Win: N/A", font=("Helvetica", 12)
        )
        self.current_avg_win_label.grid(row=18, column=0, sticky=tk.W, padx=5, pady=5)
        self.current_avg_loss_label = ttk.Label(
            self.info_frame, text="Avg Loss: N/A", font=("Helvetica", 12)
        )
        self.current_avg_loss_label.grid(row=19, column=0, sticky=tk.W, padx=5, pady=5)

        self.best_sharpe_label = ttk.Label(
            self.info_frame,
            text="Best Sharpe: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_sharpe_label.grid(row=9, column=1, sticky=tk.W, padx=5, pady=5)
        self.best_drawdown_label = ttk.Label(
            self.info_frame,
            text="Best Max DD: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_drawdown_label.grid(row=10, column=1, sticky=tk.W, padx=5, pady=5)
        self.best_netprofit_label = ttk.Label(
            self.info_frame,
            text="Best Net Pct: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_netprofit_label.grid(row=11, column=1, sticky=tk.W, padx=5, pady=5)
        self.best_trades_label = ttk.Label(
            self.info_frame,
            text="Best Trades: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_trades_label.grid(row=12, column=1, sticky=tk.W, padx=5, pady=5)
        self.best_inactivity_label = ttk.Label(
            self.info_frame,
            text="Best Inactivity: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_inactivity_label.grid(row=13, column=1, sticky=tk.W, padx=5, pady=5)
        self.best_composite_label = ttk.Label(
            self.info_frame,
            text="Best Composite: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_composite_label.grid(row=14, column=1, sticky=tk.W, padx=5, pady=5)
        self.best_days_profit_label = ttk.Label(
            self.info_frame,
            text="Best Days in Profit: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_days_profit_label.grid(row=15, column=1, sticky=tk.W, padx=5, pady=5)
        self.best_winrate_label = ttk.Label(
            self.info_frame,
            text="Best Win Rate: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_winrate_label.grid(row=16, column=1, sticky=tk.W, padx=5, pady=5)
        self.best_profit_factor_label = ttk.Label(
            self.info_frame,
            text="Best Profit Factor: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_profit_factor_label.grid(
            row=17, column=1, sticky=tk.W, padx=5, pady=5
        )
        self.best_avg_win_label = ttk.Label(
            self.info_frame,
            text="Best Avg Win: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_avg_win_label.grid(row=18, column=1, sticky=tk.W, padx=5, pady=5)
        self.best_avg_loss_label = ttk.Label(
            self.info_frame,
            text="Best Avg Loss: N/A",
            font=("Helvetica", 12),
            foreground="darkgreen",
        )
        self.best_avg_loss_label.grid(row=19, column=1, sticky=tk.W, padx=5, pady=5)

        # status indicator combines primary + secondary messages
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

        # trading control buttons
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

        # Use sidebar as container to avoid mixing pack/grid on the root
        self.frame_ai = ttk.Frame(self.sidebar)
        self.frame_ai.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.ai_output_label = ttk.Label(
            self.frame_ai, text="Latest AI Adjustments:", font=("Helvetica", 12, "bold")
        )
        self.ai_output_label.pack(anchor="n")
        self.ai_output_text = tk.Text(self.frame_ai, width=40, height=10, wrap="word")
        self.ai_output_text.pack(fill=tk.BOTH, expand=True)

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

        self.update_interval = 2000
        self.root.after(self.update_interval, self.update_dashboard)
        self.root.after(10000, self.refresh_stats)

    def _animate_attention(
        self, X: np.ndarray, Y: np.ndarray, new_data: np.ndarray
    ) -> None:
        """Animate the attention surface from the previous values."""
        if self._last_attention is None or self._last_attention.shape != new_data.shape:
            self._last_attention = new_data.copy()
        diff = (new_data - self._last_attention) / float(self.anim_steps)
        current = self._last_attention.copy()
        for _ in range(self.anim_steps):
            current += diff
            if self._surf is not None:
                self._surf.remove()
            self._surf = self.ax_details.plot_surface(X, Y, current, cmap="viridis")
            self.canvas_details.draw()
            self.canvas_details.flush_events()
        self._last_attention = new_data.copy()

    def update_dashboard(self):
        """Refresh all dashboard widgets from shared state."""
        self.ax_loss.clear()
        self.ax_loss.set_title("Training vs. Validation Loss")
        x1 = range(1, len(G.global_training_loss) + 1)
        self.ax_loss.plot(
            x1, G.global_training_loss, color="blue", marker="o", label="Train"
        )
        val_filtered = [
            (i + 1, v) for i, v in enumerate(G.global_validation_loss) if v is not None
        ]
        if val_filtered:
            xv, yv = zip(*val_filtered)
            self.ax_loss.plot(xv, yv, color="orange", marker="x", label="Val")
        self.ax_loss.legend()

        self.ax_equity_train.clear()
        self.ax_equity_train.set_title("Equity: Current (red) vs Best (green)")
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

        self.ax_details.clear()
        self.ax_details.set_title("Live Attention Weights")
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
                self.ax_details.plot(x_, avg, marker="o", color="purple")
                self._last_attention = None
        elif G.global_attention_weights_history:
            x_ = list(range(1, len(G.global_attention_weights_history) + 1))
            self.ax_details.plot(
                x_, G.global_attention_weights_history, marker="o", color="purple"
            )
            self._last_attention = None
        self.canvas_details.draw()

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

        self.yearly_perf_text.delete("1.0", tk.END)
        if G.global_best_yearly_stats_table:
            self.yearly_perf_text.insert(tk.END, G.global_best_yearly_stats_table)
        else:
            self.yearly_perf_text.insert(tk.END, "No yearly data")

        self.monthly_perf_text.delete("1.0", tk.END)
        if G.global_best_monthly_stats_table:
            self.monthly_perf_text.insert(tk.END, G.global_best_monthly_stats_table)
        else:
            self.monthly_perf_text.insert(tk.END, "No monthly data")

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
            f" SL Mult: {hp.sl}\n"
            f" TP Mult: {hp.tp}"
        )
        self.indicator_label.config(text=info)
        self.best_lr_label.config(
            text=f"Best LR: {G.global_best_lr if G.global_best_lr else 'N/A'}"
        )
        self.best_wd_label.config(
            text=f"Weight Decay: {G.global_best_wd if G.global_best_wd else 'N/A'}"
        )

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

        self.best_sharpe_label.config(text=f"Best Sharpe: {G.global_best_sharpe:.2f}")
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

        self.ai_output_text.delete("1.0", tk.END)
        self.ai_output_text.insert(tk.END, G.global_ai_adjustments)

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

        self.root.after(self.update_interval, self.update_dashboard)

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

    def on_toggle_force_nk(self) -> None:
        """Update ``G.nuke_armed`` from the checkbox state."""
        G.nuke_armed = bool(self.force_nk_var.get())

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
