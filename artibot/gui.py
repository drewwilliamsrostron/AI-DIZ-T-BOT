"""Tkinter dashboard for training progress and live prices."""

# ruff: noqa: F403, F405
import artibot.globals as G
import numpy as np
import datetime
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd


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
    """Return ``True`` when validation metrics meet risk criteria."""
    sharpe = G.global_holdout_sharpe
    dd = G.global_holdout_max_drawdown
    return sharpe >= 1.0 and dd >= -0.30


class TradingGUI:
    def __init__(self, root, ensemble):
        self.root = root
        self.ensemble = ensemble
        self.root.title("Complex AI Trading w/ Robust Backtest + Live Phemex")
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

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
        self.fig_details, self.ax_details = plt.subplots(figsize=(5, 3))
        self.canvas_details = FigureCanvasTkAgg(
            self.fig_details, master=self.frame_details
        )
        self.canvas_details.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.frame_trades = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_trades, text="Trade Details")
        self.trade_text = tk.Text(self.frame_trades, width=50, height=20)
        self.trade_text.pack(fill=tk.BOTH, expand=True)

        self.frame_yearly_perf = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_yearly_perf, text="Best Strategy Yearly Perf")
        self.yearly_perf_text = tk.Text(self.frame_yearly_perf, width=50, height=20)
        self.yearly_perf_text.pack(fill=tk.BOTH, expand=True)

        self.info_frame = ttk.Frame(root)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X)
        disclaimer_text = "NOT INVESTMENT ADVICE! Demo only.\nUse caution."
        self.disclaimer_label = ttk.Label(
            self.info_frame,
            text=disclaimer_text,
            font=("Helvetica", 9, "italic"),
            foreground="darkred",
        )
        self.disclaimer_label.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)

        self.pred_label = ttk.Label(
            self.info_frame, text="AI Prediction: N/A", font=("Helvetica", 12)
        )
        self.pred_label.grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.conf_label = ttk.Label(
            self.info_frame, text="Confidence: N/A", font=("Helvetica", 12)
        )
        self.conf_label.grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.epoch_label = ttk.Label(
            self.info_frame, text="Training Steps: 0", font=("Helvetica", 12)
        )
        self.epoch_label.grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)

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
        self.atr_label = ttk.Label(
            self.info_frame, text=f"ATR: {G.global_ATR_period}", font=("Helvetica", 12)
        )
        self.atr_label.grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.sl_label = ttk.Label(
            self.info_frame,
            text=f"SL: {G.global_SL_multiplier}",
            font=("Helvetica", 12),
        )
        self.sl_label.grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.tp_label = ttk.Label(
            self.info_frame,
            text=f"TP: {G.global_TP_multiplier}",
            font=("Helvetica", 12),
        )
        self.tp_label.grid(row=8, column=0, sticky=tk.W, padx=5, pady=5)

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

        # status indicator combines primary + secondary messages
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(
            self.info_frame,
            textvariable=self.status_var,
            font=("Helvetica", 10, "italic"),
            justify=tk.LEFT,
        )
        self.status_label.grid(
            row=18, column=0, sticky=tk.W, padx=5, pady=5, columnspan=2
        )

        # trading control buttons
        self.controls_frame = ttk.Frame(self.info_frame)
        self.controls_frame.grid(row=19, column=0, columnspan=2, pady=5)
        self.nuclear_button = ttk.Button(
            self.controls_frame,
            text="Nuclear Key",
            command=self.enable_live_trading,
            state=tk.DISABLED,
        )
        self.nuclear_button.pack(side=tk.LEFT, padx=5)
        self.close_button = ttk.Button(
            self.controls_frame,
            text="Close Active Trade",
            command=self.close_trade,
        )
        self.close_button.pack(side=tk.LEFT, padx=5)
        self.edit_button = ttk.Button(
            self.controls_frame,
            text="Edit Trade",
            command=self.edit_trade,
        )
        self.edit_button.pack(side=tk.LEFT, padx=5)

        self.frame_ai = ttk.Frame(root)
        self.frame_ai.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.ai_output_label = ttk.Label(
            self.frame_ai, text="Latest AI Adjustments:", font=("Helvetica", 12, "bold")
        )
        self.ai_output_label.pack(anchor="n")
        self.ai_output_text = tk.Text(self.frame_ai, width=40, height=10, wrap="word")
        self.ai_output_text.pack(fill=tk.BOTH, expand=True)

        self.frame_ai_log = ttk.Frame(root)
        self.frame_ai_log.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        self.ai_log_label = ttk.Label(
            self.frame_ai_log,
            text="AI Adjustments Log:",
            font=("Helvetica", 12, "bold"),
        )
        self.ai_log_label.pack(anchor="n")
        self.ai_log_text = tk.Text(self.frame_ai_log, width=40, height=10, wrap="word")
        self.ai_log_text.pack(fill=tk.BOTH, expand=True)

        self.update_interval = 2000
        self.root.after(self.update_interval, self.update_dashboard)

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
        self.ax_details.set_title("Avg Attention Weights (placeholder)")
        if G.global_attention_weights_history:
            x_ = list(range(1, len(G.global_attention_weights_history) + 1))
            self.ax_details.plot(
                x_, G.global_attention_weights_history, marker="o", color="purple"
            )
        self.canvas_details.draw()

        self.trade_text.delete("1.0", tk.END)
        if G.global_trade_details:
            summary = format_trade_details(G.global_trade_details)
            self.trade_text.insert(tk.END, summary)
        else:
            self.trade_text.insert(tk.END, "No Trade Details")

        self.yearly_perf_text.delete("1.0", tk.END)
        if G.global_best_yearly_stats_table:
            self.yearly_perf_text.insert(tk.END, G.global_best_yearly_stats_table)
        else:
            self.yearly_perf_text.insert(tk.END, "No yearly data")

        pred_str = G.global_current_prediction if G.global_current_prediction else "N/A"
        conf = G.global_ai_confidence if G.global_ai_confidence else 0.0
        steps = G.epoch_count
        self.pred_label.config(text=f"AI Prediction: {pred_str}")
        self.conf_label.config(text=f"Confidence: {conf:.2f}")
        self.epoch_label.config(text=f"Training Steps: {steps}")

        current_lr = self.ensemble.optimizers[0].param_groups[0]["lr"]
        self.lr_label.config(text=f"LR: {current_lr:.2e}")
        self.atr_label.config(text=f"ATR: {G.global_ATR_period}")
        self.sl_label.config(text=f"SL: {G.global_SL_multiplier}")
        self.tp_label.config(text=f"TP: {G.global_TP_multiplier}")
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

        self.ai_output_text.delete("1.0", tk.END)
        self.ai_output_text.insert(tk.END, G.global_ai_adjustments)
        self.ai_log_text.delete("1.0", tk.END)
        self.ai_log_text.insert(tk.END, G.global_ai_adjustments_log)

        # update status line
        primary, secondary = G.get_status_full()
        self.status_var.set(f"{primary}\n{secondary}")

        # manage trading buttons
        if should_enable_live_trading() and not G.live_trading_enabled:
            self.nuclear_button.config(state=tk.NORMAL)
        else:
            self.nuclear_button.config(state=tk.DISABLED)
        if G.live_trading_enabled:
            self.nuclear_button.config(text="Live Trading ON")

        self.root.after(self.update_interval, self.update_dashboard)

    def enable_live_trading(self):
        """Activate live trading after user confirmation."""
        G.live_trading_enabled = True
        G.set_status("Live trading enabled", "Use caution")
        self.nuclear_button.config(state=tk.DISABLED)

    def close_trade(self):
        """Cancel orders and close the current position."""
        G.cancel_open_orders()
        G.close_position()
        G.set_status("Trade closed", "All orders cancelled")

    def edit_trade(self):
        """Popup dialog to adjust SL/TP multipliers."""
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
