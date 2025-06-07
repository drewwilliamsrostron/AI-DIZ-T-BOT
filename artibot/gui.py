"""Tkinter dashboard for training progress and live prices."""

# ruff: noqa: F403, F405
from .globals import *
import numpy as np


###############################################################################
# Tkinter GUI
###############################################################################
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
        self.notebook.add(self.frame_yearly_perf, text="Yearly Perf")
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
            self.info_frame, text=f"ATR: {global_ATR_period}", font=("Helvetica", 12)
        )
        self.atr_label.grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.sl_label = ttk.Label(
            self.info_frame, text=f"SL: {global_SL_multiplier}", font=("Helvetica", 12)
        )
        self.sl_label.grid(row=7, column=0, sticky=tk.W, padx=5, pady=5)
        self.tp_label = ttk.Label(
            self.info_frame, text=f"TP: {global_TP_multiplier}", font=("Helvetica", 12)
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

        # single-line status indicator
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(
            self.info_frame,
            textvariable=self.status_var,
            font=("Helvetica", 10, "italic"),
        )
        self.status_label.grid(
            row=16, column=0, sticky=tk.W, padx=5, pady=5, columnspan=2
        )

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
        global global_equity_curve, global_best_equity_curve
        self.ax_loss.clear()
        self.ax_loss.set_title("Training vs. Validation Loss")
        x1 = range(1, len(global_training_loss) + 1)
        self.ax_loss.plot(
            x1, global_training_loss, color="blue", marker="o", label="Train"
        )
        val_filtered = [
            (i + 1, v) for i, v in enumerate(global_validation_loss) if v is not None
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
                for (t, b) in global_equity_curve
                if isinstance(t, (int, float, np.integer, np.floating))
            ]
            if valid_eq:
                ts_, bs_ = zip(*valid_eq)
                ts_dt = [datetime.datetime.fromtimestamp(t_) for t_ in ts_]
                self.ax_equity_train.plot(
                    ts_dt, bs_, color="red", marker=".", label="Current"
                )
            if global_best_equity_curve:
                best_eq = [
                    (t, b)
                    for (t, b) in global_best_equity_curve
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
            for bar in global_phemex_data:
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
        if global_backtest_profit:
            x2 = range(1, len(global_backtest_profit) + 1)
            self.ax_net_profit.plot(
                x2, global_backtest_profit, marker="o", color="green"
            )
        self.canvas_backtest.draw()

        self.ax_details.clear()
        self.ax_details.set_title("Avg Attention Weights (placeholder)")
        if global_attention_weights_history:
            x_ = list(range(1, len(global_attention_weights_history) + 1))
            self.ax_details.plot(
                x_, global_attention_weights_history, marker="o", color="purple"
            )
        self.canvas_details.draw()

        self.trade_text.delete("1.0", tk.END)
        if global_trade_details:
            self.trade_text.insert(tk.END, json.dumps(global_trade_details, indent=2))
        else:
            self.trade_text.insert(tk.END, "No Trade Details")

        self.yearly_perf_text.delete("1.0", tk.END)
        if global_yearly_stats_table:
            self.yearly_perf_text.insert(tk.END, global_yearly_stats_table)
        else:
            self.yearly_perf_text.insert(tk.END, "No yearly data")

        pred_str = global_current_prediction if global_current_prediction else "N/A"
        conf = global_ai_confidence if global_ai_confidence else 0.0
        steps = global_ai_epoch_count
        self.pred_label.config(text=f"AI Prediction: {pred_str}")
        self.conf_label.config(text=f"Confidence: {conf:.2f}")
        self.epoch_label.config(text=f"Training Steps: {steps}")

        current_lr = self.ensemble.optimizers[0].param_groups[0]["lr"]
        self.lr_label.config(text=f"LR: {current_lr:.2e}")
        self.atr_label.config(text=f"ATR: {global_ATR_period}")
        self.sl_label.config(text=f"SL: {global_SL_multiplier}")
        self.tp_label.config(text=f"TP: {global_TP_multiplier}")
        self.best_lr_label.config(
            text=f"Best LR: {global_best_lr if global_best_lr else 'N/A'}"
        )
        self.best_wd_label.config(
            text=f"Weight Decay: {global_best_wd if global_best_wd else 'N/A'}"
        )

        self.current_sharpe_label.config(text=f"Sharpe: {global_sharpe:.2f}")
        self.current_drawdown_label.config(text=f"Max DD: {global_max_drawdown:.3f}")
        self.current_netprofit_label.config(text=f"Net Pct: {global_net_pct:.2f}")
        self.current_trades_label.config(text=f"Trades: {global_num_trades}")
        if global_inactivity_penalty is not None:
            self.current_inactivity_label.config(
                text=f"Inact: {global_inactivity_penalty:.2f}"
            )
        else:
            self.current_inactivity_label.config(text="Inactivity Penalty: N/A")
        if global_composite_reward is not None:
            self.current_composite_label.config(
                text=f"Comp: {global_composite_reward:.2f}"
            )
        else:
            self.current_composite_label.config(text="Current Composite: N/A")
        if global_days_in_profit is not None:
            self.current_days_profit_label.config(
                text=f"Days in Profit: {global_days_in_profit:.2f}"
            )
        else:
            self.current_days_profit_label.config(text="Current Days in Profit: N/A")

        self.best_sharpe_label.config(text=f"Best Sharpe: {global_best_sharpe:.2f}")
        self.best_drawdown_label.config(text=f"Best Max DD: {global_best_drawdown:.3f}")
        self.best_netprofit_label.config(
            text=f"Best Net Pct: {global_best_net_pct:.2f}"
        )
        self.best_trades_label.config(text=f"Best Trades: {global_best_num_trades}")
        if global_best_inactivity_penalty is not None:
            self.best_inactivity_label.config(
                text=f"Best Inact: {global_best_inactivity_penalty:.2f}"
            )
        else:
            self.best_inactivity_label.config(text="Best Inactivity Penalty: N/A")
        if global_best_composite_reward is not None:
            self.best_composite_label.config(
                text=f"Best Comp: {global_best_composite_reward:.2f}"
            )
        else:
            self.best_composite_label.config(text="Best Composite: N/A")
        if global_best_days_in_profit is not None:
            self.best_days_profit_label.config(
                text=f"Best Days in Profit: {global_best_days_in_profit:.2f}"
            )
        else:
            self.best_days_profit_label.config(text="Best Days in Profit: N/A")

        self.ai_output_text.delete("1.0", tk.END)
        self.ai_output_text.insert(tk.END, global_ai_adjustments)
        self.ai_log_text.delete("1.0", tk.END)
        self.ai_log_text.insert(tk.END, global_ai_adjustments_log)

        # update status line
        self.status_var.set(get_status())

        self.root.after(self.update_interval, self.update_dashboard)
