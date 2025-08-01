# Regime-Specific Caching

Artibot detects market regimes without manual labels. A lightweight LSTM autoencoder compresses recent volatility and trend features before clustering the latent vectors. `G.current_regime` is updated automatically during training.

When a strategy yields positive returns, `regime_cache.save_best_for_regime()` stores the ensemble weights under `regime_model_cache/`. Later, `load_best_for_regime()` restores the cached model whenever the same regime is detected again. This allows the bot to learn which strategy performs best in each environment and recall it on demand.

No human intervention is required: regime detection, caching triggers and strategy switching all run online inside the training loop. Over time the cache becomes a library of specialised models that the bot selects from dynamically.
