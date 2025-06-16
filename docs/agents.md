# MetaTransformerRL

The meta reinforcement learner adjusts training settings and indicator
usage. Exploration follows an ε-greedy policy while optimisation uses an
Advantage Actor–Critic loss with Sharpe ratio based reward shaping.

## Actions

The agent exposes a large action space. Continuous parameters start with
``d_`` while ``toggle_*`` keys enable or disable indicators.

### Indicator toggles

``toggle_{sma,rsi,macd,atr,vortex,cmf,ichimoku,ema,donchian,kijun,tenkan,disp}``

### Parameter deltas

``lr, wd, d_sma_period, d_rsi_period, d_macd_fast, d_macd_slow, d_macd_signal,
d_atr_period, d_vortex_period, d_cmf_period, d_ema_period, d_donchian_period,
d_kijun_period, d_tenkan_period, d_displacement, d_sl, d_tp, d_long_frac,
d_short_frac``
