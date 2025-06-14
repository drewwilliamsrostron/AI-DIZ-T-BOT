# MetaTransformerRL

The meta reinforcement learner adjusts training settings and indicator
usage. Exploration follows an ε-greedy policy while optimisation uses an
Advantage Actor–Critic loss with Sharpe ratio based reward shaping.

## Actions

| key           | description                          |
|---------------|--------------------------------------|
| `d_long_frac` | change long exposure fraction        |
| `d_short_frac`| change short exposure fraction       |
| `toggle_ema`  | enable/disable EMA indicator         |
| `toggle_donchian` | toggle Donchian channels         |
| `toggle_kijun`| toggle Kijun-sen line                |
| `toggle_tenkan`| toggle Tenkan-sen line              |
| `toggle_disp` | toggle price displacement            |

Other keys adjust learning rate, weight decay and classic indicators.
