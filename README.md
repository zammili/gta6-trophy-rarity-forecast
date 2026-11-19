# Week-One Forecaster — GTA VI trophy rarity

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zammili/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zammili/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for make.fun's **gta6-trophy-rarity-forecast-01**: forecast, for every
PlayStation 5 trophy of Grand Theft Auto VI, the percent of Exophase-tracked players who
will hold it at 2026-11-26T00:00:00Z, one week after launch — as five quantiles
(5/25/50/75/95%), scored by pinball loss on the logit scale.

## How it works

`main.py` (Python 3, standard library only, deterministic, offline) reads each case in
`$MAKEFUN_INPUT_DIR` (or `/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR`
(or `/output`).

The core choice: **an earn rate is a ratio, so forecast the numerator and the denominator
separately.**

- **Collapse to distinct snapshots.** Exophase refreshes a game's counts on its own
  schedule, hours apart, so most reads repeat the previous one byte for byte. Fitting
  against every read would let one flat stretch dominate; the model keeps only the reads
  where the page actually changed.
- **Players.** `ln(players_tracked)` is fit against `ln(1 + hours since release)` by
  least squares over the distinct snapshots, the slope shrunk toward a prior of 1.0
  (launch night alone multiplied the tracked count sixfold), then anchored at the last
  snapshot and extrapolated to the target.
- **Earners, per trophy.** Earner counts are recovered from each read
  (`rate × players_tracked`), fit the same way. Fitted slopes are capped (launch-week
  elasticities do not run away), shrunk toward the class prior by the number of
  snapshots, and the fitted part is damped by `span / (span + leap)`: growth measured
  over a short window is not trusted to keep its full elasticity over a longer leap,
  because penetration saturates. The slope is also pulled toward a prior
  that depends on what the trophy asks of the player, read off its description:
  story chapters fastest (a new player reaches Chapter 1 within days), then named
  missions, one-off activities, grinds (250 stunts, $250,000 laundered, every
  meditation spot), and slowest the 100%-completion gold and the platinum. A trophy
  almost every tracked player already holds is treated differently: its earners arrive
  with the players themselves, so its slope is blended toward the player slope as the
  rate nears 100% and locked to it at 99.9%+.
- **Median.** The ratio of the two extrapolations at the target, clamped to
  [0.01, 99.99].
- **Spread.** On the logit scale, in quadrature: a random-walk term
  (0.5·√horizon-days, the house baseline's), the extrapolation uncertainty of the two
  fitted slopes over the forecast distance, a fixed 0.12 for not knowing when Exophase
  last refreshed before the target, and a small extra for the platinum/completion
  classes. Floored at 0.3, capped at 2.4.

## Why not just persistence?

The house baseline holds every rate at its last read. That fails exactly when it
matters: a refresh between checkpoint and target moves every rate at once (on launch
night the tracked crowd grew 638 → 3,868 and almost every rate fell, because the new
players held fewer trophies). Modelling earners and players separately makes dilution
and progression fall out of the arithmetic instead of being patched on afterwards.

## Running it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with .in files> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs on the same case produce byte-identical output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=nj55s4jw&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
