"""Trophy rarity forecaster for gta6-trophy-rarity-forecast-01 (Python 3, standard library only).

The idea, in one paragraph: an earn rate is a ratio, earners / players_tracked, and the two
sides of that ratio move for different reasons. Players_tracked grows as more owners sync a
profile with Exophase; earners of a trophy grow as the tracked crowd plays further into the
game. Between the page's refreshes both are frozen, so the history is a step series: collapse
it to its distinct snapshots, fit a power law in log time to each side separately
(ln(count) against ln(1 + hours since release)), shrink each trophy's earner slope toward a
prior set by what the trophy asks of the player (story, mission, grind, completion), blend
the slope toward the player slope as a rate nears saturation, anchor both fits at the last
snapshot, and read the ratio off at the target. The
quantiles are that median plus a logit-scale spread that combines a random-walk term, the
slope's extrapolation uncertainty and a fixed allowance for not knowing when Exophase last
refreshed the page before the target.

Run: python3 main.py with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set, or /input and /output
in the grader's container. Deterministic: pure arithmetic on the case file, no randomness.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS (statistics.NormalDist would do; constants keep the
# arithmetic identical on every interpreter).
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RELEASE_EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()

# Priors, as slopes of ln(count) against ln(1 + hours since release).
PLAYER_SLOPE_PRIOR = 1.0    # tracked players: launch night alone took the count 638 -> 3,868
SIGMA_RW_PER_SQRT_DAY = 0.5  # the house baseline's random-walk spread, kept as one component
STALE_SD = 0.12             # the answer is the page as of its last refresh before the target

# Earner-growth priors by what a trophy asks of the player. Story trophies ride the crowd:
# a new tracked player reaches Chapter 1 within days, so its earners grow almost as fast as
# the player count. Grinds and the completion trophies lag: their earners are the committed
# minority, and a flood of new players dilutes them.
CLASS_PRIOR = {
    "platinum": 0.40,    # needs every other trophy; week one barely moves it
    "completion": 0.45,  # 100% completion, the same few players
    "finale": 0.78,      # the final story mission
    "mission": 0.88,     # a named mission or a strand's last mission
    "grind": 0.66,       # counts, sets, totals: 250 stunts, $250,000, every spot
    "activity": 0.82,    # a one-off action most players trip over while playing
}
CLASS_EXTRA_SD = {"platinum": 0.12, "completion": 0.12}
GRIND_KEYS = ("stunts", "gold medal in all", "win 20", "50 challenges", "100 locations",
              "card sets", "earn $", "spend $", "purchase all", "discover 100",
              "meditate at every", "explore all", "rob 5 bank")


def classify(trophy):
    """(class, earner prior) from the trophy's tier and description text."""
    desc = trophy["description"].lower()
    if trophy["tier"] == "platinum":
        return "platinum", CLASS_PRIOR["platinum"]
    if "100% completion" in desc:
        return "completion", CLASS_PRIOR["completion"]
    if "chapter" in desc:
        # "Complete Chapter N.": the earlier the chapter, the faster its earners grow.
        digits = [int(w) for w in desc.replace(".", " ").split() if w.isdigit()]
        chapter = digits[0] if digits else 2
        return "mission", max(0.80, 1.02 - 0.05 * chapter)
    if desc.rstrip(".") == "complete the final mission":
        return "finale", CLASS_PRIOR["finale"]
    if any(key in desc for key in GRIND_KEYS):
        return "grind", CLASS_PRIOR["grind"]
    if desc.startswith("complete"):
        return "mission", CLASS_PRIOR["mission"]
    return "activity", CLASS_PRIOR["activity"]


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x: float) -> float:
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def parse_time(text: str) -> float:
    """ISO 8601 UTC -> hours since the release day began (2026-11-19T00:00:00Z)."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.timestamp() - RELEASE_EPOCH) / 3600.0


def u_of(hours: float) -> float:
    return math.log1p(max(0.0, hours))


def ols(xs, ys):
    """Least squares y ~ a + b x; returns (b, residual_sd, sxx)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / n)
    return b, sd, sxx


def distinct_snapshots(case):
    """Collapse the read series to the moments the page actually changed.

    A read that repeats the previous read's players_tracked and rates carries no new
    information (Exophase had not refreshed), and keeping hundreds of copies of it would
    let one flat stretch dominate every fit.
    """
    snaps = []
    for read in case["history"]:
        key = (read["players_tracked"], tuple(-1.0 if r is None else r for r in read["rates"]))
        if snaps and snaps[-1][0] == key:
            continue
        snaps.append((key, parse_time(read.get("slot") or read["at"]), read))
    return [(t, read) for _, t, read in snaps]


def fit_slope(points, prior, du, lo, hi):
    """Shrink an OLS slope toward its prior; returns (slope, slope_sd_at_unit_distance).

    points: [(u, ln_count)]; du is the extrapolation distance in u. With fewer than two
    distinct points there is nothing to fit and the prior stands alone, with a wide
    uncertainty. A fitted slope is capped to [lo, hi] (launch-week elasticities do not
    run away), shrunk toward the prior by the point count, and the fitted part is damped
    by span / (span + du): growth measured over a short window is not trusted to keep
    its full elasticity over a longer leap, because penetration saturates.
    """
    if len(points) < 2:
        return prior, 0.35
    b, sd, sxx = ols([p[0] for p in points], [p[1] for p in points])
    b = min(hi, max(lo, b))
    w = len(points) / (len(points) + 2.0)
    slope = w * b + (1.0 - w) * prior
    span = max(1.0, points[-1][0] - points[0][0])
    slope = prior + (slope - prior) * span / (span + du)
    if sxx > 0:
        slope_sd = sd / math.sqrt(sxx)
    else:
        slope_sd = 0.35
    return slope, max(0.05, min(0.6, slope_sd))


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    snaps = distinct_snapshots(case)
    if not snaps:
        raise ValueError("empty history")
    t_last, last_read = snaps[-1]
    u_last = u_of(t_last)
    horizon_days = case["horizon_minutes"] / 1440.0
    u_target = u_of(t_last + case["horizon_minutes"] / 60.0)
    du = max(0.0, u_target - u_last)

    # Players: one series for the whole case.
    n_points = [(u_of(t), math.log(read["players_tracked"])) for t, read in snaps
                if read["players_tracked"] and read["players_tracked"] > 0]
    b_n, sd_n = fit_slope(n_points, PLAYER_SLOPE_PRIOR, du, 0.0, 1.4)
    n_last = last_read["players_tracked"]
    n_target = math.exp(math.log(n_last) + b_n * du) if n_last else 0.0

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        cls, prior = classify(trophy)
        series = [(u_of(t), (read["rates"][i] / 100.0) * read["players_tracked"],
                   read["players_tracked"])
                  for t, read in snaps
                  if i < len(read["rates"]) and read["rates"][i] is not None
                  and read["players_tracked"]]
        if not series:
            median, sigma = 0.05, 1.2
        else:
            e_points = [(u, math.log(e + 0.5)) for u, e, _ in series]
            b_e, sd_e = fit_slope(e_points, prior, du, -0.3, 1.5)
            _, e_last, n_at_last = series[-1]
            r_last = 100.0 * e_last / n_at_last if n_at_last else 0.0
            # Saturation: a trophy nearly every tracked player holds cannot be diluted
            # the way a rare one can -- its earners arrive with the players themselves.
            # Blend the earner slope toward the player slope as the rate nears 100%,
            # and lock the two together once the last read shows essentially everyone.
            sat = min(1.0, max(0.0, (r_last - 40.0) / 60.0))
            if r_last >= 99.9:
                sat = 1.0
            b_e = b_e + sat * (b_n - b_e)
            e_target = math.exp(math.log(e_last + 0.5) + b_e * du) - 0.5
            e_target = min(max(e_target, 0.0), n_target)
            median = clamp_rate(100.0 * e_target / n_target) if n_target > 0 else 0.01
            rw = SIGMA_RW_PER_SQRT_DAY * math.sqrt(horizon_days)
            slope_part = math.hypot(sd_e, sd_n) * du
            extra = CLASS_EXTRA_SD.get(cls, 0.0)
            sigma = math.sqrt(rw * rw + slope_part * slope_part + STALE_SD * STALE_SD
                              + extra * extra)
            sigma = max(0.3, min(2.4, sigma))
        centre = logit(median)
        q = [round(inv_logit(centre + z * sigma), 4) for z in Z]
        q = [min(100.0, max(0.0, v)) for v in q]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
