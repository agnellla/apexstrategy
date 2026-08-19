"""Phase 4: per-circuit constants, fitted with uncertainty rather than assumed.

Two quantities the simulator needs before it can project anything:

1. **Race-progress coefficient** -- how much lap time falls per lap of race distance.
2. **Pit loss** -- total time cost of a green-flag stop, relative to staying out.

## Why it is "race progress" and not "fuel"

The brief, and the earlier ``pitwindow`` project, framed this as a fuel correction. Fuel
burn is real, but it is **not separately identifiable** from lap-level race data: track
evolution (the surface rubbering in) also makes later laps faster, and is also a function
of lap number. Any regression on lap number captures their sum. Splitting them would
require an assumption, not an estimate.

For the decision layer this does not matter, and pretending otherwise would be false
precision. The simulator needs the *total* effect of being N laps into a race, and both
components are strategy-independent -- neither is changed by when a driver pits, so both
cancel when comparing two strategies at the same lap. The coefficient is therefore named
for what it actually measures.

## Why the estimate is identified at all

Within a single stint, tyre age and lap number rise together and are perfectly collinear.
Identification comes entirely from *across-stint* variation: the same tyre age occurs at
very different race distances in stint 1 versus stint 3. Measured on the built store, the
lap-number/tyre-age correlation among clean laps is 0.34-0.48 by circuit -- far from
degenerate. ``fit_race_progress`` reports that correlation per circuit so a future dataset
where it approaches 1.0 is caught rather than silently producing a meaningless split.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import config
from ..data import store

log = logging.getLogger(__name__)

#: Plausible band for the combined race-progress coefficient, in s/lap.
#:
#: Fuel burn alone is roughly -0.05 to -0.08 s/lap (0.03-0.04 s/kg at 1.5-2.0 kg/lap),
#: but the combined effect is NOT bounded below by that. Track evolution can push either
#: way: a track that rubbers in reinforces the fuel effect, while one that overheats
#: through the race partially cancels it. The coefficient must be negative -- cars do get
#: faster as they lighten -- but its magnitude legitimately varies with lap length (longer
#: laps burn more fuel per lap) and circuit-specific evolution. A positive value, or an
#: implausibly large one, means the pipeline is broken rather than the physics.
PLAUSIBLE_RACE_PROGRESS_S_PER_LAP = (-0.40, -0.01)

#: A race with more than this share of wet-compound laps is not a dry race, and is
#: excluded from constant fitting. The MVP is dry-only (Part 14, assumption 8), and
#: mixed conditions corrupt both the pace reference and the pit-loss measurement.
MAX_WET_LAP_FRACTION = 0.05

#: Half-width of the window used to estimate a driver's local clean pace either side of
#: a pit stop. Wide enough to find clean laps, narrow enough that degradation and race
#: progress do not dominate.
PACE_WINDOW_LAPS = 8

_FORMULA = "lap_time_s ~ lap_number + C(compound) + C(compound):tyre_age + C(driver)"
#: Pooled fit adds circuit fixed effects. Without them the model tries to explain
#: baseline lap-time differences between circuits -- 71s at Austria to 97s at Suzuka --
#: using compound and lap number, which badly biases the coefficient.
_FORMULA_POOLED = _FORMULA + " + C(circuit)"


def wet_lap_fraction(laps: pd.DataFrame) -> pd.Series:
    """Share of each race's laps run on wet-compound tyres."""
    from ..data.schema import WET_COMPOUNDS

    return laps.assign(wet=laps["compound"].isin(WET_COMPOUNDS)).groupby("circuit")["wet"].mean()


def dry_circuits(laps: pd.DataFrame) -> set[str]:
    """Circuits whose race was dry enough to fit constants on.

    Silverstone 2024 fails this: 19% of its laps were on intermediates. Fitting there
    mixes conditions into both the pace reference and the pit-loss measurement, and
    silently violates the MVP's dry-only scope.
    """
    frac = wet_lap_fraction(laps)
    return set(frac[frac <= MAX_WET_LAP_FRACTION].index)


def fit_race_progress(laps: pd.DataFrame, pooled: bool = False) -> dict[str, float]:
    """Fit the race-progress coefficient on clean laps.

    Driver fixed effects absorb car pace; compound-specific slopes on tyre age absorb
    degradation, so the ``lap_number`` coefficient carries the residual effect of race
    distance. The pooled variant adds circuit fixed effects.

    Returns the coefficient (negative: later laps are faster), its standard error, and
    the collinearity diagnostic that says whether the split is trustworthy.
    """
    import statsmodels.formula.api as smf

    usable = laps.dropna(subset=["lap_time_s", "tyre_age", "lap_number", "compound", "driver"])
    if len(usable) < 50:
        raise ValueError(f"too few clean laps to fit: {len(usable)}")

    model = smf.ols(_FORMULA_POOLED if pooled else _FORMULA, data=usable).fit()
    coef = float(model.params["lap_number"])
    median_lap = float(usable["lap_time_s"].median())
    return {
        "race_progress_coef_s_per_lap": coef,
        "race_progress_se": float(model.bse["lap_number"]),
        # Normalised by lap duration, so circuits with very different lap lengths are
        # comparable: a longer lap burns more fuel and moves more seconds per lap.
        "race_progress_pct_per_lap": 100 * coef / median_lap,
        "race_progress_n_laps": int(model.nobs),
        "race_progress_r2": float(model.rsquared),
        # If this approaches 1.0 the race-progress/degradation split is no longer
        # identified. Measured at 0.34-0.48 by circuit on the built store.
        "lap_tyre_corr": float(usable["lap_number"].corr(usable["tyre_age"])),
    }


def _local_pace(driver_clean: pd.DataFrame, lap: int) -> float:
    """Median clean lap time for a driver near ``lap``, widening if the window is empty.

    Non-parametric on purpose. A fitted pace model would be circular here: the pace
    model is Phase 5 and will itself consume these constants.
    """
    for half_width in (PACE_WINDOW_LAPS, PACE_WINDOW_LAPS * 2, 10_000):
        window = driver_clean[(driver_clean["lap_number"] - lap).abs() <= half_width]
        if len(window) >= 3:
            return float(window["lap_time_s"].median())
    return float("nan")


def estimate_pit_loss(laps: pd.DataFrame, pits: pd.DataFrame) -> pd.DataFrame:
    """Per-stop pit loss: time actually lost versus having stayed out.

    Raw pit-lane transit time understates the true cost, because a driver also loses
    time slowing for the pit entry on the in-lap and rebuilding speed on the out-lap.
    The loss is therefore measured over the in-lap/out-lap pair against what those two
    laps would have taken at the driver's local clean pace::

        loss = (t_in_lap + t_out_lap) - 2 * local_clean_pace

    Green-flag, dry-compound stops only. A stop under safety car or red flag costs a
    fraction of a green-flag one; a stop onto intermediates happens in conditions where
    the dry clean-pace reference does not apply. At Silverstone 2024 the intermediate
    stops measured 42.5s against ~35s for dry ones on the same day.
    """
    from ..data.schema import WET_COMPOUNDS

    green = pits[~pits["under_caution"] & ~pits["compound_fitted"].isin(WET_COMPOUNDS)].copy()
    lap_times = laps.set_index(["race_id", "driver", "lap_number"])["lap_time_s"]
    clean = laps[laps["is_clean"]]

    rows = []
    for r in green.itertuples():
        t_in = lap_times.get((r.race_id, r.driver, r.lap))
        t_out = lap_times.get((r.race_id, r.driver, r.lap + 1))
        if pd.isna(t_in) or pd.isna(t_out):
            continue

        driver_clean = clean[(clean["race_id"] == r.race_id) & (clean["driver"] == r.driver)]
        pace = _local_pace(driver_clean, r.lap)
        if not np.isfinite(pace):
            continue

        rows.append(
            {
                "race_id": r.race_id,
                "circuit": r.circuit,
                "driver": r.driver,
                "lap": r.lap,
                "pit_lane_s": r.pit_duration_s,
                "local_pace_s": pace,
                "pit_loss_s": (t_in + t_out) - 2 * pace,
            }
        )

    return pd.DataFrame(rows)


def summarise_pit_loss(stop_losses: pd.DataFrame) -> pd.DataFrame:
    """Per-circuit pit-loss summary. Median, not mean: slow stops are a long right tail."""
    g = stop_losses.groupby("circuit")["pit_loss_s"]
    out = pd.DataFrame(
        {
            "pit_loss_median_s": g.median(),
            "pit_loss_p25_s": g.quantile(0.25),
            "pit_loss_p75_s": g.quantile(0.75),
            "pit_loss_n": g.size(),
        }
    )
    out["pit_lane_median_s"] = stop_losses.groupby("circuit")["pit_lane_s"].median()
    return out.round(3)


def build_circuit_constants() -> pd.DataFrame:
    """Fit every circuit, plus a pooled row, and write ``circuits.parquet``.

    The pooled row exists because of a real limitation: with one race per circuit,
    a per-circuit constant **cannot be validated out-of-sample**. Leave-one-race-out
    holds out the only race that circuit has, so its own constant is unavailable by
    construction -- using it anyway would be exactly the circuit-constant leakage the
    architecture doc flags (Part 13, risk 5). The pooled constant is the LORO-safe
    default; per-circuit values are an in-sample refinement.

    Monaco is fitted and stored for completeness but excluded from the pooled fit and
    flagged ``use_for_training=False`` (D3). Non-dry races are excluded from the pooled
    fit on the same basis, via ``dry_circuits``.
    """
    laps = store.fittable_laps()
    pits = store.read_table("pits")
    clean = laps[laps["is_clean"]]

    stop_losses = estimate_pit_loss(laps, pits)
    pit_summary = summarise_pit_loss(stop_losses)
    dry = dry_circuits(laps)

    rows = []
    for circuit in sorted(clean["circuit"].unique()):
        fit = fit_race_progress(clean[clean["circuit"] == circuit])
        rows.append(
            {
                "circuit": circuit,
                "scope": "circuit",
                "is_dry_race": circuit in dry,
                "use_for_training": (
                    config.CIRCUITS_BY_KEY[circuit].use_for_training and circuit in dry
                ),
                **fit,
                **pit_summary.loc[circuit].to_dict(),
            }
        )

    trainable = {c.key for c in config.TRAINING_CIRCUITS} & dry
    pooled_laps = clean[clean["circuit"].isin(trainable)]
    pooled_stops = stop_losses[stop_losses["circuit"].isin(trainable)]
    rows.append(
        {
            "circuit": "_pooled",
            "scope": "pooled",
            "is_dry_race": True,
            "use_for_training": True,
            **fit_race_progress(pooled_laps, pooled=True),
            "pit_loss_median_s": float(pooled_stops["pit_loss_s"].median()),
            "pit_loss_p25_s": float(pooled_stops["pit_loss_s"].quantile(0.25)),
            "pit_loss_p75_s": float(pooled_stops["pit_loss_s"].quantile(0.75)),
            "pit_loss_n": int(len(pooled_stops)),
            "pit_lane_median_s": float(pooled_stops["pit_lane_s"].median()),
        }
    )

    circuits = pd.DataFrame(rows)
    circuits["pit_loss_n"] = circuits["pit_loss_n"].astype(int)
    store.write_table("circuits", circuits)
    return circuits


def check_against_prior(circuits: pd.DataFrame) -> pd.DataFrame:
    """Flag coefficients that contradict the physics.

    The check is deliberately a *band*, not a floor at the fuel-only magnitude. An
    earlier version required the combined coefficient to be at least as large as fuel
    burn alone, which is wrong: track evolution can oppose the fuel effect, so a smaller
    combined value is physically ordinary rather than a bug (Challenge 4).
    """
    lo, hi = PLAUSIBLE_RACE_PROGRESS_S_PER_LAP
    beta = circuits["race_progress_coef_s_per_lap"]
    out = circuits[
        ["circuit", "race_progress_coef_s_per_lap", "race_progress_pct_per_lap", "race_progress_se"]
    ].copy()
    out["sign_ok"] = beta < 0
    out["in_band"] = beta.between(lo, hi)
    out["significant"] = beta.abs() > 2 * circuits["race_progress_se"]
    out["plausible"] = out["sign_ok"] & out["in_band"] & out["significant"]
    return out.round(4)
