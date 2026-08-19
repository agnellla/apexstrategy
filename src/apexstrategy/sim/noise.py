"""Phase 8: hierarchical noise -- sampling each effect at the level it actually lives at.

Phase 7 sampled a fresh i.i.d. residual for every lap. Over a 45-lap horizon those
average out, so an 80% interval on race time covered **11-28%** of held-out outcomes
instead of 80%. That is the variance collapse Challenge 5 predicts, and it is not fixed
by widening the per-lap distribution: the problem is not the size of the noise, it is
that real racing error is *persistent* and this treated it as independent.

## What is sampled where

| Level | Drawn once per | Represents | Estimated sd |
|---|---|---|---|
| Race progress | sim | the track trending faster or slower than expected | 0.018 s/lap |
| Reference, shared | sim | the whole field's pace level is mis-pinned | 4.43/sqrt(N) |
| Reference, per car | sim x driver | this car's pace is mis-pinned | 0.78/sqrt(n) |
| Stint level | sim x stint | this set of tyres is better or worse than nominal | 0.467 s |
| Stint slope | sim x stint | it degrades faster or slower than nominal | 0.049 s/lap |
| Pit stop | sim x stop | crew variance and the slow-stop tail (``engine``) | empirical |
| Lap | sim x driver x lap | traffic, mistakes, tyre temperature | empirical, sd 0.506 |

The first four are Gaussian and the last is not. That asymmetry is deliberate: the upper
levels are averages and slopes over many laps, where the central limit theorem genuinely
applies, while a single lap's residual is right-skewed (skew 2.30 -- a lap can go badly
wrong, but there is a floor on how fast it can go) and a Gaussian would erase exactly the
slow tail that strategy risk is about.

## The driver level is reference *sampling* error, and only the part that survives

The obvious choice would be the spread of ``estimated reference - true reference``, which
measures 0.44-1.21 s depending on the decision lap. That would be **wrong, and too wide**.

Projection error decomposes as ``(ref_hat - ref_true) - resid_future``. A driver's
persistent model misfit ``mu_d`` appears in *both* terms -- the reference estimate absorbs
it from the observed laps, and the future residuals carry it back -- so it cancels
exactly. This is the same cancellation property that makes the Phase 6 estimator immune to
a constant per-circuit offset.

Measured, ``mu_d`` has sd 0.359 s and is flat in the decision lap, confirming it is a
driver property rather than an estimation artefact. What survives is the *sampling*
fluctuation, which shrinks as laps accumulate:

| Clean laps observed | Sampling sd |
|---|---|
| <= 2 | 0.986 s |
| 3-5 | 0.635 s |
| 6-10 | 0.572 s |
| 11-20 | 0.359 s |
| > 20 | 0.241 s |

## Most of that error is shared by the whole field, not private to a car

Treating the sampling error as purely per-driver was measurably wrong, and the first
calibration run showed it in both directions at once: within-replay spread came out 1.68x
too wide while the between-replay spread came out too narrow. Splitting the residual by
circuit and decision lap explains it -- **76% of the reference-error variance at lap 5 is
a field-wide shift, falling to 45% by lap 35**:

| Decision lap | Shared sd | Per-car sd | Shared share of variance |
|---|---|---|---|
| 5 | 1.044 s | 0.583 s | 76% |
| 15 | 0.548 s | 0.433 s | 62% |
| 25 | 0.283 s | 0.284 s | 50% |
| 35 | 0.182 s | 0.204 s | 45% |

That is physically what one would expect: a field-wide reference error means the model's
pace *shape* is off for this circuit, which moves everyone together — the same
circuit-level effect D13 corrects the slope of. It matters because a shared error does not
change who beats whom, while a private one does, so the split decides whether the tool is
over- or under-confident about a *comparison*.

Both parts shrink as laps accumulate, so both take a ``k / sqrt(n)`` form: the per-car
term over that driver's clean laps (k = 0.780), the shared term over the field's
(k = 4.430). The per-car k implies an effective sample of ``(0.506/0.780)^2 = 42%`` of a
driver's observed laps — roughly one independent observation per 2.4 laps, consistent with
laps being correlated inside a stint.

## Race progress does not shrink much, and that is not a bug

The race-progress component is the error in D13's in-race correction. It falls only from
0.027 to 0.015 s/lap between decision lap 5 and 35, because it is dominated by
circuit-to-circuit heterogeneity that the shrunk estimator deliberately only partly
recovers, not by sampling noise that more laps would cure. A single constant is therefore
used. **It rests on five circuits and is the thinnest estimate in this module.**

Small as it looks, it is not negligible: 0.018 s/lap compounds over a 45-lap horizon into
``0.018 * (45*46/2) = 18.6 s`` of race-time spread, comparable to a pit stop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Minimum laps in a stint before its level and slope are estimated separately. Below
#: this the slope is noise, and the stint's laps only inform the lap-level component.
MIN_LAPS_FOR_STINT_FIT = 8

#: A race needs this many clean laps before its residual spread is an estimate rather than
#: an accident, and this many races before the spread *of* those spreads means anything.
MIN_LAPS_FOR_RACE_SCALE = 200
MIN_RACES_FOR_RACE_SCALE = 8

#: Race-level sd of the D13 race-progress correction error, in s/lap. Measured across the
#: five trainable circuits at decision laps 5-35. See the module docstring on why it is a
#: constant and why it is the thinnest number here.
DEFAULT_PROGRESS_SD = 0.018

#: Reference-pace error has a **floor**, and missing it was the calibration bug.
#:
#: The first version modelled the error as pure sampling noise, ``k / sqrt(n)``, which goes
#: to zero as laps accumulate: by lap 35 the simulator was certain it knew a driver's pace.
#: Re-measured over 190 (race, decision lap) states spanning 48 races, 24 circuits and
#: three seasons, the error is ``sqrt(floor^2 + k^2/n)`` with a floor that never shrinks:
#:
#: | Decision lap | private sd | shared sd |
#: |---|---|---|
#: | 15 | 0.561 | 0.427 |
#: | 25 | 0.503 | 0.353 |
#: | 35 | 0.423 | 0.296 |
#:
#: The sampling term was close to right all along (0.792 measured against 0.780 assumed).
#: What was missing is the part that does not average away: a driver's pace in the second
#: half of a race is genuinely not the pace they showed in the first, and no amount of
#: watching the first half fixes that. D14 argued this misfit cancels between the reference
#: estimate and the future residuals, and it does -- but only to the extent it is constant,
#: and the residue is this floor.
DEFAULT_REFERENCE_FLOOR = 0.404
DEFAULT_REFERENCE_SHARED_FLOOR = 0.248
DEFAULT_REFERENCE_K = 0.792
#: Measured raw, the field-wide reference error gives k = 4.430 -- but it correlates at
#: **0.824** with the race-progress error, sharing 68% of its variance, because they are
#: largely the same phenomenon seen twice: when D13's slope is wrong by d, the reference
#: anchored through that slope shifts with it (fitted at ~25d, consistent with a lever arm
#: of half the decision lap). Applying both at full size counted the race-level error
#: nearly twice and made the between-replay spread 1.43x too wide. This is the component
#: orthogonal to the progress term.
DEFAULT_REFERENCE_SHARED_K = 3.235

#: How much a whole race's lap-to-lap noise scales up or down, as the sd of the log ratio
#: between one race's residual sd and the pooled sd. Measured out of fold over 48 races:
#: per-race sd spans 0.431 s to 2.233 s around a pooled 0.773 s, so a race is routinely
#: **30% noisier or quieter than average** and occasionally three times noisier.
#:
#: Sampled rather than predicted, because it is not predictable. Regressing a race's spread
#: on the same circuit's spread in other seasons gives a correlation of **0.04**, and
#: conditioning on it makes the residual scatter *worse* (log-sd 0.313 against 0.262
#: unconditional). Chaos is a property of the afternoon, not of the venue. A simulator that
#: uses one pooled spread for every race is therefore over-confident at Melbourne and
#: over-cautious at Barcelona, with no way to tell which in advance; drawing the scale once
#: per simulation integrates over exactly that ignorance.
DEFAULT_RACE_SCALE_LOG_SD = 0.262

#: Two further race-level effects were measured and deliberately **not** added.
#:
#: **A shared stint-level offset.** Per-race mean stint-level error has sd 0.2345 s beyond
#: what independent sampling explains, which over a 45-lap projection is 10.6 s of race
#: time. It is not added because it is the same effect as ``reference_shared_floor``
#: (0.248 s) seen from the other side: both say "this race is systematically off for
#: everyone". Adding it would be the third instance of the double-count Phase 8 found
#: twice, and the near-equality of the two independent measurements is the evidence.
#:
#: **A shared degradation shift.** Per-race mean stint-slope error has sd 0.0301 s/lap
#: beyond sampling, and it is real -- Barcelona 2025 ran soft degradation at 0.130 s/lap
#: against 0.056 and 0.063 in the two prior years. It is not added either, for a different
#: reason: tyre age is centred within each stint, so a slope shift sums to zero over a
#: stint and moves *when* to pit without moving race time for a fixed plan. It therefore
#: cannot improve race-time calibration, which is the criterion, and would be complexity
#: bought with nothing measurable. It belongs to a pit-timing uncertainty the strategy
#: layer would need, not to this one.
#:
#: Ceiling on the field-wide reference sd. ``k / sqrt(N)`` over-predicts at very early
#: decision laps (1.65 s fitted against 1.04 s measured at lap 5) because with a handful of
#: field laps the square-root law extrapolates past anything observed. Beyond ~1 s the
#: estimate is saying "the pace level is unknown", and a wider number does not make it
#: more unknown -- it just inflates every interval by a minute of race time.
MAX_REFERENCE_SHARED_SD = 1.0


@dataclass(frozen=True)
class NoiseModel:
    """Variance components for the simulator, fitted from pace-model residuals.

    Fitted per replayed circuit with that circuit held out, like every other constant in
    the simulation layer (Part 13, risk 5).
    """

    #: Within-stint-centred residuals, bootstrapped so the right skew survives.
    lap_residuals: np.ndarray
    stint_level_sd: float
    stint_slope_sd: float
    #: Per-car reference sampling sd is ``reference_k / sqrt(that driver's clean laps)``.
    reference_k: float = DEFAULT_REFERENCE_K
    #: Irreducible part of the per-car reference error. See the module constants.
    reference_floor: float = DEFAULT_REFERENCE_FLOOR
    #: Field-wide reference sd is ``reference_shared_k / sqrt(the field's clean laps)``.
    #: Drawn once per simulation and applied to everyone, because it is the model's pace
    #: shape being wrong for this circuit rather than any one car being misjudged.
    reference_shared_k: float = DEFAULT_REFERENCE_SHARED_K
    reference_shared_floor: float = DEFAULT_REFERENCE_SHARED_FLOOR
    #: Race-level multiplier on lap and stint noise, drawn once per simulation.
    race_scale_log_sd: float = DEFAULT_RACE_SCALE_LOG_SD
    progress_sd: float = DEFAULT_PROGRESS_SD
    #: Whether the stint already running at lap L has its level treated as observed.
    #: It should be: the reference-pace estimate was built from that stint's laps, so it
    #: has already absorbed how good this set of tyres is. Drawing a fresh level for it
    #: counts the same deviation twice and was measurably over-dispersing the intervals.
    #: Its *slope* is still drawn -- knowing a stint's level so far says little about how
    #: it will decay over the laps still to come.
    continuing_stint_level_observed: bool = True

    @classmethod
    def fit(cls, laps: pd.DataFrame, pace_model) -> NoiseModel:
        """Decompose clean-lap residuals into stint-level, stint-slope and lap components.

        A random-slope decomposition rather than two separate estimates: tyre age is
        centred within each stint before fitting, which makes the level and the slope
        orthogonal so they can be sampled independently without double-counting.
        """
        from ..models.pace import TARGET, add_reference_pace, feature_frame

        laps = add_reference_pace(laps)
        clean = laps[laps["is_clean"]].dropna(subset=[TARGET, "tyre_age"]).copy()
        clean["resid"] = clean[TARGET] - pace_model.predict(feature_frame(clean))

        levels, slopes, lap_resid = [], [], []
        for _, g in clean.groupby(["race_id", "driver", "stint_number"], sort=False):
            resid = g["resid"].to_numpy()
            if len(g) < MIN_LAPS_FOR_STINT_FIT or g["tyre_age"].nunique() < 5:
                # Too short to separate a slope from a level; it still informs lap noise.
                lap_resid.extend(resid - resid.mean())
                continue
            age = g["tyre_age"].to_numpy()
            age = age - age.mean()
            slope, level = np.polyfit(age, resid, 1)
            levels.append(level)
            slopes.append(slope)
            lap_resid.extend(resid - (level + slope * age))

        if not levels:
            raise ValueError("no stint long enough to fit variance components")

        # Race-level scale, fitted rather than assumed: how much each race's residual
        # spread departs from the pooled one. Races with too few clean laps to estimate a
        # spread from are dropped rather than counted as quiet ones.
        per_race = clean.groupby("race_id")["resid"].agg(["std", "size"])
        per_race = per_race[per_race["size"] >= MIN_LAPS_FOR_RACE_SCALE]
        pooled = float(clean["resid"].std())
        scale_log_sd = (
            float(np.std(np.log(per_race["std"] / pooled)))
            if len(per_race) >= MIN_RACES_FOR_RACE_SCALE and pooled > 0
            else DEFAULT_RACE_SCALE_LOG_SD
        )

        return cls(
            lap_residuals=np.asarray(lap_resid),
            stint_level_sd=float(np.std(levels)),
            stint_slope_sd=float(np.std(slopes)),
            race_scale_log_sd=scale_log_sd,
        )

    def reference_sd(self, clean_laps_observed: int) -> float:
        """Per-car reference error: a sampling term that shrinks, plus a floor that does not."""
        n = max(clean_laps_observed, 1)
        return float(np.hypot(self.reference_floor, self.reference_k / np.sqrt(n)))

    def reference_shared_sd(self, field_clean_laps: int) -> float:
        """Field-wide reference error, same shape: sampling plus an irreducible floor."""
        n = max(field_clean_laps, 1)
        return float(
            min(
                np.hypot(self.reference_shared_floor, self.reference_shared_k / np.sqrt(n)),
                MAX_REFERENCE_SHARED_SD,
            )
        )

    def sample(
        self,
        schedule: pd.DataFrame,
        drivers: tuple[str, ...],
        clean_laps_observed: dict[str, int],
        from_lap: int,
        n_sims: int,
        rng: np.random.Generator,
        n_stint_slots: int | None = None,
    ) -> np.ndarray:
        """Draw an ``(n_sims, n_drivers, n_laps)`` additive noise field.

        Every component is added in seconds, so the total is a straightforward sum and
        the contribution of each level can be read off by zeroing the others -- which is
        what the Phase 10 explanation decomposition will need.

        ``n_stint_slots`` fixes how many stint effects are drawn rather than letting the
        schedule decide. Two pit plans then consume the generator identically and their
        difference is paired, which is what makes a strategy comparison readable -- see
        ``engine.RandomAlignment``. ``None`` is the Phase 8 behaviour.
        """
        laps = np.sort(schedule["lap_number"].unique())
        n_drivers, n_laps = len(drivers), len(laps)
        grid = (n_sims, n_drivers, n_laps)

        pivot = schedule.pivot(index="driver", columns="lap_number", values="stint_number")
        pivot = pivot.reindex(index=list(drivers), columns=laps)
        stint = pivot.to_numpy()
        age = (
            schedule.pivot(index="driver", columns="lap_number", values="tyre_age")
            .reindex(index=list(drivers), columns=laps)
            .to_numpy()
        )

        # Renumber stints from zero per driver so they index the drawn effects directly.
        stint_idx = np.zeros_like(stint, dtype=int)
        for i in range(n_drivers):
            _, stint_idx[i] = np.unique(stint[i], return_inverse=True)
        n_stints = int(stint_idx.max()) + 1

        # Tyre age centred within each stint, matching how the components were fitted.
        age_centred = np.zeros_like(age, dtype=float)
        for i in range(n_drivers):
            for s in range(n_stints):
                m = stint_idx[i] == s
                if m.any():
                    age_centred[i, m] = age[i, m] - age[i, m].mean()

        # 0. Race level: how noisy *this* afternoon turns out to be. Drawn once per
        # simulation and applied to every lap and stint effect below, because a chaotic race
        # is chaotic for everyone at once. Lognormal so the multiplier stays positive and
        # the noisy tail is longer than the quiet one, which is what the data looks like.
        race_scale = np.exp(rng.normal(0.0, self.race_scale_log_sd, size=(n_sims, 1, 1)))

        # 1. Race level: one track trend per simulation, shared by the whole field.
        progress = rng.normal(0.0, self.progress_sd, size=(n_sims, 1, 1))
        elapsed = (laps - from_lap)[None, None, :]

        # 2. Reference pace, split into the part the whole field shares and the part that
        # is private to each car. Only the private part changes who beats whom.
        shared_sd = self.reference_shared_sd(sum(clean_laps_observed.get(d, 0) for d in drivers))
        ref_shared = rng.normal(0.0, shared_sd, size=(n_sims, 1, 1))
        ref_sd = np.array([self.reference_sd(clean_laps_observed.get(d, 0)) for d in drivers])
        reference = (
            ref_shared + rng.normal(0.0, 1.0, size=(n_sims, n_drivers, 1)) * ref_sd[None, :, None]
        )

        # 3-4. Stint level and slope: persistent within a stint, resampled at each stop.
        # Unused slots are drawn and discarded so that the stream position after this
        # block -- and therefore the per-lap noise below -- is the same for every plan.
        slots = n_stints if n_stint_slots is None else n_stint_slots
        if slots < n_stints:
            raise ValueError(f"n_stint_slots={slots} is below the {n_stints} stints scheduled")
        d_idx = np.arange(n_drivers)[:, None]
        level = rng.normal(0.0, self.stint_level_sd, size=(n_sims, n_drivers, slots))
        slope = rng.normal(0.0, self.stint_slope_sd, size=(n_sims, n_drivers, slots))
        if self.continuing_stint_level_observed:
            # Stint index 0 is the one already running at lap L. Its level is in the
            # reference already; only stints started by a future stop are unknown.
            level[:, :, 0] = 0.0

        # 5. Lap level: i.i.d., bootstrapped so the slow tail survives.
        lap_noise = rng.choice(self.lap_residuals, size=grid, replace=True)

        # The reference terms are already race-level quantities measured as such, so the
        # race scale multiplies only the within-race effects. Multiplying both would count
        # the same afternoon twice, which is the double-count Phase 8 warns about.
        within_race = (
            level[:, d_idx, stint_idx]
            + slope[:, d_idx, stint_idx] * age_centred[None, :, :]
            + lap_noise
        )
        return progress * elapsed + reference + race_scale * within_race
