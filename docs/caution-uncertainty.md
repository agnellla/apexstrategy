# Cautions: diagnosis, one failed attempt, and a specification awaiting approval

Status at time of writing: **nothing implemented.** The baseline is frozen at `1e266c0`,
243 tests passing. One attempt was made, failed its acceptance criterion, and is parked on
branch `caution-regime-correction` (pushed, not merged). The specification in Part 4 has
not been approved.

Read Part 3 before reading Part 4. The specification only makes sense against the evidence
that killed the obvious alternative.

---

## Part 1 — How the problem was found

The Phase 12 mean-model audit decomposed held-out race-time error by lap category, and the
answer was not where anyone had been looking:

| Category | n | Mean bias | Share of total |
|---|---|---|---|
| **caution** | 359 | **−11.46 s** | **78%** |
| pit_out | 272 | −4.53 s | 23% |
| dirty_air | 2,792 | −0.11 s | 6% |
| clean | 6,725 | +0.08 s | −10% |

Clean laps contribute almost nothing. This also resolves an apparent contradiction in the
headline numbers: clean-lap bias is +0.268 s (pessimistic) while race-time PIT is 0.592
(optimistic). Different populations — the clean laps run slightly slow, the caution laps
run wildly fast, and cautions dominate race time.

The cause was a single caution multiplier of **1.21**, fitted on **27 caution laps, all
from one VSC**, back when the store held seven races.

Measured across 57 races:

| Condition | n | True multiplier | Model used |
|---|---|---|---|
| green (sanity check) | 54,731 | 1.000 | — |
| **safety car** | 1,622 | **1.425** | 1.21 |
| VSC deployed | 584 | 1.220 | 1.21 |
| VSC ending | 25 | 1.030 | 1.21 |
| yellow only | 1,532 | 1.006 | *(correctly not a caution)* |

Fitted on the legitimate pool only (dry, 2023–24): **SC 1.4504, VSC 1.2003,
VSC-ending 1.0310**. `ASSUMED_CAUTION_PIT_MULTIPLIER = 0.45` measured at **0.5789**.

---

## Part 2 — The attempt that failed

Implemented exactly as scoped: per-regime fitted multipliers, a fitted caution pit
multiplier, `caution_regimes` plumbed through `simulate_field`, 15 regression tests.
258 tests passing, ruff and mypy clean.

**It overshot and flipped the sign of the error.** On identical held-out folds:

| Category | n | bias before | bias after |
|---|---|---|---|
| caution | 359 | −11.461 s | **+8.963 s** |
| pit_in | 304 | −0.414 s | **+4.427 s** |
| pit_out | 272 | −4.532 s | −1.191 s |

Net race-time bias −5265 s → +4727 s: slightly smaller in magnitude, opposite in sign.
Held-out PIT mean 0.592 → 0.357, so |bias| grew from 0.092 to 0.143. 95% coverage
0.892 → 0.868, MAE 0.720 → 0.766, gate 13/20 → 12/20. Only 50% coverage improved.

Parked rather than merged. **Do not merge that branch.**

---

## Part 3 — Diagnosis: why no caution mean model works

### 3.1 The functional-form hypothesis was half right

The initial explanation was that multiplying an already-degraded pace over-slows old
tyres. Direct test — caution lap time regressed on the car's own green baseline and tyre
age, with race fixed effects:

| Regime | n | coef on green_base | coef on tyre_age | R² |
|---|---|---|---|---|
| green | 38,373 | **0.886 ± 0.011** | **+0.0341 ± 0.0010** | 0.980 |
| **safety car** | 842 | **−0.877 ± 0.516** | −0.1319 ± 0.0807 | 0.399 |
| VSC | 376 | 0.697 ± 0.695 | −0.0397 ± 0.0950 | 0.718 |

Under green, a lap tracks the car's own pace and degrades. **Under SC neither holds.** The
multiplicative form needs ≈1.45 and ≈+0.049; it gets −0.877 and −0.132.

Overshoot does grow with tyre age, as hypothesised — held-out residual by age bucket:
−1.37 s (≤5 laps), +4.25, +12.62, +19.77 s (30+), slope +0.863 ± 0.175.

### 3.2 But changing the basis fixes nothing

Leave-one-race-out on training races:

| Formulation | SC MAE | SC bias | SC age-slope | VSC MAE |
|---|---|---|---|---|
| A `m × (reference + μ + progress)` (current) | 14.78 | +2.60 | +0.438 | 8.25 |
| B `m × reference only` | 15.04 | +2.48 | +0.436 | 8.10 |
| C `m × field green median` | 14.89 | +2.57 | +0.444 | 7.99 |

**Identical.** The three bases differ by 1–2 s while the residual sd is ~17 s. Removing
degradation from the basis does not even change the age slope. The hypothesis is *rejected
as a fix* even though 3.1 is true.

### 3.3 The multiplier is not a stable quantity

Training SC ratio by circuit spans **1.19 (Las Vegas) to 1.61 (Lusail)**, race-to-race
sd **0.135**. Per-race LORO residuals run **−17.05 s to +21.48 s**, sd 12.93 s.

### 3.4 Most of a caution lap is unpredictable

SC lap time, total sd 15.0 s:

| Source | Share |
|---|---|
| between races | 36% |
| **lap-to-lap within one car** | **52%** |
| between cars in the same race | **12%** |

Only 12% is "which car", which is exactly why the basis choice is irrelevant. Median
within-car SC lap sd is **11.4 s**.

Part of the 52% is caution-period phase: the opening lap of an SC period runs at
**1.354× green** (the SC is deployed mid-lap) against **1.493×** thereafter. Splitting on
it halves SC bias (+2.60 → +1.36) and cuts MAE 5%, but leaves the age slope untouched and
does nothing for VSC (1.204 vs 1.191).

### 3.5 The +2.60 vs +8.96 discrepancy was an evaluation-definition error

Traced 496 identical held-out SC laps through both pipelines. Replay-form residual
**+9.102 s**, diagnostic-form **+9.424 s**. The difference of −0.322 s decomposes exactly:

| Component | Δ pre-caution | × 1.4504 |
|---|---|---|
| reference: estimated − true | −0.0840 s | −0.1218 s |
| model μ: scheduled − observed | −0.1654 s | −0.2399 s |
| race-progress adjustment | +0.0274 s | +0.0398 s |
| **sum** | **−0.2220 s** | **−0.3219 s** |
| *measured* | | **−0.3219 s** |

**The replay pipeline is not the cause.** Reference estimation, progress extrapolation and
tyre-age state contribute a third of a second combined. An earlier speculation that
"estimated reference amplified 1.45× plus progress drift" explained the gap was **wrong**.

The real causes: a pooled median multiplier applied to races whose true ratios span
1.19–1.61, and comparing a **15-race** training average against a **2-race** held-out
average. Training per-race residual sd is 12.93 s, so a two-race average has a standard
error of **9.15 s**; the held-out +9.4 s sits **0.48 standard errors** from the training
mean. The two numbers were never in conflict.

### 3.6 Conclusion

**The caution mean stays frozen.** No pooled, circuit-specific, regime-specific or
tyre-age-interacted mean model is defensible. Cautions are an uncertainty-heavy regime and
belong in the variance layer.

---

## Part 4 — Specification (NOT APPROVED, NOT IMPLEMENTED)

### 4.0 What the simulator carries today

| | Green lap | Caution lap |
|---|---|---|
| observed within-car sd | 1.03 s | **11.43 s** |
| simulator supplies | 1.03 s | **1.25 s** (just 1.21 × green) |

The simulator carries **11%** of the spread a caution lap actually has, and none of the
race-level severity variation at all.

### 4.1 Measured variance hierarchy

`(race, lap)` cells with ≥4 cars, usable-race pool:

| Level | Safety car | VSC |
|---|---|---|
| race | 11.11 s (47.3%) | 4.83 s (23.9%) |
| race × lap | 9.72 s (36.2%) | 6.88 s (48.6%) |
| driver × lap | 6.57 s (16.5%) | 5.18 s (27.5%) |
| **total** | **16.15 s** | **9.87 s** |

**Intraclass correlation 0.835 (SC), 0.725 (VSC).** A caution lap is overwhelmingly a
*shared* event. Sampling it independently per driver would be structurally wrong: it would
understate how much a caution moves the whole field's race time together, and would let
paired strategy comparisons cancel noise that does not cancel in reality.

Shape: SC skew **−0.50**, kurt −0.73. VSC skew +0.25, kurt −0.41. Green: skew **+6.05**,
kurt +278. The green bootstrap is wrong for cautions in scale *and in sign of skew*.

### 4.2 Race-level severity — additive, not multiplicative

Additive in seconds. The race-level term is 11.11 s of *residual* spread; a multiplicative
form re-introduces proportionality to car pace, which §3.1 rejects. Zero-mean, so it cannot
move the frozen mean. One draw per simulation, shared across all drivers, all caution laps
and both regimes.

Do **not** use the 0.135 ratio sd directly — it is a *ratio* spread that would have to be
multiplied back through car pace to become seconds, which is the coupling being removed.

**Does `race_scale` already contain this?** No. `corr(green residual sd, SC severity) =`
**−0.248** across 15 races. `race_scale` is fitted from clean-lap spread and is
uncorrelated with how severe a race's cautions are. No double-counting.

### 4.3 Lap noise — two terms, empirical, per regime

1. `race × lap`, shared across the field.
2. `driver × lap`, independent per car.

Empirical bootstrap per regime, centred to zero mean. Not the green pool.

### 4.4 Regimes — two, not three

Fold `vsc_ending` into `vsc`.

| Regime | laps | (race,lap) cells | races |
|---|---|---|---|
| SC | 845 | 73 | 16 |
| VSC | 405 | 39 | 16 |
| VSC ending | 201 | 21 | 12 |

SC and VSC differ by 1.6× in total spread — too much to pool. VSC-ending's multiplier
(1.031) is nearly green and 21 cells cannot support a third distribution.

`caution_laps: frozenset[int]` gains a companion `caution_regimes: Mapping[int, str] | None`.
**The uncertainty model needs the regime; the mean does not.**

### 4.5 Interaction with the frozen seven components

Smallest coherent change: **regime-conditional components inside `NoiseModel.sample`**, not
a parallel path in `simulate_field`. `sample` returns a single summed array, so suppression
can only happen where the components are composed. `sample` gains an optional
`caution_mask`; when `None`, behaviour and draw order are unchanged.

| # | Component | On a caution lap | Why |
|---|---|---|---|
| 1 | shared reference floor | active | sets the level |
| 2 | reference sampling | active | same |
| 3 | race noise scale | **suppressed** | scales *racing* variability; uncorrelated with caution severity |
| 4 | private reference | active | level, not variability |
| 5 | stint level | **suppressed** | tyre state does not govern a caution lap |
| 6 | stint slope | **suppressed** | same; sums to zero over a stint anyway |
| 7 | lap bootstrap | **replaced** | wrong scale, wrong skew sign |

### 4.6 Byte-identical green behaviour

Caution draws are taken **conditionally and last** in the RNG stream, after the seven
components and after pit losses. A simulation with no caution laps therefore consumes an
identical stream and produces bitwise-identical output. Assert on the array, not a metric.

### 4.7 Pit laps

**26% of caution laps coincide with a pit lap** (241 pit-in, 182 pit-out of 1,623).

| Lap type | n | resid sd | median |
|---|---|---|---|
| caution racing | 1,237 | 14.50 s | +0.10 s |
| caution pit-in | 241 | 17.66 s | −4.84 s |
| caution pit-out | 182 | 13.97 s | +11.91 s |

**No double-counting**: the caution distribution is fitted on **racing caution laps only**
and contains no pit-loss variance. Both together give √(14.5² + pit variance) ≈ 16.6 s
against an observed 17.66 s.

The pit-out median of +11.91 s is the existing pit-loss machinery working (0.82 out-lap
share × ~13.5 s caution loss). The pit-in median of −4.84 s is a **known bias in the frozen
mean** — a car peeling into the pits is not slowed by the queue the way the field is. It is
a mean error, the mean is frozen, and it must not be absorbed into variance.

`caution_pit_multiplier` stays at its current value.

### 4.8 Draw hierarchy

| Level | Term |
|---|---|
| per simulation | caution severity |
| per simulation × caution lap | shared lap term |
| per simulation × driver × lap | private lap term |
| *(none)* | no persistent per-driver caution effect is supported |

`RandomAlignment`: caution blocks are sized on **caution laps**, indexed by absolute lap
number, and **never on pit count**, so two strategies meet the same caution on the same lap.

### 4.9 Acceptance criteria

Accept only if **all** hold:

1. Green laps bitwise identical.
2. Caution-free races bitwise identical — Hungary, Austria, Suzuka must not move.
3. Caution residual sd rises toward the fitted target by regime.
4. Caution-lap bias unchanged within ±0.5 s.
5. Barcelona and Bahrain coverage improves.
6. Fixed seed reproduces exactly; paired comparisons retain the reduced-variance signature.
7. No wet race and no held-out season in the caution fit.

**Explicitly not sufficient:** aggregate 80% coverage approaching 0.80. This project has
twice produced good pooled coverage from cancelling errors.

**Rollback if:** any green lap changes; any caution-free race changes; caution bias moves
>±0.5 s; intervals widen on caution-free races; or Hungary/Austria/Suzuka coverage moves.

**On the holdout:** only **two of eleven** held-out races contain cautions. Enough to
confirm direction, not to size the effect. Acceptance is structural, not threshold-based.

### 4.10 Failure modes

| Risk | Guard |
|---|---|
| severity sampled per driver | assert two drivers share the realisation |
| green residual double-counted | component 7 replaced, not augmented |
| pit-loss double-counted | fit on racing laps only |
| SC and VSC pooled | separate distributions; assert differing sd |
| lap noise wrongly shared / wrongly independent | assert the measured ICC is reproduced |
| **wider but not better calibrated** | require coverage and PIT *shape* to improve, not sd alone |
| stream drift breaking alignment | blocks sized on caution laps, never pit count |

The sixth is the real risk. Widening is easy; calibration is what matters.

### 4.11 Why not other changes

- **No regime-specific means** — three bases gave identical MAE; the multiplier is unstable.
- **No circuit-specific caution terms** — 15 races cannot support per-circuit estimates.
- **No tyre-age interaction** — confounded with caution-period phase.
- **No change to the mean anywhere**, including the pit-in bias, which is reported rather
  than absorbed.
