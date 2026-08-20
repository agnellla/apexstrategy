# Uncertainty architecture — frozen 2026-08-20

The simulator's job is to be honest about what it does not know. This is the final
validation of the variance model, the reasoning behind each component, and the failures it
does **not** fix. Frozen at this revision: no component may be added, removed or retuned
without a measured out-of-sample calibration gain or a demonstrated implementation defect.

Validation basis: 2023 and 2024 for fitting, **2025 held out entirely** (D15), 40 training
races over 23 circuits, 24,409 clean laps. Every number below is out of sample.

---

## 1. The architecture

Seven components at four levels. Each is listed with what it represents, how it is
obtained, and — the part that matters — *why it is at that level*.

| Level | Component | Represents | Source |
|---|---|---|---|
| Race | race noise scale | how chaotic this afternoon is | fitted, lognormal, log-sd 0.262 |
| Race | race progress | track trending faster than expected | 0.018 s/lap |
| Race | reference, shared | the field's pace level is mis-pinned | √(0.248² + 3.235²/N) |
| Driver | reference, private | this car's pace is mis-pinned | √(0.404² + 0.792²/n) |
| Stint | stint level | this set of tyres is better or worse | fitted |
| Stint | stint slope | it degrades faster or slower | fitted |
| Lap | lap residual | traffic, mistakes, temperature | empirical bootstrap |
| Stop | pit loss | crew variance and the slow-stop tail | empirical bootstrap |

**Level is the whole design.** Independent per-lap noise averages out over a stint: the
first version sampled everything at lap level and 80% race-time intervals covered 0.11.
An effect has to be sampled at the level it persists at, or a forty-lap projection
integrates it away.

### The two components added at this freeze

**The reference error has a floor.** It was modelled as pure sampling noise `k/√n`, which
goes to zero, so by lap 35 the simulator was certain it knew a driver's pace — and coverage
was worst late in a race for exactly that reason. Measured over 190 states across 48 races
and three seasons:

| Decision lap | private sd | shared sd |
|---|---|---|
| 15 | 0.561 | 0.427 |
| 25 | 0.503 | 0.353 |
| 35 | 0.423 | 0.296 |

The sampling term was right all along (0.792 measured against 0.780 assumed). The floor was
simply absent. D14 argued this misfit cancels between the reference estimate and the future
residuals, and it does — but only to the extent it is constant, and the residue is the
floor. A driver's pace in the second half of a race is genuinely not the pace they showed
in the first, and no amount of watching the first half fixes that.

**A race is 30% noisier or quieter than average, and that is not predictable.** Per-race
residual sd runs 0.431 s to 2.233 s around a pooled 0.773 s. Regressing a race's spread on
the same circuit's spread in other seasons correlates at **0.04**, and conditioning on it
makes the residual scatter *worse* (0.313 against 0.262 unconditional). Chaos is a property
of the afternoon, not the venue, so it is drawn once per simulation rather than predicted.

### Two components measured and deliberately rejected

- **Shared stint-level offset (0.2345 s beyond sampling).** This is `reference_shared_floor`
  (0.248 s) seen from the other side; both say "this race is systematically off for
  everyone". Adding it would be the third instance of the double-count Phase 8 found twice,
  and the near-equality of two independent measurements is the evidence.
- **Shared degradation shift (0.0301 s/lap beyond sampling).** Real, and Barcelona 2025 is
  a clear instance. Not added because tyre age is centred within each stint, so a slope
  shift sums to zero over the stint: it moves *when* to pit, not race time. It cannot
  improve race-time calibration and belongs to a pit-timing uncertainty the strategy layer
  would need, not to this one.

---

## 2. Calibration

### Race time — the quantity the simulator exists to produce

| Split | 50% | 80% | 95% | PIT mean | PIT sd | Diagnosis |
|---|---|---|---|---|---|---|
| **held-out season (2025)** | 0.571 | **0.792** | 0.905 | 0.584 | 0.271 | width right, mean optimistic |
| in-season LORO (2023–24) | 0.652 | 0.884 | 0.959 | 0.436 | 0.236 | intervals too wide |

Nominal is 0.80; PIT is uniform at mean 0.500, sd 0.289. **On genuinely unseen races in an
unseen season, 80% intervals cover 0.792.** Before this work they covered 0.53.

PIT sd 0.271 against 0.289 says the widths are close to right. PIT mean 0.584 says what
remains is **point-estimate bias, not width**.

### By circuit, unseen season — the pooled number hides two opposite errors

| Circuit | 80% | PIT mean | Diagnosis |
|---|---|---|---|
| barcelona | 0.427 | 0.871 | severe bias, optimistic |
| bahrain | 0.675 | 0.752 | severe bias, optimistic |
| suzuka | 0.962 | 0.295 | bias, pessimistic |
| hungary | 0.938 | 0.524 | too wide |
| austria | 0.969 | 0.478 | too wide |

Across 20 (circuit, decision-lap) cells on the unseen season: 7 severe bias, 9 milder bias
split both ways, 4 too wide. **The pooled 0.792 is partly two errors cancelling.** This
keeps happening in this project and should be checked every time.

### By stint age and compound — lap level

Per-lap intervals are **conservative on both splits**, and markedly so on the unseen season
(0.968 at nominal 0.80). Every compound and every stint-age bucket reads "too wide" out of
sample. This is a real asymmetry and it is left alone: race time is the quantity the
product reports, narrowing the lap intervals would put the race-time calibration just
achieved at risk, and an over-wide lap interval errs toward admitting ignorance.

---

## 3. Four causes, kept apart

Under-coverage has causes with opposite fixes, and only the first is a variance problem.

1. **Width** — PIT spread departs from uniform, centre does not. The only failure the
   variance model should fix. Currently: in-season intervals too wide; lap level too wide.
2. **Mean bias** — PIT centre departs from 0.5. Widening intervals here buys coverage by
   hiding an error. Currently: the dominant remaining failure out of sample.
3. **Regime shift** — mean bias whose cause is that the process changed, evidenced from
   outside the residuals. Barcelona 2025, below.
4. **Irreducible** — PIT already uniform. Lap-level MAE has been flat from 5 races to 46
   while a boosted model fits training data far better and generalises worse, so ~0.53 s is
   close to the noise floor of a single lap time, not a modelling gap.

---

## 4. Barcelona 2025 — case study, no parameter added

Soft-tyre degradation, fitted per race:

| Race | slope (s/lap) | n |
|---|---|---|
| 2023 | +0.0632 ± 0.0140 | 203 |
| 2024 | +0.0559 ± 0.0130 | 260 |
| **2025** | **+0.1299 ± 0.0124** | 301 |

About five standard errors above the prior two years, on a solid sample over a 2–25 lap age
range. It is a real shift, not an artefact, and it is why Barcelona projections run
optimistic (PIT mean 0.871).

One correction to an earlier reading: the *aggregate* driver-centred residual slope is
similar for 2024 (+0.0313) and 2025 (+0.0309). That is a compound-mix effect — 2025 ran no
hards — and it is why the shift has to be looked at per compound to be seen at all.

**Is it detectable before or during the race?** Partially, and not when it matters. The
soft-compound residual slope from laps ≤ L only:

| Race | L10 | L15 | L20 | L25 | L30 |
|---|---|---|---|---|---|
| barcelona 2025 | +0.043 | +0.043 | +0.046 | +0.060 | +0.061 |
| barcelona 2024 | +0.055 | +0.029 | +0.018 | +0.027 | +0.035 |
| austria 2025 (well calibrated) | — | +0.040 | +0.028 | +0.020 | +0.023 |

By lap 25–30 the 2025 signal separates. **At laps 10–15 it does not** — Barcelona 2024 and
a well-calibrated Austria 2025 look the same. The decisions that matter happen at 15–25, so
an in-race degradation correction would be acting on noise exactly when it was most
confident, which is the failure D13 already documented for race progress (the unshrunk
estimate was *worse than no correction at all*). No parameter is added.

---

## 5. Leakage

Asserted in `tests/test_uncertainty_firewall.py`, not argued in prose.

- **Within a race.** `NoiseModel.sample` takes a schedule, driver codes, observed clean-lap
  counts, the decision lap, a count of simulations and a generator. There is no argument
  through which a future lap could arrive, and the test pins the signature so that adding
  one is a deliberate act. The only race-specific inputs are counts of laps already
  completed.
- **Across races.** `store.fittable_laps` restricts every fit to `TRAINING_SEASONS`, so the
  held-out season cannot reach the variance components. Holding a circuit out changes the
  fitted components, the pit-loss sample and the residual pool — asserted, because an
  exclusion that were a no-op would look identical from the outside.
- **The race scale is drawn blind.** It is sampled, never looked up; two states differing
  only in driver identity draw the same scale for the same seed.

## 6. The bootstrap

The lap level resamples empirical residuals rather than drawing Gaussians, because lap-time
residuals are strongly right-skewed — a lap can go badly wrong, but there is a floor on how
fast it can go. A Gaussian erases exactly the slow tail that strategy risk lives in.

Tested: the residual pool is genuinely right-skewed; resampling preserves that skew and the
99.9th percentile to within tolerance; the bootstrap's far tail exceeds a Gaussian fitted
to the same mean and sd; and holding a circuit out changes the tail, so the extreme laps
shaping it come from other races.

---

## 7. Status: frozen

The validation found **no implementation defect**. 215 tests pass. The architecture above
is frozen and ready to be a simulator component.

Known limitations, carried forward rather than fixed:

1. **The dominant remaining error is point bias, not interval width**, and it is
   circuit-specific out of sample and signed in both directions. Nothing in a variance
   model should address it.
2. **One variance model serves both splits** — calibrated out of sample, too wide in
   sample. Fixing that properly needs more seasons, not more components.
3. **Lap-level intervals are conservative.** Race time is the reported quantity and is
   calibrated; this is left alone deliberately.
4. **Five circuits, one held-out season, 11 of its 24 races have usable timing data.**
   Every calibration figure here rests on that.
