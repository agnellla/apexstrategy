# ApexStrategy — working context

F1 race **decision intelligence**. `DATA -> PREDICTIONS -> SCENARIOS -> DECISIONS`.
The product is the decision layer; the tyre/pace models are supporting components.

Full design and the running build log: **`docs/architecture-proposal.md`**. Read it before
proposing architectural changes — it records what was already considered and rejected.

## Commands

```bash
uv sync --extra dev                            # env (Python 3.11, never system 3.13)
uv run pytest                                  # all tests
uv run pytest -m "not slow"                    # offline only
uv run ruff check --fix . && uv run ruff format .
uv run python -m apexstrategy.data.build       # rebuild the Parquet store
uv run python -m apexstrategy.data.build --circuits bahrain   # one race, for iteration
uv run python -m apexstrategy.models.train     # LORO model comparison, fit + persist
uv run python -m apexstrategy.evaluation.replay              # Phase 7 gate, all circuits
uv run python -m apexstrategy.evaluation.replay --circuits hungary --from-laps 25
```

Always `uv run` — there is no active venv on PATH.

## Decisions (D1–D13) — do not silently reverse these

| | Decision |
|---|---|
| **D1** | **No tyre-life model.** `stint_end - current_lap` predicts *when a team chose to pit*, not tyre life: a decision not a physical quantity, 32% censored (measured), and circular inside an engine whose job is recommending pit laps. Degradation is modelled from pace; remaining life is *derived* in the strategy layer. Censored survival version is Phase 8+. |
| **D2** | Ranking on expected championship points — **deferred**, needs positions (see D5). |
| **D3** | Circuit set **frozen at 7** until the pipeline runs end-to-end: bahrain, suzuka, monaco, barcelona, austria, britain, hungary. **Monaco is stored but excluded from training** (pathological outlier). See also D10. |
| **D4** | **Hungary 2024 is the demo/narrative race** (McLaren undercut-and-swap-back), still in the training pool. |
| **D5** | **MVP is denominated in expected race time, not position.** Overtaking/traffic/positions are Phase 2 of the product (post-MVP). |
| **D6** | Python 3.11 via `uv`. |
| **D7** | **No telemetry ingest** (`car_data`, `position_data`) — was 337 MB of a 432 MB 4-race cache, contributes nothing at lap granularity. |
| **D8** | The sibling `pitwindow/` project is **mined, not restarted**. See the Phase 1 build log for the component-by-component disposition. |
| **D9** | **Two front-ends.** Streamlit = internal tool (diagnostics, validation). A separate **Next.js showcase on Vercel fed by precomputed scenario JSON** carries the UI/UX story — Streamlit fights design control. Consequence: **the strategy engine must emit JSON-serialisable results**, not Streamlit-coupled objects. |
| **D10** | **Silverstone 2024 excluded from fitting** — 24.3% wet-compound laps vs a 5% dry threshold. Contamination was measured, not assumed: its pit loss came out 35.3s vs 20–25s elsewhere. **Training pool is 5 circuits**, not 6. |
| **D11** | **The linear pace baseline ships; LightGBM does not.** LORO MAE 0.556 vs 0.626, degradation error 0.0137 vs 0.0174 s/lap. With 5 circuits, the GBM learns circuit-specific structure that does not transfer. Do not re-adopt a boosted model without beating these numbers. |
| **D12** | **Dirty-air laps stay excluded, not a covariate.** Tested fairly (fixed clean test set, only training data varied): +39% data does not improve clean-lap prediction. Columns remain in the store; revisit when product-Phase 2 needs a traffic penalty. |
| **D13** | **The race-progress residual is corrected in-race, with shrinkage.** The pace model has one pooled `lap_number` coefficient; true per-circuit rates span −0.027 to −0.096 s/lap, and under LORO that gap was the dominant replay error (r = 0.993). Corrected by regressing implied reference on lap number over laps ≤ L (firewall-safe), shrunk by `n/(n+300)`. **The unshrunk estimate is worse than no correction at all** — do not remove the shrinkage. |

## Design rules that are load-bearing

- **Lap times are stored RAW.** Fuel correction is a *fitted per-circuit* transform
  (Phase 4), never baked in at ingest. `pitwindow` baked in 0.035 s/kg × 100 kg and every
  downstream model silently inherited an unfitted constant. A test asserts raw lap times
  trend *downward* across a race, which detects a correction reappearing.
- **Fuel and tyre age are collinear within a stint.** Both rise with lap number. Identify
  the fuel coefficient from *across-stint* variation only. Naive within-stint regression
  attributes fuel gain to degradation and can produce negative deg.
- **Flag, never drop.** Excluded laps stay with `is_clean=False` plus an
  `exclusion_reason`. Yield decomposition feeds the data-quality panel.
- **`exclusion_reason` is a PRIORITISED label and is lossy.** Never use it to answer a
  different question. Monaco 2024 lap 1 was red-flagged but is labelled `lap_1`. Deriving
  pit `under_caution` from it silently mislabelled 16 red-flag stops as green-flag. Read
  `track_status` directly for caution state.
- **Leakage is prevented structurally, not detected.** `RaceState` (Phase 6) is built from
  a frame truncated at lap L, so it cannot contain future data.
- **No random train/test splits.** Leave-one-race-out only. Laps within a race are
  massively correlated. Circuit constants are model parameters and must be fitted
  *inside* the CV fold.
- **Uncertainty is hierarchical, not per-lap.** i.i.d. per-lap noise cancels over a stint
  and yields absurdly confident answers. Sample persistent effects (car pace today, this
  tyre set, an SC hitting everyone at once) at the level they live at.
- **Explanations are arithmetic, not SHAP.** Recommendations come from the simulator, so
  they are explained by simulator ablation whose components sum to the headline number.
- **Encode then verify.** Beliefs that could silently be wrong go into config and are
  asserted against the real source by a test. This caught a wrong event matcher on day one.

## Layout

```
src/apexstrategy/
  config.py          paths, seeds, circuit registry (D3/D4)
  data/
    schema.py        TableSpec contracts; ConditionalSpec applies tight bounds to subsets
    load.py          ONLY module that imports fastf1
    clean.py         exclusion reasons (prioritised, mutually exclusive)
    normalize.py     FastF1 -> the 6 tables
    store.py         parquet IO; validates on WRITE; `training_laps()` enforces D3+D10
  models/
    circuit_constants.py  race-progress coef + pit loss, fitted per circuit
    pace.py          LinearPaceBaseline (ships) + GBMPaceModel (rejected, D11)
    train.py         CLI: LORO comparison, fit, persist
  sim/
    state.py         RaceState/DriverState; the temporal firewall lives in build_state
    scenario.py      Scenario/PitInstruction; two-compound rule; observed_plans
    engine.py        vectorised MC; SimComponents (LORO-scoped constants)
  evaluation/
    loro.py          leave-one-race-out for the pace model
    replay.py        Phase 7 gate + CLI
```

Modules are created in the phase that first needs them — absent directories are
deliberate, not missing work.

## Store (7 tables, ~250 KB, `data/processed/`)

`laps` (8,304) · `stints` (432) · `pits` (296) · `results` (140) · `weather` (1,137) ·
`track_status` (29) · `circuits` (8).

**3,875 clean training laps** across the **5** trainable circuits (austria, bahrain,
barcelona, hungary, suzuka). Overall clean-lap yield is 59.2% after dirty-air filtering;
4,269 clean laps exist across 6 circuits before D10 removes Silverstone.

Models read via `store.training_laps()` — it applies `is_clean` plus the Monaco (D3) and
Silverstone (D10) exclusions, so no model can forget either.

### Dirty air — settled (D12)

`in_dirty_air` uses the gap entering the lap (end of lap N−1), **never** the end-of-lap
gap. Using end-of-lap would be selection-biased: closing on the car ahead is what a fast
lap looks like, so it would preferentially exclude fast laps and bias pace downward.
Consequence: a car that starts a lap P2 and takes the lead during it *is* flagged, which
is correct. Do not "fix" that.

Excluding costs 32% of otherwise-clean laps, but adding them back as a covariate was
tested and **did not help** (D12). Columns stay in the store for product-Phase 2.

## Data gotchas (all cost real debugging time)

- `TrackStatus` is **concatenated codes** for everything active during the lap:
  `"1254"` = AllClear+Yellow+SC+Red. Substring-match, never equality.
  Codes: 1=clear, 2=yellow, 4=SC, 5=red, 6=VSC deployed, 7=VSC ending.
- **Red-flag laps and pit stops legitimately run to ~2,500 s** (Monaco 2024 lap 1 spans
  the stoppage). Bound raw columns loosely and use `ConditionalSpec` for the subset that
  matters.
- **"Lapped" ≠ "retired".** Use `results.ClassifiedPosition`; non-numeric (R/D/W/N/E)
  means not classified. Inferring retirement from lap counts mislabels lapped finishers.
- Silverstone 2024 had a wet phase — 183 INTERMEDIATE/WET laps, excluded (MVP is dry-only).
- **Suzuka 2024 was red-flagged on lap 1.** The first timed lap after a restart carries an
  ordinary `AllClear` status but runs 33–53% slow; six such laps distorted Suzuka's Phase 4
  constant threefold before the `restart` exclusion was added.
- FastF1's Ergast/jolpi backend times out intermittently. Non-fatal; retry.
- `session.total_laps` and `session.event['RoundNumber']` are the reliable identity source.

## Phase status

Done: **0** (repo/env), **1** (ingest + contracts), **2+3** (gaps + dirty air),
**4** (circuit constants), **5** (pace model), **6** (RaceState/Scenario + firewall),
**7 (simulator + replay validation — the hard gate, PASSED)**.
Next: **8** (hierarchical MC + calibration), 9 (strategy engine).

### Simulator + replay gate (Phase 7)

**Gate: median absolute race-time error < 1% of the projected window. 17/20 replays pass;
14/15 from decision lap 15 onwards.** Assessed on race time only, per D5. A 1,000-sim
full-race replay runs in 0.15 s.

| Decision lap | Clean-lap MAE | Race-time error | 80% coverage |
|---|---|---|---|
| 5 | 1.214 s | 0.946% | 0.11 |
| 15 | 0.886 s | 0.518% | 0.20 |
| 25 | 0.702 s | 0.417% | 0.28 |
| 35 | 0.758 s | 0.519% | 0.21 |

Failures: `suzuka@L5`, `suzuka@L15`, `austria@L5` — both modes already flagged upstream
(Suzuka is the known weak circuit; lap-5 states have almost no observed pace).

- **Coverage is 0.11–0.28 against a nominal 0.80, and that is expected.** Sampling is
  i.i.d. per lap, so error cancels over 40 laps. This is Challenge 5; Phase 8 fixes it.
  These are the numbers to beat — do not "improve" coverage by widening per-lap noise.
- **Replay is leave-one-race-out.** Both the pace model *and* the pit-loss sample are
  refitted per replayed circuit with that circuit held out. Circuit constants leak exactly
  like model weights (Part 13, risk 5).
- **Pit loss must be measured in the simulator's currency.** Phase 4 charges a stop
  against local median clean pace; the simulator charges against `reference + model(lap)`,
  and the model expects an age-1 tyre to be fast. Reusing the Phase 4 number credited every
  out-lap with a fresh-tyre bonus it never enjoyed: −1.73 s/out-lap, the largest single
  error contributor. `engine.fit_pit_loss` fixes this (pooled 23.2 s, in-lap share 0.144).
  **Do not point the simulator at `circuits.parquet`'s pit loss.**
- **No lap loop, deliberately.** With overtaking and traffic deferred (D5) no driver's lap
  time depends on another's, so the race vectorises over `(n_sims, n_drivers, n_laps)`.
  `build_schedule` still emits the per-lap frame, so restoring a loop at product-Phase 2
  is local to `simulate_field`.
- **Errors are decomposed by lap category and sum exactly to the total.** The unmodelled
  traffic bias is −0.07 s/dirty-air lap, −11% of total error, and one-sided (optimistic)
  as predicted.

### Race state (Phase 6)

- **The firewall is tested by corruption.** `test_future_laps_cannot_influence_the_state`
  overwrites every lap after L with garbage and demands a bitwise-identical state. A
  truncated-vs-untruncated comparison only proves the two code paths agree; this proves no
  future row is read.
- **Reference pace is de-trended through the model**, not taken as a raw median: bias
  +0.790 s → +0.045 s, MAE 0.810 → 0.463. A constant per-circuit model offset then cancels
  exactly; only the shape of the error survives.
- **The fallback matters more than the estimator.** At lap 15 only 74% of drivers have 3
  clean laps; at lap 5, 9%. Falling back to the *field* reference hands a backmarker the
  field's pace — it was the largest error in the first replay. Now a low quantile (0.1) of
  the driver's own green laps, because traffic is a one-sided penalty.
- **`total_laps` is a required argument**, never looked up inside, so a post-race lap count
  cannot be passed by accident.

### Caution data is too thin to calibrate on

**27 caution laps across all seven races (0.2%)** — Austria's late VSC plus two lap-1 red
flags. The caution lap multiplier (1.21) rests on one event, and
`ASSUMED_CAUTION_PIT_MULTIPLIER = 0.45` is **assumed, not fitted**: no green-flag-to-caution
stop pairs exist in the store. Phase 8's SC slider cannot be calibrated from this circuit
set and needs multi-year data (Challenge 3).

### Pace model (Phase 5 output, `models/pace_model.joblib`)

Target is a **pace delta** — `lap_time_s − median(driver's clean laps that race)` — not
raw lap time. Raw is not LORO-evaluable (circuit medians span 71–97s), and strategy needs
differences anyway.

Selected model is `LinearPaceBaseline` (D11), residual sd **0.73s**. `predict_dist`
bootstraps empirical residuals — non-parametric because lap-time residuals are
right-skewed and a Gaussian understates the slow tail that drives strategy risk.

**Closed at Phase 6.** `sim.state.estimate_reference_pace` rebuilds the reference from
laps ≤ L by de-trending through the model. See the Phase 6 notes below.

**Coverage is good at 4 of 5 circuits but Suzuka is materially under-covered**
(50/80/95 → 0.33/0.57/0.78). Expected: sampling is i.i.d. per lap, which Challenge 5 says
is insufficient. Phase 8 fixes this; these are the baseline numbers to beat.

Evaluate degradation recovery, not just MAE — a model can score well on MAE by predicting
each driver's mean pace while learning nothing about tyre decay.

### Circuit constants (Phase 4 output, `circuits.parquet`)

It is a **race-progress** coefficient, not a fuel coefficient: fuel burn and track
evolution are not separately identifiable from lap-level data, and the simulator needs
only their sum (both are strategy-independent, so both cancel when comparing strategies
at the same lap). Do not rename it back to "fuel".

Pooled: **−0.0534 s/lap** (R² 0.97), which lands inside the independently-derived
fuel-only physical prior of −0.05 to −0.08 — the strongest validation so far. Pooled pit
loss **21.9s** over 204 dry green-flag stops.

- **Prefer the pooled constant as the default.** With one race per circuit, per-circuit
  constants **cannot be validated out-of-sample** (LORO holds out the only race a circuit
  has). Per-circuit values are an in-sample refinement.
- **Suzuka is the weak spot**: R² 0.31, SE 10× Bahrain's, coefficient ~3× pooled. Treat
  with suspicion until the pace model gives a second opinion.
- Pooled fits **must** include `C(circuit)` — without it R² collapses to 0.11 because the
  model tries to explain 71s-vs-97s baseline lap times using compound and lap number.

Phase 7 measured what this costs: under LORO the pooled coefficient is the **dominant**
projection error (r = 0.993 against the per-circuit gap), now corrected in-race by D13.

## Style

Match the existing register: docstrings explain *why*, and state limitations honestly
rather than hiding them. Comments earn their place by recording a decision or a trap,
not by narrating the code.
