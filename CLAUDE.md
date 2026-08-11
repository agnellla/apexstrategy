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
```

Always `uv run` — there is no active venv on PATH.

## Decisions (D1–D12) — do not silently reverse these

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
    build.py         CLI entry point
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
**4** (circuit constants), **5** (pace model).
Next: **6** (RaceState/Scenario + temporal firewall), **7 (simulator + replay validation
— the hard gate)**, 8 (hierarchical MC + calibration).

### Pace model (Phase 5 output, `models/pace_model.joblib`)

Target is a **pace delta** — `lap_time_s − median(driver's clean laps that race)` — not
raw lap time. Raw is not LORO-evaluable (circuit medians span 71–97s), and strategy needs
differences anyway. **Open gap:** the reference uses whole-race data; the simulator must
rebuild it from laps ≤ L. Close this in Phase 6/7.

Selected model is `LinearPaceBaseline` (D11), residual sd **0.73s**. `predict_dist`
bootstraps empirical residuals — non-parametric because lap-time residuals are
right-skewed and a Gaussian understates the slow tail that drives strategy risk.

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

Phase 7 is the credibility gate: if the simulator cannot reproduce a real race given the
true strategies, every counterfactual downstream is worthless. Do not build the strategy
engine before it passes.

## Style

Match the existing register: docstrings explain *why*, and state limitations honestly
rather than hiding them. Comments earn their place by recording a decision or a trap,
not by narrating the code.
