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

## Decisions (D1–D8) — do not silently reverse these

| | Decision |
|---|---|
| **D1** | **No tyre-life model.** `stint_end - current_lap` predicts *when a team chose to pit*, not tyre life: a decision not a physical quantity, 32% censored (measured), and circular inside an engine whose job is recommending pit laps. Degradation is modelled from pace; remaining life is *derived* in the strategy layer. Censored survival version is Phase 8+. |
| **D2** | Ranking on expected championship points — **deferred**, needs positions (see D5). |
| **D3** | Circuit set **frozen at 7** until the pipeline runs end-to-end: bahrain, suzuka, monaco, barcelona, austria, britain, hungary. **Monaco is stored but excluded from training** (pathological outlier). Training pool = 6. |
| **D4** | **Hungary 2024 is the demo/narrative race** (McLaren undercut-and-swap-back), still in the training pool. |
| **D5** | **MVP is denominated in expected race time, not position.** Overtaking/traffic/positions are Phase 2 of the product (post-MVP). |
| **D6** | Python 3.11 via `uv`. |
| **D7** | **No telemetry ingest** (`car_data`, `position_data`) — was 337 MB of a 432 MB 4-race cache, contributes nothing at lap granularity. |
| **D8** | The sibling `pitwindow/` project is **mined, not restarted**. See the Phase 1 build log for the component-by-component disposition. |

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
    store.py         parquet IO; validates on WRITE; `training_laps()` enforces D3
    build.py         CLI entry point
```

Modules are created in the phase that first needs them — absent directories are
deliberate, not missing work.

## Store (6 tables, ~250 KB, `data/processed/`)

`laps` (8,304) · `stints` (432) · `pits` (296) · `results` (140) · `weather` (1,137) ·
`track_status` (29). **4,269 clean training laps** across the 6 non-Monaco circuits
(59.2% clean-lap yield, after dirty-air filtering).

Models read via `store.training_laps()` — it applies both `is_clean` and the Monaco
exclusion so no model can forget D3.

### Dirty air — a live decision for Phase 5

`in_dirty_air` uses the gap entering the lap (end of lap N−1), **never** the end-of-lap
gap. Using end-of-lap would be selection-biased: closing on the car ahead is what a fast
lap looks like, so it would preferentially exclude fast laps and bias pace downward.
Consequence: a car that starts a lap P2 and takes the lead during it *is* flagged, which
is correct. Do not "fix" that.

Excluding dirty-air laps costs **32% of otherwise-clean laps** at the 1.5 s threshold.
An open alternative for Phase 5: retain those laps and add `entering_gap_s` as a pace-model
covariate instead. That recovers the data *and* yields the traffic-penalty estimate the
product's Phase 2 needs. Decide by comparing LORO error both ways — not yet settled.

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
- FastF1's Ergast/jolpi backend times out intermittently. Non-fatal; retry.
- `session.total_laps` and `session.event['RoundNumber']` are the reliable identity source.

## Phase status

Done: **0** (repo/env), **1** (ingest + contracts), **2+3** (gaps + dirty air, merged).
Next: **4** (circuit constants: fuel β and pit loss, fitted per circuit with standard
errors), 5 (pace model), 6 (RaceState), **7 (simulator + replay validation — the hard
gate)**.

Phase 7 is the credibility gate: if the simulator cannot reproduce a real race given the
true strategies, every counterfactual downstream is worthless. Do not build the strategy
engine before it passes.

## Style

Match the existing register: docstrings explain *why*, and state limitations honestly
rather than hiding them. Comments earn their place by recording a decision or a trap,
not by narrating the code.
