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
uv run python -m apexstrategy.evaluation.calibration        # Phase 8 coverage + PIT
uv run python -m apexstrategy.strategy                       # Phase 9 recommendation, demo race
uv run python -m apexstrategy.strategy --circuit austria --lap 30 --driver VER --explain
```

Always `uv run` — there is no active venv on PATH.

## Decisions (D1–D14) — do not silently reverse these

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
| **D14** | **Driver-level uncertainty is reference *sampling* error, split shared/private.** A driver's persistent model misfit (sd 0.359s) cancels between the reference estimate and the future residuals, so it is not sampled at all. What remains is **mostly field-wide** — 76% of its variance at lap 5, 45% at lap 35 — and is drawn once per sim, not per driver: a shared error cannot change who beats whom, a private one can. Do not merge them back together. |
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
  tyre set, an SC hitting everyone at once) at the level they live at. Built in Phase 8
  (`sim/noise.py`); coverage went 0.11–0.28 → 0.814 with point estimates unchanged.
- **Check a variance component for double-counting before adding it.** Both errors found
  in Phase 8 were the same effect entering twice from two directions. Calibration that is
  wrong in *both* directions at once (too wide within, too narrow between) is the
  signature: it means an effect is at the wrong level, not the wrong size.
- **Explanations are arithmetic, not SHAP.** Recommendations come from the simulator, so
  they are explained by regrouping the simulator's own addends into components that sum to
  the headline number. Not by *ablation*, which was the Phase 10 plan and is wrong here:
  disabling a component changes what the RNG consumes, so an ablated run is not the same
  run minus a term and the differences would not sum.
- **Encode then verify.** Beliefs that could silently be wrong go into config and are
  asserted against the real source by a test. This caught a wrong event matcher on day one.

## Layout

Modules are created in the phase that first needs them — absent directories are
deliberate, not missing work. Two structural invariants a directory listing does not
show: **`data/load.py` is the only module that imports fastf1**, and **`data/store.py`
validates on write**, not on read.

## Store (7 tables, `data/processed/`)

The **5** trainable circuits are austria, bahrain, barcelona, hungary, suzuka. Clean-lap
yield is 59.2% after dirty-air filtering.

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
**7 (simulator + replay validation — the hard gate, PASSED)**, **8** (hierarchical MC +
calibration), **9** (strategy engine: candidates, race-time ranking, risk),
**10** (explanation decomposition, components reconciling to 1e-15s).
Next: **11** (Streamlit: 3 pages, provenance colouring — MVP complete).

### Invariants from Phases 4–10

Numbers, tables and the full reasoning live in `docs/architecture-proposal.md`. These are
the conclusions that are expensive to rediscover:

- **Strategies are compared against common random draws, never independent ones.** Race
  time has sd ~33s, almost all shared; the difference between two plans is a few seconds.
  Paired, a comparison sharpens from sd 45.7s to 2.9s. Same seed is *not* enough —
  `RandomAlignment` declares the stint and stop draw sizes, because a plan with more stops
  draws a bigger array and decorrelates everything after it, including the per-lap noise.
  That is exactly the one-stop-vs-two-stop comparison the product exists to make.
  **`align=None` reproduces Phase 7/8 bitwise; keep it that way.**
- **Pairing is what makes the argmax honest.** Picking the best of ~88 noisy candidates is
  a winner's-curse setup; measured against a held-out seed the shrinkage is 0.00s median,
  because pairing puts `delta_se_s` at ~0.1s. Drop the pairing and the selection bias
  returns along with the variance.
- **Risk is CVaR on the *difference*, not on race time.** Absolute-time CVaR is shared
  noise and ranks nothing. `regret_s` grows with the size of the candidate set — compare
  it within a ranking, never across two.
- **The max-stint cap must bind on the "stay out" plan.** It is the plan most likely to
  break the pace model (linear degradation extrapolates past the longest stint in the
  store) and the one with no stop to filter on. Exempting it invented a 28.6s gain. A
  *baseline* supplied for audit stays exempt on purpose — what a car ran is a fact — and
  `recommend` warns instead.
- **Candidate filters measure stints from where the tyre was fitted**, not from the
  decision lap, so an immediate stop (the undercut, D4) is representable.
  `Scenario.is_legal` is used for the two-compound rule only; its first-stint bound is the
  wrong constraint here and a test pins the divergence.
- **"The engine disagrees with what the team did" is not a validation.** It is one-sided
  by construction (the baseline is one of the candidates), the executed plan is sometimes
  a damage stop rather than a strategy, and counterfactuals are unobservable in principle.
- **Explanations keep the simulator's addends, they do not re-run ablations.**
  `simulate_field(capture=True)` returns the three terms that sum to each lap time.
  Ablation by zeroing `pit_loss_s` would change `rng.choice`'s rejection sampling and move
  the stream, so the differences would not have summed. Capture is opt-in: 24 MB per run.
- **Reference pace and race progress contribute exactly zero to a comparison** — both
  plans run the same laps, so both cancel. That is why they are absent from the
  decomposition, and two tests assert it rather than the docstring claiming it.
- **The compound/tyre-age split goes through `predict` only**, as a two-factor Shapley
  average over both orderings. Reading the linear model's coefficients would work today
  and break silently under the model swap D11 leaves open.
- **The sampling row is the honesty check, not a component.** It is mean-zero, so its size
  is Monte Carlo residue; if it is not small against the headline, the headline is not
  settled. Plans with different stop counts keep an irreducible share of it.

- **Read this before "improving" calibration: the pooled pass is partly two errors
  cancelling.** Four circuits' intervals are still too wide; Suzuka's *point-estimate*
  bias pulls the pooled number back to nominal. Narrowing would fix four circuits and
  break the fifth, on a sample of five races. Reported, not tuned away.
- **Two variance double-counts are easy to reintroduce:** the continuing stint's level is
  already in the reference (don't redraw it), and the field-wide reference term correlates
  with the race-progress term (use the component orthogonal to it).
- **Validate uncertainty against the error *structure*, not one number** — within-replay
  and between-replay spread separately. The within-replay ratio is what governs strategy
  comparisons.
- **Upper noise levels are Gaussian, the lap level is not.** Levels and slopes are
  averages over many laps (CLT applies); a single lap's residual is right-skewed and a
  Gaussian would erase the slow tail that strategy risk lives in.

- **Replay is leave-one-race-out.** Both the pace model *and* the pit-loss sample are
  refitted per replayed circuit with that circuit held out. Circuit constants leak exactly
  like model weights (Part 13, risk 5).
- **Do not point the simulator at `circuits.parquet`'s pit loss.** Phase 4 charges a stop
  against local median clean pace; the simulator charges against `reference + model(lap)`,
  and the model expects an age-1 tyre to be fast. Reusing the Phase 4 number credits every
  out-lap with a fresh-tyre bonus it never enjoyed — it was the largest single error
  contributor. `engine.fit_pit_loss` measures it in the right currency.
- **No lap loop, deliberately.** With overtaking and traffic deferred (D5) no driver's lap
  time depends on another's, so the race vectorises over `(n_sims, n_drivers, n_laps)`.
  `build_schedule` still emits the per-lap frame, so restoring a loop at product-Phase 2
  is local to `simulate_field`.
- **The firewall is tested by corruption, not comparison.**
  `test_future_laps_cannot_influence_the_state` overwrites every lap after L with garbage
  and demands a bitwise-identical state. A truncated-vs-untruncated comparison only proves
  the two code paths agree; this proves no future row is read.
- **Reference pace is de-trended through the model**, not taken as a raw median. A constant
  per-circuit model offset then cancels exactly; only the shape of the error survives.
- **The reference fallback matters more than the estimator.** At lap 15 only 74% of drivers
  have 3 clean laps; at lap 5, 9%. Falling back to the *field* reference hands a backmarker
  the field's pace — it was the largest error in the first replay. Use a low quantile (0.1)
  of the driver's own green laps, because traffic is a one-sided penalty.
- **`total_laps` is a required argument**, never looked up inside, so a post-race lap count
  cannot be passed by accident.
- **Caution data is too thin to calibrate on** — 27 caution laps across all seven races
  (0.2%). The caution lap multiplier rests on one event, and
  `ASSUMED_CAUTION_PIT_MULTIPLIER` is **assumed, not fitted**: no green-flag-to-caution
  stop pairs exist in the store. Phase 8's SC slider needs multi-year data (Challenge 3).
- **The pace target is a delta**, not raw lap time: raw is not LORO-evaluable because
  circuit medians span 71–97s, and strategy needs differences anyway.
- **Evaluate degradation recovery, not just MAE** — a model can score well on MAE by
  predicting each driver's mean pace while learning nothing about tyre decay.
- **It is a race-progress coefficient, not a fuel coefficient.** Fuel burn and track
  evolution are not separately identifiable from lap-level data, and the simulator needs
  only their sum (both are strategy-independent, so both cancel when comparing strategies
  at the same lap). Do not rename it back to "fuel".
- **Prefer the pooled circuit constant as the default.** With one race per circuit,
  per-circuit constants **cannot be validated out-of-sample** (LORO holds out the only race
  a circuit has). Per-circuit values are an in-sample refinement, and Suzuka's is the
  weak one.
- **Pooled fits must include `C(circuit)`** — without it R² collapses because the model
  tries to explain 71s-vs-97s baseline lap times using compound and lap number.

## Style

Match the existing register: docstrings explain *why*, and state limitations honestly
rather than hiding them. Comments earn their place by recording a decision or a trap,
not by narrating the code.
