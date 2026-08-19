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
uv run python -m apexstrategy.data.build --seasons 2024 2025   # rebuild the store (BOTH seasons)
uv run python -m apexstrategy.evaluation.season_holdout        # fit 2024, predict 2025
uv run python -m apexstrategy.data.build --circuits bahrain   # one race, for iteration
uv run python -m apexstrategy.models.train     # LORO model comparison, fit + persist
uv run python -m apexstrategy.evaluation.replay              # Phase 7 gate, all circuits
uv run python -m apexstrategy.evaluation.replay --circuits hungary --from-laps 25
uv run python -m apexstrategy.evaluation.calibration        # Phase 8 coverage + PIT
uv run python -m apexstrategy.evaluation.calibration_report # coverage on UNSEEN races/seasons
uv run python -m apexstrategy.evaluation.validate           # learning curves, under/overfit
uv run python -m apexstrategy.strategy                       # Phase 9 recommendation, demo race
uv run python -m apexstrategy.strategy --circuit austria --lap 30 --driver VER --explain
uv run python -m apexstrategy.showcase          # Phase 11 scenario grid -> web/public/data
cd web && npm run dev                           # the Next.js showcase (D9)
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
| **D11** | **The linear family ships; LightGBM does not. Re-tested at 5× the data and confirmed.** At 23 circuits / 25,429 laps: LORO MAE **0.610 vs 0.672**, worst fold **1.12 vs 2.11**, sd across folds **0.16 vs 0.35**. A grouped-fold capacity sweep makes the mechanism explicit — the GBM fits training data far better (0.35 train) and generalises worse (0.55 held-out, gap +0.20), and more capacity widens the gap. Extra capacity buys memorisation of individual races here, not transferable structure. See D18 for what *did* improve. |
| **D12** | **Dirty-air laps stay excluded, not a covariate.** Tested fairly (fixed clean test set, only training data varied): +39% data does not improve clean-lap prediction. Columns remain in the store; revisit when product-Phase 2 needs a traffic penalty. |
| **D15** | **2025 is a held-out season, never fitted on.** `config.TRAINING_SEASON = 2024`, `HOLDOUT_SEASONS = (2025,)`, enforced in `store.trainable_laps`/`fittable_laps` and asserted by `tests/test_season_gate.py`, not remembered at call sites. LORO within 2024 only asks the model to generalise across *circuits*; a held-out season asks it to generalise across cars, tyres and regulations, which is the question anyone deploying it would ask. **Result: 18/20 replays pass on 2024, 11/20 on 2025.** Do not report the 2024 number without the 2025 one. |
| **D16** | **Races are discovered from the schedule, not hand-encoded.** Keyed on FastF1's `Location` (the venue), because `EventName` is marketing and a sponsor can rename a race without moving it. Aliases exist for the seven original keys and for venue drift (`Miami`/`Miami Gardens` are one circuit). The `round_2024` encode-then-verify check is scoped to 2024: calendars reorder, and applying a 2024 belief to a 2025 session rejects correctly matched races. |
| **D17** | **Exclusions are rules, not lists.** D10's wet-race exclusion is now a measured 5% wet-lap threshold applied per *race* — which is what D10 actually said. It reproduces the original judgement exactly (Silverstone 2024 measures 24.3%) and catches six more nobody had listed, including São Paulo 2024 at 100%. Silverstone **2023 is dry and belongs in the pool**; the old per-circuit exclusion threw it away. Monaco stays a named venue: no rule catches "pathological". |
| **D18** | **`QuadraticPaceBaseline` ships; the linear form is superseded.** Not for MAE (0.535 vs 0.539, inside fold noise) but for **bias**: the linear form's out-of-fold error curves from −0.11 s on fresh rubber to −0.26 s past forty laps, and a model linear in tyre age cannot represent that. The quadratic flattens it to −0.001/+0.005 and wins on every fold. Per-lap it is negligible; over a stint it is the stay-out decision. **No per-circuit intercept** — `reference_pace` already absorbs constant circuit offsets (Phase 6), and adding one makes the model undefined at unseen circuits. |
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
  a frame truncated at lap L, so it cannot contain future data. The same discipline applies
  across seasons (D15): every model-facing reader goes through `store.fittable_laps` /
  `trainable_laps`, which filter to `TRAINING_SEASON`. A raw `read_table("laps")` in
  modelling code is a leak, because `circuit` alone no longer identifies a race.
- **A caution-contaminated lap-time scale poisons the whole state.** `build_state`'s
  fallback lap time must come from *green* recent laps. Austria 2025 ran laps 1–3 behind a
  safety car and the unfiltered median handed every driver an 86.0 s reference against a
  71 s true pace, for a 21.5% race-time error. 2024 never triggered it. Found only by the
  held-out season.
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

**70 races ingested across 2023–2025; 57 have usable timing data** (13 of 2025's return
nothing from the API, so the holdout is 11 races, not a full season — say so rather than
implying a complete year). 25 circuits, 62,613 laps.

Training pool after the computed rules: **40 races, 23 circuits, 24,409 clean laps**,
against 5 races and 3,868 before.

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
**10** (explanation decomposition, components reconciling to 1e-15s),
**11** (precomputed scenario grid + Next.js showcase — **MVP complete**).
Next: **12** (Hungary 2024 post-race audit, the D4 narrative demo).

**Streamlit was dropped, not deferred.** D9 gave it the internal-diagnostics job and the
Phase 7–10 CLIs already do that job with richer output. Building it would have been
duplicate surface for no new information. D9's *other* half, the Next.js showcase, is
built and is where the UI/UX story lives.

### Underfit or overfit? Neither, and the diagnosis is what mattered

Learning curve for the linear form, folds are whole races:

| Training races | Train MAE | Held-out MAE | Gap |
|---|---|---|---|
| 5 | 0.517 | 0.590 | +0.073 |
| 23 | 0.537 | 0.542 | +0.005 |
| 46 | 0.538 | 0.509 | −0.029 |

**Flat.** Nine times the data moves training error by 0.02 s and the generalisation gap is
zero. All three generalisation cases (new circuit / new season / both) land within 0.04 s
of each other, which is not a virtue: a model with nothing to overfit has nothing to fail
to transfer.

The GBM sweep supplies the other half. It reaches 0.35 s on training data and 0.55 s held
out, and *more* capacity widens the gap. So the ~0.53 s held-out floor is not a capacity
problem: it is close to the irreducible noise in a single lap time.

What the residuals *did* justify was one specific term (D18). MAE was the wrong thing to
look at — the systematic bias was in tyre age, it curved, and a linear-in-age model cannot
represent a curve.

### Calibration on unseen seasons (D19) — the current headline

**Held-out season, 80% intervals cover 0.792** against nominal 0.80 (was 0.53).
50% covers 0.571, 95% covers 0.905. PIT sd 0.271 against a uniform 0.289, so widths are
about right; PIT mean 0.584 says what is left is **point-estimate bias, not spread**.

Two structural fixes did it, both found by measuring components that had been *frozen*:

1. **The reference error has a floor.** It was modelled as pure sampling noise `k/√n`,
   which goes to zero, so by lap 35 the simulator was certain it knew a driver's pace.
   Measured over 190 states across 48 races and three seasons it is `√(floor² + k²/n)`
   with floor **0.404 s** per car and **0.248 s** field-wide. The sampling term was right
   all along (0.792 measured vs 0.780 assumed); the floor was simply absent.
2. **A race is 30% noisier or quieter than average, and you cannot know which.** Per-race
   residual sd runs 0.431–2.233 s around a pooled 0.773 s. Correlation with the same
   circuit in other seasons is **0.04**, and conditioning on it makes things worse, so it
   is sampled once per simulation rather than predicted (`race_scale_log_sd`, fitted).

**Two further race-level effects were measured and deliberately not added.** A shared
stint-level offset (0.2345 s) is the same thing as the shared reference floor (0.248 s)
seen from the other side — adding it would be the third instance of the Phase 8
double-count. A shared degradation shift (0.030 s/lap) is real, but tyre age is centred
within a stint so a slope shift sums to zero over it: it moves *when* to pit, not race
time, and cannot improve the criterion being measured.

**The pooled number is again partly two errors cancelling** — this keeps happening and
should always be checked. By circuit on the held-out season: Barcelona 0.427 and Bahrain
0.675 (both badly biased optimistic), against Suzuka 0.962 and Austria 0.969
(over-covered). Barcelona 2025 ran soft degradation at **0.130 s/lap against 0.056 and
0.063** in the two prior years — a genuine regime change the training data does not
contain. **Do not widen intervals to cover it**; that would hide an informative failure.

### The out-of-season result, read this before quoting any accuracy number

**Per-lap pace transfers. Race-time accumulation and uncertainty do not.**

| Decision lap | Clean-lap MAE 2024 → 2025 | Race-time error 2024 → 2025 | 80% coverage 2025 |
|---|---|---|---|
| 15 | 0.92 → **0.64** s | 0.62 → **0.68** % | 0.71 |
| 25 | 0.75 → **0.61** s | 0.50 → **0.94** % | 0.53 |
| 35 | 0.81 → **0.59** s | 0.52 → **0.75** % | 0.62 |

Gate: **17/20 pass on 2024, 13/20 on 2025** (was 11/20 before D17/D18).

Clean-lap MAE is *better* on the held-out season at every lap, so the degradation and
race-progress structure is a property of racing rather than of 2024. But race-time error
roughly doubles and 80% intervals cover ~0.55. The per-lap errors are small and
**correlated across a race** in a way the 2024-fitted variance components do not capture,
so they accumulate into race-time bias while the intervals stay too narrow.

Barcelona 2025 fails at all four decision laps and worsens with lap (1.57% → 2.38%), which
is a circuit-level bias rather than noise. The uncertainty model is the part that most
needs refitting before any of this is quoted live.

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
