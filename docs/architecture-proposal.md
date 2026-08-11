# ApexStrategy — Architecture Proposal v0.1

**Status:** Proposal for review. No implementation code yet.
**Date:** 2026-08-11

---

## Part 0 — Verdict and the six challenges that change the design

The product vision is sound and genuinely differentiated. `DATA → PREDICTIONS → SCENARIOS → DECISIONS` is the right spine, and the insistence on uncertainty + explanation is what separates this from the hundreds of F1 dashboards on GitHub.

But six things in the brief are wrong, under-specified, or will quietly destroy the project's credibility if implemented as written. These are the decisions worth arguing about before any code exists.

### Challenge 1 — `TyreLifeModel` has a mis-specified target. This is the most important issue.

The brief defines:

```
RemainingLife = final lap of current stint − current lap
```

This is **not tyre life**. It is *the lap on which the team chose to pit*. Three fatal problems:

1. **It is a decision, not a physical quantity.** Stint length is the output of a strategy call, confounded by track position, rivals, safety cars, and team orders. You would be building a model of "what teams typically do."
2. **It is circular.** The product's purpose is to recommend pit laps. A model trained to predict the pit lap teams actually chose, embedded in an engine that recommends pit laps, will recommend what teams already did. The counterfactual engine can then never disagree with history — which kills the entire post-race audit feature (Module F).
3. **It is heavily censored.** Stints ending at the chequered flag, under a safety car, or at a retirement were not tyre-limited. In a typical race, a large share of stints end for non-tyre reasons. Regressing on all of them biases every prediction downward.

**Recommendation: delete `TyreLifeModel` as a primary model.** Replace it with:

- **A pace/degradation model** (physically grounded, well-identified, plenty of data — every clean lap is a training row instead of every stint being one row).
- **"Remaining useful life" becomes a *derived output of the strategy layer***, not an input: the lap at which projected cumulative pace loss exceeds the pit-loss-adjusted benefit of fresh rubber. That is a real definition of tyre life, and it falls out of the economics for free.
- **Optionally, later: a `TeamPitBehaviourModel`**, honestly labelled as *behavioural*, used only to make simulated rivals act plausibly. This is legitimately useful and intellectually honest — it just isn't a physics model and must never be called one.

If you want to keep an explicit tyre-life model for the portfolio story, the statistically correct framing is **survival analysis with right-censoring** (`lifelines`, Cox PH or AFT), where SC-ended / race-end / retirement stints are censored observations. That is a genuine differentiator and shows real statistical maturity. But it should be a Phase 8+ enrichment, not the MVP foundation.

**Trade-off:** you lose a headline "ML model" from the MVP. You gain a system whose recommendations can actually disagree with what the team did — which is the entire product.

### Challenge 2 — You cannot simulate finishing positions without simulating rivals, and full multi-agent simulation is a research project.

"Expected finishing position" requires knowing where 19 other cars are. If you re-simulate only the ego driver, rivals stay on rails. If you simulate rivals fully, you need them to *react* to your pit stop — a game-theoretic problem with no clean solution and no ground truth.

Most portfolio projects here either ignore the problem (and produce fiction) or drown in it.

**Recommendation: tiered rival fidelity, declared explicitly in the UI on every result.**

| Tier | Rival behaviour | Valid for | Phase |
|---|---|---|---|
| **T0 — Frozen** | Rivals replay observed lap times and pit laps | Short-horizon undercut/overcut questions (≤10 laps) | MVP |
| **T1 — Simulated pace, historical pits** | Rivals driven by the pace model; pit laps fixed to actual | Full-race projections, unreactive field | MVP+ |
| **T2 — Reactive** | Rule-based rival response (cover the undercut, react to SC) | "What would really have happened" | Later |

T0 is honest, cheap, and *correct* for the highest-value question (undercut vs. extend over a few laps). Shipping T0 with the assumption stated is far stronger than shipping a fake T2. The tiering itself is a good portfolio narrative: it shows you understand the validity envelope of your own model.

### Challenge 3 — Safety cars will dominate your outcome distributions, and you cannot predict them. Do not try.

SC/VSC is plausibly the single largest source of strategic variance in F1. It is also close to irreducible noise: roughly one SC/VSC event per race on average, meaning a full 2024 season gives you on the order of tens of events. That is nowhere near enough to fit a per-lap, per-circuit hazard model with any skill.

If you build an ML safety-car predictor, it will be noise — and because SC dominates the variance, that noise will propagate into and corrupt *every* recommendation the system makes.

**Recommendation:** treat SC as an **exogenous scenario parameter**, not a prediction.
- Estimate a circuit-level base hazard rate empirically from multi-year data (a simple per-lap Poisson rate, ideally with a hierarchical/Bayesian shrinkage prior toward the global mean, because per-circuit counts are tiny).
- Expose it as a **user-controlled slider** in the dashboard: *"assume SC probability = X"*.
- Report strategy rankings **both** with and without SC risk, since the ranking often flips.

This is the project's best "knew when *not* to use ML" story. Lead with it in the write-up.

### Challenge 4 — Fuel load is not in the data. Fuel and tyre-age effects are partially collinear.

FastF1 provides no fuel data. Fuel correction is therefore an *assumption*, and a consequential one — it is the same order of magnitude as the tyre effect you are trying to measure.

Worse: within a single stint, fuel burn and tyre age both increase monotonically with lap number. They are **structurally collinear**, and you cannot separate them from one stint. Naively regressing lap time on tyre age inside a stint attributes fuel gain to tyre degradation and will *understate* degradation, sometimes to the point of showing negative deg.

**Recommendation:**
- Identify the fuel coefficient from **across-stint variation** — the same tyre age occurs at very different fuel loads in stint 1 vs. stint 3. Fit lap time ~ `f(tyre_age, compound) + β·lap_number` pooled across stints, where `β` captures fuel. This is identified only because stints start at different points in the race.
- Estimate `β` **per circuit** from data rather than hardcoding a textbook constant, and carry its standard error into the simulator.
- Sanity-check against the physical prior (order 0.03–0.04 s/lap per kg, ~1.5–2.0 kg burned per lap → roughly 0.05–0.08 s/lap). If your fitted value lands far outside that, the pipeline is broken, not the physics.
- Write this up as a **documented, versioned assumption** with a sensitivity analysis. Reviewers who know F1 will look for exactly this.

### Challenge 5 — Naive Monte Carlo will collapse your variance and produce absurdly confident answers.

The brief asks (correctly) to design sampling before implementing. Here is the trap: if you sample independent noise per lap per model and sum it, the errors average out over 50 laps. Your 90% interval on race time will come out at ±2 seconds, and the system will confidently claim P(win) = 94%. It will be nonsense.

Real racing uncertainty is dominated by **persistent, correlated** effects: a car is simply fast today; degradation is worse than expected *for the whole stint*; a safety car affects *everyone at once*.

**Recommendation: hierarchical variance decomposition.** Per Monte Carlo draw, sample at the level the effect actually lives at:

| Level | Sampled once per | Represents |
|---|---|---|
| Driver/car pace offset | sim × driver | Car is fast/slow today (persistent all race) |
| Degradation multiplier | sim × stint | This set is better/worse than nominal |
| Track evolution / temp drift | sim × race | Conditions trend |
| **SC events** | sim × race, **shared across all drivers** | The critical correlated shock |
| Pit stop duration | sim × stop | Crew variance + slow-stop tail (use a right-skewed dist, not Gaussian) |
| Lap noise | sim × driver × lap | Traffic, mistakes, tyre temp |

The variance at each level is estimable empirically from residuals (or properly via a mixed-effects model, `statsmodels`). **This is the intellectual centrepiece of the project** — it is the part that a Data Science interviewer will actually probe. Budget real time for it.

**Validation gate:** if your 80% predictive interval doesn't cover ~80% of held-out actual outcomes, the decomposition is wrong. Calibration is a first-class test, not an afterthought.

### Challenge 6 — Explanations must be arithmetic, not SHAP.

The brief rightly warns against fake explanations, then suggests SHAP. SHAP on the pace model explains *the pace model*; it cannot explain *a strategy recommendation*, because the recommendation comes from the simulator, not the regressor.

**Recommendation: decompose the strategy delta directly.** Because you control the simulator, you can run ablations and produce an explanation that is *arithmetically true*:

```
Pit now vs. extend 5 laps:  −2.8 s expected

  tyre pace advantage      −5.4 s
  pit-lane loss            +2.4 s
  traffic on rejoin        +1.4 s
  SC option value          −1.2 s
  ─────────────────────────────
  net                      −2.8 s
```

This is a real explanation, it sums correctly, and it is far more compelling than a SHAP bar chart. Use SHAP only inside the model-diagnostics view, clearly scoped as "what drives the pace model," never as "why we recommend this."

---

## Part 1 — Product architecture

The brief's diagram is broadly right. Four changes:

```
  FastF1  ──►  Race Store (Parquet)  ──►  Feature Layer
                                                │
                        ┌───────────────────────┼──────────────────┐
                        ▼                       ▼                  ▼
                  Pace/Deg Model          Pit Loss Model    Overtake/Traffic Model
                   (ML, learned)        (empirical/stats)   (logistic, calibrated)
                        └───────────────────────┼──────────────────┘
                                                ▼
                                    ┌───────────────────────┐
                                    │  Race State (t)       │  ◄── single source of truth
                                    └───────────┬───────────┘
                                                ▼
                                    ┌───────────────────────┐
                                    │  Vectorised MC Sim    │  ◄── SC hazard (exogenous)
                                    │  hierarchical noise   │  ◄── rival policy tier
                                    └───────────┬───────────┘
                                                ▼
                                    ┌───────────────────────┐
                                    │  Strategy Engine      │
                                    │  generate → evaluate  │
                                    │  → rank → decompose   │
                                    └───────────┬───────────┘
                                                ▼
                                    ┌───────────────────────┐
                                    │  Streamlit Dashboard  │
                                    └───────────────────────┘
```

Changes vs. the brief:
1. **`TyreLifeModel` removed** from the model layer (Challenge 1); remaining life is derived downstream.
2. **Overtaking/traffic promoted to a first-class model.** The brief has it as model #5, but without it *every position output is fiction*. It is load-bearing for the MVP, not optional.
3. **SC hazard enters the simulator as an exogenous input**, not as a model in the prediction layer.
4. **Race State is an explicit shared object**, not an implicit dataframe row — it is the interface between offline (data/ML) and online (sim/strategy) halves of the system.

---

## Part 2 — Repository structure

**Challenge:** the proposed tree has ~25 modules for a system that doesn't exist yet. `features/{tyre,pace,weather,fuel,race_state}` as five packages, plus `evaluation/backtesting`, plus `utils/` is premature abstraction. Empty-ish modules make a repo look *less* professional, not more.

**Recommendation — start here, split when a file exceeds ~400 lines:**

```
apexstrategy/
  pyproject.toml            # deps + tool config, not requirements.txt
  README.md
  CLAUDE.md
  docs/
    architecture-proposal.md
    assumptions.md          # the fuel/pit-loss/SC constants + their provenance
    validation-report.md    # generated
  src/apexstrategy/
    config.py               # paths, circuit constants, seeds
    data/
      load.py               # FastF1 → normalised frames
      schema.py             # pydantic models = the data contract
      clean.py              # clean-lap filtering (the crown jewels)
      store.py              # parquet read/write
    features/
      build.py              # one module until it isn't
    models/
      pace.py               # deg + fuel + compound + driver
      pit_loss.py
      overtake.py
      base.py               # fit/predict/predict_dist interface
    sim/
      state.py              # RaceState / DriverState
      scenario.py           # Scenario / PitInstruction
      engine.py             # vectorised MC core
      noise.py              # hierarchical sampling
    strategy/
      candidates.py
      evaluate.py
      explain.py
    evaluation/
      replay.py             # THE validation gate
      calibration.py
  app/
    Home.py                 # streamlit multipage
    pages/
  data/                     # gitignored
    cache/                  # fastf1
    processed/              # parquet
  models/                   # gitignored joblib
  tests/
  notebooks/                # exploration only, never imported
```

Deliberately absent: `utils/` (becomes a junk drawer), `data/validators/` (pydantic schema *is* the validator), separate `simulation/monte_carlo.py` vs `simulation/simulator.py` (same file until proven otherwise).

**Note:** `pyproject.toml` + editable install, so `from apexstrategy.sim import ...` works everywhere without `sys.path` hacks. This matters more for portfolio credibility than the folder count does.

---

## Part 3 — Data architecture

**Two-layer store, FastF1 touched exactly once per session.**

```
FastF1 API ──► ff1 cache (.ff1pkl) ──► normalised Parquet ──► feature frames
   slow, flaky        ~65 MB/race          ~2 MB/race         in-memory
```

**Recommendation: skip telemetry entirely for the MVP.** Measured from your existing cache: `car_data` is 168 MB of 432 MB across 4 races (~42 MB/race). Lap-level timing, weather, track status, and race control are all you need for strategy, and they are ~1–3 MB/race. Dropping telemetry cuts the full-season footprint from ~2.6 GB to a few hundred MB — which matters, because **your disk is 89% full with 21 GB free**.

**Reuse the existing cache.** `pitwindow/data/cache/` already has Bahrain, Monaco, Austria, and Britain 2024 warm. Point `fastf1.Cache` at a shared location rather than re-downloading. (Hungary is *not* cached and will need a first fetch.)

Normalised tables (Parquet, one file per table, partitioned by season/round):

| Table | Grain | Key fields |
|---|---|---|
| `laps` | driver × lap | lap_time, sector times, compound, tyre_age, stint, position, is_pit_in/out, track_status, cumulative_time |
| `stints` | driver × stint | start/end lap, compound, start_age, end_reason (**pit / race_end / retirement / SC**) |
| `pits` | driver × stop | lap, stationary time, pit lane delta, under_sc flag |
| `weather` | ~1 min | air/track temp, humidity, rainfall, wind |
| `track_status` | change events | SC / VSC / yellow / red, start & end lap |
| `results` | driver | grid, finish, status, points |
| `circuits` | circuit | pit lane loss, laps, overtaking factor, fuel coefficient (**derived, versioned**) |

**Format choice: Parquet, not SQLite.** The access pattern is "load whole race, compute in pandas" — columnar files win, they version well, and they need no schema migrations. Add **DuckDB** only if cross-race querying becomes painful; it reads the Parquet in place with zero migration cost. SQLite buys nothing here.

**`circuits` is a derived, versioned artifact, not hand-typed constants.** Pit loss and fuel coefficients get *estimated* and written out with their standard errors, then consumed by the simulator. Hardcoding them is where projects lose credibility.

---

## Part 4 — Data flow

```
1. ingest      session → ff1 cache                        (once, slow)
2. normalise   ff1 objects → typed frames → parquet       (pydantic-validated)
3. clean       laps → clean_laps (+ exclusion reasons)    ◄── audit trail retained
4. calibrate   clean_laps → circuit constants             (fuel β, pit loss, overtake)
5. features    clean_laps + context → model matrices
6. train       matrices → fitted models + residual variances
7. state       laps @ lap L → RaceState                   (temporal firewall here)
8. simulate    RaceState × Scenario × N → outcome samples
9. strategize  candidates → evaluate → rank → decompose
10. present    Streamlit
```

Steps 1–6 are **offline/batch**. Steps 7–10 are **online/interactive** and must be fast (<2 s). This split is the main architectural boundary in the system.

**Step 3 deserves emphasis — it is where most of the real work lives.** Exclusions: in/out laps, lap 1, laps under SC/VSC/yellow/red, laps with a car within ~1.0–1.5 s ahead (dirty air), post-damage laps, deleted laps, and per-driver outliers. **Retain every excluded lap with its exclusion reason** — this powers a data-quality view in the dashboard and is exactly the kind of rigour that reads as professional.

Expect a **large fraction of laps to be excluded**. A race is ~1,000–1,300 driver-laps; clean racing laps may be a few hundred. Measure this early — if a single race yields too few clean laps to fit a per-compound deg curve, that finding drives the whole training strategy (see Part 5).

---

## Part 5 — ML model architecture

**Guiding rule (agreeing with the brief, then applying it harder than the brief does): use ML only where the relationship is unknown and data is plentiful.** By that test, most of this system should not be ML.

| Component | Method | Why |
|---|---|---|
| **Pace / degradation** | **ML — gradient boosting (LightGBM)** + linear baseline | Genuinely unknown nonlinear interaction of compound × age × temp × circuit × driver. Thousands of clean-lap rows. ML earns its place. |
| **Fuel correction** | **Deterministic**, coefficient fitted per circuit | Physics is known and linear. ML would only overfit. |
| **Pit loss** | **Empirical/statistical** — per-circuit distribution from observed stops | ~40 observations/circuit/season. Fit a distribution, don't train a model. |
| **Overtaking** | **Logistic regression**, calibrated per circuit | Needs to be *calibrated* and *interpretable* far more than it needs to be accurate. Small data. |
| **Traffic / rejoin position** | **Deterministic simulation** | It is arithmetic on gaps, not prediction. Simulate it. |
| **Safety car** | **Exogenous Poisson rate** | Irreducible (Challenge 3). |
| **Tyre remaining life** | **Derived from the strategy layer** | Not a model (Challenge 1). |
| Team pit behaviour | ML, *later*, labelled behavioural | Only for T2 rival realism. |

So: **one real ML model in the MVP**, plus statistical components. That is the correct answer, and defending it is a stronger portfolio signal than shipping six mediocre models. The brief already says this ("Do NOT assume every model must be ML") — I'm just applying it more aggressively than the model list implies.

**LightGBM over XGBoost:** native categorical handling (circuit, compound, driver — all high-cardinality categoricals you'd otherwise one-hot), faster on small data, better small-sample behaviour. Marginal call, but it fits this data shape better.

**Every model implements a common interface:**

```python
class Model(Protocol):
    def fit(self, X, y) -> Self: ...
    def predict(self, X) -> np.ndarray: ...  # point estimate
    def predict_dist(self, X, n: int, rng) -> np.ndarray: ...  # (n, len(X)) samples
```

`predict_dist` is mandatory, not optional. The simulator only ever calls `predict_dist`. This forces uncertainty to be a first-class citizen from day one rather than something bolted on in Phase 7 — retrofitting it later would mean rewriting the simulator.

**Baselines (required before any boosted model is accepted):**
- Pace: per-compound linear in tyre age. If LightGBM doesn't beat this by a clear margin on leave-one-race-out, ship the linear model.
- Pit loss: circuit median.
- Overtake: base rate per circuit.
- Strategy: "pit when tyre is N laps old" and "copy what the team actually did."

The last one is the real bar. **If the strategy engine cannot beat "do what the team did," the product has no claim.** Be prepared for the honest possibility that it doesn't — real F1 teams are very good at this. A defensible finding is "our engine matches team decisions 78% of the time and the disagreements cluster in situation X"; that is a *good* result, not a failure.

---

## Part 6 — Race state representation

Immutable, validated, self-contained. Everything derivable is derived, so it cannot go inconsistent.

```python
@dataclass(frozen=True)
class DriverState:
    code: str
    team: str
    gap_to_leader_s: float  # SINGLE SOURCE OF TRUTH for track position
    compound: str
    tyre_age: int
    stint_number: int
    compounds_used: frozenset[str]
    pit_stops: int
    laps_completed: int
    status: Literal["running", "retired"]
    # position, gaps to adjacent cars, is_lapped are all DERIVED properties


@dataclass(frozen=True)
class RaceState:
    circuit: str
    season: int
    round: int
    lap: int
    total_laps: int
    track_status: Literal["green", "vsc", "sc", "yellow", "red"]
    drivers: tuple[DriverState, ...]
    air_temp: float
    track_temp: float
    rainfall: bool
```

**Key decisions:**
- **Position is derived from `gap_to_leader_s`, never stored.** Storing both invites contradiction. Sorting by gap is the only definition of order.
- **`compounds_used` is in the state** because the two-compound rule is a *hard constraint* on legal strategies (see Part 9). Systems that forget this recommend illegal strategies.
- **Frozen/immutable.** The simulator produces new states rather than mutating; this makes it trivially parallel and testable.
- **The state contains no future information by construction.** This is the temporal firewall (Part 13).

**Caveat to verify early:** deriving reliable per-lap gaps from FastF1 is real engineering work — lapped cars, retirements, and stale `Position` values all bite. Budget a dedicated phase for it; it is not a one-liner.

---

## Part 7 — Scenario representation

```python
@dataclass(frozen=True)
class PitInstruction:
    lap: int
    compound: Literal["SOFT", "MEDIUM", "HARD", "INTER", "WET"]


@dataclass(frozen=True)
class Scenario:
    ego: str
    plan: tuple[PitInstruction, ...]  # sorted; final stint implicitly runs to flag
    rival_policy: Literal["frozen", "sim_pace", "reactive"] = "frozen"
    sc_rate_per_lap: float | None = None  # None → circuit empirical rate
    label: str = ""
```

Deliberately minimal. A scenario is *only* the ego driver's pit plan plus the assumptions under which it's evaluated. Pace deltas, push/conserve modes, and driver-specific tweaks are **not** in v1 — they multiply the search space and cannot be validated against anything.

---

## Part 8 — Monte Carlo simulation design

**Vectorise over the simulation dimension from day one.** This is the one place where worrying about performance early is correct, because retrofitting it is a full rewrite.

- Loop over **laps** (~50–70 iterations, in Python).
- Vectorise over **(n_sims × n_drivers)** in NumPy.
- 1,000 sims × 20 drivers × 60 laps = 1.2 M lap-updates → sub-second. A naive triple loop is 30–60 s, which makes the dashboard unusable and the developer loop miserable.

**On sample count:** 1,000 sims is statistically sufficient for the MVP. Monte Carlo standard error on a probability near 0.5 is `sqrt(0.25/1000) ≈ 1.6%`. Reporting "P(win) = 54%" with ±1.6% MC error is fine; 5,000 sims only buys ±0.7% and the *model* error dwarfs both. Use 1,000 interactively, 5,000 for final reports.

**Per-lap update, executed in this order (order matters):**

```
for lap in remaining_laps:
    1. resolve SC state          (shared shock across all drivers — sampled once per sim)
    2. execute scheduled pits    (add pit loss; SC pits are much cheaper — this is the
                                  key interaction, and getting it right is most of the value)
    3. sample base lap time      = pace_model.predict_dist(compound, age, fuel, temp, driver)
                                   + driver_offset[sim]        (persistent)
                                   × deg_multiplier[sim,stint] (persistent within stint)
                                   + lap_noise[sim,driver,lap] (iid)
    4. apply dirty-air penalty   if gap_ahead < threshold
    5. resolve overtakes         Bernoulli(p) from calibrated logistic; enforce
                                 minimum following gap for cars that fail to pass
    6. accumulate cumulative_time; recompute order
```

**The correlated-sampling design (Challenge 5) is implemented in step 3's decomposition and step 1's shared shock.** These are the two things that stop variance from collapsing.

**Dependencies handled explicitly, not by independent sampling:**
- SC is **shared across drivers within a sim** — it must be, since it's what makes strategies co-vary.
- Degradation multiplier is **shared within a stint**, not resampled per lap.
- Driver pace offset is **shared across the whole race**.
- Traffic is **not sampled at all** — it emerges from the simulated gaps. Sampling it independently would double-count.

**Reproducibility:** one seeded `np.random.Generator` per scenario evaluation, seed recorded in the result object. Identical inputs must give identical outputs — this is a test (Part 11).

---

## Part 9 — Strategy engine design

**Generate → filter → evaluate → rank → decompose.**

**Generate.** Cartesian product of `pit_lap ∈ [now, now+K]` × `compound ∈ available` × `n_stops ∈ {remaining feasible}`. Naively this explodes; prune hard:
- **Regulatory filter (hard):** a dry race requires two different dry compounds. Discard plans violating it. This constraint is *strategically load-bearing* — it's often the entire reason a stop happens when it does.
- Physical filter: no stint below ~5 laps or beyond a compound's plausible maximum.
- Coarse grid first (every 3rd lap), then refine ±2 laps around the leaders.

Target **~15–30 surviving candidates**, not thousands. At 1,000 sims each, that is ~30 M lap-updates — a few seconds vectorised. Acceptable interactively.

**Evaluate.** Each candidate → outcome sample array → summary statistics.

**Rank — and here the brief needs correcting.** The brief ranks on expected finishing position and race time. Neither is the true objective.

**Recommendation: rank on expected championship points.** Position value is wildly nonlinear — P10→P9 is worth 1 point, P4→P3 is worth 3, P2→P1 is worth 7. "Expected finishing position 2.4" is a meaningless average over an ordinal scale. Points are the actual utility function teams optimise, they're linear-in-expectation (so averaging is valid), and using them is a genuine product-thinking signal.

Report per strategy:
- Full **position distribution** (the primary output — not a point estimate)
- **Expected points** (ranking metric)
- P(win), P(podium), P(points)
- **CVaR@10%** — mean outcome in the worst 10% of sims → this is "risk", defined concretely instead of as the label "High"
- Expected race time (secondary/diagnostic)

**Explain** via simulator ablation decomposition (Challenge 6), not SHAP.

---

## Part 10 — Dashboard architecture

Streamlit is right for this — agreed, no reason to change. Multipage app, with two rules:

1. **All heavy work behind `@st.cache_data` / `@st.cache_resource`.** Models load once; race data caches per session; simulations cache per (state, scenario, seed).
2. **The UI never computes.** Pages call `strategy.evaluate(...)` and render. No modelling logic in `app/`. This keeps the core testable and headless.

**Provenance colouring is a hard requirement**, applied consistently across every view — the brief asks for it and it's the difference between an honest tool and a misleading one:

| Class | Meaning | Treatment |
|---|---|---|
| **Observed** | Actually happened | Solid, neutral |
| **Predicted** | Model output, current state | Solid + interval band |
| **Simulated** | Forward projection | Dashed + fan chart |
| **Counterfactual** | Did not happen | Dashed + distinct hue + explicit label |

**MVP pages: three, not six.**
1. **Race Overview** — stint timeline, positions, track status, *data-quality panel* (clean-lap yield, exclusion reasons)
2. **Strategy Simulator** — set state at lap L, define candidates, run, compare distributions
3. **Decision Panel** — the recommendation with its arithmetic decomposition

Deferred: Driver Strategy View, Tyre Analysis, Strategy Audit. These are the ones most likely to drift into "generic F1 dashboard" (§14 of the brief) — add them once the decision loop is proven.

---

## Part 11 — Testing strategy

`pytest`. The tests that actually matter here are not unit tests of feature functions:

1. **Leakage tests (structural + asserted).** Property test: `build_state(race, lap=L)` must produce identical output when the source frame is truncated at lap L versus the full race. If it doesn't, something reads the future. This turns leakage from a review question into a CI failure.
2. **Replay test — the headline validation gate.** Simulate a real race with everyone's *actual* strategies and compare to reality: finishing order rank correlation, race-time MAE. *If the simulator cannot reproduce a race it has the true strategies for, every counterfactual it produces is worthless.* This is the single most important test in the project.
3. **Simulator invariants.** Laps conserved; cumulative time monotonic; no negative gaps; retired cars stay retired; two-compound rule satisfied in every simulated plan; positions form a valid permutation.
4. **Determinism.** Same seed → identical output, bitwise.
5. **Calibration tests.** Held-out coverage of 50/80/95% intervals within tolerance.
6. **Data contract tests.** Pydantic schema validation on ingest; fail loudly on FastF1 shape changes (it has breaking releases — **pin the version**).
7. **Synthetic-world test.** Generate a fake race from known ground-truth parameters, run the whole pipeline, assert recovery of those parameters. This is how you know the estimator works when you have no real ground truth for counterfactuals.

Test 7 and test 2 together are what let you claim the system is validated. Prioritise them over coverage percentage.

---

## Part 12 — Evaluation methodology

**Split strategy: leave-one-race-out (LORO), plus a strictly chronological holdout.**

Never random-split laps — laps within a race are massively correlated (same track, weather, cars), so random splits leak and will report absurdly good MAE. LORO matches the real use case: *predict a race you have never seen*. Report mean ± spread across held-out races, since between-race variance is the number that matters.

Chronological holdout (train ≤ round N, test > N) additionally guards against season-level drift (car development, regulation changes). Both, because they catch different failures.

**Metric hierarchy — component metrics are necessary but not sufficient:**

| Level | Metric | Target (hypothesis, to be revised from data) |
|---|---|---|
| Pace model | MAE on clean laps (LORO) | < 0.30 s |
| Pit loss | MAE | < 1.5 s |
| Overtake | Brier score + calibration curve | beats base rate |
| **Simulator** | **Race-time MAE, finish-order Spearman ρ** | **the gate** |
| **Uncertainty** | **Interval coverage @50/80/95** | **within ±5pp** |
| Strategy | Agreement with actual team decisions; regret vs. observed outcome | characterise, don't optimise |

The brief is right that better MAE ≠ better decisions. **Concretely: evaluate at the decision level.** For every real pit decision in the dataset, ask the engine at lap L−1 what it recommends, then compare. Where the engine disagrees with a team, that disagreement is either an insight or a bug — and investigating which is the most interesting analysis in the project.

**Be honest about the ceiling:** counterfactual outcomes have no ground truth. You can validate the *simulator* (replay) and the *calibration*, but you can never validate a specific counterfactual claim. Say so explicitly in the UI and the write-up. Overclaiming here is the most common way projects like this lose credibility with expert reviewers.

---

## Part 13 — Data leakage risks

Ranked by likelihood of actually happening:

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Target leakage in tyre life** — `EndLap`, `StintLength`, `MaxTyreLife` | Eliminated architecturally by dropping the model (Challenge 1) |
| 2 | **Temporal firewall breach** — features computed on full-race frames | `build_state(race, lap=L)` truncates *before* any computation. Structural prevention + property test. |
| 3 | **Cross-race leakage** — random lap splits | LORO only. No random splits, ever. |
| 4 | **Normalisation leakage** — scalers/encoders fit on full data | Fit inside sklearn `Pipeline`, inside the CV fold. |
| 5 | **Circuit-constant leakage** — pit loss / fuel β fitted on the test race | Fit constants per-fold too. Easy to miss; treat circuit constants as model parameters, not data. |
| 6 | **Survivorship** — retirements excluded from training | Retain and censor explicitly. |
| 7 | **Outcome-informed cleaning** — dropping "anomalous" laps that are anomalous *because* of what happened later | Exclusion rules must depend only on information available at that lap. |

**Risk 5 is the subtle one** most projects miss. If pit loss for Hungary is estimated using Hungary's own race and Hungary is your test race, you've leaked. Circuit constants are learned parameters and must live inside the CV fold.

**The architectural principle:** make leakage *impossible* rather than *detected*. The `RaceState` object cannot contain future data because it's constructed from a truncated frame. That beats any amount of vigilance.

---

## Part 14 — Key assumptions

To be recorded in `docs/assumptions.md`, each with provenance, value, uncertainty, and a sensitivity note:

1. Fuel burn is linear; effect is linear in mass; start fuel ≈ regulation max. *(Coefficient fitted, not assumed; but linearity is assumed.)*
2. Degradation is a function of tyre age, compound, and temperature — **not** of driving style or individual set variation (unobservable in the data).
3. Pit lane loss is circuit-constant, modulo SC/VSC state.
4. Clean-air pace is recoverable by filtering to laps with no car within ~1.0–1.5 s ahead.
5. Rivals do not react to the ego strategy (T0/T1). **Explicitly false in reality**; bounded by restricting T0 claims to short horizons.
6. SC hazard is constant per lap within a circuit (a Poisson process). Reality has lap-1 and late-race clustering; revisit if data supports it.
7. Driver pace offsets are stable within a race.
8. Weather is known/static over the simulation horizon — **dry races only in the MVP.** Wet races need a compound-crossover model and are out of scope.
9. Track evolution is monotonic improvement.
10. Points maximisation is the objective (ignores championship context, reliability management, team orders).

Assumptions 5 and 8 are the biggest limits on validity. Say so up front rather than letting a reviewer find them.

---

## Part 15 — Risks and unknowns

| Risk | Severity | Notes / mitigation |
|---|---|---|
| **Insufficient clean laps per race** | **High** | Measure in Phase 2. If a race yields too few, pooling across races becomes mandatory earlier — this is why multi-race moves to Phase 5, not 11. |
| **Deg signal swamped by fuel collinearity** | **High** | Cross-stint identification (Challenge 4). If it fails, fall back to a physically-priored coefficient and widen uncertainty. |
| **Simulator fails replay validation** | **High** | This is a project-defining gate. Budget iteration time; do not build the strategy engine on an unvalidated simulator. |
| **Variance collapse → overconfident probabilities** | **High** | Hierarchical noise + calibration tests (Challenge 5). |
| **Strategy engine can't beat "what the team did"** | Medium | Likely, at least initially. Reframe as characterisation, not superiority. Still a strong result. |
| **Overtaking model uncalibratable at low-passing circuits** | Medium | Hungary has very few on-track passes to learn from — pool across circuits with a circuit effect. |
| FastF1 breaking changes / API flakiness | Medium | Pin version; local Parquet insulates downstream code. |
| Gap derivation harder than expected | Medium | Dedicated phase; lapped cars and retirements are the traps. |
| **Disk space (21 GB free, 89% full)** | Medium | Skip telemetry; reuse existing cache. |
| Scope creep into a generic dashboard | Medium | Three MVP pages only. |
| **Total scope** | **High** | Full vision is realistically 150–250 hours. MVP as scoped below is ~40–60. |

---

## Part 16 — MVP vs. later

**MVP — a coherent, defensible decision-intelligence product:**
- Dry races only, 2024, ~6–8 circuits (reuse the 4 cached + Hungary + a high-deg track)
- Clean-lap pipeline with retained exclusion audit
- Pace/degradation model (LightGBM) + linear baseline, `predict_dist` on both
- Empirical pit-loss distribution; fitted per-circuit fuel coefficient
- Calibrated overtaking model (pooled, circuit effect)
- `RaceState` / `Scenario` with temporal firewall
- Vectorised MC engine, hierarchical noise, exogenous SC slider, **T0 + T1 rivals**
- **Replay validation report** — the credibility centrepiece
- Strategy engine: ~15–30 candidates, points-based ranking, CVaR risk, arithmetic decomposition
- Streamlit: 3 pages, provenance colouring

**Later:** survival-based tyre-life model (censored); T2 reactive rivals; team pit-behaviour model; full post-race audit page; wet races and compound crossover; 2023+2025 expansion; full backtest suite across all decisions; SHAP diagnostics view; driver-specific deg effects.

The MVP already tells the complete portfolio story: *data engineering → validated simulation → propagated uncertainty → ranked decisions with honest explanations*. Everything in "later" deepens it but none of it is required for coherence.

---

## Part 17 — Recommended tech stack

| Layer | Choice | Rationale / alternative rejected |
|---|---|---|
| Python | **3.11** | Not 3.13 (currently installed): scientific-stack wheel availability is more reliable one or two minors back. Use `uv` or `pyenv`. |
| Env | **`uv` + `pyproject.toml`** | Fast, reproducible lockfile. Rejected: bare `requirements.txt`. |
| Data | fastf1 (**pinned**), pandas, numpy, pyarrow | — |
| Storage | **Parquet**; DuckDB only if needed | Rejected SQLite: no benefit for whole-race column reads. |
| Validation | **pydantic v2** | Schema *is* the validator; replaces a bespoke `validators/` module. |
| ML | scikit-learn + **LightGBM** | Over XGBoost: native categoricals, better small-data behaviour. |
| Stats | statsmodels (mixed effects for variance components); `lifelines` later | — |
| Sim | **NumPy, vectorised** | Rejected: numba/JAX — unnecessary at this scale, adds friction. |
| Dashboard | **Streamlit + Plotly** | Per the brief. Plotly over matplotlib for interactivity. |
| Testing | pytest, hypothesis (property tests for leakage) | — |
| Quality | ruff (lint+format), mypy on `sim/` and `strategy/` only | Full-repo mypy is not worth the friction here. |

Everything free and offline-capable. No paid APIs. Total footprint a few hundred MB.

---

## Part 18 — Recommended development order

The brief's vertical-slice philosophy is right. Three sequencing changes:

- **Multi-race moves from Phase 11 → Phase 4.** Waiting until the end risks baking Hungaroring-specific assumptions into the models, and Hungary is *unrepresentative*: very low overtaking, distinctive deg profile. Single-race models will also likely be data-starved. Pull several circuits as soon as the pipeline runs.
- **Replay validation becomes its own early phase (7), before the strategy engine.** Building strategy on an unvalidated simulator risks discovering at Phase 12 that the foundation is wrong.
- **A thin dashboard arrives earlier (Phase 9, minimal).** It forces clean API design and sustains motivation.

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | Repo, `pyproject`, pinned deps, shared cache pointed at existing 4 races | `import apexstrategy` works; tests run |
| 1 | Ingest + normalise → Parquet, pydantic-validated | 5 races stored; schema tests pass |
| 2 | Clean-lap pipeline + exclusion audit | **Clean-lap yield measured** — informs everything downstream |
| 3 | Gap/track-position derivation | Reconstructed order matches official results |
| 4 | **Expand to 6–8 circuits** | Pooled dataset built |
| 5 | Circuit constants (fuel β, pit loss) fitted with CIs | Values pass physical sanity checks |
| 6 | Pace model + baseline, `predict_dist`, LORO eval | Beats baseline, or baseline ships |
| 7 | **`RaceState` + simulator + REPLAY VALIDATION** | **Reproduces real races within tolerance — hard gate** |
| 8 | Hierarchical MC noise + calibration | 80% intervals cover ~80% |
| 9 | Overtaking model; thin Streamlit (1 page) | End-to-end scenario runs in UI |
| 10 | Strategy engine: candidates, points ranking, CVaR | 20 candidates evaluated < 5 s |
| 11 | Explanation decomposition | Components sum to the total |
| 12 | Full 3-page dashboard, provenance colouring | MVP complete |
| 13 | Decision-level backtest across 2024 | Agreement rate characterised |

Phase 7 is the hard gate. Everything after it is only as trustworthy as the replay result.

---

## Decisions taken (2026-08-11)

- **D1 — `TyreLifeModel` is dropped as a primary model.** Pace/degradation is the modelled quantity; remaining useful life is derived in the strategy layer. Censored survival version deferred to Phase 8+.
- **D2 — Strategies rank on expected championship points *eventually*.** Superseded for the MVP by D5: points require finishing positions, which are Phase 2.
- **D3 — Circuit set fixed at seven, frozen until the pipeline runs end-to-end:** Bahrain, Austria, Britain, Hungary, Suzuka, Barcelona, Monaco. **Monaco is stored but excluded from model training and evaluation** as a pathological outlier (near-zero overtaking, atypical pit-lane geometry). Initial modelling pool is therefore **six circuits**.
- **D4 — Hungary 2024 is the dedicated demo/narrative race** for the post-race strategy audit (the McLaren undercut-and-swap-back sequence). It remains in the training pool; the audit story is built on it specifically.
- **D5 — The MVP decision layer is denominated in expected race time, not finishing position.** Overtaking, traffic and position outcomes move to **Phase 2** — retained in the architecture, deferred in the build. The MVP answers "how much time does this strategy gain or lose?"; Phase 2 converts time into positions and then into points (D2).
- **D6 — Python 3.11**, managed by `uv`, not the system 3.13.
- **D7 — Telemetry (`car_data`, `position_data`) is not ingested.** Measured: 337 MB of the 432 MB existing cache. Lap-level essentials are ~1.2 MB/race.
- **D9 — Two front-ends, with different jobs.** Streamlit remains the *internal* tool (diagnostics, data-quality, replay validation) because it is fast to build and nobody needs it to be beautiful. A separate **Next.js showcase on Vercel**, fed by **precomputed scenario JSON**, carries the UI/UX story. Rationale: Streamlit's layout primitives actively fight design control, so making it beautiful is slow *and* still looks like Streamlit. Precomputing a discretised scenario grid (pit lap × compound × SC assumption) offline removes the need for any backend — no cold starts, instant interaction, zero hosting cost. Trade-off accepted: users can only explore grid points, not arbitrary scenarios, which is invisible in a portfolio demo. **Consequence for earlier phases:** the strategy engine must emit JSON-serialisable results, not objects coupled to Streamlit.
- **D10 — Silverstone 2024 is excluded from model fitting.** Measured at **24.3% wet-compound laps** against a 5% dry-race threshold. The MVP is dry-only (Part 14, assumption 8), and the contamination is measurable rather than theoretical: its pit-loss estimate came out at 35.3 s against 20–25 s everywhere else, because in/out laps ran in damp conditions while the clean-pace reference came from dry laps. Stored for completeness and for later wet-weather work. **The training pool is now five circuits**, not six.

### Consequences of D5 for the sections above

D5 is a scope reduction, not an architecture change. Affected sections:

- **Part 5** — the overtaking model leaves the MVP row set. The MVP has **one ML model** (pace/degradation) plus fitted statistical components.
- **Part 8** — lap-loop steps 4–5 (dirty air, overtake resolution) become no-ops in the MVP; cars pass freely on cumulative time. **This is an explicit known bias:** projected race times will be optimistic for a car that would really be stuck in traffic. It must be surfaced in the UI and is the single largest MVP caveat.
- **Part 9** — ranking is on expected race-time delta with a CVaR@10% risk column. Position distributions are not reported until Phase 2.
- **Part 12** — the Phase 7 replay gate is assessed on **race-time MAE only**; finish-order Spearman ρ becomes a Phase 2 criterion.
- **Part 14** — assumption 10 (points maximisation) is deferred; the MVP objective is time.
- **Parts 16 & 18** — overtaking and position/points reporting move from MVP to Phase 2. Revised phase table below.

### Revised phase table (supersedes Part 18)

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **0** | Repo, `pyproject`, `uv` + Python 3.11, cache seeded from existing races | `import apexstrategy` works; tests run |
| 1 | Ingest + normalise 7 races → Parquet, pydantic-validated | Schema tests pass on all 7 |
| 2 | Clean-lap pipeline + exclusion audit | **Clean-lap yield measured** |
| 3 | Gap/track-position derivation | Reconstructed order matches official results |
| 4 | Circuit constants (fuel β, pit loss) fitted with CIs | Values pass physical sanity checks |
| 5 | Pace model + linear baseline, `predict_dist`, LORO over 6 circuits | Beats baseline, or baseline ships |
| 6 | `RaceState` + `Scenario` + temporal firewall | Leakage property test passes |
| 7 | Vectorised simulator + **REPLAY VALIDATION** | **Race-time MAE within tolerance — hard gate** |
| 8 | Hierarchical MC noise + calibration | 80% intervals cover ~80% |
| 9 | Strategy engine: candidates, race-time ranking, CVaR | 20 candidates evaluated < 5 s |
| 10 | Explanation decomposition | Components sum to the total |
| 11 | Streamlit: 3 pages, provenance colouring | **MVP complete** |
| 12 | Hungary 2024 post-race audit (D4) | Narrative demo ships |
| **P2** | Overtaking + traffic → positions → points | Finish-order ρ meets gate |

Phase 7 remains the hard gate. Everything after it is only as trustworthy as the replay result.

## Open questions

None blocking. Revisit after the Phase 7 replay gate.

---

## Build log

### Phase 0 — complete (2026-08-11)

Exit criterion met: `import apexstrategy` works, 13/13 tests pass, ruff clean.

| Item | Detail |
|---|---|
| Environment | `uv` 0.12.3, **Python 3.11.15** (uv-managed; system 3.13 untouched) |
| Locked versions | fastf1 3.5.3, pandas 2.3.3, numpy 2.4.6, lightgbm 4.7.0, scikit-learn 1.9.0, pydantic 2.13.4, streamlit 1.59.1 — exact pins in `uv.lock` |
| Package | `src/` layout, hatchling, editable install |
| Circuit registry | `config.py` encodes D3/D4; Monaco flagged `use_for_training=False` with its reason |
| Cache seeded | 4 races (Bahrain, Monaco, Austria, Britain) from `pitwindow`, telemetry stripped: **432 MB → 4.8 MB** |
| Tests | 8 config + 5 smoke; `slow` marker isolates network tests |
| Quality | ruff configured (E, F, I, UP, B, SIM); mypy scoped to `sim/` + `strategy/` |
| VCS | `git init`; `data/` and `models/` gitignored |

**Convention established — encode then verify.** Beliefs that could silently be wrong are
written into config and asserted against the real source by a test, rather than being
trusted. This immediately paid for itself: `test_rounds_match_real_schedule` caught that
the Barcelona event matcher was `"Spain"` (the `Country` column) when `EventName` is
`"Spanish Grand Prix"`. All seven round numbers were correct. `"Austria"` was also
tightened to `"Austrian"` to guarantee it can never collide with `"Australian"`.

**Guard tests worth keeping.** `test_cache_contains_no_telemetry` fails if `car_data` or
`position_data` re-enter the cache, since that silently inflates the footprint ~90x (D7).
`test_python_is_311` pins the interpreter (D6). `test_circuit_set_is_frozen_at_seven`
enforces D3's freeze until the pipeline runs end-to-end.

**Footprint note.** `.venv` is 718 MB. With ~21 GB free at start, headroom is adequate but
not generous; the telemetry decision (D7) matters more than it first appeared.

**Not yet built** (deliberately — no premature abstraction): `data/`, `features/`,
`models/`, `sim/`, `strategy/`, `evaluation/`, `app/`. These are created in the phase that
first needs them.

### Phase 1 — complete (2026-08-12)

Exit criterion met: all 7 races ingested, normalised, and passing their contracts. 52/52
tests pass, ruff clean.

**D8 — the prior `pitwindow` project is mined, not restarted.** Roughly half its data
layer was directly reusable; the other half was built around the model D1 deletes and
carried the Challenge 4 collinearity bug.

| pitwindow component | Disposition | Reason |
|---|---|---|
| `loader.py` | **Ported** | Clean "dumb fetch" isolation of the FastF1 API; already `telemetry=False` |
| weather `merge_asof` | **Ported** | The explicit timedelta-resolution coercion is hard-won knowledge |
| flag-don't-drop cleaning | **Ported and extended** | Now records *which* reason, not just a boolean |
| `race_id` convention | **Ported** | — |
| `add_gap_features` | **Deferred to Phase 3** | Right approach, but a Python loop that its own docstring admits breaks around pit stops and lapped traffic |
| `fuel_correction.py` | **Demoted to a prior** | Baked 0.035 s/kg × 100 kg into stored lap times at ingest, so every downstream model inherited an unfitted constant |
| `build_processed_stints` | **Rebuilt** | The D1-era artifact; `degradation_slope` regressed fuel-corrected time on lap number *within* a stint, which is Challenge 4 exactly |
| `ended_by_pit` | **Replaced** | See below |

**Store:** 6 tables, 228 KB total. 8,304 laps / 432 stints / 296 pits / 140 results /
1,137 weather / 29 track-status intervals. **6,062 clean training laps** across the six
non-Monaco circuits.

#### Three bugs the contracts caught

1. **`ended_by_pit` was wrong for every non-finisher.** pitwindow computed it as
   "stint's last lap ≠ race's last lap", which labels *lapped finishers* and
   *retirements* as pit stops. Monaco 2024 alone: 4 genuine retirements and 9 lapped
   finishers, all 13 mislabelled. Correcting it required the `results` table
   (`ClassifiedPosition`), which pitwindow never built. Now a 3-way `end_reason`
   (pit / race_end / retirement) with `is_censored`.

2. **Red-flagged laps blew the lap-time bound — correctly.** Monaco 2024 lap 1 carries
   track status `1254` (AllClear + Yellow + SC + Red) and a ~2,495 s lap time, because
   the race was suspended mid-lap. The data was right and the bound was wrong. Fix:
   loosen the raw bound and add `ConditionalSpec`, which applies a tight bound
   (40–300 s) only to laps that are `is_clean` — the rows that actually train models.

3. **A prioritised label was used as a semantic flag.** `under_caution` on pit stops was
   derived from `exclusion_reason`, but that label is lossy by construction: Monaco's
   red-flag stops are labelled `lap_1`, since lap-1 has higher priority than red-flag.
   All 16 came back green-flagged, which would have poisoned the Phase 4 pit-loss fit
   with ~2,400 s "stops". Now read from `track_status` directly. Regression-tested.

#### Findings that inform later phases

- **32.4% of stints are censored** (140/432). A third of stint lengths are not
  strategy-driven observations. Direct empirical support for D1.
- **Clean-lap yield is 87.1%**, higher than expected — but dirty air is not yet
  filtered (Phase 3), so the real figure will be materially lower. Excluding wet laps
  costs Silverstone alone 183 laps (19% of that race).
- **Green-flag pit-lane times are already physically sensible**: Hungary 21.8 s median,
  Britain 29.4 s. Good Phase 4 input, and a useful prior sanity check.
- **Lap times are stored raw**, verified by a test asserting they trend *downward*
  across a race (fuel burn-off). This detects a fuel correction silently reappearing
  at ingest.

### Phases 2 + 3 — complete (2026-08-12), merged

Merged because dirty-air exclusion (Phase 2) is impossible without gaps (Phase 3).
67/67 tests pass, ruff clean.

**Gap reconstruction.** FastF1 has no gap column. Gaps are differenced from cumulative
session time between drivers at the same lap number. Added to `laps`:
`gap_to_leader_s`, `gap_ahead_s`, `entering_gap_s`, `derived_position`, `in_dirty_air`.

**Exit criterion met, with a caveat about what it proves.** Reconstructed order agrees
with FastF1's reported `position` on **99.9–100%** of laps at every circuit. But
FastF1's `Position` derives from the same timing feed, so this is a *consistency* check
on the ordering logic, not independent validation of the underlying times. The stronger
structural check: null `gap_ahead_s` occurs exactly **447** times against exactly **447**
(race, lap) groups — precisely one leader per lap, which the construction could not pass
by accident.

#### The entering-gap decision

`in_dirty_air` for lap *N* uses the gap measured at the end of lap *N−1* — the traffic
state the car *entered* the lap in — not the gap at the end of lap *N*.

This is load-bearing. Using the end-of-lap gap would be **selection-biased**: a car
having a fast lap closes on the car ahead and would be flagged, while the same car
having a slow lap drops back and would not. Excluding preferentially-fast laps would
bias the pace model downward. The entering gap is causally prior to the lap time it
explains.

A consequence worth stating: a car that starts a lap in second and takes the lead during
it *is* flagged, because it genuinely ran part of that lap in a wake. In the built store
9 such laps exist, and all 9 were P2 or P3 the previous lap. A test initially asserted
"the leader is never in dirty air" and failed against real data — the assertion encoded
the end-of-lap definition that was deliberately rejected. The test was wrong, not the code.

#### Clean-lap yield collapses, as predicted

| | Before dirty air | After |
|---|---|---|
| Overall clean | 87.1% | **59.2%** |
| Clean training laps (6 circuits) | 6,062 | **4,269** |

Per circuit, worst to best: Britain 41.0% (wet phase *and* traffic), Monaco 52.5%
(everyone runs nose-to-tail), Suzuka 57.6%, Bahrain 63.0%, Austria 63.4%,
Hungary 65.5%, Barcelona 65.9%.

#### Threshold sensitivity — a judgement call, made visible

Share of otherwise-clean laps excluded at each candidate cutoff:

| Threshold | Laps excluded | % of otherwise-clean |
|---|---|---|
| 0.5 s | 460 | 6.4% |
| 1.0 s | 1,576 | 21.8% |
| **1.5 s (chosen)** | **2,312** | **32.0%** |
| 2.0 s | 2,949 | 40.8% |
| 3.0 s | 3,862 | 53.4% |

1.5 s is roughly the range over which aerodynamic wake measurably costs lap time (DRS
detection sits at 1.0 s), but it is a judgement, not a measured constant — hence the
sensitivity table shipping alongside it in the build output.

#### Open recommendation for Phase 5

Discarding 32% of otherwise-usable laps is expensive. A statistically more efficient
alternative is to **retain dirty-air laps and include `entering_gap_s` as a covariate**
in the pace model, letting it estimate the wake penalty rather than assuming those laps
are unusable. That would recover the data *and* produce the traffic-penalty estimate
that the product's Phase 2 (overtaking/positions) needs anyway — for free.

The store already supports both: `entering_gap_s` and `in_dirty_air` are retained on
every lap, and `is_clean` remains the strict clean-air definition. Decide at Phase 5 by
comparing LORO error both ways. **This is a live decision, not a settled one.**

### Phase 4 — complete (2026-08-12)

Circuit constants fitted with uncertainty and written to `circuits.parquet`.
69/69 tests pass, ruff clean.

#### It is a race-progress coefficient, not a fuel coefficient

The brief called for a fuel correction. Fuel burn is real, but it is **not separately
identifiable** from lap-level race data: track evolution also makes later laps faster and
is also a function of lap number. Any regression on lap number captures their sum.
Splitting them requires an assumption, not an estimate.

For the decision layer this is immaterial, and claiming otherwise would be false
precision. The simulator needs the *total* effect of being N laps into a race, and both
components are strategy-independent — neither changes with when a driver pits, so both
cancel when comparing two strategies at the same lap. The constant is named for what it
measures.

#### Identification held up

Within a stint, tyre age and lap number are perfectly collinear. Identification comes
entirely from across-stint variation, and the data supports it: the lap-number/tyre-age
correlation among clean laps is **0.34–0.48** by circuit, far from degenerate. Tyre age
8–12 occurs at laps 6–12 in stint 1 and laps 36–65 in stint 3. `fit_race_progress`
reports that correlation per circuit so a future dataset approaching 1.0 is caught rather
than silently producing a meaningless split.

#### Results

| Circuit | β (s/lap) | β (%/lap) | R² | Pit loss (s) | n stops |
|---|---|---|---|---|---|
| austria | −0.027 | −0.037 | 0.75 | 20.8 | 45 |
| hungary | −0.041 | −0.049 | 0.69 | 20.4 | 40 |
| barcelona | −0.053 | −0.065 | 0.76 | 22.5 | 42 |
| bahrain | −0.068 | −0.071 | 0.84 | 24.8 | 41 |
| suzuka | −0.149 | −0.154 | 0.31 | 22.1 | 36 |
| *britain (excluded)* | *−0.059* | *−0.064* | *0.52* | *35.3* | *22* |
| *monaco (excluded)* | *−0.116* | *−0.147* | *0.71* | *23.0* | *7* |
| **_pooled** | **−0.053** | **−0.064** | **0.97** | **21.9** | **204** |

**The strongest validation in the project so far:** the pooled coefficient of
**−0.0534 s/lap** lands squarely inside the fuel-only physical prior of −0.05 to −0.08,
derived independently from published fuel-effect figures. A data-driven estimate agreeing
with physics it never saw is meaningful evidence the pipeline is sound.

Pit-loss values are likewise consistent with published F1 figures (Hungary ~20 s,
Bahrain ~25 s), and were measured over the in-lap/out-lap pair against local clean pace
rather than from raw pit-lane transit, which understates the true cost.

#### Three corrections made during the phase

1. **The pooled fit was misspecified.** Without circuit fixed effects it tried to explain
   baseline lap-time differences — 71 s at Austria to 97 s at Suzuka — using compound and
   lap number. R² was 0.11 and the coefficient −0.153, larger than every individual
   circuit. Adding `C(circuit)` took R² to **0.97** and the coefficient to −0.053.
2. **The prior check was wrong-headed.** It required the combined coefficient to be at
   least as large in magnitude as fuel burn alone, and flagged Austria and Hungary as
   implausible. That is backwards: track evolution can *oppose* the fuel effect, so a
   smaller combined value is physically ordinary. Replaced with a plausibility band plus
   a significance test.
3. **Wet stops contaminated pit loss.** Silverstone's intermediate stops measured 42.5 s
   against ~35 s for dry stops the same day. Now excluded, along with the whole race (D10).

#### Known weak spot

**Suzuka's estimate is the least reliable**: R² 0.31 and SE 0.0146, roughly 10× the SE at
Bahrain, on only 522 clean laps. Its coefficient (−0.149) is nearly 3× the pooled value.
The pooled constant should be preferred there until the pace model gives a second opinion.

#### A limitation that cannot be engineered away

With **one race per circuit**, per-circuit constants **cannot be validated out-of-sample**.
Leave-one-race-out holds out the only race a circuit has, so its own constant is
unavailable by construction — using it anyway is exactly the circuit-constant leakage
flagged in Part 13, risk 5. The pooled constant is therefore the LORO-safe default and
per-circuit values are an in-sample refinement. Resolving this properly needs multiple
seasons per circuit, which is post-MVP.

### Phase 5 — next

Pace/degradation model: LightGBM plus a linear baseline, both exposing `predict_dist`,
evaluated leave-one-race-out across the five training circuits. Two decisions land here:
whether dirty-air laps return as a covariate rather than an exclusion (Phase 2+3 log),
and whether the boosted model earns its place over the baseline.
