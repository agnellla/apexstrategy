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
- **D14 — Driver-level uncertainty is reference *sampling* error, split shared/private
  (2026-08-17).** A driver's persistent model misfit (sd 0.359 s) cancels between the
  reference estimate and the future residuals, so it is not sampled. What remains is
  mostly field-wide — 76% of its variance at lap 5, 45% by lap 35 — and is drawn once per
  simulation rather than per driver, because a shared error cannot change who beats whom
  while a private one can. Do not merge the two back together.
- **D13 — The race-progress residual is estimated in-race and shrunk (2026-08-17).** The
  pace model carries one pooled race-progress coefficient; the true per-circuit rate spans
  −0.027 to −0.096 s/lap, and under LORO that mismatch was the dominant replay error
  (r = 0.993 against the Phase 4 coefficient gap). It is corrected by regressing the
  implied reference on lap number over laps ≤ L — firewall-safe, since it uses only the
  race in progress — shrunk by `n/(n+300)`. Shrinkage is load-bearing: the **unshrunk**
  estimate is worse than no correction at all.
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

### Phase 5 — complete (2026-08-12)

Pace/degradation model fitted, evaluated LORO, and persisted. 91/91 tests pass.
**Both open questions were settled by evidence, and both went against the sophisticated
option.**

#### The target is a pace *delta*, not raw lap time

Raw lap time is not LORO-evaluable. LORO holds out an entire circuit, and median clean
lap times span 71 s (Austria) to 97 s (Suzuka) — a model predicting raw time for an
unseen circuit would predict roughly the average of the others and post a huge error
that says nothing about whether it understands tyre degradation.

The target is therefore `lap_time_s − median(driver's clean laps in that race)`. This is
also what strategy actually needs: "how much do I lose staying out three more laps?" is
a difference. The absolute level only sets race duration, and the simulator knows it
from observed state.

**Known gap, flagged not hidden:** the reference is computed over the whole race. In the
simulator it must come from laps observed up to the decision lap only, so the model is
evaluated under a slightly easier reference than it will face live. Closed at Phase 6/7.

#### D11 — the linear baseline ships; LightGBM does not

| Model | MAE (mean) | MAE (worst) | RMSE | R² | Deg error |
|---|---|---|---|---|---|
| **linear baseline** | **0.556 s** | 0.743 s | 0.768 | **0.424** | **0.0137 s/lap** |
| lightgbm | 0.626 s | 0.747 s | 0.829 | 0.327 | 0.0174 s/lap |

The boosted model loses on every metric, including the one that matters most. With only
five circuits and LORO holding an entire circuit out, LightGBM learns circuit-specific
structure that does not transfer, while the constrained linear form generalises. This is
precisely the outcome Part 7 of this document demanded be tested for rather than assumed.

**Degradation recovery matters more than MAE** and is reported separately, because a
model can post excellent error by predicting each driver's mean pace while learning
nothing about how tyres decay — and such a model is useless to the strategy layer.
The linear model recovers the compound ordering (SOFT > MEDIUM > HARD) at every held-out
circuit, at physically sensible rates: HARD 0.020–0.043 s/lap, MEDIUM 0.024–0.073,
SOFT 0.056–0.122.

Worst case is Suzuka SOFT (actual 0.122 vs model 0.075), but on n=33 laps the "actual"
is itself unreliable.

#### D12 — dirty-air laps stay excluded

Adding them back with `entering_gap_s` as a covariate recovers 39% more data
(3,875 → 5,387 laps) but does **not** improve prediction. Tested fairly, holding the
test set fixed to clean laps so only the training data varies:

| Training data | MAE on clean laps |
|---|---|
| **linear, clean only** | **0.556 s** |
| linear, +dirty +gap | 0.581 s |
| lgbm, clean only | 0.626 s |
| lgbm, +dirty +gap | 0.600 s |

The extra data *helps* LightGBM and *hurts* the linear model — expected, since
`LinearPaceBaseline`'s formula has no traffic term, so those laps add noise it cannot
explain. Either way the clean-only linear model wins. The columns stay in the store, so
the decision is revisitable when the product's Phase 2 needs a traffic-penalty estimate.

#### A data-quality bug found while investigating outliers

Six "clean" laps sat 33–53% above their drivers' own race medians, all at **Suzuka
lap 3** — the first timed lap after the red-flag restart. Their track status had already
reverted to `AllClear`, so every existing rule let them through.

These six laps were enough to distort Phase 4: they dragged Suzuka's race-progress
coefficient to roughly three times pooled. Adding a `restart` exclusion — keyed off
race-control state and the *presence* of a lap time, never its value, so it is not
outcome-informed cleaning — fixed it:

| | Before | After |
|---|---|---|
| Suzuka R² | 0.31 | **0.876** |
| Suzuka SE | 0.0146 | **0.0021** |
| Suzuka coefficient | −0.149 | −0.096 |
| **Pooled R²** | 0.9945 | **0.9945** |
| Pooled coefficient | −0.0534 | −0.0479 |

#### Uncertainty: honest but incomplete

`predict_dist` bootstraps empirical residuals (non-parametric, because lap-time residuals
are right-skewed and a Gaussian would understate the slow tail that matters for risk).
Residual sd is 0.73 s. Coverage on held-out circuits:

| Circuit | 50% | 80% | 95% |
|---|---|---|---|
| austria | 0.538 | 0.834 | 0.974 |
| bahrain | 0.495 | 0.838 | 0.983 |
| barcelona | 0.486 | 0.809 | 0.948 |
| hungary | 0.436 | 0.732 | 0.935 |
| **suzuka** | **0.332** | **0.569** | **0.781** |

Four of five circuits are close to nominal. **Suzuka is materially under-covered** — its
residual spread differs from the circuits the model trained on. This is expected at this
stage: sampling is i.i.d. per lap, which is exactly the thing Challenge 5 says is
insufficient. Phase 8 adds the hierarchical structure, and these numbers are the baseline
improvement is measured against.

### Phase 6 — complete (2026-08-17)

`RaceState` / `Scenario` built, temporal firewall enforced structurally, and the Phase 5
reference-pace gap closed. Merged into the Phase 7 run because the simulator is the only
consumer that can tell whether the state is right.

#### The firewall is tested by corruption, not by comparison

`build_state` truncates to laps ≤ L as its first statement, so every field is a function
of the truncated frame by construction. The property test does not merely compare the
truncated and untruncated code paths — that only proves the two agree. It **overwrites
every lap after L with garbage** (999 s lap times, 10⁹ cumulative times, spurious pit
flags) and requires a bitwise-identical state. That proves no future row is read at all.

`total_laps` is a required argument rather than a lookup, so a caller cannot accidentally
pass a post-race lap count in place of the scheduled distance.

#### Closing the reference-pace gap — de-trending through the model

Phase 5 flagged that the pace target is measured against a whole-race median which the
simulator cannot have. The fix inverts the model rather than ignoring it: each observed
clean lap contributes `lap_time − model.predict(lap)`, an estimate of the same whole-race
reference the model trained against, and the reference is the median of those.

Measured over five circuits at decision laps 5–35 (n=397 driver-states):

| Estimator | Bias | MAE |
|---|---|---|
| raw median of clean laps (Phase 5 behaviour) | +0.790 s | 0.810 s |
| **de-trended through the model** | **+0.045 s** | **0.463 s** |

A useful property falls out: a *constant* per-circuit offset in the pace model cancels
exactly, because the implied reference absorbs `−c` and the projection adds it back. Only
the **shape** of the model's error survives — which is precisely what Phase 7 then found.

#### The fallback mattered more than the estimator

At lap 15 only **74%** of drivers have three clean laps to their name; at lap 5, **9%**.
A car in traffic generates almost none. The first version fell back to the *field's*
reference pace, which hands a backmarker the pace of the field — and it was the single
largest error source in the first replay run (Hungary's worst four race-time errors were
MAG, BOT, ALB and RIC, all backmarkers).

The replacement uses the fact that traffic is a **one-sided** penalty: a lap in a wake is
slower than clean-air pace, never faster. So the fallback takes a low quantile of the
de-trended implied references over the driver's green non-pit laps. On the drivers that
need it, the 10th percentile beat 0.2/0.3/0.5 on both bias and MAE (+0.23 s / 0.75 s
against +0.66 s / 0.95 s for the median).

#### "Retired" without knowing the future

At lap L nobody knows who is about to retire, and the state must not either. A driver is
running if their last line crossing is recent relative to the leader's — a lapped car
crosses the line every lap, a stopped car does not. A car that retires *during* lap L+1
therefore still reads as running at lap L, which is correct: that is all the pit wall
knows at that moment.

### Phase 7 — complete (2026-08-17). **The gate passes, with its failures characterised.**

Vectorised simulator plus leave-one-race-out replay validation. 134/134 tests pass,
ruff clean. A 1,000-sim full-race replay runs in **0.15 s**.

#### Result

Gate (per the D5 consequences above, assessed on race time only): **median absolute
race-time error < 1% of the projected window**.

**17 / 20 replays pass. From decision lap 15 onwards, 14 / 15 pass.**

| Decision lap | Clean-lap MAE | Clean-lap bias | Race-time error | 80% coverage |
|---|---|---|---|---|
| 5 | 1.214 s | +0.419 s | 0.946% | 0.11 |
| 15 | 0.886 s | +0.274 s | **0.518%** | 0.20 |
| 25 | 0.702 s | +0.277 s | **0.417%** | 0.28 |
| 35 | 0.758 s | +0.214 s | **0.519%** | 0.21 |

The three failures are `suzuka@L5` (2.26%), `suzuka@L15` (1.20%) and `austria@L5`
(1.03%). Both failure modes were already flagged upstream: Suzuka is the circuit Phase 4
(R² 0.31 before the restart fix) and Phase 5 (coverage 0.33/0.57/0.78) both named as the
weak spot, and lap-5 states carry almost no observed pace to build a reference from.
Reported, not excused.

**Coverage is bad and expected to be** — 0.11–0.28 against a nominal 0.80. Sampling is
i.i.d. per lap, so error cancels over 40 laps and the intervals collapse. This is
Challenge 5 exactly, it is what Phase 8 exists to fix, and these are the numbers it will
be measured against.

#### The dominant error source was identified, not merely observed

The first run showed a clean-lap bias that varied by circuit and would not average away.
It is **entirely explained by the race-progress transfer error**: the pace model carries
one pooled `lap_number` coefficient, while Phase 4 measured the true rate varying from
−0.027 s/lap (Austria) to −0.096 s/lap (Suzuka). Under LORO the model cannot know which
it faces.

Regressing each circuit's implied-reference residual on lap number recovers exactly that
gap — **r = 0.993** against the Phase 4 coefficient difference:

| Circuit | β own | β pooled | gap | implied-reference slope |
|---|---|---|---|---|
| bahrain | −0.068 | −0.048 | −0.020 | −0.019 |
| suzuka | −0.096 | −0.048 | −0.048 | −0.042 |
| barcelona | −0.053 | −0.048 | −0.005 | −0.007 |
| austria | −0.027 | −0.048 | +0.021 | +0.022 |
| hungary | −0.041 | −0.048 | +0.007 | +0.011 |

This is the Phase 4 limitation — per-circuit constants cannot be validated out-of-sample
with one race per circuit — showing up downstream as a measured cost in seconds, rather
than as a caveat.

#### D13 — the race-progress residual is corrected in-race, with shrinkage

Because the residual rate is observable *from the race in progress*, it can be estimated
from laps ≤ L without breaching the firewall — which is what a race engineer does when
the track comes to them faster than expected. The slope is fitted field-wide (fuel burn
and track evolution are shared by everyone; pooling turns twenty noisy slopes into one),
with driver level removed first.

The shrinkage is not decoration. Swept over the full replay suite:

| Shrink | Clean-lap MAE | Abs bias | Race-time error |
|---|---|---|---|
| 0 (unshrunk) | 1.075 s | 0.739 s | 0.877% |
| 150 | 0.881 s | 0.485 s | 0.620% |
| **300 (chosen)** | **0.890 s** | **0.485 s** | **0.599%** |
| 600 | 0.909 s | 0.491 s | 0.592% |
| off | 0.960 s | 0.538 s | 0.636% |

**The unshrunk estimate is worse than no correction at all** — at lap 5 it extrapolates
noise over 50 laps. Shrunk, it improves all three metrics at every decision lap. A
variant controlling for tyre age per compound was tried and was *worse* (0.909 / 0.611 /
0.764), so it was rejected.

#### A coupling bug the replay caught: pit loss in the wrong currency

The first run charged −1.73 s on every out-lap, the largest single error contributor
(−913 s pooled). Neither estimate was wrong on its own — the *coupling* was. Phase 4
measures pit loss against the driver's local median clean pace; the simulator charges it
against `reference + model.predict(lap)`, and the model expects a tyre at age 1 to be
**fast**. Reusing the Phase 4 number therefore credited every out-lap with a fresh-tyre
bonus it never actually enjoyed.

`fit_pit_loss` now measures the loss as the residual over what the model predicts for
those two laps, making the two definitions consistent by construction. Pooled loss comes
out at **23.2 s** against Phase 4's 21.9 s, with an in-lap share of **0.144**. Out-lap
error fell from −1.73 s to −0.32 s.

#### Errors are decomposed, not averaged away

Per-category contributions sum exactly to the total race-time error, so the cost of what
the MVP deliberately does not model is a number rather than a caveat:

| Category | Laps | Mean error | Total | Share |
|---|---|---|---|---|
| pit_in | 527 | −0.554 s | −292 s | −15% |
| **dirty_air** | **3,194** | **−0.070 s** | **−222 s** | **−11%** |
| pit_out | 527 | −0.324 s | −171 s | −9% |
| caution | 108 | +2.221 s | +240 s | +12% |
| clean | 12,257 | +0.197 s | +2,420 s | +122% |

The negative `dirty_air` contribution is the traffic bias D5 accepts, and it is
**one-sided** as predicted: projections run optimistic for cars that were really stuck
behind someone. At −0.07 s per dirty-air lap it is far smaller than the −1.0 s median
dirty-air penalty measured in Phase 3, because the reference-pace estimator already
absorbs part of a habitually-blocked driver's deficit into their level.

#### Why the simulator has no lap loop

The architecture doc specifies a loop over laps vectorised over sims, because per-lap
steps 4–5 (dirty air, overtake resolution) couple drivers and must be resolved in track
order. D5 defers both, and with them gone **no driver's lap time depends on any other's**.
The tyre schedule is deterministic given the plan, so the whole race is one vectorised
expression over `(n_sims, n_drivers, n_laps)`. This is a consequence of a scoping
decision, not a shortcut: `build_schedule` still produces the per-lap frame a loop would
walk, so restoring one at product-Phase 2 is local to `simulate_field`.

#### Finding: this circuit set is almost caution-free

Across all seven races there are **27 caution laps, 0.2% of all laps** — Austria's late
VSC, plus Suzuka's and Monaco's lap-1 red flags. Consequences:

- The caution lap-time multiplier (1.21) rests on one event and is nearly inert here.
- `caution_pit_multiplier` is an **assumed 0.45, not fitted**: the store contains no
  green-flag-to-caution stop pairs to fit it from, and Monaco's 16 stops were taken during
  a stoppage where the cost is zero for a different reason. Flagged rather than quietly
  fitted on unsuitable data.
- Phase 8's safety-car scenario slider **cannot be calibrated from this circuit set** and
  will need multi-year data, exactly as Challenge 3 anticipated.

### Phase 8 — complete (2026-08-17). **Exit criterion met, and partly by luck.**

Hierarchical noise and calibration. 152/152 tests pass, ruff clean.

#### Result

| Level | Nominal | Phase 7 (i.i.d.) | **Phase 8** |
|---|---|---|---|
| 50% | 0.50 | — | 0.559 |
| **80%** | **0.80** | **0.11–0.28** | **0.814** |
| 95% | 0.95 | — | 0.897 |

Point estimates are untouched — the Phase 7 gate still passes 17/20 with the same three
failures, which is the check that the decomposition changed only the spread.

#### The honest caveat, stated first because it matters

**The pooled pass is partly two errors cancelling.** Split by circuit:

| | 50% | 80% | 95% | PIT mean |
|---|---|---|---|---|
| all five circuits | 0.559 | **0.814** | 0.897 | 0.409 |
| excluding Suzuka | 0.649 | 0.912 | 0.956 | 0.461 |

Suzuka's *point estimates* are biased (the circuit Phases 4, 5 and 7 all flagged), which
pushes its coverage down; the other four circuits' intervals are still too wide, which
pushes theirs up. Pooled, they land on nominal. Closing the remaining gap by narrowing the
intervals would fix four circuits and break the fifth, on a sample of five races — so it
is reported rather than tuned away. More seasons per circuit is the real fix.

#### The variance components

| Level | Drawn once per | Estimated sd |
|---|---|---|
| Race progress | sim | 0.018 s/lap |
| Reference, field-wide | sim | 2.31/√N, capped at 1.0 s |
| Reference, per car | sim × driver | 0.78/√n |
| Stint level | sim × stint | 0.467 s |
| Stint slope | sim × stint | 0.049 s/lap |
| Pit stop | sim × stop | empirical, bootstrapped |
| Lap | sim × driver × lap | empirical, sd 0.506, skew 2.30 |

The upper levels are Gaussian and the lap level is not. That is deliberate: the upper
levels are averages and slopes over many laps where the CLT applies, while a single lap's
residual is right-skewed and a Gaussian would erase the slow tail strategy risk is about.
Components were fitted by a random-slope decomposition with tyre age centred within each
stint, which makes level and slope orthogonal so they can be sampled independently.

#### D14 — the driver level is reference *sampling* error, and it is mostly shared

The obvious driver-level term is the spread of `estimated reference − true reference`
(0.44–1.21 s). Using it is wrong twice over.

**First, most of it cancels.** Projection error is `(ref_hat − ref_true) − resid_future`,
and a driver's persistent model misfit `mu_d` appears in both terms — the reference
absorbs it from the observed laps and the future residuals give it back. Measured, `mu_d`
has sd 0.359 s and is flat in the decision lap, confirming it is a driver property. Only
the sampling fluctuation survives.

**Second, most of what survives is field-wide, not private to a car.** The first
calibration run was wrong in both directions at once — within-replay spread 1.68× too
wide, between-replay too narrow. Splitting the residual by circuit and decision lap
explains it: **76% of the reference-error variance at lap 5 is a field-wide shift**,
falling to 45% by lap 35. That is what one would expect — a field-wide reference error
means the model's pace *shape* is wrong for this circuit, which moves everyone together.

The distinction is not cosmetic. A shared error cannot change who beats whom; a private
one can. The split decides whether the tool is over- or under-confident about a
*comparison*, which is the only thing it is actually asked for.

#### Two double-counts the calibration exposed

1. **The continuing stint's level was drawn twice.** The reference estimate is built from
   the laps of the stint currently running, so it has already absorbed how good that set
   of tyres is. Drawing a fresh level for it counted the same deviation twice. Its
   *slope* is still drawn — a stint's level so far says little about how it will decay.
2. **The field-wide reference term and the race-progress term are largely the same
   thing.** They correlate at **0.824**, sharing 68% of variance: when D13's slope is
   wrong by δ, the reference anchored through that slope moves with it (fitted at ~25δ,
   consistent with a lever arm of half the decision lap). Applying both at full size made
   the between-replay spread 1.43× too wide. The shared reference term is now the
   component orthogonal to progress: k = 2.314 rather than 4.430.

After both, the variance decomposition lines up with the observed error structure:

| | Empirical | Simulated | Ratio |
|---|---|---|---|
| within-replay (idiosyncratic) | 25.8 s | 27.7 s | **1.07** |
| between-replay (field-wide) | 25.8 s | 29.8 s | 1.16 |

The within-replay ratio is the one that governs strategy comparisons, and it is the one
that came out right. The between-replay figure rests on five numbers per decision lap and
should not be over-read.

#### Calibration is measured by PIT, not only by coverage

Coverage at three levels can still hide a badly shaped distribution, so each outcome's
probability integral transform is reported: the share of simulated draws at or below the
truth, Uniform(0,1) under perfect calibration. Pooled it is mean 0.409 (target 0.500),
sd 0.267 (target 0.289), tail share 0.134 (target 0.100).

The reading is precise in a way a coverage number is not: **sd 0.267 says the intervals
are still slightly too wide; mean 0.409 says the point estimate carries a residual bias**
— and the by-circuit table above says that bias is Suzuka.

### Phase 9 — complete (2026-08-18). **Exit criterion met with 15× margin.**

The strategy engine: generate, prune, evaluate, rank. 185/185 tests pass, ruff clean,
mypy clean over `sim/` and `strategy/` apart from two pre-existing `pace_model: object`
attribute errors in `sim/engine.py`.

#### Result

Exit criterion is **20 candidates evaluated in under 5 s**. Measured over 100 decisions
(5 circuits × decision laps 10/20/30/40 × the top 5 runners at each):

| | median | max |
|---|---|---|
| candidates evaluated | 88 | 153 |
| **seconds per 20 candidates** | **0.27** | **0.34** |

Nothing failed and no state produced an empty candidate set. The typical run evaluates
~90 candidates in under 3 s, so the doc's "target 15–30 surviving candidates" was set
against a simulator slower than the one that got built. The grid was left honest rather
than trimmed to hit a number: computing broadly is cheap, and `table(top=N)` presents
narrowly.

#### Candidates are compared against common random draws, and that is the whole design

This is the single most consequential decision in the phase. Absolute projected race time
over a 50-lap horizon has **sd 33 s**, nearly all of it shared across strategies — how
fast the track came to the field today, how far the reference pace was off. The difference
between two sensible pit plans is a handful of seconds. Measured at Hungary from lap 20,
a lap-30 stop against a lap-36 stop:

| Draws | sd of the difference | mean difference |
|---|---|---|
| independent seeds | 45.7 s | 8.9 s |
| **common seed** | **2.9 s** | 8.9 s |

Under independent draws a nine-second gap reads as a coin flip. Every reported quantity is
therefore a function of the paired difference.

**Seeding both runs identically is not sufficient.** `NoiseModel.sample` draws one block
per *stint* and `_apply_pit_losses` one per *stop*, so a plan carrying more stops than any
rival draws a larger array, leaves the generator somewhere else, and decorrelates
everything after it — including the per-lap noise. That breaks exactly the comparison the
product exists to make. One-stop against two-stop with the field frozen: sd **25.5 s** on
the same seed, **14.9 s** once draw sizes are declared (`sim.engine.RandomAlignment`).

The residual 14.9 s is real and must not be engineered away: a two-stop plan is genuinely
exposed to one more set of tyres than a one-stop, and that stint-level effect is risk the
two plans do not share. Pairing removes the luck they have in common; what survives is the
strategy.

`RandomAlignment` defaults to `None`, which reproduces the Phase 7/8 draw order exactly —
verified by MD5 of the full `(n_sims, n_drivers, n_laps)` array before and after the
change, and by the replay gate being unmoved.

#### Pairing also makes the *selection* nearly unbiased

The recommendation is the best of ~88 noisy candidates, which is a winner's-curse setup:
the argmax of a noisy criterion overstates its own advantage. Measured by re-evaluating
each chosen plan against the baseline on a **held-out seed** (30 decisions):

| | median | mean |
|---|---|---|
| gain as selected | 1.54 s | 3.22 s |
| gain on held-out draws | 1.47 s | 3.21 s |
| **shrinkage** | **0.00 s** | **0.01 s** |

Negligible, and it is negligible *because* of the pairing: `delta_se_s` is ~0.1 s, so
selecting the maximum over 88 candidates costs almost nothing. Without common draws the
per-candidate standard error would be ~1.4 s and the selection bias would run to several
seconds. The variance reduction is not just cosmetic — it is what makes the argmax honest.

#### Risk: CVaR on the difference, not on the race time

The doc specifies CVaR@10%. Applied to absolute race time it mostly measures the shared
uncertainty every candidate carries equally, so the column ranks nothing — it is reported
(`cvar10_s`) but does no work. The column that carries risk is **`cvar10_delta_s`**: the
mean of the worst tenth of the paired difference against the baseline. It is independent
of how many candidates were enumerated and reads directly as "if this switch goes badly,
this is how badly".

It discriminates where expectation does not. Hungary lap 20, VER: the top-ranked plan
gains 1.05 s over what was executed with a downside of 4.45 s, while a plan keeping the
executed second stop gains 0.52 s with a downside of 1.25 s. Same race, same lap, a
genuine expectation-versus-downside trade the ranking would otherwise hide.

`regret_s` / `cvar10_regret_s` are also reported — regret against whichever plan won that
same simulation, the only risk numbers that stay non-zero for the recommendation itself.
**They grow with the size of the candidate set** (more options give the oracle more to win
with), so they are comparable within a ranking and never across two.

#### The physical filter has to bind on "stay out", and did not

The max-stint cap is fitted from the store as the longest stint each dry compound has been
seen to complete, as a share of race distance — HARD 0.586, MEDIUM 0.485, SOFT 0.379 —
LORO-scoped like every other circuit constant. Two properties are worth stating: it is a
*lower* bound on what is possible, because many of those stints ended at the flag
(`is_censored`); and it is pooled across circuits, because one race per circuit makes a
per-circuit maximum a single observation.

The first version exempted the no-stop plan from it, on the reasoning that a car already
past the cap must still get candidates. That reasoning is right for a stint with a stop
after it — the shortest reachable length is `tyre_age + 1` — and wrong for staying out,
where every remaining lap is a *choice*, not a constraint. The consequence was that the
plan most likely to break the pace model was the one plan that could not be filtered:
Hamilton at Hungary lap 40 was offered a 54-lap stint against a 41-lap cap, and the
degradation term, being linear in tyre age, extrapolated thirteen laps past any stint in
the store. The engine came back claiming an extra stop was worth **28.6 s** — a statement
about the extrapolation, not about the tyres.

Fixed by capping at `max(compound cap, tyre_age + 1)` uniformly, with running to the flag
retained as a last resort when the grid is genuinely empty (a handful of laps from the
end, where no stop leaves a stint worth running).

A **baseline** supplied for audit stays exempt, deliberately: what a car actually ran is a
fact, and refusing to evaluate it would make the post-race audit useless. `recommend` logs
a warning instead, which fires on exactly the two states in the 100-decision sweep where
the executed plan runs past the cap.

#### The stint the car is already on counts, so the undercut is representable

`Scenario.is_legal` measures the first stint from the decision lap, which makes an
immediate stop look like a zero-lap stint and rejects it. That is the wrong constraint for
candidate generation — a driver twenty laps into a set can pit next lap, and that move is
the Hungary 2024 demo (D4). The filters in `strategy.candidates` therefore measure stints
from where the tyre was fitted, adding `tyre_age` to the segment still to run;
`is_legal` is still called for the two-compound rule it owns. A test pins both halves of
that divergence so it cannot be tidied up later.

#### Refinement moves only the next stop

Coarse grid (step 4 within a 20-lap window for the next stop, step 8 beyond it), then the
leaders' next stop re-searched to the lap. Refining *all* stops of six two-stop plans at
±2 generates 150 combinations and swamps the coarse set it is meant to sharpen; refining
only the next one keeps the pass linear in stop count. It is also the more defensible
choice by the module's own argument — a stop thirty laps out will be re-decided at its own
decision lap with evidence this state does not have, so picking it to the lap is false
precision.

Both passes share one seed and one `RandomAlignment`, so refined plans are paired with
coarse ones and the union ranks as a single comparison.

#### What the engine says about the races it was not trained on

Over the 98 of 100 decisions whose executed plan is inside the model's support:

| | |
|---|---|
| executed plan within 2 MC standard errors of the recommendation | 24 / 98 |
| stop-count disagreement | 9 / 98 |
| median time the engine says was on the table | 1.40 s |
| 90th percentile / max | 6.14 s / 18.4 s |

Agreement rises steadily with the decision lap — 4% at lap 10, 20% at 20, 28% at 30, 48%
at 40 — which is the same story the replay gate tells: an early state has almost no
observed pace to build a reference from. By circuit it tracks model quality too: Suzuka
0.60 agreement and a median gain of 0.00 s, Bahrain 0.50 / 0.30 s, against Barcelona
0.00 / 5.16 s.

**None of this is a validation, and it must not be read as one.** Three reasons, all
one-directional:

1. **The comparison is one-sided by construction.** The baseline is one of the evaluated
   candidates, so the reported gain is `max(0, ...)` before anything else happens.
2. **The executed plan is not always a strategy.** Verstappen's Austria plan after lap 20
   contains three stops including a lap-64 SOFT — the post-collision damage stop. The
   store does not distinguish a forced stop from a chosen one, so the engine's "you lost
   18 s" is arithmetically true and strategically vacuous. It accounts for the four
   largest disagreements in the sweep.
3. **Counterfactuals are unobservable.** No amount of agreement or disagreement with a
   real pit wall tests a strategy nobody ran. Phase 7 validated reproduction and Phase 8
   validated spread; the counterfactual itself is not checkable, and that limit belongs in
   the UI rather than in an argument.

Barcelona's 5.16 s median with zero agreement is the one number here worth chasing, and it
is left open rather than explained away.

#### Notes

- `simulate_field` is fast enough (0.028 s per candidate at 1,000 sims over a full field)
  that the ego-only optimisation D5 permits — no driver's lap time depends on another's,
  so projecting the ego alone would be exact rather than approximate — was measured,
  found unnecessary, and not built. A test asserts the independence property instead, so
  that coupling rivals in product-Phase 2 breaks it loudly.
- The replay gate now passes **18/20**, not the 17/20 recorded at Phase 7. The improvement
  is Phase 8's: hierarchical noise changed the per-sim median slightly and moved
  `austria@L5` from 1.00% to 0.9622%. Phase 9 changed nothing here, verified bitwise.

### Phase 10 — complete (2026-08-18). **Components reconcile to 1e-15 s.**

Explanation by decomposition (Challenge 6). 200/200 tests pass, ruff clean, and mypy is
now clean over `sim/` and `strategy/` — the two long-standing `pace_model: object` errors
are gone, replaced by a `PaceModel` protocol that documents the seam D11 is argued over.

#### Result

Exit criterion is **components sum to the total**. Measured residual on every explanation
produced: **< 1e-14 s** against a headline of order 1–30 s, and `explain` raises rather
than reports if it ever exceeds 1e-6 s. Hungary, lap 20, VER, two-stop against one-stop:

| Component | Seconds |
|---|---|
| fresher rubber | −31.20 |
| compound choice | −0.63 |
| pit-lane time | +24.30 |
| sampling residue | +0.53 |
| **net** | **−7.01** |

The extra stop costs 24.3 s in the pit lane and buys 31.2 s of fresher rubber. This is not
a fit, a surrogate or an attribution method — it is the arithmetic the simulator already
did, regrouped.

#### Keeping the addends beats reconstructing them

The simulator adds every effect in seconds, so `simulate_field(capture=True)` returns a
`LapBreakdown` of the three addends and `pace_s + noise_s + pit_s` reconstructs
`lap_times_s`. The alternative — ablation re-runs with one component disabled — was
rejected: zeroing the pit-loss array changes `rng.choice`'s rejection sampling and moves
the stream, so the "ablated" run is not the same run minus a term, and the differences
would not have summed.

Capture costs three more full-size arrays (24 MB at 1,000 sims), which is worth paying for
the two plans being explained and not the hundred being ranked, so it is opt-in and off by
default. The default path is bitwise unchanged — verified by MD5 of the full lap-times
array, same as the Phase 9 check.

The reconstruction is exact to float round-off (~1e-14 s) rather than bitwise, because the
caution multiplier is applied to the sum and divided back out of the parts. Making it
bitwise would mean changing the order of operations in the validated path, which is not
worth 1e-14 s.

#### The pace difference is split without asking the model how it works

A per-lap pace difference mixes two causes: a different compound, and a different tyre age
under it. Reading the fitted coefficients would work today — `pace_delta ~ C(compound) +
C(compound):tyre_age + lap_number` is about as separable as a model gets — and would
silently become wrong the moment a model with interactions replaced it, which is exactly
the substitution D11 leaves open.

Both orderings are evaluated and averaged instead, which needs nothing but `predict`:

```
age      = ½[ mu(cA, ageA) − mu(cA, ageB) ] + ½[ mu(cB, ageA) − mu(cB, ageB) ]
compound = ½[ mu(cA, ageA) − mu(cB, ageA) ] + ½[ mu(cA, ageB) − mu(cB, ageB) ]
```

For two factors that is the Shapley value, and it sums to `mu(cA, ageA) − mu(cB, ageB)` as
an algebraic identity. A test checks the two halves against the closed form for a
deliberately separable stub model, so the split is verified against the right answer and
not merely against itself.

The counterfactual frames move `tyre_age`, `is_fresh_tyre` and `stint_number` together,
because all three describe when the set was fitted; pulling one across without the others
would build a frame describing a car that cannot exist.

#### Two whole effects cancel, and that is worth showing

Reference pace and the D13 race-progress correction do not appear in the decomposition
because they contribute exactly zero to it: both plans run the same laps of the same race,
so both terms enter both totals identically and vanish in the subtraction. The two largest
terms in an *absolute* projection contribute nothing to a *comparison* — the same fact
that makes Phase 9's pairing work. Two tests assert it by varying the reference pace by
10 s and the progress slope by 0.04 s/lap and demanding an unchanged explanation.

#### The sampling row is an honesty check, not a component

The hierarchical noise is mean zero, so its contribution to a difference is zero in
expectation and Monte Carlo residue in practice. Reporting it is what stops the
decomposition quietly attributing residue to a cause: **if the sampling row is not small
against the headline, the headline is not settled and the answer is more draws.**

It is not automatically small. Plans with different stop counts are exposed to different
numbers of tyre sets, and the stint-level effect on the extra set does not cancel — the
irreducible 14.9 s of spread `RandomAlignment` documents. Hence 4,000 draws by default
here against the ranking's 1,000, which leaves it at ~0.5 s on a 7 s headline.

#### What the decomposition does not contain, stated rather than omitted

The doc's worked example has four terms and this has two of them, so the missing pair
ships inside the result object (`to_dict()["not_included"]`) rather than in a design
document nobody reads at the point of use:

- **Traffic on rejoin** — not modelled at all (D5). The replay puts the omission at
  −0.07 s per dirty-air lap, ~11% of total replay error, one-sided: a plan that rejoins
  into traffic is flattered by every number here.
- **Safety-car option value** — cautions are exogenous and never predicted (Challenge 3),
  so there is no distribution to take an expectation over. What can honestly be produced
  is a contrast under an *assumed* caution: `caution_option_value` returns −1.13 s for
  VER's two-stop against one-stop with a caution assumed on laps 30–32. The sign is more
  trustworthy than the size — it leans on the caution lap multiplier (one VSC) and the
  caution pit multiplier (**assumed, never fitted**, because the store contains no
  green-flag-to-caution stop pairs).

### Phase 11 — next (superseded, see below)

Streamlit, three pages, provenance colouring (Part 10). The rule that matters is that the
UI never computes: pages call `strategy.evaluate` / `strategy.explain` and render, so the
core stays testable and headless. Everything those return is already JSON-serialisable
(D9), which is also what the Next.js showcase will consume.


### Phase 11 — complete (2026-08-19). **MVP ships, and the first version was the wrong product.**

Precomputed scenario grid plus a Next.js showcase on Vercel (D9). Streamlit is **dropped,
not deferred**: D9 gave it the internal-diagnostics job and the Phase 7-10 CLIs already do
that job with richer output, so building it would have been duplicate surface for no new
information.

#### The redesign, and why it was needed

The first build was a methodology showcase: stint timeline, a twelve-row ranked table, a
decomposition waterfall, a data-quality panel, a method page. Everything on it was true and
almost none of it was *useful*, because it showed the engine rather than the decision. It
was the analyst's view shipped to a strategist.

The test that settled it: at Hungary lap 20 the twelve exported candidates reduce to
**two** genuinely different decisions, HARD or MEDIUM on the next set, spread across four
adjacent laps, and four of the twelve were inside Monte Carlo noise of each other. A ranked
list presents that as an ordering. It is not one.

Rebuilt around the question actually being asked on the lap:

| Panel | What it answers |
|---|---|
| **BOX L22**, window 21-23 | the call, and how long it stays open |
| Worth / confidence / cost of waiting | is it worth it, is it settled, what does delay cost |
| **The window curve** | the whole candidate set collapsed onto the live axis |
| Four shapes, not twelve labels | the choices that are actually different |
| Safety-car exposure | the first thing anyone asks about a plan |
| "Behind the number" (collapsed) | the machinery, available and out of the way |

The window curve is the graphic that did the most work and it came free from the existing
ranking: group every evaluated plan by its *next stop lap* and take the best. At Hungary
lap 20 that is a floor at laps 21-23 and a cliff after 25, rising to +5.9 s by lap 33. The
width of the floor is the window; the slope past it is what waiting costs. A ranked table
hides both.

Two things were added to the export to support it, both computable from work already done:
the window curve, and the safety-car contrast (`caution_option_value`, +0.22 s at that
state, so the call does not change). The SC panel ships with its caveats attached, because
it leans on the two thinnest constants in the project.

#### What it still cannot tell her, said on the page

There is no overtaking model (D5), so nothing here knows about track position. It will say
a stop is worth a second and it cannot say whether the car comes out ahead of the one it is
racing. On a circuit where that is the whole question, the tool is silent by construction.
The method page now leads with this rather than burying it.

#### Design

Dark instrument surface, committed to rather than switched on `prefers-color-scheme`: this
imitates a pit wall screen, and rendering it as a cream document on half the machines that
open it would be two identities rather than one. No glow, glass, gradient, shadow or
motion; what makes a timing screen read as advanced is density and figures that do not
shift width as they tick.

Chart colour is validated with the runnable checker rather than chosen by eye. The first
attempt failed hard: the muted earth palette put the blue below the chroma floor and
collapsed the moss/rust pair to CVD delta-E 3.5. The shipped compound hues clear every gate
in both themes. The light-theme amber lands at 2.7:1 on paper, below 3:1, so it ships under
the relief rule with visible labels and a table view on every chart.

Three encoding mistakes were caught only by rendering the page and looking at it:

1. Fading a stint that ran to the flag produced a 55% amber and a 55% blue that read as two
   more compounds. Hatching it at 15px bar height produced a tint that read the same way.
   Both were spending the encoding on information the reader already has, since the last
   stint of anyone who finished ran to the flag by definition. The bars now carry no
   modifier and the palette stays three colours meaning three compounds.
2. The stint gap went from 2px to 4px: Hamilton and Perez each ran two consecutive stints on
   the same compound, and at 2px the stop was invisible.
3. Grouping candidates by shape silently absorbed the plan that was actually executed, so
   the comparison hid what the team did. The baseline is now always its own row.

Plus one that was invisible until measured: the font variables were declared on `<body>`
while `globals.css` composes them at `:root`, so every `font-family` resolved to nothing
and the entire site rendered in Times.

### Phase 12 — next

The Hungary 2024 post-race audit (D4): the McLaren undercut-and-swap-back as a narrative
walk-through of the same decision surface, lap by lap.
