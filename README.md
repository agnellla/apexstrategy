# ApexStrategy

**An F1 race decision intelligence platform.**

Most public F1 analytics are descriptive — they tell you *what* happened. ApexStrategy
is built to answer the question a race engineer actually faces:

> Given the current race state and its uncertainty, what should we do next?

```
DATA  ->  PREDICTIONS  ->  SCENARIOS  ->  DECISIONS
```

The tyre and pace models are supporting components. The product is the decision layer
sitting above them: a probabilistic race simulator that evaluates candidate pit
strategies, ranks them, quantifies the risk of each, and explains the trade-off in
terms that sum arithmetically.

---

## Status

**Phase 0 complete** — repository, environment, and circuit registry.
See [`docs/architecture-proposal.md`](docs/architecture-proposal.md) for the full design,
the decisions log (D1–D7), and the phase plan.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) (`brew install uv`).

```bash
uv sync --extra dev        # creates .venv on Python 3.11, installs everything
uv run pytest              # offline tests
uv run pytest -m slow      # tests that may hit the network
```

## What this is not

Deliberately *not* a telemetry viewer, a lap-chart dashboard, or a tyre-degradation
notebook. Charts exist to support decisions, not to decorate them.

## Design commitments

These are the choices that shape the codebase. Full reasoning in the architecture doc.

- **No "tyre life" model.** Predicting `stint_end - current_lap` predicts *when a team
  chose to pit*, not how long the tyre lasts — a decision, not a physical quantity, and
  circular inside an engine whose job is recommending pit laps. Degradation is modelled
  from pace instead; remaining useful life is *derived* in the strategy layer as the lap
  where projected pace loss exceeds the pit-loss-adjusted benefit of fresh rubber.
- **No safety-car predictor.** ~1 event per race means a season yields only tens of
  observations. Since SC dominates strategic variance, a noisy SC model would corrupt
  every downstream recommendation. It is an exogenous Poisson rate with a user-facing
  slider instead.
- **Uncertainty is hierarchical, not per-lap.** Sampling i.i.d. noise each lap makes
  errors cancel over a stint and produces absurdly confident answers. Persistent effects
  (car pace today, this tyre set, a safety car hitting everyone at once) are sampled at
  the level they actually live at.
- **Explanations are arithmetic, not SHAP.** A strategy recommendation comes from the
  simulator, so it is explained by simulator ablation — a decomposition whose components
  sum to the headline number. SHAP explains the pace model and is scoped to diagnostics.
- **Leakage is prevented structurally, not detected.** `RaceState` is built from a frame
  truncated at lap *L*, so it cannot contain future information. A property test enforces it.

## Scope

| | |
|---|---|
| Season | 2024 |
| Circuits | Bahrain, Suzuka, Barcelona, Austria, Britain, Hungary (+ Monaco, stored but excluded from training) |
| Demo race | Hungarian GP 2024 — the McLaren undercut-and-swap-back sequence |
| MVP objective | Expected **race-time** delta per strategy |
| Phase 2 | Traffic and overtaking → finishing positions → championship points |

Monaco is excluded from model fitting as a pathological outlier: near-zero overtaking and
atypical pit-lane geometry would distort pooled pace, degradation, and pit-loss estimates.

**Known MVP bias:** with overtaking deferred to Phase 2, the simulator lets cars pass
freely on cumulative time. Projected race times are therefore optimistic for a car that
would really be stuck in traffic. This is surfaced in the UI rather than hidden.

## Data

FastF1 only. No paid APIs. Telemetry (`car_data`, `position_data`) is **not** ingested —
it was 337 MB of a 432 MB four-race cache while contributing nothing to lap-level strategy
modelling. The working cache is ~1.2 MB per race.

## Layout

```
src/apexstrategy/
  config.py       paths, seeds, circuit registry
  data/           FastF1 ingest, schema contracts, clean-lap filtering, Parquet store
  features/       feature construction
  models/         pace/degradation, pit loss, (Phase 2) overtaking
  sim/            RaceState, Scenario, vectorised Monte Carlo engine
  strategy/       candidate generation, evaluation, ranking, explanation
  evaluation/     replay validation, calibration
app/              Streamlit
docs/             architecture, assumptions, validation reports
tests/
```
