"""Declarative table contracts, enforced on every write to the store.

Specs are pydantic models (typed, self-validating configuration); enforcement against
a DataFrame is vectorised pandas rather than per-row pydantic, because these tables run
to tens of thousands of rows and row-wise validation would dominate build time without
catching anything extra.

The contract exists so that a FastF1 upgrade, or a silent change in what a column means,
fails loudly at ingest instead of quietly at model-fitting time.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict

# Compounds. INTERMEDIATE/WET are ingested but excluded from clean laps -- the MVP is
# dry-race only (architecture-proposal.md, Part 14, assumption 8).
DRY_COMPOUNDS = frozenset({"SOFT", "MEDIUM", "HARD"})
WET_COMPOUNDS = frozenset({"INTERMEDIATE", "WET"})
ALL_COMPOUNDS = DRY_COMPOUNDS | WET_COMPOUNDS | {"UNKNOWN"}

#: Why a lap is excluded from clean-pace analysis. Ordered by priority: a lap gets the
#: first reason that applies, so counts across reasons partition the excluded set rather
#: than double-counting. ``"clean"`` means the lap survived every filter.
EXCLUSION_REASONS = (
    "clean",
    "no_lap_time",
    "lap_1",
    "pit_in",
    "pit_out",
    "red_flag",
    "safety_car",
    "vsc",
    "yellow",
    "deleted",
    "inaccurate",
    "wet_compound",
    "dirty_air",  # populated in Phase 3, once gaps exist
)

#: How a stint ended. Drives censoring for the deferred survival model (D1): only
#: ``"pit"`` is an uncensored observation of a strategy-driven stint end.
STINT_END_REASONS = ("pit", "race_end", "retirement")


class SchemaError(AssertionError):
    """A DataFrame violated its table contract."""


class ColumnSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    kind: str  # "int" | "float" | "str" | "bool"
    nullable: bool = False
    allowed: frozenset[str] | None = None
    min_value: float | None = None
    max_value: float | None = None

    def check(self, s: pd.Series, table: str) -> list[str]:
        errs: list[str] = []
        where = f"{table}.{self.name}"

        if not self.nullable and s.isna().any():
            errs.append(f"{where}: {int(s.isna().sum())} null(s) in a non-nullable column")

        present = s.dropna()
        if present.empty:
            return errs

        if self.kind == "int" and not pd.api.types.is_integer_dtype(s.dropna().infer_objects()):
            if not (present % 1 == 0).all():
                errs.append(f"{where}: expected integral values, got {s.dtype}")
        elif self.kind == "float" and not pd.api.types.is_numeric_dtype(present):
            errs.append(f"{where}: expected numeric, got {s.dtype}")
        elif self.kind == "bool" and not pd.api.types.is_bool_dtype(present):
            errs.append(f"{where}: expected bool, got {s.dtype}")

        if self.allowed is not None:
            bad = set(present.astype(str).unique()) - set(self.allowed)
            if bad:
                errs.append(f"{where}: unexpected value(s) {sorted(bad)}")

        if self.min_value is not None and pd.api.types.is_numeric_dtype(present):
            below = present < self.min_value
            if below.any():
                errs.append(f"{where}: {int(below.sum())} value(s) below {self.min_value}")

        if self.max_value is not None and pd.api.types.is_numeric_dtype(present):
            above = present > self.max_value
            if above.any():
                errs.append(f"{where}: {int(above.sum())} value(s) above {self.max_value}")

        return errs


class ConditionalSpec(BaseModel):
    """A constraint that applies only to the subset where ``when`` is True.

    Exists because raw and clean rows warrant genuinely different bounds. A red-flagged
    lap legitimately lasts 40 minutes of elapsed time, so the raw column cannot be
    tightly bounded -- but a lap that survived cleaning and is about to train a pace
    model absolutely can be, and that is the bound worth enforcing.
    """

    model_config = ConfigDict(frozen=True)

    when: str
    column: ColumnSpec
    negate: bool = False

    @property
    def label(self) -> str:
        return f"{'~' if self.negate else ''}{self.when}"


class TableSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    grain: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    conditional: tuple[ConditionalSpec, ...] = ()

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def validate_frame(self, df: pd.DataFrame) -> None:
        """Raise ``SchemaError`` describing every violation at once.

        Reporting all failures together matters during pipeline development: fixing
        them one exception at a time across a 7-race build is needlessly slow.
        """
        errs: list[str] = []

        missing = [c for c in self.column_names if c not in df.columns]
        if missing:
            raise SchemaError(f"{self.name}: missing column(s) {missing}")

        extra = [c for c in df.columns if c not in self.column_names]
        if extra:
            errs.append(f"{self.name}: unexpected column(s) {extra}")

        for col in self.columns:
            errs.extend(col.check(df[col.name], self.name))

        for cond in self.conditional:
            mask = df[cond.when].fillna(False).astype(bool)
            if cond.negate:
                mask = ~mask
            subset = df.loc[mask, cond.column.name]
            errs.extend(
                e.replace(f"{self.name}.", f"{self.name}[{cond.label}].")
                for e in cond.column.check(subset, self.name)
            )

        if all(g in df.columns for g in self.grain):
            dupes = int(df.duplicated(subset=list(self.grain)).sum())
            if dupes:
                errs.append(f"{self.name}: {dupes} row(s) duplicate the grain {self.grain}")

        if errs:
            raise SchemaError(f"{self.name} failed validation:\n  - " + "\n  - ".join(errs))


_ID = (
    ColumnSpec(name="race_id", kind="str"),
    ColumnSpec(name="season", kind="int"),
    ColumnSpec(name="round", kind="int", min_value=1, max_value=24),
    ColumnSpec(name="circuit", kind="str"),
)

LAPS = TableSpec(
    name="laps",
    grain=("race_id", "driver", "lap_number"),
    columns=(
        *_ID,
        ColumnSpec(name="driver", kind="str"),
        ColumnSpec(name="team", kind="str"),
        ColumnSpec(name="lap_number", kind="int", min_value=1),
        # Raw, uncorrected. Fuel correction is a FITTED transform applied in Phase 4,
        # deliberately not baked in here -- see normalize.py module docstring.
        # The loose upper bound admits red-flagged laps, whose elapsed time spans the
        # stoppage (Monaco 2024 lap 1 is ~2500s). The bound that matters is the
        # conditional one on clean laps below.
        ColumnSpec(name="lap_time_s", kind="float", nullable=True, min_value=30, max_value=3600),
        ColumnSpec(name="cumulative_time_s", kind="float", nullable=True, min_value=0),
        ColumnSpec(name="compound", kind="str", allowed=ALL_COMPOUNDS),
        ColumnSpec(name="tyre_age", kind="float", nullable=True, min_value=0),
        ColumnSpec(name="is_fresh_tyre", kind="bool", nullable=True),
        ColumnSpec(name="stint_number", kind="int", min_value=1),
        ColumnSpec(name="position", kind="float", nullable=True, min_value=1, max_value=20),
        # Reconstructed from cumulative lap time (Phase 3). Kept alongside FastF1's
        # reported `position` so the two can be reconciled rather than conflated.
        ColumnSpec(name="derived_position", kind="int", min_value=1, max_value=20),
        ColumnSpec(name="gap_to_leader_s", kind="float", nullable=True, min_value=0),
        # Null for the race leader, who has nobody ahead.
        ColumnSpec(name="gap_ahead_s", kind="float", nullable=True, min_value=0),
        # Gap entering the lap (end of the previous lap): causally prior to this lap's
        # time, which end-of-lap gap is not. See gaps.add_gaps.
        ColumnSpec(name="entering_gap_s", kind="float", nullable=True, min_value=0),
        ColumnSpec(name="in_dirty_air", kind="bool"),
        ColumnSpec(name="is_pit_in", kind="bool"),
        ColumnSpec(name="is_pit_out", kind="bool"),
        ColumnSpec(name="track_status", kind="str", nullable=True),
        ColumnSpec(name="is_clean", kind="bool"),
        ColumnSpec(name="exclusion_reason", kind="str", allowed=frozenset(EXCLUSION_REASONS)),
        ColumnSpec(name="track_temp", kind="float", nullable=True),
        ColumnSpec(name="air_temp", kind="float", nullable=True),
        ColumnSpec(name="humidity", kind="float", nullable=True),
        ColumnSpec(name="wind_speed", kind="float", nullable=True),
        ColumnSpec(name="rainfall", kind="bool", nullable=True),
    ),
    conditional=(
        # A lap that survived cleaning is about to train a pace model. Nulls or
        # implausible times here are a pipeline bug, not real racing.
        ConditionalSpec(
            when="is_clean",
            column=ColumnSpec(
                name="lap_time_s", kind="float", nullable=False, min_value=40, max_value=300
            ),
        ),
        ConditionalSpec(
            when="is_clean",
            column=ColumnSpec(name="compound", kind="str", allowed=DRY_COMPOUNDS),
        ),
    ),
)

STINTS = TableSpec(
    name="stints",
    grain=("race_id", "driver", "stint_number"),
    columns=(
        *_ID,
        ColumnSpec(name="driver", kind="str"),
        ColumnSpec(name="stint_number", kind="int", min_value=1),
        ColumnSpec(name="compound", kind="str", allowed=ALL_COMPOUNDS),
        ColumnSpec(name="start_lap", kind="int", min_value=1),
        ColumnSpec(name="end_lap", kind="int", min_value=1),
        ColumnSpec(name="n_laps", kind="int", min_value=1),
        ColumnSpec(name="start_tyre_age", kind="float", nullable=True, min_value=0),
        ColumnSpec(name="end_reason", kind="str", allowed=frozenset(STINT_END_REASONS)),
        # True unless the stint ended in a pit stop; the stint was cut short by
        # something other than the strategy call, so its length is a censored
        # observation of "how long would they have run".
        ColumnSpec(name="is_censored", kind="bool"),
    ),
)

PITS = TableSpec(
    name="pits",
    grain=("race_id", "driver", "stop_number"),
    columns=(
        *_ID,
        ColumnSpec(name="driver", kind="str"),
        ColumnSpec(name="stop_number", kind="int", min_value=1),
        ColumnSpec(name="lap", kind="int", min_value=1),
        # Loose bound for the same reason as lap_time_s: a stop made during a red-flag
        # stoppage spans it (Monaco 2024 lap 1 stops run to ~2400s). Green-flag stops
        # are bounded conditionally below.
        ColumnSpec(name="pit_duration_s", kind="float", nullable=True, min_value=0, max_value=3600),
        ColumnSpec(name="compound_fitted", kind="str", nullable=True, allowed=ALL_COMPOUNDS),
        ColumnSpec(name="under_caution", kind="bool"),
    ),
    conditional=(
        # Green-flag stops are the only ones Phase 4 fits pit loss on, so this is the
        # bound that protects that model. Total pit-lane time, not stationary time.
        ConditionalSpec(
            when="under_caution",
            negate=True,
            column=ColumnSpec(
                name="pit_duration_s", kind="float", nullable=True, min_value=10, max_value=120
            ),
        ),
    ),
)

TRACK_STATUS = TableSpec(
    name="track_status",
    grain=("race_id", "start_lap", "status"),
    columns=(
        *_ID,
        ColumnSpec(name="status", kind="str"),
        ColumnSpec(name="message", kind="str", nullable=True),
        ColumnSpec(name="start_lap", kind="int", min_value=0),
        ColumnSpec(name="end_lap", kind="int", min_value=0),
    ),
)

WEATHER = TableSpec(
    name="weather",
    grain=("race_id", "time_s"),
    columns=(
        *_ID,
        ColumnSpec(name="time_s", kind="float", min_value=0),
        ColumnSpec(name="track_temp", kind="float", nullable=True),
        ColumnSpec(name="air_temp", kind="float", nullable=True),
        ColumnSpec(name="humidity", kind="float", nullable=True),
        ColumnSpec(name="wind_speed", kind="float", nullable=True),
        ColumnSpec(name="rainfall", kind="bool", nullable=True),
    ),
)

RESULTS = TableSpec(
    name="results",
    grain=("race_id", "driver"),
    columns=(
        *_ID,
        ColumnSpec(name="driver", kind="str"),
        ColumnSpec(name="team", kind="str"),
        ColumnSpec(name="grid_position", kind="float", nullable=True, min_value=0),
        ColumnSpec(name="finish_position", kind="float", nullable=True, min_value=1),
        ColumnSpec(name="classified_position", kind="str", nullable=True),
        ColumnSpec(name="status", kind="str", nullable=True),
        ColumnSpec(name="points", kind="float", nullable=True, min_value=0),
        ColumnSpec(name="laps_completed", kind="int", min_value=0),
        # Derived from classified_position: 'R'/'D'/'W'/'N' are non-classified.
        ColumnSpec(name="is_retired", kind="bool"),
        ColumnSpec(name="total_laps", kind="int", min_value=1),
    ),
)

ALL_TABLES: tuple[TableSpec, ...] = (LAPS, STINTS, PITS, TRACK_STATUS, WEATHER, RESULTS)
TABLES_BY_NAME = {t.name: t for t in ALL_TABLES}
