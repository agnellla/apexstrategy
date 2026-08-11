"""Unit tests for the table contracts, on synthetic frames. Fast, no data needed."""

from __future__ import annotations

import pandas as pd
import pytest

from apexstrategy.data.schema import (
    ColumnSpec,
    ConditionalSpec,
    SchemaError,
    TableSpec,
)


def _spec(**kw) -> TableSpec:
    return TableSpec(
        name="t",
        grain=("id",),
        columns=(
            ColumnSpec(name="id", kind="int"),
            ColumnSpec(name="v", kind="float", nullable=True, min_value=0, max_value=10),
            ColumnSpec(name="flag", kind="bool"),
        ),
        **kw,
    )


def test_valid_frame_passes() -> None:
    _spec().validate_frame(pd.DataFrame({"id": [1, 2], "v": [1.0, None], "flag": [True, False]}))


def test_missing_column_raises() -> None:
    with pytest.raises(SchemaError, match="missing column"):
        _spec().validate_frame(pd.DataFrame({"id": [1], "v": [1.0]}))


def test_out_of_range_raises() -> None:
    with pytest.raises(SchemaError, match="above 10"):
        _spec().validate_frame(pd.DataFrame({"id": [1], "v": [99.0], "flag": [True]}))


def test_null_in_non_nullable_raises() -> None:
    with pytest.raises(SchemaError, match="non-nullable"):
        _spec().validate_frame(pd.DataFrame({"id": [None], "v": [1.0], "flag": [True]}))


def test_duplicate_grain_raises() -> None:
    with pytest.raises(SchemaError, match="duplicate the grain"):
        _spec().validate_frame(pd.DataFrame({"id": [1, 1], "v": [1.0, 2.0], "flag": [True, True]}))


def test_all_errors_reported_together() -> None:
    """One exception should list every violation, not just the first."""
    with pytest.raises(SchemaError) as exc:
        _spec().validate_frame(
            pd.DataFrame({"id": [1, 1], "v": [99.0, -5.0], "flag": [True, True]})
        )
    msg = str(exc.value)
    assert "above 10" in msg and "below 0" in msg and "duplicate the grain" in msg


def test_disallowed_value_raises() -> None:
    spec = TableSpec(
        name="t",
        grain=("id",),
        columns=(
            ColumnSpec(name="id", kind="int"),
            ColumnSpec(name="c", kind="str", allowed=frozenset({"A", "B"})),
        ),
    )
    with pytest.raises(SchemaError, match=r"unexpected value\(s\) \['Z'\]"):
        spec.validate_frame(pd.DataFrame({"id": [1], "c": ["Z"]}))


class TestConditionalSpec:
    """Conditional bounds are what let raw and clean rows carry different contracts."""

    spec = TableSpec(
        name="t",
        grain=("id",),
        columns=(
            ColumnSpec(name="id", kind="int"),
            ColumnSpec(name="v", kind="float", nullable=True, max_value=3600),
            ColumnSpec(name="ok", kind="bool"),
        ),
        conditional=(
            ConditionalSpec(
                when="ok",
                column=ColumnSpec(name="v", kind="float", nullable=False, max_value=100),
            ),
        ),
    )

    def test_excluded_rows_may_exceed_the_tight_bound(self) -> None:
        """A red-flagged lap legitimately runs to 2500s; it just cannot be 'clean'."""
        self.spec.validate_frame(
            pd.DataFrame({"id": [1, 2], "v": [2500.0, 90.0], "ok": [False, True]})
        )

    def test_included_rows_must_satisfy_the_tight_bound(self) -> None:
        with pytest.raises(SchemaError, match=r"t\[ok\].v"):
            self.spec.validate_frame(pd.DataFrame({"id": [1], "v": [2500.0], "ok": [True]}))

    def test_negate_inverts_the_subset(self) -> None:
        spec = TableSpec(
            name="t",
            grain=("id",),
            columns=(
                ColumnSpec(name="id", kind="int"),
                ColumnSpec(name="v", kind="float", nullable=True, max_value=3600),
                ColumnSpec(name="caution", kind="bool"),
            ),
            conditional=(
                ConditionalSpec(
                    when="caution",
                    negate=True,
                    column=ColumnSpec(name="v", kind="float", nullable=True, max_value=120),
                ),
            ),
        )
        # Under caution: unbounded. Green flag: bounded.
        spec.validate_frame(pd.DataFrame({"id": [1], "v": [2400.0], "caution": [True]}))
        with pytest.raises(SchemaError, match=r"t\[~caution\].v"):
            spec.validate_frame(pd.DataFrame({"id": [1], "v": [2400.0], "caution": [False]}))
