"""Tests for plays Silver transforms.

Each test drives the transform functions with a small hand-built DataFrame
(1–3 rows, fully controlled). We are testing *our* logic, not Spark.
"""

from __future__ import annotations

from datetime import date

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from sonicwave_ingestion.plays.silver import (
    SILVER_SCHEMA,
    _cast,
    _conform,
    _dedup,
    _split,
    _validate,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Schema that mimics a Bronze DataFrame row (all strings + typed provenance).
_BRONZE_SCHEMA = StructType(
    [
        StructField("play_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("content_id", StringType(), True),
        StructField("device_id", StringType(), True),
        StructField("played_at", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("updated_at", StringType(), True),
        StructField("ms_played", StringType(), True),
        StructField("ingested_at", TimestampType(), True),
        StructField("source_file", StringType(), True),
        StructField("snapshot_date", StringType(), True),
    ]
)


def _row(
    play_id: str | None = "p1",
    user_id: str | None = "u1",
    content_id: str | None = "c1",
    device_id: str | None = "d1",
    played_at: str | None = "2026-03-01T10:00:00",
    created_at: str | None = "2026-03-01T10:00:00",
    updated_at: str | None = None,
    ms_played: str | None = "180000",
    snapshot_date: str = "2026-03-01",
) -> tuple[
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    str | None,
    None,
    None,
    str,
]:
    """Return a tuple matching _BRONZE_SCHEMA — override only what a test needs."""
    return (
        play_id,
        user_id,
        content_id,
        device_id,
        played_at,
        created_at,
        updated_at,
        ms_played,
        None,  # ingested_at
        None,  # source_file
        snapshot_date,
    )


# ---------------------------------------------------------------------------
# Quarantine / validation
# ---------------------------------------------------------------------------


def test_clean_row_passes(spark: SparkSession) -> None:
    """A well-formed row must pass validation and land in Silver."""
    df = spark.createDataFrame([_row()], schema=_BRONZE_SCHEMA)
    valid, quarantine = _split(_validate(_cast(df)))

    assert valid.count() == 1
    assert quarantine.count() == 0


@pytest.mark.parametrize(
    ("play_id", "user_id", "ms_played", "expected_reason"),
    [
        ("p1", None, "180000", "null user_id"),
        ("p1", "u1", "-5000", "ms_played <= 0"),
        ("p1", "u1", "NaN", "invalid ms_played"),
        (None, "u1", "180000", "null play_id"),
    ],
)
def test_quarantine_bad_rows(
    spark: SparkSession,
    play_id: str | None,
    user_id: str | None,
    ms_played: str,
    expected_reason: str,
) -> None:
    """Each class of bad row must be quarantined with a matching reject_reason."""
    df = spark.createDataFrame(
        [_row(play_id=play_id, user_id=user_id, ms_played=ms_played)],
        schema=_BRONZE_SCHEMA,
    )
    valid, quarantine = _split(_validate(_cast(df)))

    assert valid.count() == 0, "bad row must not reach Silver"
    assert quarantine.count() == 1

    reason = quarantine.collect()[0]["reject_reason"]
    assert expected_reason in reason, f"expected '{expected_reason}' in '{reason}'"


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def test_dedup_keeps_earliest_created_at(spark: SparkSession) -> None:
    """Two rows with the same play_id — the one with the earlier created_at survives."""
    rows = [
        _row(play_id="p1", ms_played="180000", created_at="2026-03-01T10:00:00"),
        _row(play_id="p1", ms_played="999999", created_at="2026-03-01T11:00:00"),
    ]
    df = spark.createDataFrame(rows, schema=_BRONZE_SCHEMA)
    result = _dedup(_cast(df))

    assert result.count() == 1
    assert result.collect()[0]["ms_played_c"] == 180000


# ---------------------------------------------------------------------------
# Late-arriving event
# ---------------------------------------------------------------------------


def test_late_event_dated_by_event_time(spark: SparkSession) -> None:
    """A play whose played_at is T but created_at is T+1 must get event_date = T.

    The row is stored in the T+1 snapshot partition (snapshot_date stays T+1),
    but event_date is derived from played_at — not from snapshot_date.
    """
    df = spark.createDataFrame(
        [
            _row(
                played_at="2026-03-01T21:30:00",
                created_at="2026-03-02T06:00:00",
                snapshot_date="2026-03-02",
            )
        ],
        schema=_BRONZE_SCHEMA,
    )
    valid, _ = _split(_validate(_cast(df)))
    result = _conform(_dedup(valid))

    row = result.collect()[0]
    assert row["event_date"] == date(2026, 3, 1), "event_date must follow event time"
    assert row["snapshot_date"] == "2026-03-02", "snapshot_date must follow arrival"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_silver_schema(spark: SparkSession) -> None:
    """Silver output must have exactly the declared columns and correct types."""
    df = spark.createDataFrame([_row()], schema=_BRONZE_SCHEMA)
    valid, _ = _split(_validate(_cast(df)))
    result = _conform(_dedup(valid))

    # Column names
    expected_cols = {f.name for f in SILVER_SCHEMA}
    assert set(result.columns) == expected_cols

    # Key typed columns
    type_map = {f.name: type(f.dataType) for f in result.schema}
    assert type_map["played_at"] is TimestampType
    assert type_map["created_at"] is TimestampType
    assert type_map["ms_played"] is LongType
    assert type_map["event_date"] is DateType
