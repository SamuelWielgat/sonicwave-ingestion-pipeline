"""Tests for users Silver transforms, including SCD2 logic.

Each test drives the transform functions with small hand-built DataFrames
(1–3 rows, fully controlled). We are testing *our* logic, not Spark.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DateType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from sonicwave_ingestion.users.silver import (
    SILVER_SCHEMA,
    _apply_scd2,
    _cast,
    _dedup,
    _split,
    _validate,
)

# ---------------------------------------------------------------------------
# Schemas for test DataFrames
# ---------------------------------------------------------------------------

# Matches the output of _dedup + valid_from computation — input to _apply_scd2.
_INCOMING_SCHEMA = StructType(
    [
        StructField("user_id", StringType(), True),
        StructField("email", StringType(), True),
        StructField("country", StringType(), True),
        StructField("plan_tier", StringType(), True),
        StructField("valid_from", TimestampType(), True),
        StructField("ingested_at", TimestampType(), True),
        StructField("source_file", StringType(), True),
        StructField("snapshot_date", DateType(), True),
    ]
)

# Bronze-like schema — all source strings + provenance.
_BRONZE_SCHEMA = StructType(
    [
        StructField("user_id", StringType(), True),
        StructField("email", StringType(), True),
        StructField("country", StringType(), True),
        StructField("plan_tier", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("updated_at", StringType(), True),
        StructField("ingested_at", TimestampType(), True),
        StructField("source_file", StringType(), True),
        StructField("snapshot_date", DateType(), True),
    ]
)

# Reusable timestamps — keep tests readable.
_T0 = datetime(2026, 1, 1, 8, 0, 0)   # user creation time
_T1 = datetime(2026, 3, 2, 9, 0, 0)   # first change
_T2 = datetime(2026, 3, 3, 8, 0, 0)   # second change
_D0 = date(2026, 3, 1)                 # snapshot dates
_D1 = date(2026, 3, 2)
_D2 = date(2026, 3, 3)


# ---------------------------------------------------------------------------
# Row helpers
# ---------------------------------------------------------------------------


def _incoming(
    spark: SparkSession,
    user_id: str = "u1",
    email: str = "u@test.com",
    country: str = "PL",
    plan_tier: str = "free",
    valid_from: datetime = _T0,
    snapshot_date: date = _D0,
) -> object:  # DataFrame
    return spark.createDataFrame(
        [(user_id, email, country, plan_tier, valid_from, None, None, snapshot_date)],
        schema=_INCOMING_SCHEMA,
    )


def _existing(
    spark: SparkSession,
    user_id: str = "u1",
    email: str = "u@test.com",
    country: str = "PL",
    plan_tier: str = "free",
    valid_from: datetime = _T0,
    valid_to: datetime | None = None,
    is_current: bool = True,
    snapshot_date: date = _D0,
) -> object:  # DataFrame
    return spark.createDataFrame(
        [
            (
                user_id,
                email,
                country,
                plan_tier,
                valid_from,
                valid_to,
                is_current,
                None,
                None,
                snapshot_date,
            )
        ],
        schema=SILVER_SCHEMA,
    )


def _empty_existing(spark: SparkSession) -> object:  # DataFrame
    return spark.createDataFrame([], SILVER_SCHEMA)


# ---------------------------------------------------------------------------
# SCD2 — core logic
# ---------------------------------------------------------------------------


def test_new_user_creates_v1(spark: SparkSession) -> None:
    """A user absent from Silver gets a first version: is_current=True, valid_to=null."""
    result = _apply_scd2(_incoming(spark), _empty_existing(spark))

    assert result.count() == 1
    row = result.collect()[0]
    assert row["valid_from"] == _T0
    assert row["valid_to"] is None
    assert row["is_current"] is True


def test_tracked_attr_change_opens_new_version(spark: SparkSession) -> None:
    """Changing plan_tier must open a new SCD2 version and close the old one.

    Before: u1  free  valid_from=T0  is_current=True
    After:  u1  free  valid_from=T0  valid_to=T1   is_current=False  ← closed
            u1  premium  valid_from=T1  valid_to=null  is_current=True   ← new
    """
    old_version = _existing(spark, plan_tier="free", valid_from=_T0, snapshot_date=_D0)
    new_incoming = _incoming(spark, plan_tier="premium", valid_from=_T1, snapshot_date=_D1)

    result = _apply_scd2(new_incoming, old_version).orderBy("valid_from")

    assert result.count() == 2
    rows = result.collect()

    v1 = rows[0]
    assert v1["plan_tier"] == "free"
    assert v1["valid_to"] == _T1
    assert v1["is_current"] is False

    v2 = rows[1]
    assert v2["plan_tier"] == "premium"
    assert v2["valid_to"] is None
    assert v2["is_current"] is True


def test_unchanged_user_no_new_version(spark: SparkSession) -> None:
    """Re-processing a user with identical attributes must not open a new version.

    Same valid_from in incoming and existing → dedup removes the duplicate.
    """
    existing = _existing(spark, valid_from=_T0, snapshot_date=_D0)
    # Incoming from a later snapshot but the user hasn't changed — valid_from is still T0.
    incoming = _incoming(spark, valid_from=_T0, snapshot_date=_D1)

    result = _apply_scd2(incoming, existing)

    assert result.count() == 1
    assert result.collect()[0]["is_current"] is True


def test_rerun_same_snapshot_idempotent(spark: SparkSession) -> None:
    """Running _apply_scd2 twice with the same incoming must yield the same result.

    This is the idempotency guarantee: apply(incoming, apply(incoming, empty)) == apply(incoming, empty).
    """
    incoming = _incoming(spark, valid_from=_T0, snapshot_date=_D0)

    # Force Spark to materialise — re-use the same incoming object twice.
    first_run = _apply_scd2(incoming, _empty_existing(spark))
    second_run = _apply_scd2(incoming, first_run)

    assert first_run.count() == second_run.count()
    assert second_run.filter("is_current").count() == 1


def test_three_versions_chain(spark: SparkSession) -> None:
    """Two successive tracked-attribute changes produce three correctly linked versions."""
    v1 = _existing(spark, country="PL", plan_tier="free", valid_from=_T0, snapshot_date=_D0)
    inc_t1 = _incoming(spark, country="PL", plan_tier="premium", valid_from=_T1, snapshot_date=_D1)
    inc_t2 = _incoming(spark, country="DE", plan_tier="premium", valid_from=_T2, snapshot_date=_D2)

    after_t1 = _apply_scd2(inc_t1, v1)
    result = _apply_scd2(inc_t2, after_t1).orderBy("valid_from")

    assert result.count() == 3
    rows = result.collect()

    assert rows[0]["plan_tier"] == "free"   and rows[0]["valid_to"] == _T1
    assert rows[1]["plan_tier"] == "premium" and rows[1]["valid_to"] == _T2
    assert rows[2]["country"] == "DE"        and rows[2]["valid_to"] is None
    assert rows[2]["is_current"] is True


# ---------------------------------------------------------------------------
# valid_from derivation
# ---------------------------------------------------------------------------


def test_valid_from_prefers_updated_at(spark: SparkSession) -> None:
    """valid_from must be updated_at when present, falling back to created_at."""
    df = spark.createDataFrame(
        [("u1", "u@test.com", "PL", "free", "2026-01-01T08:00:00", "2026-03-02T09:00:00", None, None, _D1)],
        schema=_BRONZE_SCHEMA,
    )
    df_cast = _cast(df)
    valid, _ = _split(_validate(df_cast))
    deduped = _dedup(valid)
    row = deduped.withColumn(
        "valid_from",
        __import__("pyspark.sql.functions", fromlist=["coalesce", "col"]).coalesce(
            __import__("pyspark.sql.functions", fromlist=["col"]).col("updated_at_c"),
            __import__("pyspark.sql.functions", fromlist=["col"]).col("created_at_c"),
        ),
    ).collect()[0]

    assert row["valid_from"] == datetime(2026, 3, 2, 9, 0, 0)  # updated_at, not created_at


# ---------------------------------------------------------------------------
# Validation / quarantine
# ---------------------------------------------------------------------------


def test_quarantine_null_email(spark: SparkSession) -> None:
    """A row with null email must be quarantined with a matching reject_reason."""
    df = spark.createDataFrame(
        [("u1", None, "PL", "free", "2026-01-01T08:00:00", None, None, None, _D0)],
        schema=_BRONZE_SCHEMA,
    )
    valid, quarantine = _split(_validate(_cast(df)))

    assert valid.count() == 0
    assert quarantine.count() == 1
    assert "null email" in quarantine.collect()[0]["reject_reason"]


# ---------------------------------------------------------------------------
# Dedup within snapshot
# ---------------------------------------------------------------------------


def test_dedup_within_snapshot_keeps_one_row(spark: SparkSession) -> None:
    """Two identical rows for the same user_id must collapse to one version."""
    rows = [
        ("u1", "u@test.com", "PL", "free", "2026-01-01T08:00:00", None, None, None, _D0),
        ("u1", "u@test.com", "PL", "free", "2026-01-01T08:00:00", None, None, None, _D0),
    ]
    df = spark.createDataFrame(rows, schema=_BRONZE_SCHEMA)
    valid, _ = _split(_validate(_cast(df)))
    result = _dedup(valid)

    assert result.count() == 1
