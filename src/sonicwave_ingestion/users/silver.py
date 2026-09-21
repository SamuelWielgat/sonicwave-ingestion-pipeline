"""users Silver: typed schema, SCD2 history, validation, and quarantine."""

from __future__ import annotations

import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# Silver schema — SCD2 dimension; no snapshot partitioning (full overwrite).
# ---------------------------------------------------------------------------
SILVER_SCHEMA = StructType(
    [
        StructField("user_id", StringType(), False),
        StructField("email", StringType(), True),
        StructField("country", StringType(), True),
        StructField("plan_tier", StringType(), True),
        StructField("valid_from", TimestampType(), False),
        StructField("valid_to", TimestampType(), True),    # null = current version
        StructField("is_current", BooleanType(), False),
        StructField("ingested_at", TimestampType(), True),
        StructField("source_file", StringType(), True),
        StructField("snapshot_date", DateType(), True),
    ]
)

# Columns carried into the union — SCD2 fields are always recomputed, never copied.
_VERSION_COLS: list[str] = [
    "user_id",
    "email",
    "country",
    "plan_tier",
    "valid_from",
    "ingested_at",
    "source_file",
    "snapshot_date",
]


# ---------------------------------------------------------------------------
# Transform steps
# ---------------------------------------------------------------------------


def _cast(df: DataFrame) -> DataFrame:
    """Cast string timestamp columns to TimestampType."""
    return (
        df.withColumn("created_at_c", F.to_timestamp("created_at"))
        .withColumn("updated_at_c", F.to_timestamp("updated_at"))
    )


def _validate(df: DataFrame) -> DataFrame:
    """Add reject_reason column; empty string means the row is valid.

    Rules:
        - user_id must not be null (can't build SCD2 key without it)
        - email must not be null (brief-specified required field for users)
    """
    return df.withColumn(
        "reject_reason",
        F.concat_ws(
            ", ",
            F.when(F.col("user_id").isNull(), F.lit("null user_id")),
            F.when(F.col("email").isNull(), F.lit("null email")),
        ),
    )


def _split(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Return (valid, quarantine) based on reject_reason."""
    valid = df.filter(F.col("reject_reason") == "")
    quarantine = df.filter(F.col("reject_reason") != "")
    return valid, quarantine


def _dedup(df: DataFrame) -> DataFrame:
    """Keep one row per user_id within the snapshot.

    Ordered by coalesce(updated_at_c, created_at_c) ascending — earliest version
    timestamp wins on ties (mirrors the SCD2 version key logic below).
    """
    w = Window.partitionBy("user_id").orderBy(
        F.coalesce(F.col("updated_at_c"), F.col("created_at_c")).asc_nulls_last()
    )
    return (
        df.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def _apply_scd2(incoming: DataFrame, existing: DataFrame) -> DataFrame:
    """Build the full SCD2 dimension from incoming + existing Silver.

    Algorithm:
        1. Union version records (strip SCD2 fields; they are always recomputed).
        2. Dedup by (user_id, valid_from) — idempotency guard.
           Same user + same version timestamp = same version.
           Re-running the same snapshot can't produce a new valid_from because
           updated_at only advances when the source system makes a real change.
        3. Recompute SCD2 fields via window functions:
               valid_to   = LEAD(valid_from) OVER (PARTITION BY user_id ORDER BY valid_from)
               is_current = (valid_to IS NULL)

    Tracked attributes that drive new versions: country, plan_tier.
    """
    all_versions = existing.select(_VERSION_COLS).union(incoming.select(_VERSION_COLS))

    # Idempotency guard: one row per (user_id, valid_from), latest snapshot wins.
    w_dedup = Window.partitionBy("user_id", "valid_from").orderBy(
        F.col("snapshot_date").desc_nulls_last()
    )
    all_versions = (
        all_versions.withColumn("_rn", F.row_number().over(w_dedup))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    # Recompute SCD2 fields for the full history.
    w_scd2 = Window.partitionBy("user_id").orderBy("valid_from")
    return all_versions.withColumn(
        "valid_to", F.lead("valid_from").over(w_scd2)
    ).withColumn(
        "is_current", F.col("valid_to").isNull()
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_existing_silver(spark: SparkSession, silver_path: str) -> DataFrame:
    """Read the existing Silver dimension; return an empty frame on first run."""
    if os.path.isdir(silver_path):
        return spark.read.parquet(silver_path)
    return spark.createDataFrame([], SILVER_SCHEMA)


def _write_silver(df: DataFrame, output_path: str) -> None:
    """Overwrite the entire Silver dimension (SCD2 is always fully recomputed)."""
    df.write.mode("overwrite").parquet(output_path)


def _write_quarantine(df: DataFrame, output_path: str) -> None:
    """Write rejected rows partitioned by snapshot_date, dynamic overwrite."""
    (
        df.write.option("partitionOverwriteMode", "dynamic")
        .mode("overwrite")
        .partitionBy("snapshot_date")
        .parquet(output_path)
    )


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def run(
    spark: SparkSession,
    bronze_path: str,
    snapshot_date: str,
    silver_output: str,
    quarantine_output: str,
) -> None:
    """Process one users Bronze snapshot into the full SCD2 Silver dimension.

    Reads only the snapshot_date partition from Bronze, unions with existing
    Silver history, and rewrites the complete dimension — all SCD2 fields are
    recomputed from scratch on every run, making re-runs safe.
    """
    df = spark.read.parquet(bronze_path).filter(
        F.col("snapshot_date") == snapshot_date
    )

    df_cast = _cast(df)
    df_flagged = _validate(df_cast)
    valid, quarantine = _split(df_flagged)

    deduped = _dedup(valid)
    incoming = deduped.withColumn(
        "valid_from",
        F.coalesce(F.col("updated_at_c"), F.col("created_at_c")),
    )

    existing = _read_existing_silver(spark, silver_output)
    result = _apply_scd2(incoming, existing)

    _write_silver(result, silver_output)
    _write_quarantine(quarantine, quarantine_output)
