"""plays Silver: typed schema, validation, quarantine, dedup, and event dating."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DateType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from pyspark.sql.window import Window

# ---------------------------------------------------------------------------
# Silver schema — typed, explicit, never inferred.
# nullable=False marks fields our validation rules treat as required.
# ---------------------------------------------------------------------------
SILVER_SCHEMA = StructType(
    [
        StructField("play_id", StringType(), False),
        StructField("user_id", StringType(), False),
        StructField("content_id", StringType(), True),
        StructField("device_id", StringType(), True),
        StructField("played_at", TimestampType(), False),
        StructField("created_at", TimestampType(), True),
        StructField("updated_at", TimestampType(), True),
        StructField("ms_played", LongType(), False),
        StructField("event_date", DateType(), False),
        StructField("ingested_at", TimestampType(), True),
        StructField("source_file", StringType(), True),
        StructField("snapshot_date", StringType(), False),
    ]
)


# ---------------------------------------------------------------------------
# Transform steps — each is a pure DataFrame → DataFrame function so they
# can be tested independently with hand-built sample DataFrames.
# ---------------------------------------------------------------------------


def _cast(df: DataFrame) -> DataFrame:
    """Attempt type casts on all typed Silver columns.

    Results land in *_c columns alongside the original strings so the
    validation step can distinguish null-source from cast-failure.
    """
    return (
        df.withColumn("played_at_c", F.to_timestamp("played_at"))
        .withColumn("created_at_c", F.to_timestamp("created_at"))
        .withColumn("updated_at_c", F.to_timestamp("updated_at"))
        .withColumn("ms_played_c", F.col("ms_played").cast(LongType()))
    )


def _validate(df: DataFrame) -> DataFrame:
    """Add reject_reason column.

    An empty string means the row passed all rules.
    Rules:
        - play_id must not be null
        - user_id must not be null
        - played_at must parse to a timestamp (required for event_date)
        - ms_played must parse to a long AND be > 0
    """
    return df.withColumn(
        "reject_reason",
        F.concat_ws(
            ", ",
            F.when(F.col("play_id").isNull(), F.lit("null play_id")),
            F.when(F.col("user_id").isNull(), F.lit("null user_id")),
            F.when(F.col("played_at_c").isNull(), F.lit("invalid played_at")),
            F.when(F.col("ms_played_c").isNull(), F.lit("invalid ms_played")),
            F.when(
                F.col("ms_played_c").isNotNull() & (F.col("ms_played_c") <= 0),
                F.lit("ms_played <= 0"),
            ),
        ),
    )


def _split(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Split into (valid, quarantine) on reject_reason."""
    valid = df.filter(F.col("reject_reason") == "")
    quarantine = df.filter(F.col("reject_reason") != "")
    return valid, quarantine


def _dedup(df: DataFrame) -> DataFrame:
    """Keep the earliest record per play_id, ordered by created_at.

    Uses row_number() over a window — the standard dedup pattern for Silver.
    Rows with a null created_at are ranked last (asc_nulls_last).
    """
    w = Window.partitionBy("play_id").orderBy(F.col("created_at_c").asc_nulls_last())
    return (
        df.withColumn("_rn", F.row_number().over(w))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )


def _conform(df: DataFrame) -> DataFrame:
    """Select final Silver columns and derive event_date from played_at.

    event_date is derived from played_at (event time), not from snapshot_date
    (arrival time) — this is what makes late-arriving events date correctly.
    A play that happened on T but arrived in the T+1 drop gets event_date=T.
    """
    return df.select(
        F.col("play_id"),
        F.col("user_id"),
        F.col("content_id"),
        F.col("device_id"),
        F.col("played_at_c").alias("played_at"),
        F.col("created_at_c").alias("created_at"),
        F.col("updated_at_c").alias("updated_at"),
        F.col("ms_played_c").alias("ms_played"),
        F.to_date(F.col("played_at_c")).alias("event_date"),
        F.col("ingested_at"),
        F.col("source_file"),
        F.col("snapshot_date"),
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_silver(df: DataFrame, output_path: str) -> None:
    """Write Silver plays partitioned by snapshot_date, dynamic overwrite."""
    (
        df.write.option("partitionOverwriteMode", "dynamic")
        .mode("overwrite")
        .partitionBy("snapshot_date")
        .parquet(output_path)
    )


def _write_quarantine(df: DataFrame, output_path: str) -> None:
    """Write rejected rows partitioned by snapshot_date, dynamic overwrite."""
    (
        df.write.option("partitionOverwriteMode", "dynamic")
        .mode("overwrite")
        .partitionBy("snapshot_date")
        .parquet(output_path)
    )


# ---------------------------------------------------------------------------
# Public entry-point called by the ingest script
# ---------------------------------------------------------------------------


def run(
    spark: SparkSession,
    bronze_path: str,
    snapshot_date: str,
    silver_output: str,
    quarantine_output: str,
) -> None:
    """Process one plays Bronze snapshot into typed Silver + quarantine.

    Reads only the snapshot_date partition from Bronze, so this stage is
    independently re-runnable without re-landing Bronze.
    """
    df = spark.read.parquet(bronze_path).filter(
        F.col("snapshot_date") == snapshot_date
    )

    df_cast = _cast(df)
    df_flagged = _validate(df_cast)
    valid, quarantine = _split(df_flagged)

    _write_silver(_conform(_dedup(valid)), silver_output)
    _write_quarantine(quarantine, quarantine_output)
