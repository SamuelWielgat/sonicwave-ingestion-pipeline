"""Shared Bronze-layer helpers: permissive source read and partitioned Parquet write."""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType


def read_source(
    spark: SparkSession,
    path: str,
    snapshot_date: str,
    columns: list[str],
) -> DataFrame:
    """Read one snapshot drop permissively (all strings) and add provenance columns.

    Every source column is read as StringType — types are enforced in Silver, not here.
    Missing fields (e.g. a null user_id that JSON omits entirely) become null strings.
    Provenance columns added:
        ingested_at   — wall-clock time this run landed the data
        source_file   — path of the source file each row came from
        snapshot_date — the drop date this snapshot belongs to
    """
    schema = StructType([StructField(c, StringType(), True) for c in columns])
    return (
        spark.read.schema(schema)
        .option("mode", "PERMISSIVE")
        .json(path)
        .withColumn("ingested_at", F.current_timestamp())
        .withColumn("source_file", F.input_file_name())
        .withColumn("snapshot_date", F.lit(snapshot_date))
    )


def write_bronze(df: DataFrame, output_path: str) -> None:
    """Write Bronze DataFrame as Parquet, partitioned by snapshot_date.

    Dynamic partition overwrite ensures only the partition being processed
    is rewritten — re-running the same snapshot date is idempotent.
    """
    (
        df.write.option("partitionOverwriteMode", "dynamic")
        .mode("overwrite")
        .partitionBy("snapshot_date")
        .parquet(output_path)
    )
