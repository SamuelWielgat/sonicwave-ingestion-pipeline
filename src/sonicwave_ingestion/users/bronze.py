"""users Bronze: source column list and snapshot run function."""

from __future__ import annotations

from pyspark.sql import SparkSession

from sonicwave_ingestion import bronze

# All columns the users source drop carries — all land as strings in Bronze.
SOURCE_COLUMNS: list[str] = [
    "user_id",
    "email",
    "country",
    "plan_tier",
    "created_at",
    "updated_at",
]


def run(
    spark: SparkSession,
    source_path: str,
    snapshot_date: str,
    output_path: str,
) -> None:
    """Land one users snapshot drop into Bronze.

    Reads <source_path>/<snapshot_date>/ permissively, attaches provenance
    columns, and writes Parquet partitioned by snapshot_date.
    """
    df = bronze.read_source(
        spark,
        path=f"{source_path}/{snapshot_date}",
        snapshot_date=snapshot_date,
        columns=SOURCE_COLUMNS,
    )
    bronze.write_bronze(df, output_path)
