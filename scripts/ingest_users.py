"""Entry-point: users ingestion pipeline (source -> Bronze -> Silver).

Parses CLI arguments and delegates all work to the sonicwave_ingestion package.
No business logic lives here.

Usage:
    python scripts/ingest_users.py \\
        --source ./data/source/users \\
        --snapshot-date 2026-03-01
"""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession

from sonicwave_ingestion.users import bronze


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest one users snapshot: source -> Bronze -> Silver."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the users source folder (e.g. ./data/source/users).",
    )
    parser.add_argument(
        "--snapshot-date",
        required=True,
        help="Snapshot date to process (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Root output directory (default: ./data).",
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("sonicwave-users")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    bronze.run(
        spark,
        source_path=args.source,
        snapshot_date=args.snapshot_date,
        output_path=f"{args.output_dir}/bronze/users",
    )

    spark.stop()


if __name__ == "__main__":
    main()
