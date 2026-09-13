"""SonicWave source-data seed generator.

Generates three daily source snapshots (T, T+1, T+2) for two SonicWave
tables and writes them, through a Spark session, into a local landing folder:

    <output>/<table>/<snapshot_date>/part-*.json

This is the *raw source* your ingestion pipeline consumes — the daily drop a
source system leaves for you. It is deliberately **permissive and messy**:
everything lands as text (JSON strings), and each snapshot carries real-world
problems your pipeline has to survive — new rows, a late-arriving event, a
dimension attribute that changes, and a handful of malformed rows.

**Every row carries two source timestamps:**

    created_at   when the source row was born (never null)
    updated_at   when it was last changed -- NULL if it has never changed
                 since creation. So `updated_at IS NULL` means "still in its
                 original state"; a non-null value marks a real update.

Together they give you a proper change signal: order SCD2 versions by
`coalesce(updated_at, created_at)`. For `plays`, `created_at` is the source
*record* time, so a **late** play has `played_at` on an earlier day than its
`created_at` (recorded a day after it happened).

The two tables and what each one is here to make you handle:

    plays     high-volume EVENT feed  -> partition-overwrite idempotency;
                                         a LATE event (played_at < created_at);
                                         plus a duplicate id + malformed rows
                                         (negative / non-numeric duration, null
                                         user) to dedup and quarantine
    users     SCD2 DIMENSION          -> a tracked attribute CHANGES across days
                                         (updated_at flips from null to a time);
                                         plus a duplicate row + a null-email row
                                         to dedup and quarantine

Run it:

    uv run python seed/generate_seed.py                 # -> ./data/source
    uv run python seed/generate_seed.py --output ./data/source

**This seed is yours.** Change it, add rows, invent a specific edge case you
want your pipeline to prove it handles — the data below is a starting point,
not a fixed input.
"""

from __future__ import annotations

import argparse

from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

# Three consecutive daily snapshots: T, T+1, T+2.
T, T1, T2 = "2026-03-01", "2026-03-02", "2026-03-03"
SNAPSHOTS = [T, T1, T2]


# ---------------------------------------------------------------------------
# users — a FULL snapshot of the user table each day (a dimension extract).
# The pipeline compares consecutive snapshots to build SCD2 history.
#   * created_at = when the user was first created (always present)
#   * updated_at = null until an attribute changes, then the change time
#   * user 3 changes plan_tier (T+1) then country (T+2)  -> two SCD2 versions
#   * user 5 changes country (T+1)
#   * new users appear at T+1 (id 9) and T+2 (id 10)
#   * messy: a duplicate row (dedup) and a row with a null email (quarantine)
# ---------------------------------------------------------------------------
USER_COLS = ["user_id", "email", "country", "plan_tier", "created_at", "updated_at"]


def _user(uid, email, country, tier, created_at, updated_at=None):
    return {
        "user_id": uid,
        "email": email,
        "country": country,
        "plan_tier": tier,
        "created_at": created_at,
        "updated_at": updated_at,  # None -> "never updated since creation"
    }


_USERS_BASE = [
    _user("1", "alice@sonicwave.io", "PL", "free", "2026-01-10T08:00:00"),
    _user("2", "bob@sonicwave.io", "US", "premium", "2026-01-11T08:00:00"),
    _user("3", "cara@sonicwave.io", "PL", "free", "2026-01-12T08:00:00"),
    _user("4", "dan@sonicwave.io", "DE", "premium", "2026-01-13T08:00:00"),
    _user("5", "eve@sonicwave.io", "US", "free", "2026-01-14T08:00:00"),
    _user("6", "finn@sonicwave.io", "GB", "premium", "2026-01-15T08:00:00"),
    _user("7", "gina@sonicwave.io", "PL", "free", "2026-01-16T08:00:00"),
    _user("8", "hugo@sonicwave.io", "FR", "premium", "2026-01-17T08:00:00"),
]


def users_for(snapshot: str) -> list[dict]:
    rows = [dict(u) for u in _USERS_BASE]
    by_id = {u["user_id"]: u for u in rows}

    if snapshot == T:
        # Nobody has changed yet -> every updated_at is null.
        # A duplicate row of user 4 in the same drop -> your dedup must fire.
        rows.append(dict(by_id["4"]))
        return rows

    # T+1: user 3 upgrades, user 5 relocates (updated_at set), user 9 joins.
    by_id["3"].update(plan_tier="premium", updated_at="2026-03-02T09:00:00")
    by_id["5"].update(country="CA", updated_at="2026-03-02T10:00:00")
    if snapshot == T1:
        rows.append(_user("9", "iris@sonicwave.io", "ES", "free", "2026-03-02T11:00:00"))
        return rows

    # T+2: user 3 relocates too -> its updated_at advances to the newer change;
    # user 10 joins; and a malformed row (null email) Silver must quarantine.
    by_id["3"].update(country="DE", updated_at="2026-03-03T08:00:00")
    rows.append(_user("9", "iris@sonicwave.io", "ES", "free", "2026-03-02T11:00:00"))
    rows.append(_user("10", "jack@sonicwave.io", "IT", "premium", "2026-03-03T09:00:00"))
    rows.append(_user("11", None, "PL", "free", "2026-03-03T09:30:00"))  # null email
    return rows


# ---------------------------------------------------------------------------
# plays — an INCREMENTAL event feed: each drop is that day's new plays.
#   * played_at  = when the play happened (event time)
#   * created_at = when the source recorded it; == played_at for on-time events,
#                  but LATER for a late arrival
#   * updated_at = null (events are immutable here)
#   * messy at T: a negative ms_played, a null user_id, a duplicate play_id
#   * a non-numeric ms_played at T+2
# ---------------------------------------------------------------------------
PLAY_COLS = [
    "play_id", "user_id", "content_id", "device_id",
    "played_at", "created_at", "updated_at", "ms_played",
]


def _play(pid, uid, cid, did, played_at, ms_played, created_at=None):
    return {
        "play_id": pid,
        "user_id": uid,
        "content_id": cid,
        "device_id": did,
        "played_at": played_at,
        "created_at": created_at if created_at is not None else played_at,
        "updated_at": None,
        "ms_played": ms_played,
    }


def _generated_plays(snapshot: str, start_id: int, n: int = 30) -> list[dict]:
    rows = []
    for i in range(n):
        played_at = f"{snapshot}T{i % 24:02d}:{i * 7 % 60:02d}:00"
        rows.append(
            _play(
                str(start_id + i),
                str((i % 8) + 1),
                str((i % 10) + 1),
                str((i % 5) + 1),
                played_at,
                str(120000 + (i * 1234) % 60000),
            )
        )
    return rows


def plays_for(snapshot: str) -> list[dict]:
    if snapshot == T:
        rows = _generated_plays(T, 1000)
        rows += [
            # negative duration -> range check must reject it
            _play("1900", "2", "3", "1", f"{T}T12:00:00", "-5000"),
            # null user_id -> required-field check must reject it
            _play("1901", None, "4", "2", f"{T}T13:00:00", "180000"),
            # duplicate play_id (same key as the first generated row) -> dedup
            _play("1000", "1", "1", "1", f"{T}T00:00:00", "200000"),
        ]
        return rows

    if snapshot == T1:
        rows = _generated_plays(T1, 2000)
        # LATE: happened on day T but recorded (created_at) on T+1.
        rows.append(_play("2900", "5", "7", "3", f"{T}T21:30:00", "175000",
                          created_at=f"{T1}T06:00:00"))
        return rows

    rows = _generated_plays(T2, 3000)
    rows += [
        # another late arrival: played on T+1, recorded on T+2
        _play("3900", "6", "2", "4", f"{T1}T22:15:00", "160000",
              created_at=f"{T2}T06:00:00"),
        # unparseable duration -> cast failure your validation must catch
        _play("3901", "3", "9", "2", f"{T2}T14:00:00", "NaN"),
    ]
    return rows


# ---------------------------------------------------------------------------
# Writing: one all-string DataFrame per (table, snapshot), written as JSON.
# All columns are strings on purpose — a permissive landing where everything
# arrives as text, so your Silver layer is what applies types and rules.
# ---------------------------------------------------------------------------
TABLES = {
    "users": (USER_COLS, users_for),
    "plays": (PLAY_COLS, plays_for),
}


def write_table(spark: SparkSession, name: str, columns, builder, out_dir: str) -> None:
    schema = StructType([StructField(c, StringType(), True) for c in columns])
    for snapshot in SNAPSHOTS:
        rows = builder(snapshot)
        data = [tuple(r.get(c) for c in columns) for r in rows]
        df = spark.createDataFrame(data, schema=schema)
        path = f"{out_dir}/{name}/{snapshot}"
        df.coalesce(1).write.mode("overwrite").json(path)
        print(f"  {name:8s} {snapshot}  ({len(rows):3d} rows) -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SonicWave source snapshots.")
    parser.add_argument(
        "--output",
        default="./data/source",
        help="Landing folder for the generated snapshots (default: ./data/source).",
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("sonicwave-seed")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"Generating source snapshots into {args.output} ...")
    for name, (columns, builder) in TABLES.items():
        write_table(spark, name, columns, builder, args.output)
    print("Done. Point your pipeline at --source", args.output)

    spark.stop()


if __name__ == "__main__":
    main()
