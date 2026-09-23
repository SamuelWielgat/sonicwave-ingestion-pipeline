# SonicWave Ingestion Pipeline

A medallion ingestion pipeline (Bronze → Silver) for a music streaming platform.
Two source tables — `plays` (append-only events) and `users` (SCD2 dimension) —
processed from daily JSON drops into typed, validated, idempotent Parquet.

[![CI](https://github.com/SamuelWielgat/sonicwave-ingestion-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/SamuelWielgat/sonicwave-ingestion-pipeline/actions/workflows/ci.yml)

---

## Quick start

```bash
# 1. Install dependencies
uv sync --extra dev

# 2. Generate seed data (three daily snapshots: T, T+1, T+2)
uv run python seed/generate_seed.py

# 3. Run the pipeline — one snapshot at a time, in order
for DATE in 2026-03-01 2026-03-02 2026-03-03; do
    uv run python scripts/ingest_plays.py --source ./data/source/plays --snapshot-date $DATE
    uv run python scripts/ingest_users.py --source ./data/source/users --snapshot-date $DATE
done
```

Output lands in `data/`:

```
data/
├── bronze/
│   ├── plays/snapshot_date=2026-03-01/
│   └── users/snapshot_date=2026-03-01/
├── silver/
│   ├── plays/snapshot_date=2026-03-01/   ← partitioned by snapshot_date
│   └── users/                            ← full overwrite (SCD2 dimension)
└── quarantine/
    ├── plays/snapshot_date=2026-03-01/
    └── users/snapshot_date=2026-03-01/
```

### Verify the output

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[1]").getOrCreate()

# plays Silver — 30 / 31 / 31 rows per snapshot
spark.read.parquet("data/silver/plays").groupBy("snapshot_date").count().show()

# users Silver — full SCD2 dimension (13 rows after T+2)
spark.read.parquet("data/silver/users").orderBy("user_id", "valid_from").show()
```

---

## Project structure

```
src/sonicwave_ingestion/
├── bronze.py            # shared: read_source(), write_bronze()
├── plays/
│   ├── bronze.py        # plays column list + run()
│   └── silver.py        # cast, validate, quarantine, dedup, event_date, write
└── users/
    ├── bronze.py        # users column list + run()
    └── silver.py        # cast, validate, quarantine, dedup, SCD2, write

scripts/
├── ingest_plays.py      # CLI entry-point → plays bronze + silver
└── ingest_users.py      # CLI entry-point → users bronze + silver

tests/
├── conftest.py          # SparkSession fixture
├── test_plays_silver.py # 8 tests: quarantine, dedup, late event, schema
└── test_users_silver.py # 8 tests: SCD2 new version, close, idempotency, valid_from
```

---

## Defended choices

### Bronze: all strings, no schema enforcement

Bronze is a verbatim copy of the source — every field lands as `StringType`.
Schema enforcement happens in Silver, not here.
This means Bronze never rejects a row: a malformed `ms_played="NaN"` lands
in Bronze unchanged, and Silver decides what to do with it (quarantine).
If the source schema changes, Bronze still lands the data; only Silver rules need updating.

### Silver: explicit `StructType`, no inference

Every Silver column is declared explicitly in `SILVER_SCHEMA`.
Spark schema inference is banned — it reads the whole file to guess types
and can change between runs if the sample changes.
Explicit schemas are a contract: a column that disappears from the source fails fast.

### Validation + quarantine — not crash, not silent drop

Invalid rows (null `play_id`, `ms_played <= 0`, null `email`, etc.) go to
`data/quarantine/` partitioned by `snapshot_date`.
They are never silently dropped and the pipeline never crashes.
An analyst can inspect `quarantine/plays/snapshot_date=2026-03-03/` to see
exactly which rows were rejected on which day and why (`reject_reason` column).

### `plays`: `event_date` from `played_at`, not `snapshot_date`

A play that happened on T but arrived in the T+1 drop gets `event_date = T`.
`snapshot_date` records *when we ingested it*; `event_date` records *when it happened*.
This is the only correct basis for daily play-count aggregations — using
`snapshot_date` would misdate late arrivals by one day.

### `users` SCD2 key: `coalesce(updated_at, created_at)`

The version timestamp that drives `valid_from` is `coalesce(updated_at, created_at)`:
- `updated_at` is set by the source system only when attributes actually change.
- If `updated_at` is null, the user was never modified — `created_at` is the version.

Tracked attributes that open a new version: `country`, `plan_tier`.
A new SCD2 version only opens when one of those changes — not on every snapshot.

### Shared `bronze.py` vs per-table `silver.py`

`sonicwave_ingestion/bronze.py` provides `read_source()` and `write_bronze()` —
functions that are truly identical across all tables (permissive JSON read,
provenance columns, Parquet write partitioned by `snapshot_date`).

Silver is intentionally **not** shared. `plays` and `users` have different validation
rules, different typed schemas, and fundamentally different write strategies
(partition overwrite vs full SCD2 dimension overwrite). Sharing that logic
would couple unrelated things.

---

## Idempotency — two different mechanisms

Re-running any snapshot produces identical output. The mechanism differs per table:

### `plays` — dynamic partition overwrite

```
write.option("partitionOverwriteMode", "dynamic").partitionBy("snapshot_date")
```

When snapshot T is re-run, only the `snapshot_date=T` partition is replaced.
Partitions T+1 and T+2 are untouched.
This works because plays are append-only — the same Bronze snapshot always
produces the same Silver rows.

### `users` — union + dedup by `(user_id, valid_from)`

For the SCD2 dimension there are no partitions to overwrite — the whole table
is rewritten on every run. Idempotency comes from the dedup step:

```
all_versions = existing_silver ∪ incoming_snapshot
dedup by (user_id, valid_from)  ← same user + same version timestamp = same version
```

`valid_from = coalesce(updated_at, created_at)` only advances when the source
system makes a real change. Re-running the same snapshot brings in the same
`valid_from` → the dedup eliminates the duplicate → `LEAD()` recomputes the same
`valid_to` and `is_current` → output is identical.

---

## Late data handling

### `plays` — late events are dated correctly

A play with `played_at = 2026-03-01T21:30` that arrives in the T+1 snapshot
(`snapshot_date = 2026-03-02`) gets:

```
event_date    = 2026-03-01   ← from played_at (event time)
snapshot_date = 2026-03-02   ← from Bronze partition (arrival time)
```

The play is stored in the T+1 partition but dated to T.
Aggregations over `event_date` will count it on the correct day.

### `users` — late attribute changes are versioned correctly

A user whose `plan_tier` changed on T but whose record arrived in the T+1 drop
gets `valid_from = updated_at` (the source-side timestamp of the change, not
the ingestion timestamp). The SCD2 history reflects when the change *happened*,
not when we saw it.

---

## Development

```bash
# Run tests
uv run pytest -v

# Lint + format
uv run ruff check src/ tests/
uv run ruff format src/ scripts/

# Type check
uv run mypy src/

# All checks at once (runs on every git commit via pre-commit)
uv run pre-commit run --all-files
```

## CI / CD

| Workflow | Trigger | Steps |
|---|---|---|
| `ci.yml` | every PR + push to `main` | `uv sync` → `ruff` → `mypy` → `pytest` |
| `release.yml` | tag `v*` | `uv build` → upload wheel + sdist artifact |

