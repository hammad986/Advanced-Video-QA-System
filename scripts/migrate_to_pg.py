#!/usr/bin/env python3
"""Migrate all data from SQLite (data/saas.db) to PostgreSQL (DATABASE_URL).

Usage
-----
  python scripts/migrate_to_pg.py [--dry-run]

The script:
  1. Reads every row from the SQLite database.
  2. Upserts each row into the target PostgreSQL database, skipping
     rows whose primary key already exists (idempotent — safe to re-run).
  3. Prints a summary of rows migrated per table.

Prerequisites
-------------
  * DATABASE_URL must be set in the environment.
  * Both databases must be reachable from the machine running this script.
  * Run AFTER `uvicorn` has been started at least once (so init_db() has
    created the PG tables).

Safety
------
  * The script never deletes data from either database.
  * Use --dry-run to preview what would be migrated without writing.
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

SQLITE_PATH = Path(os.environ.get("VIDEO_QA_SAAS_DB", "data/saas.db"))
PG_URL      = os.environ.get("DATABASE_URL", "")


def _connect_pg():
    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        sys.exit("psycopg2-binary is not installed. Run: pip install psycopg2-binary")
    conn = psycopg2.connect(PG_URL)
    conn.autocommit = False
    return conn


def _connect_sqlite():
    if not SQLITE_PATH.exists():
        sys.exit(f"SQLite DB not found at {SQLITE_PATH}")
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_table(
    sqlite_conn,
    pg_conn,
    table: str,
    pk: str,
    dry_run: bool,
) -> int:
    sqlite_cur = sqlite_conn.execute(f"SELECT * FROM {table}")
    rows = sqlite_cur.fetchall()
    if not rows:
        print(f"  {table}: 0 rows — nothing to migrate.")
        return 0

    cols = [d[0] for d in sqlite_cur.description]
    placeholders = ", ".join(["%s"] * len(cols))
    col_list     = ", ".join(cols)
    upsert_sql   = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT ({pk}) DO NOTHING"
    )

    inserted = 0
    pg_cur   = pg_conn.cursor()
    for row in rows:
        values = [row[c] for c in cols]
        if not dry_run:
            pg_cur.execute(upsert_sql, values)
            inserted += pg_cur.rowcount
        else:
            inserted += 1  # count as "would insert" in dry-run

    if not dry_run:
        pg_conn.commit()

    label = "would migrate" if dry_run else "migrated"
    print(f"  {table}: {inserted}/{len(rows)} rows {label}.")
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite → PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, no writes")
    args = parser.parse_args()

    if not PG_URL:
        sys.exit("DATABASE_URL is not set. Cannot connect to PostgreSQL.")

    print(f"SQLite source : {SQLITE_PATH}")
    print(f"PostgreSQL    : {PG_URL.split('@')[-1] if '@' in PG_URL else PG_URL[:40]}")
    print(f"Dry run       : {args.dry_run}\n")

    sqlite_conn = _connect_sqlite()
    pg_conn     = _connect_pg()

    tables = [
        ("users",              "id"),
        ("videos",             "video_id"),
        ("rate_limit_events",  "id"),
    ]

    total = 0
    t0 = time.time()
    for table, pk in tables:
        # Check table exists in SQLite
        check = sqlite_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not check:
            print(f"  {table}: not found in SQLite — skipping.")
            continue
        total += _migrate_table(sqlite_conn, pg_conn, table, pk, args.dry_run)

    elapsed = time.time() - t0
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Done. {total} total rows in {elapsed:.1f}s.")
    sqlite_conn.close()
    pg_conn.close()


if __name__ == "__main__":
    main()
