"""DB layer — dual SQLite / PostgreSQL backend.

When DATABASE_URL is present in the environment, PostgreSQL is used.
Otherwise the system falls back to SQLite at data/saas.db.

Design
------
* _Row     — dict subclass that also supports integer positional access
             so that `row[0]` works for COUNT(*) results.
* _Cur     — thin cursor wrapper with .fetchone() / .fetchall().
* _Conn    — unified connection wrapper with .execute() / .executescript().
* _conn()  — context manager that yields a _Conn, commits on exit,
             rolls back on exception.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(os.environ.get("VIDEO_QA_SAAS_DB", "data/saas.db"))
_DATABASE_URL = os.environ.get("DATABASE_URL")
_USE_PG = bool(_DATABASE_URL)


# ── Unified row / cursor helpers ───────────────────────────────────────

class _Row(dict):
    """Dict row that also supports positional integer access (COUNT(*) etc.)."""
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


class _Cur:
    def __init__(self, cursor):
        self._c = cursor

    def fetchone(self) -> Optional[_Row]:
        row = self._c.fetchone()
        return _Row(dict(row)) if row is not None else None

    def fetchall(self) -> List[_Row]:
        return [_Row(dict(r)) for r in (self._c.fetchall() or [])]


class _Conn:
    def __init__(self, raw, is_pg: bool):
        self._raw = raw
        self._is_pg = is_pg

    def execute(self, sql: str, params=()) -> _Cur:
        if self._is_pg:
            import psycopg2.extras
            cur = self._raw.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql.replace("?", "%s"), params)
            return _Cur(cur)
        else:
            return _Cur(self._raw.execute(sql, params))

    def executescript(self, sql: str) -> None:
        """Run a multi-statement DDL script (no parameter binding)."""
        if self._is_pg:
            cur = self._raw.cursor()
            for stmt in sql.strip().split(";"):
                s = stmt.strip()
                if s:
                    cur.execute(s)
        else:
            self._raw.executescript(sql)

    def commit(self):
        self._raw.commit()

    def rollback(self):
        self._raw.rollback()

    def close(self):
        self._raw.close()


@contextmanager
def _conn():
    if _USE_PG:
        import psycopg2
        raw = psycopg2.connect(_DATABASE_URL)
        c = _Conn(raw, is_pg=True)
        try:
            yield c
            raw.commit()
        except Exception:
            raw.rollback()
            raise
        finally:
            raw.close()
    else:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = sqlite3.connect(str(DB_PATH))
        raw.row_factory = sqlite3.Row
        c = _Conn(raw, is_pg=False)
        try:
            yield c
            raw.commit()
        finally:
            raw.close()


# ── Schema init ────────────────────────────────────────────────────────

def init_db() -> None:
    """Create / migrate tables. Safe to call on every boot."""
    if _USE_PG:
        _init_pg()
    else:
        _init_sqlite()


def _init_pg() -> None:
    # Step 1 — ensure base tables exist (no new columns here so it's safe against old schemas)
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id                        TEXT PRIMARY KEY,
                email                     TEXT UNIQUE NOT NULL,
                password_hash             TEXT,
                created_at                DOUBLE PRECISION NOT NULL
            );

            CREATE TABLE IF NOT EXISTS videos (
                video_id    TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL REFERENCES users(id),
                filename    TEXT NOT NULL,
                status      TEXT NOT NULL,
                error       TEXT,
                created_at  DOUBLE PRECISION NOT NULL,
                updated_at  DOUBLE PRECISION NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_videos_user ON videos(user_id);

            CREATE TABLE IF NOT EXISTS rate_limit_events (
                id   BIGSERIAL PRIMARY KEY,
                key  TEXT NOT NULL,
                ts   DOUBLE PRECISION NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rle_key_ts ON rate_limit_events(key, ts)
        """)

    # Step 2 — idempotent column additions (must run before any index on those columns)
    _add_column_if_missing("users",  "otp_hash",                  "TEXT")
    _add_column_if_missing("users",  "otp_expiry",                "DOUBLE PRECISION")
    _add_column_if_missing("users",  "otp_attempts",              "INTEGER DEFAULT 0")
    _add_column_if_missing("users",  "otp_verified",              "INTEGER DEFAULT 0")
    _add_column_if_missing("users",  "tokens_invalidated_before", "DOUBLE PRECISION DEFAULT 0")
    _add_column_if_missing("users",  "email_verified",            "INTEGER DEFAULT 1")
    _add_column_if_missing("users",  "email_ver_hash",            "TEXT")
    _add_column_if_missing("users",  "email_ver_expiry",          "DOUBLE PRECISION")
    _add_column_if_missing("users",  "google_id",                 "TEXT")
    _add_column_if_missing("users",  "auth_provider",             "TEXT DEFAULT 'local'")
    _add_column_if_missing("videos", "job_id",                    "TEXT")
    _add_column_if_missing("videos", "progress",                  "INTEGER DEFAULT 0")
    _add_column_if_missing("videos", "stage",                     "TEXT DEFAULT 'queued'")
    _add_column_if_missing("videos", "file_url",                  "TEXT")
    _add_column_if_missing("videos", "source_url",                "TEXT")

    # Step 3 — idempotent indexes (columns guaranteed to exist now)
    with _conn() as c:
        c.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_id "
            "ON users(google_id) WHERE google_id IS NOT NULL"
        )
        c.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_job_id "
            "ON videos(job_id) WHERE job_id IS NOT NULL"
        )


def _init_sqlite() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS videos (
                video_id   TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                filename   TEXT NOT NULL,
                status     TEXT NOT NULL,
                error      TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                job_id     TEXT UNIQUE,
                progress   INTEGER DEFAULT 0,
                stage      TEXT DEFAULT 'queued',
                file_url   TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_videos_user   ON videos(user_id);
            CREATE INDEX IF NOT EXISTS idx_videos_job_id ON videos(job_id);

            CREATE TABLE IF NOT EXISTS rate_limit_events (
                id  INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                ts  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rle_key_ts ON rate_limit_events(key, ts)
        """)
    for col, col_def in [
        ("otp_hash",                  "TEXT"),
        ("otp_expiry",                "REAL"),
        ("otp_attempts",              "INTEGER DEFAULT 0"),
        ("otp_verified",              "INTEGER DEFAULT 0"),
        ("tokens_invalidated_before", "REAL DEFAULT 0"),
        ("email_verified",            "INTEGER DEFAULT 1"),
        ("email_ver_hash",            "TEXT"),
        ("email_ver_expiry",          "REAL"),
        ("google_id",                 "TEXT"),
        ("auth_provider",             "TEXT DEFAULT 'local'"),
        ("job_id",                    "TEXT"),
        ("progress",                  "INTEGER DEFAULT 0"),
        ("stage",                     "TEXT DEFAULT 'queued'"),
        ("file_url",                  "TEXT"),
        ("source_url",               "TEXT"),
    ]:
        _add_column_if_missing("users" if col in (
            "otp_hash", "otp_expiry", "otp_attempts", "otp_verified",
            "tokens_invalidated_before", "email_verified", "email_ver_hash",
            "email_ver_expiry", "google_id", "auth_provider",
        ) else "videos", col, col_def)
    # Unique indexes for SQLite (ALTER TABLE can't add UNIQUE)
    with _conn() as c:
        c.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_id "
            "ON users(google_id) WHERE google_id IS NOT NULL"
        )
        c.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_videos_job_id_uq "
            "ON videos(job_id) WHERE job_id IS NOT NULL"
        )


def _add_column_if_missing(table: str, column: str, col_def: str) -> None:
    if _USE_PG:
        pg_def = col_def.replace("REAL", "DOUBLE PRECISION")
        # Strip UNIQUE from col_def — handled via separate CREATE UNIQUE INDEX
        pg_def = pg_def.replace(" UNIQUE", "")
        with _conn() as c:
            c.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {pg_def}"
            )
    else:
        with _conn() as c:
            existing = {
                row["name"]
                for row in c.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if column not in existing:
                safe_def = col_def.replace(" UNIQUE", "")
                c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {safe_def}")


# ── Users ──────────────────────────────────────────────────────────────

def create_user(
    email: str,
    password_hash: str,
    auth_provider: str = "local",
    google_id: Optional[str] = None,
) -> Dict[str, Any]:
    user_id = uuid.uuid4().hex
    now = time.time()
    email_verified = 1 if auth_provider != "local" else 0
    with _conn() as c:
        c.execute(
            """INSERT INTO users
               (id, email, password_hash, created_at, email_verified, auth_provider, google_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, email.lower(), password_hash, now,
             email_verified, auth_provider, google_id),
        )
    return {
        "id": user_id, "email": email.lower(), "created_at": now,
        "email_verified": email_verified, "auth_provider": auth_provider,
        "google_id": google_id,
    }


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            """SELECT id, email, password_hash, created_at,
                      otp_hash, otp_expiry, otp_attempts, otp_verified,
                      tokens_invalidated_before,
                      email_verified, email_ver_hash, email_ver_expiry,
                      google_id, auth_provider
               FROM users WHERE email = ?""",
            (email.lower(),),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            """SELECT id, email, created_at, tokens_invalidated_before,
                      email_verified, auth_provider
               FROM users WHERE id = ?""",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_google_id(google_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            """SELECT id, email, password_hash, created_at,
                      tokens_invalidated_before, email_verified, auth_provider, google_id
               FROM users WHERE google_id = ?""",
            (google_id,),
        ).fetchone()
    return dict(row) if row else None


def link_google_account(user_id: str, google_id: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET google_id = ?, email_verified = 1 WHERE id = ?",
            (google_id, user_id),
        )


def update_password(user_id: str, password_hash: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
        )


# ── Email Verification ─────────────────────────────────────────────────

def set_email_verification(user_id: str, ver_hash: str, expiry: float) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET email_ver_hash = ?, email_ver_expiry = ? WHERE id = ?",
            (ver_hash, expiry, user_id),
        )


def mark_email_verified(user_id: str) -> None:
    with _conn() as c:
        c.execute(
            """UPDATE users
               SET email_verified = 1, email_ver_hash = NULL, email_ver_expiry = NULL
               WHERE id = ?""",
            (user_id,),
        )


# ── OTP ────────────────────────────────────────────────────────────────

def set_otp(user_id: str, otp_hash: str, expiry: float) -> None:
    with _conn() as c:
        c.execute(
            """UPDATE users
               SET otp_hash = ?, otp_expiry = ?, otp_attempts = 0, otp_verified = 0
               WHERE id = ?""",
            (otp_hash, expiry, user_id),
        )


def get_otp_data(user_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            "SELECT otp_hash, otp_expiry, otp_attempts, otp_verified FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def increment_otp_attempts(user_id: str) -> int:
    with _conn() as c:
        c.execute(
            "UPDATE users SET otp_attempts = COALESCE(otp_attempts, 0) + 1 WHERE id = ?",
            (user_id,),
        )
        row = c.execute(
            "SELECT otp_attempts FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return row["otp_attempts"] if row else 0


def mark_otp_verified(user_id: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET otp_verified = 1, otp_hash = NULL WHERE id = ?",
            (user_id,),
        )


def clear_otp(user_id: str) -> None:
    with _conn() as c:
        c.execute(
            """UPDATE users
               SET otp_hash = NULL, otp_expiry = NULL, otp_attempts = 0, otp_verified = 0
               WHERE id = ?""",
            (user_id,),
        )


def invalidate_user_tokens(user_id: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET tokens_invalidated_before = ? WHERE id = ?",
            (time.time(), user_id),
        )


# ── Rate Limiting ──────────────────────────────────────────────────────

def check_rate_limit(key: str, max_count: int, window_seconds: float) -> bool:
    cutoff = time.time() - window_seconds
    with _conn() as c:
        row = c.execute(
            "SELECT COUNT(*) AS cnt FROM rate_limit_events WHERE key = ? AND ts > ?",
            (key, cutoff),
        ).fetchone()
    return (row["cnt"] if row else 0) < max_count


def record_rate_event(key: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO rate_limit_events (key, ts) VALUES (?, ?)",
            (key, time.time()),
        )
    _prune_old_rate_events()


def _prune_old_rate_events() -> None:
    cutoff = time.time() - 86400
    with _conn() as c:
        c.execute("DELETE FROM rate_limit_events WHERE ts < ?", (cutoff,))


# ── Videos ─────────────────────────────────────────────────────────────

def register_video(
    video_id: str,
    user_id: str,
    filename: str,
    status: str = "uploaded",
    job_id: Optional[str] = None,
    file_url: Optional[str] = None,
    source_url: Optional[str] = None,
) -> None:
    now = time.time()
    with _conn() as c:
        c.execute(
            """INSERT INTO videos
               (video_id, user_id, filename, status, error,
                created_at, updated_at, job_id, progress, stage, file_url, source_url)
               VALUES (?, ?, ?, ?, NULL, ?, ?, ?, 0, ?, ?, ?)""",
            (video_id, user_id, filename, status,
             now, now, job_id, status, file_url, source_url),
        )


def update_video_status(
    video_id: str, status: str, error: Optional[str] = None
) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE videos SET status = ?, error = ?, updated_at = ? WHERE video_id = ?",
            (status, error, time.time(), video_id),
        )


def update_video_file_url(video_id: str, file_url: str) -> None:
    """Set the stored file_url for a video (used by URL-sourced jobs after download)."""
    with _conn() as c:
        c.execute(
            "UPDATE videos SET file_url = ?, updated_at = ? WHERE video_id = ?",
            (file_url, time.time(), video_id),
        )


def update_job_progress(
    job_id: str,
    progress: int,
    stage: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Update processing progress for the video associated with job_id."""
    with _conn() as c:
        c.execute(
            """UPDATE videos
               SET progress = ?, stage = ?, status = ?, error = ?, updated_at = ?
               WHERE job_id = ?""",
            (progress, stage, status, error, time.time(), job_id),
        )


def get_video(video_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            """SELECT video_id, user_id, filename, status, error,
                      created_at, updated_at, job_id, progress, stage, file_url
               FROM videos WHERE video_id = ?""",
            (video_id,),
        ).fetchone()
    return dict(row) if row else None


def get_video_by_job_id(job_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            """SELECT video_id, user_id, filename, status, error,
                      created_at, updated_at, job_id, progress, stage, file_url
               FROM videos WHERE job_id = ?""",
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def list_user_videos(user_id: str) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            """SELECT video_id, filename, status, error,
                      created_at, updated_at, job_id, progress, stage, file_url
               FROM videos WHERE user_id = ? ORDER BY created_at DESC""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]
