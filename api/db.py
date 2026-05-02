"""Lightweight SQLite store for users, sessions, and video registry.

Uses the stdlib `sqlite3` driver — no SQLAlchemy needed for this surface area.
The DB file lives at `data/saas.db`. All write paths use `with _conn() as c:`
so commits/rollbacks are automatic.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(os.environ.get("VIDEO_QA_SAAS_DB", "data/saas.db"))


def _ensure_dirs() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn():
    _ensure_dirs()
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    """Create tables and add new columns if they do not exist. Safe to call on every boot."""
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id                       TEXT PRIMARY KEY,
                email                    TEXT UNIQUE NOT NULL,
                password_hash            TEXT NOT NULL,
                created_at               REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS videos (
                video_id   TEXT PRIMARY KEY,
                user_id    TEXT NOT NULL,
                filename   TEXT NOT NULL,
                status     TEXT NOT NULL,
                error      TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_videos_user ON videos(user_id);

            CREATE TABLE IF NOT EXISTS rate_limit_events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                key        TEXT NOT NULL,
                ts         REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_rle_key_ts ON rate_limit_events(key, ts);
            """
        )

    # Safe ALTER TABLE migrations — add columns only if missing
    _add_column_if_missing("users", "otp_hash",                  "TEXT")
    _add_column_if_missing("users", "otp_expiry",                "REAL")
    _add_column_if_missing("users", "otp_attempts",              "INTEGER DEFAULT 0")
    _add_column_if_missing("users", "otp_verified",              "INTEGER DEFAULT 0")
    _add_column_if_missing("users", "tokens_invalidated_before", "REAL DEFAULT 0")


def _add_column_if_missing(table: str, column: str, col_def: str) -> None:
    """ALTER TABLE … ADD COLUMN only when the column doesn't exist yet."""
    with _conn() as c:
        existing = {
            row[1]
            for row in c.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")


# ── Users ──────────────────────────────────────────────────────────────

def create_user(email: str, password_hash: str) -> Dict[str, Any]:
    user_id = uuid.uuid4().hex
    now = time.time()
    with _conn() as c:
        c.execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, email.lower(), password_hash, now),
        )
    return {"id": user_id, "email": email.lower(), "created_at": now}


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            """SELECT id, email, password_hash, created_at,
                      otp_hash, otp_expiry, otp_attempts, otp_verified,
                      tokens_invalidated_before
               FROM users WHERE email = ?""",
            (email.lower(),),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            """SELECT id, email, created_at, tokens_invalidated_before
               FROM users WHERE id = ?""",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def update_password(user_id: str, password_hash: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id),
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
    """Increment and return the new attempts count."""
    with _conn() as c:
        c.execute(
            "UPDATE users SET otp_attempts = COALESCE(otp_attempts, 0) + 1 WHERE id = ?",
            (user_id,),
        )
        row = c.execute("SELECT otp_attempts FROM users WHERE id = ?", (user_id,)).fetchone()
    return row["otp_attempts"] if row else 0


def mark_otp_verified(user_id: str) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE users SET otp_verified = 1 WHERE id = ?",
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
    """Mark all tokens issued before now as invalid."""
    with _conn() as c:
        c.execute(
            "UPDATE users SET tokens_invalidated_before = ? WHERE id = ?",
            (time.time(), user_id),
        )


# ── Rate Limiting (sliding window) ────────────────────────────────────

def check_rate_limit(key: str, max_count: int, window_seconds: float) -> bool:
    """Return True if the action is ALLOWED (count < max_count within window)."""
    cutoff = time.time() - window_seconds
    with _conn() as c:
        count = c.execute(
            "SELECT COUNT(*) FROM rate_limit_events WHERE key = ? AND ts > ?",
            (key, cutoff),
        ).fetchone()[0]
    return count < max_count


def record_rate_event(key: str) -> None:
    with _conn() as c:
        c.execute(
            "INSERT INTO rate_limit_events (key, ts) VALUES (?, ?)",
            (key, time.time()),
        )
    _prune_old_rate_events()


def _prune_old_rate_events() -> None:
    """Remove events older than 24 hours to keep the table small."""
    cutoff = time.time() - 86400
    with _conn() as c:
        c.execute("DELETE FROM rate_limit_events WHERE ts < ?", (cutoff,))


# ── Videos ─────────────────────────────────────────────────────────────

def register_video(video_id: str, user_id: str, filename: str, status: str = "uploaded") -> None:
    now = time.time()
    with _conn() as c:
        c.execute(
            """INSERT INTO videos (video_id, user_id, filename, status, error, created_at, updated_at)
               VALUES (?, ?, ?, ?, NULL, ?, ?)""",
            (video_id, user_id, filename, status, now, now),
        )


def update_video_status(video_id: str, status: str, error: Optional[str] = None) -> None:
    with _conn() as c:
        c.execute(
            "UPDATE videos SET status = ?, error = ?, updated_at = ? WHERE video_id = ?",
            (status, error, time.time(), video_id),
        )


def get_video(video_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            """SELECT video_id, user_id, filename, status, error, created_at, updated_at
               FROM videos WHERE video_id = ?""",
            (video_id,),
        ).fetchone()
    return dict(row) if row else None


def list_user_videos(user_id: str) -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            """SELECT video_id, filename, status, error, created_at, updated_at
               FROM videos WHERE user_id = ? ORDER BY created_at DESC""",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]
