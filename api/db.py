"""Lightweight SQLite store for users, sessions, and video registry.

Uses the stdlib `sqlite3` driver — no SQLAlchemy needed for this surface area.
The DB file lives at `data/saas.db`. All write paths use `with _conn() as c:`
so commits/rollbacks are automatic.
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
    """Create tables if they do not exist. Safe to call on every boot."""
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            TEXT PRIMARY KEY,
                email         TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at    REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS videos (
                video_id   TEXT PRIMARY KEY,         -- namespaced as "{user_id}__{slug}"
                user_id    TEXT NOT NULL,
                filename   TEXT NOT NULL,
                status     TEXT NOT NULL,            -- uploaded | processing | ready | failed
                error      TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_videos_user ON videos(user_id);
            """
        )


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
            "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
            (email.lower(),),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute(
            "SELECT id, email, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


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
