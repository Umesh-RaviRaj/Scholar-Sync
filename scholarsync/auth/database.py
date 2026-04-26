"""
SQLite database setup for authentication.

Uses aiosqlite for non-blocking database operations.
Stores the auth database as a local file — no external services needed.
"""

from __future__ import annotations

import os
import aiosqlite
from pathlib import Path

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)

# ── Module-level connection path ────────────────────────────────────
_db_path: str | None = None


def _get_db_path() -> str:
    """
    Resolve the SQLite database file path to an absolute path.

    Always anchored to the project root (directory containing the
    scholarsync package), so the same db file is used regardless of
    which directory the server is launched from.
    """
    global _db_path
    if _db_path is None:
        settings = get_settings()
        raw_path = settings.auth_db_path

        # If path is relative, anchor it to the project root
        # (two levels up from this file: auth/ -> scholarsync/ -> project root)
        if not Path(raw_path).is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent
            _db_path = str(project_root / raw_path)
        else:
            _db_path = raw_path

        # Ensure parent directory exists
        Path(_db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info("Auth database path resolved to: %s", _db_path)

    return _db_path


async def get_auth_db() -> aiosqlite.Connection:
    """
    Open and return an async SQLite connection.

    Caller is responsible for closing the connection (use async with).
    Row factory is set to aiosqlite.Row for dict-like access.
    """
    db = await aiosqlite.connect(_get_db_path())
    db.row_factory = aiosqlite.Row
    # Enable WAL mode for better concurrent read performance
    await db.execute("PRAGMA journal_mode=WAL")
    # Enable foreign keys (required for ON DELETE CASCADE)
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_auth_db() -> None:
    """
    Initialize the auth database — create tables and indexes.
    Called once at application startup. Safe to call multiple times.
    """
    db_path = _get_db_path()
    logger.info("Initializing auth database at: %s", db_path)

    async with aiosqlite.connect(db_path) as db:
        # 1. Users table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS auth_users (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                username    TEXT    UNIQUE NOT NULL,
                password_hash TEXT  NOT NULL,
                role        TEXT    DEFAULT 'user',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Index for fast username lookups
        await db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_auth_users_username
            ON auth_users (username)
        """)

        # 2. Chat threads table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id          TEXT PRIMARY KEY,
                user_id     INTEGER NOT NULL,
                title       TEXT NOT NULL DEFAULT 'New Review',
                mode        TEXT NOT NULL DEFAULT 'normal',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES auth_users(id) ON DELETE CASCADE
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)")

        # 3. Messages table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id          TEXT PRIMARY KEY,
                chat_id     TEXT NOT NULL,
                role        TEXT NOT NULL,
                type        TEXT NOT NULL DEFAULT 'text',
                content     TEXT,
                file_name   TEXT,
                file_size   TEXT,
                report_markdown TEXT,
                timestamp   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id)")

        await db.commit()

    logger.info("Auth/Chat database initialized successfully")

