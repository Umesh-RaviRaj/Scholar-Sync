"""
Chat service — business logic for user-scoped chats and messages using SQLite.
"""

from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from scholarsync.auth.database import get_auth_db
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)

def _generate_uuid() -> str:
    return str(uuid.uuid4())

async def create_chat(user_id: int, title: str = "New Review", mode: str = "normal") -> dict:
    """Create a new chat thread for a user."""
    chat_id = _generate_uuid()
    db = await get_auth_db()
    try:
        await db.execute(
            "INSERT INTO chats (id, user_id, title, mode) VALUES (?, ?, ?, ?)",
            (chat_id, user_id, title, mode)
        )
        await db.commit()
        return {
            "id": chat_id,
            "title": title,
            "mode": mode,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
    finally:
        await db.close()

async def list_chats(user_id: int) -> list[dict]:
    """List all chats for a specific user."""
    db = await get_auth_db()
    try:
        cursor = await db.execute(
            "SELECT id, title, mode, created_at FROM chats WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()

async def get_messages(user_id: int, chat_id: str) -> list[dict]:
    """Fetch all messages for a chat, verifying ownership."""
    db = await get_auth_db()
    try:
        # Verify ownership first
        cursor = await db.execute(
            "SELECT id FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id)
        )
        if not await cursor.fetchone():
            return []

        cursor = await db.execute(
            "SELECT id, role, type, content, file_name, file_size, report_markdown, timestamp FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
            (chat_id,)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()

async def add_message(user_id: int, chat_id: str, msg_data: dict) -> bool:
    """Add a message to a chat, verifying ownership."""
    db = await get_auth_db()
    try:
        # Verify ownership
        cursor = await db.execute(
            "SELECT id FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id)
        )
        if not await cursor.fetchone():
            return False

        msg_id = _generate_uuid()
        await db.execute(
            """INSERT INTO messages 
               (id, chat_id, role, type, content, file_name, file_size, report_markdown) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                msg_id, 
                chat_id, 
                msg_data["role"], 
                msg_data.get("type", "text"),
                msg_data.get("content"),
                msg_data.get("file_name"),
                msg_data.get("file_size"),
                msg_data.get("report_markdown")
            )
        )
        # Update chat timestamp
        await db.execute(
            "UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (chat_id,)
        )
        await db.commit()
        return True
    finally:
        await db.close()

async def delete_chat(user_id: int, chat_id: str) -> bool:
    """Delete a chat and its messages, verifying ownership."""
    db = await get_auth_db()
    try:
        cursor = await db.execute(
            "DELETE FROM chats WHERE id = ? AND user_id = ?",
            (chat_id, user_id)
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()

async def update_title(user_id: int, chat_id: str, title: str) -> bool:
    """Update the title of a chat."""
    db = await get_auth_db()
    try:
        cursor = await db.execute(
            "UPDATE chats SET title = ? WHERE id = ? AND user_id = ?",
            (title, chat_id, user_id)
        )
        await db.commit()
        return cursor.rowcount > 0
    finally:
        await db.close()
