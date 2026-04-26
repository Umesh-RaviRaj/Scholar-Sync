"""
User Chat API Router — endpoints for user-scoped chat management.

All endpoints require local JWT authentication and enforce
strict user-level data isolation via user_id checks.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from scholarsync.auth.router import get_current_user_local
from scholarsync.auth import chat_service
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/user-chat", tags=["User Chat"])


# ── Request / Response Models ───────────────────────────────────────

class CreateChatReq(BaseModel):
    title: str = "New Review"
    mode: str = "normal"

class UpdateTitleReq(BaseModel):
    title: str = Field(..., min_length=1)

class AddMessageReq(BaseModel):
    role: str          # 'user' | 'bot'
    type: str = "text" # 'text' | 'file' | 'report' | 'progress'
    content: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[str] = None
    report_markdown: Optional[str] = None


# ── POST /user-chat/create ──────────────────────────────────────────

@router.post("/create")
async def create_chat(
    request: CreateChatReq,
    user: dict = Depends(get_current_user_local),
):
    """Create a new chat thread for the authenticated user."""
    result = await chat_service.create_chat(
        user_id=user["id"],
        title=request.title,
        mode=request.mode,
    )
    return result


# ── GET /user-chat/list ─────────────────────────────────────────────

@router.get("/list")
async def list_chats(user: dict = Depends(get_current_user_local)):
    """List all chats for the authenticated user."""
    chats = await chat_service.list_chats(user_id=user["id"])
    return {"chats": chats}


# ── GET /user-chat/{chat_id}/messages ────────────────────────────────

@router.get("/{chat_id}/messages")
async def get_messages(
    chat_id: str,
    user: dict = Depends(get_current_user_local),
):
    """Get all messages for a chat. Verifies the chat belongs to this user."""
    messages = await chat_service.get_messages(
        user_id=user["id"],
        chat_id=chat_id,
    )
    return {"messages": messages}


# ── POST /user-chat/{chat_id}/message ────────────────────────────────

@router.post("/{chat_id}/message")
async def add_message(
    chat_id: str,
    request: AddMessageReq,
    user: dict = Depends(get_current_user_local),
):
    """Add a message to a chat. Verifies ownership."""
    success = await chat_service.add_message(
        user_id=user["id"],
        chat_id=chat_id,
        msg_data=request.model_dump(),
    )
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found or access denied.")
    return {"status": "ok"}


# ── PATCH /user-chat/{chat_id}/title ─────────────────────────────────

@router.patch("/{chat_id}/title")
async def update_title(
    chat_id: str,
    request: UpdateTitleReq,
    user: dict = Depends(get_current_user_local),
):
    """Update a chat's title. Verifies ownership."""
    success = await chat_service.update_title(
        user_id=user["id"],
        chat_id=chat_id,
        title=request.title,
    )
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found or access denied.")
    return {"status": "ok"}


# ── DELETE /user-chat/{chat_id} ──────────────────────────────────────

@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    user: dict = Depends(get_current_user_local),
):
    """Delete a chat and all its messages. Verifies ownership."""
    deleted = await chat_service.delete_chat(
        user_id=user["id"],
        chat_id=chat_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found or access denied.")
    return {"status": "ok", "chat_id": chat_id}
