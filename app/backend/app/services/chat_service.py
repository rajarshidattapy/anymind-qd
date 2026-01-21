from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import List, Optional

from qdrant_client.http import models as qm

from app.models.schemas import Chat, ChatCreate, ChatUpdate, MemorySize
from app.services.message_service import MessageService
from app.services.qdrant_service import get_qdrant_service, make_base_payload


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class ChatService:
    COLLECTION = "chats"

    def __init__(self) -> None:
        self.qdrant = get_qdrant_service()
        self.messages = MessageService()

    async def create_chat(self, agent_id: str, chat: ChatCreate, wallet: str) -> Chat:
        now = _utc_now()
        chat_id = str(uuid.uuid4())

        payload = {
            **make_base_payload("chat"),
            "id": chat_id,
            # Required schema
            "chat_id": chat_id,
            "agent_id": agent_id,
            "wallet": wallet,
            "title": chat.name,
            "created_at": _iso(now),
            # Compatibility fields (existing API model)
            "name": chat.name,
            "memory_size": chat.memory_size.value if isinstance(chat.memory_size, MemorySize) else str(chat.memory_size),
            "timestamp": _iso(now),
            "message_count": 0,
            "last_message": None,
            "capsule_id": chat.capsule_id,
            "user_wallet": wallet,
            "web_search_enabled": getattr(chat, "web_search_enabled", False),
        }

        self.qdrant.upsert_record(self.COLLECTION, chat_id, payload)

        return Chat(
            id=chat_id,
            name=chat.name,
            memory_size=chat.memory_size,
            last_message=None,
            timestamp=now,
            message_count=0,
            messages=[],
            agent_id=agent_id,
            capsule_id=chat.capsule_id,
            user_wallet=wallet,
            web_search_enabled=getattr(chat, "web_search_enabled", False),
        )

    async def list_chats(self, agent_id: str, wallet: Optional[str]) -> List[Chat]:
        must = [qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id))]
        if wallet:
            must.append(qm.FieldCondition(key="wallet", match=qm.MatchValue(value=wallet)))
        qfilter = qm.Filter(must=must)

        out: List[qm.Record] = []
        offset = None
        while True:
            points, next_offset = self.qdrant.query_by_filter(self.COLLECTION, qfilter=qfilter, limit=200, offset=offset)
            out.extend(points)
            if not next_offset:
                break
            offset = next_offset

        chats: List[Chat] = []
        for p in out:
            payload = p.payload or {}
            ts_raw = payload.get("timestamp") or payload.get("created_at")
            try:
                ts = datetime.fromisoformat(ts_raw) if ts_raw else _utc_now()
            except Exception:
                ts = _utc_now()

            mem_raw = payload.get("memory_size") or "Small"
            try:
                mem_size = MemorySize(mem_raw)
            except Exception:
                mem_size = MemorySize.SMALL

            chat_id = str(payload.get("chat_id") or payload.get("id") or p.id)
            msgs = await self.messages.list_messages(chat_id, wallet=payload.get("wallet") or wallet)

            chats.append(
                Chat(
                    id=chat_id,
                    name=str(payload.get("name") or payload.get("title") or ""),
                    memory_size=mem_size,
                    last_message=payload.get("last_message"),
                    timestamp=ts,
                    message_count=int(payload.get("message_count") or len(msgs)),
                    messages=msgs,
                    agent_id=str(payload.get("agent_id") or agent_id),
                    capsule_id=payload.get("capsule_id"),
                    user_wallet=payload.get("wallet") or payload.get("user_wallet") or wallet,
                    web_search_enabled=bool(payload.get("web_search_enabled") or False),
                )
            )

        # Newest first
        chats.sort(key=lambda c: c.timestamp, reverse=True)
        return chats

    async def get_chat(self, chat_id: str, wallet: Optional[str]) -> Optional[Chat]:
        rec = self.qdrant.get_by_id(self.COLLECTION, chat_id)
        if not rec or not rec.payload:
            return None
        payload = rec.payload

        if wallet and payload.get("wallet") != wallet:
            return None

        ts_raw = payload.get("timestamp") or payload.get("created_at")
        try:
            ts = datetime.fromisoformat(ts_raw) if ts_raw else _utc_now()
        except Exception:
            ts = _utc_now()

        mem_raw = payload.get("memory_size") or "Small"
        try:
            mem_size = MemorySize(mem_raw)
        except Exception:
            mem_size = MemorySize.SMALL

        msgs = await self.messages.list_messages(chat_id, wallet=payload.get("wallet") or wallet)
        return Chat(
            id=chat_id,
            name=str(payload.get("name") or payload.get("title") or ""),
            memory_size=mem_size,
            last_message=payload.get("last_message"),
            timestamp=ts,
            message_count=int(payload.get("message_count") or len(msgs)),
            messages=msgs,
            agent_id=payload.get("agent_id"),
            capsule_id=payload.get("capsule_id"),
            user_wallet=payload.get("wallet") or payload.get("user_wallet"),
            web_search_enabled=bool(payload.get("web_search_enabled") or False),
        )

    async def update_chat(self, chat_id: str, chat_update: ChatUpdate, wallet: Optional[str]) -> Chat:
        existing = await self.get_chat(chat_id, wallet)
        if not existing:
            raise Exception("Chat not found")

        # Update fields
        if chat_update.name is not None:
            existing.name = chat_update.name
        if chat_update.memory_size is not None:
            existing.memory_size = chat_update.memory_size
        if chat_update.web_search_enabled is not None:
            existing.web_search_enabled = chat_update.web_search_enabled

        rec = self.qdrant.get_by_id(self.COLLECTION, chat_id)
        payload = (rec.payload or {}) if rec else {}

        payload.update(
            {
                "name": existing.name,
                "title": existing.name,
                "memory_size": existing.memory_size.value,
                "web_search_enabled": existing.web_search_enabled,
                "updated_at": _iso(_utc_now()),
            }
        )
        self.qdrant.upsert_record(self.COLLECTION, chat_id, payload)
        return existing

    async def update_chat_counters(self, chat_id: str, wallet: Optional[str], message_count: int, last_message: str) -> None:
        rec = self.qdrant.get_by_id(self.COLLECTION, chat_id)
        if not rec or not rec.payload:
            return
        payload = rec.payload
        if wallet and payload.get("wallet") != wallet:
            return
        payload["message_count"] = message_count
        payload["last_message"] = last_message
        payload["updated_at"] = _iso(_utc_now())
        self.qdrant.upsert_record(self.COLLECTION, chat_id, payload)

    async def delete_chat(self, chat_id: str, wallet: Optional[str]) -> None:
        chat = await self.get_chat(chat_id, wallet)
        if not chat:
            return

        # Delete associated memories (mem0)
        try:
            from app.services.memory_service import MemoryService
            MemoryService().delete_chat_memories(chat.agent_id or "", chat_id)
        except Exception:
            # Memory is optional; chat deletion should still proceed
            pass

        # Delete messages then chat
        await self.messages.delete_messages_for_chat(chat_id, wallet=wallet)
        self.qdrant.delete_by_id(self.COLLECTION, chat_id)

