from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import List, Optional

from qdrant_client.http import models as qm

from app.models.schemas import Message, MessageCreate, MessageRole
from app.services.embedding_service import EmbeddingService
from app.core.config import settings
from app.services.qdrant_service import get_qdrant_service, make_base_payload, QdrantService


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class MessageService:
    COLLECTION = "messages"

    def __init__(self) -> None:
        self.qdrant = get_qdrant_service()
        self.embedder = EmbeddingService()

    async def add_message(
        self,
        chat_id: str,
        agent_id: str,
        wallet: str,
        message: MessageCreate,
    ) -> Message:
        now = _utc_now()
        message_id = str(uuid.uuid4())

        content = message.content or ""
        vec = await self.embedder.embed_text(content, expected_dim=settings.QDRANT_MESSAGE_VECTOR_SIZE)

        payload = {
            **make_base_payload("message"),
            "id": message_id,
            # Required schema
            "message_id": message_id,
            "chat_id": chat_id,
            "agent_id": agent_id,
            "wallet": wallet,
            "role": message.role.value if isinstance(message.role, MessageRole) else str(message.role),
            "content": content,
            "created_at": _iso(now),
            "updated_at": _iso(now),
            # Compatibility fields (existing API model uses timestamp)
            "timestamp": _iso(now),
        }

        self.qdrant.upsert_record(
            self.COLLECTION,
            message_id,
            payload,
            vector={QdrantService.MESSAGE_VECTOR_NAME: vec},
        )

        return Message(
            id=message_id,
            role=message.role,
            content=content,
            timestamp=now,
        )

    async def list_messages(
        self,
        chat_id: str,
        wallet: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Message]:
        must = [qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id))]
        if wallet:
            must.append(qm.FieldCondition(key="wallet", match=qm.MatchValue(value=wallet)))
        qfilter = qm.Filter(must=must)

        out: List[qm.Record] = []
        offset = None
        while True:
            points, next_offset = self.qdrant.query_by_filter(
                self.COLLECTION, qfilter=qfilter, limit=min(200, limit - len(out)), offset=offset
            )
            out.extend(points)
            if not next_offset or len(out) >= limit:
                break
            offset = next_offset

        # Sort chronologically (oldest first)
        def _ts(p: qm.Record) -> str:
            if not p.payload:
                return ""
            return str(p.payload.get("created_at") or p.payload.get("timestamp") or "")

        out.sort(key=_ts)

        messages: List[Message] = []
        for p in out:
            payload = p.payload or {}
            ts_raw = payload.get("timestamp") or payload.get("created_at")
            try:
                ts = datetime.fromisoformat(ts_raw) if ts_raw else None
            except Exception:
                ts = None

            role_val = payload.get("role") or "user"
            try:
                role = MessageRole(role_val)
            except Exception:
                role = MessageRole.USER

            messages.append(
                Message(
                    id=str(payload.get("id") or p.id),
                    role=role,
                    content=str(payload.get("content") or ""),
                    timestamp=ts,
                )
            )
        return messages

    async def semantic_recall(
        self,
        chat_id: str,
        agent_id: str,
        wallet: str,
        query: str,
        k: int = 5,
    ) -> List[Message]:
        vec = await self.embedder.embed_text(query, expected_dim=settings.QDRANT_MESSAGE_VECTOR_SIZE)
        qfilter = qm.Filter(
            must=[
                qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id)),
                qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id)),
                qm.FieldCondition(key="wallet", match=qm.MatchValue(value=wallet)),
            ]
        )
        hits = self.qdrant.search(
            self.COLLECTION,
            vector_name=QdrantService.MESSAGE_VECTOR_NAME,
            query_vector=vec,
            qfilter=qfilter,
            limit=k,
        )
        results: List[Message] = []
        for h in hits:
            payload = h.payload or {}
            ts_raw = payload.get("timestamp") or payload.get("created_at")
            try:
                ts = datetime.fromisoformat(ts_raw) if ts_raw else None
            except Exception:
                ts = None
            role_val = payload.get("role") or "user"
            try:
                role = MessageRole(role_val)
            except Exception:
                role = MessageRole.USER
            results.append(
                Message(
                    id=str(payload.get("id") or h.id),
                    role=role,
                    content=str(payload.get("content") or ""),
                    timestamp=ts,
                )
            )
        return results

    async def delete_messages_for_chat(self, chat_id: str, wallet: Optional[str] = None) -> None:
        must = [qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id))]
        if wallet:
            must.append(qm.FieldCondition(key="wallet", match=qm.MatchValue(value=wallet)))
        self.qdrant.delete_by_filter(self.COLLECTION, qm.Filter(must=must))

