from __future__ import annotations

from typing import Dict, List, Optional, Any
import logging

from qdrant_client.http import models as qm

from app.core.config import settings
from app.services.qdrant_service import get_qdrant_service, make_base_payload

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Semantic memory service powered by mem0 OSS, with Qdrant as the only persistence layer.

    - No hosted Mem0 platform client.
    - No local ChromaDB fallback.
    - If mem0 cannot be initialized, memory features are disabled (but persistence remains Qdrant-only).
    """

    def __init__(self) -> None:
        self.memory: Any = None
        self.use_platform = False  # Kept for compatibility; always False here.

        if not settings.MEM0_ENABLED:
            self.memory = None
            return

        # mem0 OSS requires an LLM + embedder; default providers typically need OPENAI_API_KEY.
        if not settings.OPENAI_API_KEY:
            self.memory = None
            return

        try:
            from mem0 import Memory  # type: ignore
        except Exception:
            self.memory = None
            return

        # Configure mem0 to store its vectors in Qdrant (collection: mem0_memories)
        config = {
            "version": "v1.1",
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0_memories",
                    "url": settings.QDRANT_URL,
                    "api_key": settings.QDRANT_API_KEY or None,
                    "embedding_model_dims": settings.QDRANT_MESSAGE_VECTOR_SIZE,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": settings.OPENAI_API_KEY,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": settings.OPENAI_EMBEDDING_MODEL,
                    "api_key": settings.OPENAI_API_KEY,
                },
            },
        }

        try:
            self.memory = Memory.from_config(config)
        except Exception:
            self.memory = None

    def _is_available(self) -> bool:
        return self.memory is not None

    def get_chat_memories(
        self,
        agent_id: str,
        chat_id: str,
        query: str,
        memory_size: str = "Medium",
        limit: Optional[int] = None,
        capsule_id: Optional[str] = None,
    ) -> List[Dict]:
        if not self._is_available():
            return []

        if limit is None:
            limits = {"Small": 3, "Medium": 5, "Large": 10}
            limit = limits.get(memory_size, 5)

        try:
            metadata = {"chat_id": chat_id, "agent_id": agent_id}
            if capsule_id:
                metadata["capsule_id"] = capsule_id
            return self.memory.search(query=query, user_id=agent_id, metadata=metadata, limit=limit)  # type: ignore[misc]
        except Exception:
            return []

    def store_chat_memory(
        self,
        agent_id: str,
        chat_id: str,
        messages: List[Dict[str, str]],
        capsule_id: Optional[str] = None,
    ) -> bool:
        if not self._is_available():
            return False
        if not messages or len(messages) < 2:
            return False

        try:
            metadata = {"chat_id": chat_id, "agent_id": agent_id}
            if capsule_id:
                metadata["capsule_id"] = capsule_id

            result = self.memory.add(messages=messages, user_id=agent_id, metadata=metadata)  # type: ignore[misc]

            # Persist "pointer" records to Qdrant for traceability/linkage
            mem_ids: List[str] = []
            if isinstance(result, dict):
                mid = result.get("id") or result.get("memory_id")
                if mid:
                    mem_ids.append(str(mid))
            elif isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        mid = item.get("id") or item.get("memory_id")
                        if mid:
                            mem_ids.append(str(mid))

            if mem_ids:
                qdrant = get_qdrant_service()
                for mid in mem_ids:
                    pointer_id = f"mem0:{mid}"
                    payload = {
                        **make_base_payload("mem0_pointer"),
                        "id": pointer_id,
                        "mem0_memory_id": mid,
                        "agent_id": agent_id,
                        "chat_id": chat_id,
                        "capsule_id": capsule_id,
                    }
                    qdrant.upsert_record("mem0_pointers", pointer_id, payload)

            return True
        except Exception:
            return False

    def format_memory_context(self, memories: List[Dict]) -> str:
        if not memories:
            return ""
        lines = []
        for mem in memories:
            text = mem.get("memory", "")
            if text:
                lines.append(f"- {text}")
        return "\n".join(lines)

    def get_all_chat_memories(
        self,
        agent_id: str,
        chat_id: str,
        capsule_id: Optional[str] = None,
    ) -> List[Dict]:
        if not self._is_available():
            return []
        try:
            metadata = {"chat_id": chat_id, "agent_id": agent_id}
            if capsule_id:
                metadata["capsule_id"] = capsule_id
            # Open-source mem0 doesn't guarantee a get_all API; use a broad search.
            return self.memory.search(query="", user_id=agent_id, metadata=metadata, limit=100)  # type: ignore[misc]
        except Exception:
            return []

    def delete_chat_memories(self, agent_id: str, chat_id: str) -> bool:
        """
        Best-effort:
        - Delete pointer records (always possible).
        - Actual mem0 memory deletion depends on mem0 capabilities.
        """
        try:
            qdrant = get_qdrant_service()
            qdrant.delete_by_filter(
                "mem0_pointers",
                qm.Filter(
                    must=[
                        qm.FieldCondition(key="agent_id", match=qm.MatchValue(value=agent_id)),
                        qm.FieldCondition(key="chat_id", match=qm.MatchValue(value=chat_id)),
                    ]
                ),
            )
        except Exception:
            pass

        # mem0 OSS delete-by-metadata is not guaranteed; keep behavior non-fatal.
        return False

