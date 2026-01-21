from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import settings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    vectors: Dict[str, qm.VectorParams]


class QdrantService:
    """
    Single persistence layer for the backend.

    Rules:
    - Qdrant must be reachable; failures should surface immediately.
    - Uses payload as primary store; vectors only where needed.
    - Collections are created if missing.
    """

    DUMMY_VECTOR_NAME = "__dummy"
    DUMMY_VECTOR_SIZE = 1

    MESSAGE_VECTOR_NAME = "content"
    CAPSULE_VECTOR_NAME = "description"

    def __init__(self) -> None:
        if not settings.QDRANT_URL:
            raise RuntimeError("QDRANT_URL is required")

        # QdrantClient accepts url=... for both http(s) endpoints and local.
        self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY or None)

        # Fail loudly if unreachable and ensure collections exist.
        self.ping()
        self._ensure_collections()

    def ping(self) -> None:
        # Any request that hits the server is fine; collections is lightweight.
        self.client.get_collections()

    # ---------------------------------------------------------------------
    # Collection bootstrap
    # ---------------------------------------------------------------------

    def _collection_specs(self) -> List[CollectionSpec]:
        dummy = qm.VectorParams(size=self.DUMMY_VECTOR_SIZE, distance=qm.Distance.COSINE)
        msg = qm.VectorParams(size=settings.QDRANT_MESSAGE_VECTOR_SIZE, distance=qm.Distance.COSINE)
        cap = qm.VectorParams(size=settings.QDRANT_CAPSULE_VECTOR_SIZE, distance=qm.Distance.COSINE)

        return [
            CollectionSpec("agents", {self.DUMMY_VECTOR_NAME: dummy}),
            CollectionSpec("chats", {self.DUMMY_VECTOR_NAME: dummy}),
            CollectionSpec("messages", {self.MESSAGE_VECTOR_NAME: msg}),
            CollectionSpec("preferences", {self.DUMMY_VECTOR_NAME: dummy}),
            CollectionSpec("capsules", {self.CAPSULE_VECTOR_NAME: cap}),
            CollectionSpec("staking", {self.DUMMY_VECTOR_NAME: dummy}),
            CollectionSpec("earnings", {self.DUMMY_VECTOR_NAME: dummy}),
            # Link mem0 memory IDs back to chats/agents (payload-only)
            CollectionSpec("mem0_pointers", {self.DUMMY_VECTOR_NAME: dummy}),
        ]

    def _ensure_collections(self) -> None:
        for spec in self._collection_specs():
            if self.client.collection_exists(spec.name):
                continue
            self.client.create_collection(
                collection_name=spec.name,
                vectors_config=spec.vectors,
            )

    # ---------------------------------------------------------------------
    # CRUD helpers
    # ---------------------------------------------------------------------

    def upsert_record(
        self,
        collection: str,
        id: str,
        payload: Dict[str, Any],
        vector: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        """
        Upsert a single record.
        - payload is stored as Qdrant payload.
        - vector should be a dict of named vectors.
        """
        point_vector = vector or {self.DUMMY_VECTOR_NAME: [0.0]}
        point = qm.PointStruct(id=id, payload=payload, vector=point_vector)
        self.client.upsert(collection_name=collection, points=[point])

    def set_payload(self, collection: str, id: str, payload: Dict[str, Any]) -> None:
        """
        Update payload for an existing point without touching vectors.
        Use this for collections where vectors are required and you are not updating them.
        """
        self.client.set_payload(
            collection_name=collection,
            payload=payload,
            points=[id],
            wait=True,
        )

    def get_by_id(self, collection: str, id: str, with_vectors: bool = False) -> Optional[qm.Record]:
        records = self.client.retrieve(
            collection_name=collection,
            ids=[id],
            with_payload=True,
            with_vectors=with_vectors,
        )
        return records[0] if records else None

    def query_by_filter(
        self,
        collection: str,
        qfilter: Optional[qm.Filter] = None,
        limit: int = 100,
        offset: Optional[qm.PointId] = None,
        with_vectors: bool = False,
    ) -> Tuple[List[qm.Record], Optional[qm.PointId]]:
        points, next_offset = self.client.scroll(
            collection_name=collection,
            scroll_filter=qfilter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return points, next_offset

    def delete_by_filter(self, collection: str, qfilter: qm.Filter) -> None:
        self.client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(filter=qfilter),
            wait=True,
        )

    def delete_by_id(self, collection: str, id: str) -> None:
        self.client.delete(
            collection_name=collection,
            points_selector=qm.PointIdsList(points=[id]),
            wait=True,
        )

    def search(
        self,
        collection: str,
        vector_name: str,
        query_vector: List[float],
        qfilter: Optional[qm.Filter],
        limit: int = 10,
    ) -> List[qm.ScoredPoint]:
        return self.client.search(
            collection_name=collection,
            query_vector=qm.NamedVector(name=vector_name, vector=query_vector),
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )


# -------------------------------------------------------------------------
# Singleton lifecycle (initialized on FastAPI startup)
# -------------------------------------------------------------------------

_qdrant_singleton: Optional[QdrantService] = None


def init_qdrant_service() -> QdrantService:
    global _qdrant_singleton
    _qdrant_singleton = QdrantService()
    return _qdrant_singleton


def get_qdrant_service() -> QdrantService:
    if _qdrant_singleton is None:
        raise RuntimeError("QdrantService not initialized. Did startup run?")
    return _qdrant_singleton


def make_base_payload(record_type: str) -> Dict[str, Any]:
    now = _utc_now_iso()
    return {
        "type": record_type,
        "created_at": now,
        "updated_at": now,
    }

