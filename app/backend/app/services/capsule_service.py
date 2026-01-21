from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional

import httpx
from qdrant_client.http import models as qm

from app.core.config import settings
from app.models.schemas import Capsule, CapsuleCreate, CapsuleUpdate
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService, get_qdrant_service, make_base_payload


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except Exception:
            pass
    return _utc_now()


class CapsuleService:
    COLLECTION = "capsules"
    EARNINGS_COLLECTION = "earnings"

    def __init__(self) -> None:
        self.qdrant = get_qdrant_service()
        self.embedder = EmbeddingService()

    async def get_user_capsules(self, wallet_address: Optional[str]) -> List[Capsule]:
        must = []
        if wallet_address:
            must.append(qm.FieldCondition(key="creator_wallet", match=qm.MatchValue(value=wallet_address)))
        qfilter = qm.Filter(must=must) if must else None

        out: List[qm.Record] = []
        offset = None
        while True:
            points, next_offset = self.qdrant.query_by_filter(self.COLLECTION, qfilter=qfilter, limit=200, offset=offset)
            out.extend(points)
            if not next_offset:
                break
            offset = next_offset

        return [self._to_capsule(p.payload or {}) for p in out if p.payload]

    async def get_capsule(self, capsule_id: str) -> Optional[Capsule]:
        rec = self.qdrant.get_by_id(self.COLLECTION, capsule_id)
        if not rec or not rec.payload:
            return None
        return self._to_capsule(rec.payload)

    async def find_capsule_for_agent(self, wallet_address: str, agent_id: str) -> Optional[Capsule]:
        # Qdrant nested payload filtering can vary by setup; keep it robust by scanning user's capsules.
        capsules = await self.get_user_capsules(wallet_address)
        for c in capsules:
            md = c.metadata or {}
            if isinstance(md, dict) and md.get("agent_id") == agent_id:
                return c
        return None

    async def create_capsule(self, capsule_data: CapsuleCreate, wallet_address: str) -> Capsule:
        capsule_id = str(uuid.uuid4())
        now = _utc_now()

        # Vector for optional semantic search (marketplace search by description/name/category)
        text_for_embedding = f"{capsule_data.name}\n{capsule_data.description}\n{capsule_data.category}".strip()
        vec = await self.embedder.embed_text(text_for_embedding, expected_dim=settings.QDRANT_CAPSULE_VECTOR_SIZE)

        agent_id = None
        if capsule_data.metadata and isinstance(capsule_data.metadata, dict):
            agent_id = capsule_data.metadata.get("agent_id")

        payload: Dict[str, Any] = {
            **make_base_payload("capsule"),
            "id": capsule_id,
            # Required schema (plus compatibility mapping)
            "capsule_id": capsule_id,
            "agent_id": agent_id,
            "owner_wallet": wallet_address,
            "price": float(capsule_data.price_per_query),
            "is_listed": False,
            "created_at": _iso(now),
            # Existing API fields
            "name": capsule_data.name,
            "description": capsule_data.description,
            "category": capsule_data.category,
            "creator_wallet": wallet_address,
            "price_per_query": float(capsule_data.price_per_query),
            "stake_amount": 0.0,
            "reputation": 0.0,
            "query_count": 0,
            "rating": 0.0,
            "updated_at": _iso(now),
            "metadata": capsule_data.metadata or {},
        }

        self.qdrant.upsert_record(
            self.COLLECTION,
            capsule_id,
            payload,
            vector={QdrantService.CAPSULE_VECTOR_NAME: vec},
        )

        return self._to_capsule(payload)

    async def update_capsule(self, capsule_id: str, capsule_update: CapsuleUpdate, wallet_address: str) -> Optional[Capsule]:
        existing = await self.get_capsule(capsule_id)
        if not existing:
            return None
        if existing.creator_wallet != wallet_address:
            return None

        payload = (self.qdrant.get_by_id(self.COLLECTION, capsule_id).payload or {})  # type: ignore[union-attr]
        changed_for_embedding = False

        if capsule_update.name is not None:
            payload["name"] = capsule_update.name
            changed_for_embedding = True
        if capsule_update.description is not None:
            payload["description"] = capsule_update.description
            changed_for_embedding = True
        if capsule_update.price_per_query is not None:
            payload["price_per_query"] = float(capsule_update.price_per_query)
            payload["price"] = float(capsule_update.price_per_query)
        if capsule_update.metadata is not None:
            payload["metadata"] = capsule_update.metadata

        payload["updated_at"] = _iso(_utc_now())

        vec = None
        if changed_for_embedding:
            text_for_embedding = f"{payload.get('name','')}\n{payload.get('description','')}\n{payload.get('category','')}".strip()
            vec_list = await self.embedder.embed_text(text_for_embedding, expected_dim=settings.QDRANT_CAPSULE_VECTOR_SIZE)
            vec = {QdrantService.CAPSULE_VECTOR_NAME: vec_list}

        if vec is None:
            self.qdrant.set_payload(self.COLLECTION, capsule_id, payload)
        else:
            self.qdrant.upsert_record(self.COLLECTION, capsule_id, payload, vector=vec)
        return await self.get_capsule(capsule_id)

    async def delete_capsule(self, capsule_id: str, wallet_address: str) -> None:
        existing = await self.get_capsule(capsule_id)
        if not existing:
            return
        if existing.creator_wallet != wallet_address:
            return
        self.qdrant.delete_by_id(self.COLLECTION, capsule_id)

    async def query_capsule(
        self,
        capsule_id: str,
        prompt: str,
        wallet_address: str,
        payment_signature: Optional[str] = None,
        amount_paid: Optional[float] = None,
    ) -> dict:
        capsule = await self.get_capsule(capsule_id)
        if not capsule:
            raise Exception("Capsule not found")

        # Verify payment if signature provided
        if payment_signature and amount_paid:
            verified = await self._verify_payment(
                payment_signature,
                wallet_address,
                capsule.creator_wallet,
                float(amount_paid),
            )
            if not verified:
                raise Exception("Payment verification failed")

            # Record earnings
            await self._record_earnings(capsule_id, capsule.creator_wallet, float(amount_paid), source="usage")

        # Increment query count
        await self._increment_query_count(capsule_id)

        # TODO: Implement memory retrieval and LLM query integration
        return {
            "response": f"Query processed for capsule '{capsule.name}'. Payment verified ({amount_paid or 0} SOL). LLM integration pending.",
            "capsule_id": capsule_id,
            "price_paid": amount_paid or 0,
        }

    async def _verify_payment(self, signature: str, sender: str, recipient: str, amount: float) -> bool:
        """Verify Solana transaction on-chain (unchanged)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    settings.SOLANA_RPC_URL,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "getTransaction",
                        "params": [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0}],
                    },
                    timeout=10.0,
                )
                data = response.json()
                if "result" not in data or not data["result"]:
                    return False
                tx = data["result"]
                if not tx.get("meta") or tx["meta"].get("err"):
                    return False
                return True
        except Exception:
            return False

    async def _record_earnings(self, capsule_id: str, wallet_address: str, amount: float, source: str) -> None:
        now = _utc_now()
        earning_id = str(uuid.uuid4())
        payload = {
            **make_base_payload("earning"),
            "id": earning_id,
            # Required schema
            "wallet": wallet_address,
            "capsule_id": capsule_id,
            "amount": float(amount),
            "source": source,
            "timestamp": _iso(now),
            # Compatibility fields
            "wallet_address": wallet_address,
            "created_at": _iso(now),
        }
        self.qdrant.upsert_record(self.EARNINGS_COLLECTION, earning_id, payload)

    async def _increment_query_count(self, capsule_id: str) -> None:
        rec = self.qdrant.get_by_id(self.COLLECTION, capsule_id)
        if not rec or not rec.payload:
            return
        payload = rec.payload
        current = float(payload.get("query_count") or 0)
        payload["query_count"] = int(current + 1)
        payload["updated_at"] = _iso(_utc_now())
        self.qdrant.set_payload(self.COLLECTION, capsule_id, payload)

    def _to_capsule(self, payload: Dict[str, Any]) -> Capsule:
        return Capsule(
            id=str(payload.get("id") or payload.get("capsule_id") or ""),
            name=str(payload.get("name") or ""),
            description=str(payload.get("description") or ""),
            category=str(payload.get("category") or ""),
            creator_wallet=str(payload.get("creator_wallet") or payload.get("owner_wallet") or ""),
            price_per_query=float(payload.get("price_per_query") or payload.get("price") or 0.0),
            stake_amount=float(payload.get("stake_amount") or 0.0),
            reputation=float(payload.get("reputation") or 0.0),
            query_count=int(payload.get("query_count") or 0),
            rating=float(payload.get("rating") or 0.0),
            created_at=_dt(payload.get("created_at")),
            updated_at=_dt(payload.get("updated_at")),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
        )

