from __future__ import annotations

from typing import List

from qdrant_client.http import models as qm

from app.models.schemas import Capsule, MarketplaceFilters
from app.services.capsule_service import CapsuleService
from app.services.qdrant_service import get_qdrant_service


class MarketplaceService:
    COLLECTION = "capsules"

    def __init__(self) -> None:
        self.qdrant = get_qdrant_service()
        self.capsules = CapsuleService()

    async def browse_capsules(self, filters: MarketplaceFilters, limit: int, offset: int) -> List[Capsule]:
        """Browse marketplace with filters - only shows capsules that have been staked"""
        must = [qm.FieldCondition(key="stake_amount", range=qm.Range(gt=0))]
        if filters.category:
            must.append(qm.FieldCondition(key="category", match=qm.MatchValue(value=filters.category)))
        if filters.min_reputation is not None:
            must.append(qm.FieldCondition(key="reputation", range=qm.Range(gte=filters.min_reputation)))
        if filters.max_price is not None:
            must.append(qm.FieldCondition(key="price_per_query", range=qm.Range(lte=filters.max_price)))

        qfilter = qm.Filter(must=must)

        # Pull a window large enough for sorting + pagination (simple & predictable)
        fetch_limit = min(max(offset + limit, 50), 1000)
        points, _ = self.qdrant.query_by_filter(self.COLLECTION, qfilter=qfilter, limit=fetch_limit)
        capsules = [self.capsules._to_capsule(p.payload or {}) for p in points if p.payload]

        # Apply sorting
        sort_by = filters.sort_by or "popular"
        if sort_by == "popular":
            capsules.sort(key=lambda c: c.query_count, reverse=True)
        elif sort_by == "newest":
            capsules.sort(key=lambda c: c.created_at, reverse=True)
        elif sort_by == "price_low":
            capsules.sort(key=lambda c: c.price_per_query)
        elif sort_by == "price_high":
            capsules.sort(key=lambda c: c.price_per_query, reverse=True)
        elif sort_by == "rating":
            capsules.sort(key=lambda c: c.rating, reverse=True)

        return capsules[offset : offset + limit]

    async def get_trending_capsules(self, limit: int) -> List[Capsule]:
        must = [qm.FieldCondition(key="stake_amount", range=qm.Range(gt=0))]
        points, _ = self.qdrant.query_by_filter(self.COLLECTION, qfilter=qm.Filter(must=must), limit=min(limit * 5, 200))
        capsules = [self.capsules._to_capsule(p.payload or {}) for p in points if p.payload]
        capsules.sort(key=lambda c: c.query_count, reverse=True)
        return capsules[:limit]

    async def get_categories(self) -> List[str]:
        # Qdrant doesn't have an easy "distinct" payload query; scan a reasonable set.
        points, _ = self.qdrant.query_by_filter(self.COLLECTION, qfilter=None, limit=1000)
        cats = set()
        for p in points:
            if p.payload and p.payload.get("category"):
                cats.add(str(p.payload["category"]))
        if not cats:
            return ["Finance", "Gaming", "Health", "Technology", "Education"]
        return sorted(cats)

    async def search_capsules(self, query: str, limit: int) -> List[Capsule]:
        """Search capsules by name or description - only shows capsules that have been staked"""
        q = (query or "").strip().lower()
        must = [qm.FieldCondition(key="stake_amount", range=qm.Range(gt=0))]
        points, _ = self.qdrant.query_by_filter(self.COLLECTION, qfilter=qm.Filter(must=must), limit=1000)
        capsules = [self.capsules._to_capsule(p.payload or {}) for p in points if p.payload]

        if not q:
            return capsules[:limit]

        filtered = [
            c
            for c in capsules
            if q in (c.name or "").lower() or q in (c.description or "").lower()
        ]
        return filtered[:limit]

