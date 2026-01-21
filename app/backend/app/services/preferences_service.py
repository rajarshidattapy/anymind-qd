from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from qdrant_client.http import models as qm

from app.services.qdrant_service import get_qdrant_service, make_base_payload


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PreferencesService:
    COLLECTION = "preferences"

    def __init__(self) -> None:
        self.qdrant = get_qdrant_service()

    async def get_preferences(self, wallet: str) -> Dict[str, Any]:
        rec_id = f"pref:{wallet}"
        record = self.qdrant.get_by_id(self.COLLECTION, rec_id)
        if not record or not record.payload:
            return {}
        return (record.payload.get("preferences") or {})  # type: ignore[return-value]

    async def upsert_preferences(self, wallet: str, new_prefs: Dict[str, Any]) -> Dict[str, Any]:
        rec_id = f"pref:{wallet}"
        existing = await self.get_preferences(wallet)
        merged = {**existing, **new_prefs}

        payload = {
            **make_base_payload("preferences"),
            "id": rec_id,
            "wallet": wallet,
            "preferences": merged,
            "updated_at": _utc_now_iso(),
        }
        # Keep original created_at if record exists
        existing_record = self.qdrant.get_by_id(self.COLLECTION, rec_id)
        if existing_record and existing_record.payload and existing_record.payload.get("created_at"):
            payload["created_at"] = existing_record.payload["created_at"]

        self.qdrant.upsert_record(self.COLLECTION, rec_id, payload)
        return merged

    async def clear_preferences(self, wallet: str) -> None:
        rec_id = f"pref:{wallet}"
        self.qdrant.delete_by_id(self.COLLECTION, rec_id)

