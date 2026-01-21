from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional

import httpx
from qdrant_client.http import models as qm

from app.core.config import settings
from app.models.schemas import WalletBalance, Earnings, StakingInfo, StakingCreate
from app.services.qdrant_service import get_qdrant_service, make_base_payload


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


class WalletService:
    CAPSULES_COLLECTION = "capsules"
    STAKING_COLLECTION = "staking"
    EARNINGS_COLLECTION = "earnings"

    def __init__(self) -> None:
        self.qdrant = get_qdrant_service()
        self.solana_rpc_url = settings.SOLANA_RPC_URL

    async def get_balance(self, wallet_address: str) -> WalletBalance:
        """Get SOL balance for a wallet (unchanged)"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.solana_rpc_url,
                    json={"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [wallet_address]},
                    timeout=10.0,
                )
                data = response.json()
                if "result" in data:
                    balance_lamports = data["result"]["value"]
                    balance_sol = balance_lamports / 1e9
                    return WalletBalance(wallet_address=wallet_address, balance=balance_sol, currency="SOL")
        except Exception:
            pass
        return WalletBalance(wallet_address=wallet_address, balance=0.0, currency="SOL")

    async def get_earnings(self, wallet_address: str, period: Optional[str] = None) -> Earnings:
        must = [qm.FieldCondition(key="wallet", match=qm.MatchValue(value=wallet_address))]
        qfilter = qm.Filter(must=must)

        out: List[qm.Record] = []
        offset = None
        while True:
            points, next_offset = self.qdrant.query_by_filter(self.EARNINGS_COLLECTION, qfilter=qfilter, limit=200, offset=offset)
            out.extend(points)
            if not next_offset:
                break
            offset = next_offset

        rows: List[Dict[str, Any]] = []
        total = 0.0
        for p in out:
            payload = p.payload or {}
            amount = float(payload.get("amount") or 0.0)
            total += amount
            rows.append(
                {
                    "id": payload.get("id") or p.id,
                    "wallet_address": payload.get("wallet_address") or payload.get("wallet") or wallet_address,
                    "wallet": payload.get("wallet") or wallet_address,
                    "capsule_id": payload.get("capsule_id"),
                    "amount": amount,
                    "created_at": payload.get("created_at") or payload.get("timestamp"),
                    "timestamp": payload.get("timestamp"),
                    "source": payload.get("source"),
                }
            )

        # No period filtering implemented previously either.
        return Earnings(wallet_address=wallet_address, total_earnings=total, capsule_earnings=rows, period=period)

    async def get_staking_info(self, wallet_address: str) -> List[StakingInfo]:
        must = [qm.FieldCondition(key="staker_wallet", match=qm.MatchValue(value=wallet_address))]
        qfilter = qm.Filter(must=must)

        out: List[qm.Record] = []
        offset = None
        while True:
            points, next_offset = self.qdrant.query_by_filter(self.STAKING_COLLECTION, qfilter=qfilter, limit=200, offset=offset)
            out.extend(points)
            if not next_offset:
                break
            offset = next_offset

        staking: List[StakingInfo] = []
        for p in out:
            payload = p.payload or {}
            staking.append(
                StakingInfo(
                    capsule_id=str(payload.get("capsule_id") or ""),
                    wallet_address=str(payload.get("wallet_address") or payload.get("staker_wallet") or ""),
                    stake_amount=float(payload.get("amount") or payload.get("stake_amount") or 0.0),
                    staked_at=_dt(payload.get("staked_at") or payload.get("timestamp")),
                )
            )
        # Newest first
        staking.sort(key=lambda s: s.staked_at, reverse=True)
        return staking

    async def create_staking(self, staking: StakingCreate, wallet_address: str) -> StakingInfo:
        now = _utc_now()
        stake_id = str(uuid.uuid4())

        payload = {
            **make_base_payload("staking"),
            "id": stake_id,
            # Required schema
            "capsule_id": staking.capsule_id,
            "staker_wallet": wallet_address,
            "amount": float(staking.stake_amount),
            "timestamp": _iso(now),
            # Compatibility fields
            "wallet_address": wallet_address,
            "stake_amount": float(staking.stake_amount),
            "staked_at": _iso(now),
        }
        self.qdrant.upsert_record(self.STAKING_COLLECTION, stake_id, payload)

        # Update capsule stake_amount (+ listed)
        capsule = self.qdrant.get_by_id(self.CAPSULES_COLLECTION, staking.capsule_id)
        if capsule and capsule.payload:
            cap = capsule.payload
            current = float(cap.get("stake_amount") or 0.0)
            new_stake = current + float(staking.stake_amount)
            cap["stake_amount"] = new_stake
            cap["is_listed"] = bool(new_stake > 0)
            cap["updated_at"] = _iso(_utc_now())
            self.qdrant.set_payload(self.CAPSULES_COLLECTION, staking.capsule_id, cap)

        return StakingInfo(
            capsule_id=staking.capsule_id,
            wallet_address=wallet_address,
            stake_amount=float(staking.stake_amount),
            staked_at=now,
        )

