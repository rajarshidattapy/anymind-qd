from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import List, Optional

from qdrant_client.http import models as qm

from app.core.crypto import decrypt_secret, encrypt_secret
from app.models.schemas import (
    Agent,
    AgentCreate,
    AgentUpdate,
    Chat,
    ChatCreate,
    ChatUpdate,
    Message,
    MessageCreate,
)
from app.services.chat_service import ChatService
from app.services.message_service import MessageService
from app.services.qdrant_service import get_qdrant_service, make_base_payload


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


class AgentService:
    COLLECTION = "agents"

    def __init__(self) -> None:
        self.qdrant = get_qdrant_service()
        self.chats = ChatService()
        self.messages = MessageService()

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------

    async def get_user_agents(self, wallet_address: Optional[str]) -> List[Agent]:
        must = []
        if wallet_address:
            must.append(qm.FieldCondition(key="wallet", match=qm.MatchValue(value=wallet_address)))
        qfilter = qm.Filter(must=must) if must else None

        out: List[qm.Record] = []
        offset = None
        while True:
            points, next_offset = self.qdrant.query_by_filter(self.COLLECTION, qfilter=qfilter, limit=200, offset=offset)
            out.extend(points)
            if not next_offset:
                break
            offset = next_offset

        agents: List[Agent] = []
        for p in out:
            payload = p.payload or {}
            agent_id = str(payload.get("agent_id") or payload.get("id") or p.id)

            agents.append(
                Agent(
                    id=agent_id,
                    name=str(payload.get("name") or ""),
                    display_name=str(payload.get("display_name") or payload.get("description") or payload.get("name") or ""),
                    platform=str(payload.get("platform") or "openrouter"),
                    api_key_configured=bool(payload.get("api_key_encrypted") or payload.get("api_key")),
                    model=payload.get("model"),
                    user_wallet=payload.get("wallet") or payload.get("user_wallet"),
                    api_key=None,  # never expose
                )
            )
        return agents

    async def get_agent(self, agent_id: str, wallet_address: Optional[str]) -> Optional[Agent]:
        rec = self.qdrant.get_by_id(self.COLLECTION, agent_id)
        if not rec or not rec.payload:
            return None
        payload = rec.payload

        # Authorization (if wallet provided, enforce ownership)
        if wallet_address and payload.get("wallet") != wallet_address:
            return None

        encrypted = payload.get("api_key_encrypted") or payload.get("api_key")
        api_key = decrypt_secret(encrypted) if encrypted else None

        return Agent(
            id=str(payload.get("agent_id") or agent_id),
            name=str(payload.get("name") or ""),
            display_name=str(payload.get("display_name") or payload.get("description") or payload.get("name") or ""),
            platform=str(payload.get("platform") or "openrouter"),
            api_key_configured=bool(payload.get("api_key_encrypted")),
            model=payload.get("model"),
            user_wallet=payload.get("wallet") or payload.get("user_wallet"),
            api_key=api_key,
        )

    async def create_agent(self, agent_data: AgentCreate, wallet_address: str) -> Agent:
        now = _utc_now()
        agent_id = f"custom-{uuid.uuid4().hex[:8]}"

        encrypted = encrypt_secret(agent_data.api_key)

        payload = {
            **make_base_payload("agent"),
            "id": agent_id,
            # Required schema
            "agent_id": agent_id,
            "wallet": wallet_address,
            "name": agent_data.name,
            "description": agent_data.display_name,
            "model": agent_data.model,
            "api_key": encrypted,  # encrypted at rest
            "is_public": False,
            "created_at": _iso(now),
            # Stored fields
            "display_name": agent_data.display_name,
            "platform": agent_data.platform,
            "api_key_encrypted": encrypted,
            "user_wallet": wallet_address,
            "updated_at": _iso(now),
            "api_key_configured": True,
        }

        self.qdrant.upsert_record(self.COLLECTION, agent_id, payload)

        # Do not return api_key in response
        return Agent(
            id=agent_id,
            name=agent_data.name,
            display_name=agent_data.display_name,
            platform=agent_data.platform,
            api_key_configured=True,
            model=agent_data.model,
            user_wallet=wallet_address,
            api_key=None,
        )

    async def update_agent(self, agent_id: str, agent_update: AgentUpdate, wallet_address: str) -> Agent:
        rec = self.qdrant.get_by_id(self.COLLECTION, agent_id)
        if not rec or not rec.payload:
            raise Exception("Agent not found")
        payload = rec.payload
        if payload.get("wallet") != wallet_address:
            raise Exception("Agent not found or unauthorized")

        if agent_update.display_name is not None:
            payload["display_name"] = agent_update.display_name
            payload["description"] = agent_update.display_name
        if agent_update.model is not None:
            payload["model"] = agent_update.model

        payload["updated_at"] = _iso(_utc_now())
        self.qdrant.upsert_record(self.COLLECTION, agent_id, payload)

        return Agent(
            id=agent_id,
            name=str(payload.get("name") or ""),
            display_name=str(payload.get("display_name") or payload.get("description") or payload.get("name") or ""),
            platform=str(payload.get("platform") or "openrouter"),
            api_key_configured=bool(payload.get("api_key_encrypted")),
            model=payload.get("model"),
            user_wallet=payload.get("wallet"),
            api_key=None,
        )

    async def delete_agent(self, agent_id: str, wallet_address: str) -> bool:
        agent = await self.get_agent(agent_id, wallet_address)
        if not agent:
            raise Exception(f"Agent {agent_id} not found or unauthorized")

        # Delete all chats (and their messages/memories)
        chats = await self.get_agent_chats(agent_id, wallet_address)
        for chat in chats:
            await self.delete_chat(chat.id, wallet_address)

        # Delete the agent record
        self.qdrant.delete_by_id(self.COLLECTION, agent_id)
        return True

    # ------------------------------------------------------------------
    # Chats (delegated)
    # ------------------------------------------------------------------

    async def get_agent_chats(self, agent_id: str, wallet_address: Optional[str]) -> List[Chat]:
        return await self.chats.list_chats(agent_id, wallet_address)

    async def create_chat(self, agent_id: str, chat_data: ChatCreate, wallet_address: str) -> Chat:
        return await self.chats.create_chat(agent_id, chat_data, wallet_address)

    async def get_chat(self, chat_id: str, wallet_address: Optional[str]) -> Optional[Chat]:
        return await self.chats.get_chat(chat_id, wallet_address)

    async def update_chat(self, chat_id: str, chat_update: ChatUpdate, wallet_address: Optional[str]) -> Chat:
        return await self.chats.update_chat(chat_id, chat_update, wallet_address)

    async def delete_chat(self, chat_id: str, wallet_address: Optional[str]) -> None:
        await self.chats.delete_chat(chat_id, wallet_address)

    # ------------------------------------------------------------------
    # Messages (delegated + chat counters)
    # ------------------------------------------------------------------

    async def add_message(self, chat_id: str, message: MessageCreate, wallet_address: str) -> Message:
        chat = await self.get_chat(chat_id, wallet_address)
        if not chat or not chat.agent_id:
            raise Exception("Chat not found")
        msg = await self.messages.add_message(chat_id, chat.agent_id, wallet_address, message)

        # Update chat counters
        msgs = await self.messages.list_messages(chat_id, wallet=wallet_address, limit=5000)
        await self.chats.update_chat_counters(chat_id, wallet_address, len(msgs), message.content[:100])
        return msg

