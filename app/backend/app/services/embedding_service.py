from __future__ import annotations

from typing import List

import httpx

from app.core.config import settings


class EmbeddingService:
    """
    Text embedding service (used for Qdrant vectors).
    Uses OpenAI embeddings via HTTP to avoid extra SDK deps.
    """

    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is required for embeddings")

    async def embed_text(self, text: str, expected_dim: int = 1536) -> List[float]:
        text = (text or "").strip()
        if not text:
            # Represent empty content deterministically
            return [0.0] * expected_dim

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.OPENAI_EMBEDDING_MODEL,
                    "input": text,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            vec = data["data"][0]["embedding"]

        if len(vec) != expected_dim:
            raise RuntimeError(
                f"Embedding dim mismatch: got {len(vec)} expected {expected_dim}"
            )
        return vec

