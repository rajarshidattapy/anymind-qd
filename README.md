# Anymind: Multi‑Agent Systems Memory Runtime (Qdrant‑First)

Instead of fine‑tuning models or rebuilding prompts, Anymind turns interactions into **retrievable memory** that agents can use across sessions, tools, and apps.

Anymind is a **Multi‑Agentic framework (MAS)** runtime that makes agents stateful by default: chats, messages, preferences, capsules, staking, earnings, and semantic recall all persist through a **single storage layer: Qdrant**.



## What it does (MAS Track)
- **Agents**: configurable LLM backends (provider/model/key) with a stable identity
- **Chats**: multiple isolated sessions per agent (scoped memory + settings)
- **Messages**: persistent conversation history + vectorized content for recall
- **Preferences**: per‑wallet/user settings that survive restarts
- **Capsules / Marketplace**: publishable “memory capsules” with discovery/search
- **Runtime**: FastAPI endpoints used by the React client (no route/contract changes)

## Why Qdrant
Qdrant is used as the **only persistence layer**:
- **Structured store via payloads** (agents, chats, preferences, staking, earnings)
- **Vector store** for semantic retrieval (message content; optional capsule search)
- **Named collections** per entity type (not one giant collection)

If Qdrant is unavailable, the backend fails loudly on startup.

## Core Concepts
- **Agent**: a runnable configuration (provider/model) + identity
- **Chat**: a scoped session for an agent (history + settings)
- **Message**: a durable event in a chat (vectorized for recall)
- **Memory**: retrieved context injected into LLM calls (via `mem0` on top of Qdrant)

## SDK Usage (Read‑Only)
```python
from mantlememo import Agent

agent = Agent(
    agent_id="agent_123",
    chat_id="governance_chat",
)

response = agent.chat("Analyze proposal X")
````

## Architecture

* **Frontend**: React (wallet header for user identity)
* **Runtime**: FastAPI
* **Persistence**: Qdrant (payloads + vectors; single storage layer)
* **Semantic Memory**: mem0 (Qdrant‑backed)
* **Optional module**: Ethereum smart contract integration (staking/payments)

## Getting Started

### Prerequisites
- A running Qdrant instance (local or hosted)
- `OPENAI_API_KEY` for embeddings (messages semantic search) and mem0 OSS

### Run Qdrant locally
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Running the app
```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

## Ethereum module (staking/payments)
This repo still includes a web3 UI flow and contract integration, but it is **not the core of the MAS runtime**.

- **Network**: Ethereum Sepolia (Chain ID: 11155111)
- **Contract**: `0x4287F8a28AC24FF1b0723D26747CBE7F39C9C167`
- **Explorer**: [View on Explorer](https://sepolia.etherscan.io/address/0x4287F8a28AC24FF1b0723D26747CBE7F39C9C167)

## What MantleMemo is NOT

* ❌ No model fine-tuning
* ❌ No raw data on-chain
* ❌ No prompt marketplace
* ❌ No AI inference on-chain

## Status

✅ **MAS runtime**: multi-agent chats + memory + persistence through Qdrant  
✅ **Semantic recall**: message embeddings + retrieval (Qdrant vectors)  
✅ **Marketplace data**: capsules + staking/earnings stored in Qdrant payloads  
✅ **SDK Access**: persistent agent intelligence via API and SDK

