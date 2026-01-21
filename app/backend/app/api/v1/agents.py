from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from app.models.schemas import (
    Chat, ChatCreate, ChatUpdate, Message, MessageCreate,
    Agent, AgentCreate, AgentUpdate, LLMResponse, CapsuleCreate, StakingCreate
)
from app.services.agent_service import AgentService
from app.services.llm_service import LLMService
from app.services.capsule_service import CapsuleService
from app.services.wallet_service import WalletService
from app.core.auth_dependencies import get_wallet_address
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=List[Agent])
async def list_agents(wallet_address: Optional[str] = Depends(get_wallet_address)):
    """List all agents for a user"""
    service = AgentService()
    return await service.get_user_agents(wallet_address)


@router.post("/", response_model=Agent)
async def create_agent(agent: AgentCreate, wallet_address: Optional[str] = Depends(get_wallet_address)):
    """Create a new agent/LLM configuration"""
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    service = AgentService()
    return await service.create_agent(agent, wallet_address)


@router.put("/{agent_id}", response_model=Agent)
async def update_agent(
    agent_id: str,
    agent_update: AgentUpdate,
    wallet_address: Optional[str] = Depends(get_wallet_address)
):
    """Update an agent's display name or model"""
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    service = AgentService()
    
    try:
        return await service.update_agent(agent_id, agent_update, wallet_address)
    except Exception as e:
        # logger.error(f"Error updating agent {agent_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str, 
    wallet_address: Optional[str] = Depends(get_wallet_address)
):
    """Delete an agent/LLM configuration and all associated chats"""
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    service = AgentService()
    
    try:
        await service.delete_agent(agent_id, wallet_address)
        return {"success": True, "message": "Agent deleted successfully"}
    except Exception as e:
        # logger.error(f"Error deleting agent {agent_id}: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{agent_id}/chats", response_model=List[Chat])
async def list_chats(agent_id: str, wallet_address: Optional[str] = Depends(get_wallet_address)):
    """List all chats for an agent"""
    service = AgentService()
    return await service.get_agent_chats(agent_id, wallet_address)


@router.post("/{agent_id}/chats", response_model=Chat)
async def create_chat(agent_id: str, chat: ChatCreate, wallet_address: Optional[str] = Depends(get_wallet_address)):
    """Create a new chat"""
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    service = AgentService()
    
    # Verify agent exists
    agent = await service.get_agent(agent_id, wallet_address)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return await service.create_chat(agent_id, chat, wallet_address)


@router.get("/{agent_id}/chats/{chat_id}", response_model=Chat)
async def get_chat(agent_id: str, chat_id: str, wallet_address: Optional[str] = Depends(get_wallet_address)):
    """Get a specific chat with messages"""
    service = AgentService()
    chat = await service.get_chat(chat_id, wallet_address)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.put("/{agent_id}/chats/{chat_id}", response_model=Chat)
async def update_chat(
    agent_id: str,
    chat_id: str,
    chat_update: ChatUpdate,
    wallet_address: Optional[str] = Depends(get_wallet_address)
):
    """Update chat metadata"""
    service = AgentService()
    return await service.update_chat(chat_id, chat_update, wallet_address)


@router.post("/{agent_id}/chats/{chat_id}/messages", response_model=LLMResponse)
async def send_message(
    agent_id: str,
    chat_id: str,
    message: MessageCreate,
    wallet_address: Optional[str] = Depends(get_wallet_address)
):
    """Send a message to an agent and get response"""
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    service = AgentService()
    llm_service = LLMService()
    
    # Get chat history
    # logger.debug(f"Looking up chat {chat_id} for wallet {wallet_address}")
    chat = await service.get_chat(chat_id, wallet_address)
    if not chat:
        # logger.warning(f"Chat {chat_id} not found for wallet {wallet_address}")
        raise HTTPException(status_code=404, detail=f"Chat not found (chat_id: {chat_id}, wallet: {wallet_address})")
    
    # Use the chat's agent_id if available, otherwise use the URL agent_id
    # This ensures we use the correct agent that the chat was created with
    actual_agent_id = chat.agent_id if chat.agent_id else agent_id
    # if actual_agent_id != agent_id:
    #     logger.info(f"Using chat's agent_id ({actual_agent_id}) instead of URL agent_id ({agent_id})")
    
    # Get agent config (with API key for internal use)
    agent = await service.get_agent(actual_agent_id, wallet_address)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found (agent_id: {actual_agent_id})")
    
    # Save user message first
    user_msg = await service.add_message(chat_id, message, wallet_address)
    
    # Get LLM response with memory integration
    messages_history = [{"role": m.role.value, "content": m.content} for m in chat.messages]
    messages_history.append({"role": message.role.value, "content": message.content})
    
    try:
        # Get memory_size from chat
        memory_size = chat.memory_size.value if hasattr(chat.memory_size, 'value') else str(chat.memory_size)
        
        # Get capsule_id from chat for memory scope isolation
        capsule_id = chat.capsule_id if hasattr(chat, 'capsule_id') else None
        
        # Get web_search_enabled from chat
        web_search_enabled = getattr(chat, 'web_search_enabled', False)
        
        response = await llm_service.get_completion(
            agent_id=actual_agent_id,  # Use the actual agent_id from chat
            messages=messages_history,
            agent_config=agent,
            chat_id=chat_id,  # Pass chat_id for memory retrieval
            memory_size=memory_size,  # Pass memory_size setting
            capsule_id=capsule_id,  # Pass capsule_id for memory scope isolation
            web_search_enabled=web_search_enabled  # Pass web_search_enabled flag
        )
        
        # Save assistant message
        assistant_msg = MessageCreate(role="assistant", content=response.content)
        await service.add_message(chat_id, assistant_msg, wallet_address)
        
        return response
    except Exception as e:
        # Log error but don't remove user message (user can see it failed)
        # logger.error(f"Error getting LLM response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get AI response: {str(e)}")


@router.post("/{agent_id}/chats/{chat_id}/messages/stream")
async def send_message_stream(
    agent_id: str,
    chat_id: str,
    message: MessageCreate,
    wallet_address: Optional[str] = Depends(get_wallet_address)
):
    """Send a message to an agent and get streaming response (Server-Sent Events)"""
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    service = AgentService()
    llm_service = LLMService()
    
    # Get chat history
    # logger.debug(f"Looking up chat {chat_id} for wallet {wallet_address}")
    chat = await service.get_chat(chat_id, wallet_address)
    if not chat:
        # logger.warning(f"Chat {chat_id} not found for wallet {wallet_address}")
        raise HTTPException(status_code=404, detail=f"Chat not found (chat_id: {chat_id}, wallet: {wallet_address})")
    
    # Use the chat's agent_id if available
    actual_agent_id = chat.agent_id if chat.agent_id else agent_id
    # if actual_agent_id != agent_id:
    #     logger.info(f"Using chat's agent_id ({actual_agent_id}) instead of URL agent_id ({agent_id})")
    
    # Get agent config (with API key for internal use)
    agent = await service.get_agent(actual_agent_id, wallet_address)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found (agent_id: {actual_agent_id})")
    
    # Save user message first
    user_msg = await service.add_message(chat_id, message, wallet_address)
    
    # Get LLM response with memory integration
    messages_history = [{"role": m.role.value, "content": m.content} for m in chat.messages]
    messages_history.append({"role": message.role.value, "content": message.content})
    
    # Get memory_size and capsule_id from chat
    memory_size = chat.memory_size.value if hasattr(chat.memory_size, 'value') else str(chat.memory_size)
    capsule_id = chat.capsule_id if hasattr(chat, 'capsule_id') else None
    web_search_enabled = getattr(chat, 'web_search_enabled', False)
    
    async def generate_stream():
        full_content = ""
        try:
            async for chunk in llm_service.get_completion_stream(
                agent_id=actual_agent_id,
                messages=messages_history,
                agent_config=agent,
                chat_id=chat_id,
                memory_size=memory_size,
                capsule_id=capsule_id,
                web_search_enabled=web_search_enabled
            ):
                full_content += chunk
                # Send chunk as SSE
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            # Save assistant message after streaming completes
            if full_content:
                assistant_msg = MessageCreate(role="assistant", content=full_content)
                await service.add_message(chat_id, assistant_msg, wallet_address)
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            # logger.error(f"Error in streaming: {e}", exc_info=True)
            error_data = json.dumps({'error': str(e)})
            yield f"data: {error_data}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable buffering for nginx
        }
    )


@router.get("/{agent_id}/chats/{chat_id}/messages", response_model=List[Message])
async def get_messages(
    agent_id: str,
    chat_id: str,
    wallet_address: Optional[str] = Depends(get_wallet_address)
):
    """Get all messages for a chat"""
    service = AgentService()
    chat = await service.get_chat(chat_id, wallet_address)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat.messages


@router.get("/{agent_id}/chats/{chat_id}/memories")
async def get_chat_memories(
    agent_id: str,
    chat_id: str,
    wallet_address: Optional[str] = Depends(get_wallet_address)
):
    """Get all stored memories for a chat (for verification/tracking)"""
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    service = AgentService()
    llm_service = LLMService()
    
    # Verify chat exists and belongs to user
    chat = await service.get_chat(chat_id, wallet_address)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Verify agent
    actual_agent_id = chat.agent_id if chat.agent_id else agent_id
    agent = await service.get_agent(actual_agent_id, wallet_address)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Get all memories for this chat
    from app.services.memory_service import MemoryService
    memory_service = MemoryService()
    # Get capsule_id from chat for memory filtering
    capsule_id = chat.capsule_id if hasattr(chat, 'capsule_id') else None
    memories = memory_service.get_all_chat_memories(actual_agent_id, chat_id, capsule_id)
    
    return {
        "chat_id": chat_id,
        "agent_id": actual_agent_id,
        "memory_count": len(memories),
        "memories": memories,
        "using_platform": memory_service.use_platform
    }


@router.delete("/{agent_id}/chats/{chat_id}")
async def delete_chat(agent_id: str, chat_id: str, wallet_address: Optional[str] = Depends(get_wallet_address)):
    """Delete a chat"""
    service = AgentService()
    await service.delete_chat(chat_id, wallet_address)
    return {"success": True, "message": "Chat deleted"}


@router.post("/{agent_id}/stake")
async def stake_on_agent(
    agent_id: str,
    stake_data: Dict[str, Any],
    wallet_address: Optional[str] = Depends(get_wallet_address)
):
    """Stake on an agent - creates a capsule if needed and stakes on it"""
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    agent_service = AgentService()
    capsule_service = CapsuleService()
    wallet_service = WalletService()
    
    # Verify agent exists and belongs to user
    agent = await agent_service.get_agent(agent_id, wallet_address)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found or unauthorized")
    
    # Check if capsule already exists for this agent
    # We'll use agent_id as part of capsule metadata to track the relationship
    existing_capsule = await capsule_service.find_capsule_for_agent(wallet_address, agent_id)
    
    # Create capsule if it doesn't exist
    if not existing_capsule:
        # logger.info(f"Creating new capsule for agent {agent_id}")
        pass
        capsule_data = CapsuleCreate(
            name=agent.display_name or agent.name,
            description=stake_data.get("description", f"Memory capsule for {agent.display_name or agent.name}"),
            category=stake_data.get("category", "General"),
            price_per_query=stake_data.get("price_per_query", 0.05),
            metadata={
                "agent_id": agent_id,
                "agent_name": agent.name,
                "agent_display_name": agent.display_name,
                "platform": agent.platform,
                "model": agent.model
            }
        )
        capsule = await capsule_service.create_capsule(capsule_data, wallet_address)
        capsule_id = capsule.id
        # logger.info(f"Created capsule {capsule_id} for agent {agent_id}")
    else:
        capsule_id = existing_capsule.id
        # logger.info(f"Using existing capsule {capsule_id} for agent {agent_id}")
    
    # Create staking entry
    staking_create = StakingCreate(
        capsule_id=capsule_id,
        stake_amount=stake_data.get("stake_amount", 0)
    )
    
    # Create staking entry (this also updates the capsule stake amount)
    staking_info = await wallet_service.create_staking(staking_create, wallet_address)
    
    return {
        "success": True,
        "capsule_id": capsule_id,
        "staking_info": staking_info,
        "message": "Successfully staked on agent"
    }

