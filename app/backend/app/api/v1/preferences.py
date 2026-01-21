"""
API endpoints for user preferences stored in Qdrant
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from pydantic import BaseModel
from app.core.auth_dependencies import get_wallet_address
from app.services.preferences_service import PreferencesService

router = APIRouter(prefix="/preferences", tags=["preferences"])


class UserPreferences(BaseModel):
    default_model: Optional[str] = None
    memory_behavior: Optional[str] = None
    active_tab: Optional[str] = None
    active_sub_tab: Optional[str] = None


@router.get("/", response_model=UserPreferences)
async def get_preferences(wallet_address: Optional[str] = Depends(get_wallet_address)):
    """
    Get user preferences from cache
    Requires wallet address for authentication
    """
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    preferences = await PreferencesService().get_preferences(wallet_address)
    return UserPreferences(**preferences)


@router.post("/", response_model=UserPreferences)
async def update_preferences(
    preferences: UserPreferences,
    wallet_address: Optional[str] = Depends(get_wallet_address)
):
    """
    Update user preferences in cache
    Requires wallet address for authentication
    """
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    updated = await PreferencesService().upsert_preferences(
        wallet_address, preferences.model_dump(exclude_none=True)
    )
    return UserPreferences(**updated)


@router.delete("/")
async def clear_preferences(wallet_address: Optional[str] = Depends(get_wallet_address)):
    """
    Clear user preferences
    Requires wallet address for authentication
    """
    if not wallet_address:
        raise HTTPException(status_code=401, detail="Wallet address required")
    
    await PreferencesService().clear_preferences(wallet_address)
    
    return {"message": "Preferences cleared"}

