from fastapi import FastAPI, status as http_status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
import os

from app.api.v1 import agents, marketplace, capsules, wallet, auth, preferences
from app.core.config import settings
from app.services.qdrant_service import init_qdrant_service, get_qdrant_service

# Configure logging
log_level = logging.INFO
if os.getenv("DEBUG", "False").lower() == "true":
    log_level = logging.DEBUG
elif os.getenv("ENVIRONMENT") == "production":
    log_level = logging.WARNING  # Less verbose in production

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Mantlememo API...")
    # Initialize Qdrant (single persistence layer) and hard-fail if unreachable
    init_qdrant_service()
    
    # Initialize memory service (warm up)
    try:
        from app.services.memory_service import MemoryService
        memory_service = MemoryService()
        if memory_service._is_available():
            logger.info("Memory service initialized successfully")
        else:
            logger.warning("Memory service not available (mem0 may not be configured)")
    except Exception as e:
        logger.warning(f"Memory service initialization failed: {e}")
    
    yield
    # Shutdown
    logger.info("Shutting down Mantlememo API...")


app = FastAPI(
    title="Mantlememo API",
    description="Backend API for Mantlememo is running!",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - Use configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # Use configured origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Wallet-Address"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
app.include_router(capsules.router, prefix="/api/v1/capsules", tags=["Capsules"])
app.include_router(marketplace.router, prefix="/api/v1/marketplace", tags=["Marketplace"])
app.include_router(wallet.router, prefix="/api/v1/wallet", tags=["Wallet"])
app.include_router(preferences.router, prefix="/api/v1", tags=["Preferences"])


@app.get("/")
async def root():
    return {
        "message": "Mantlememo API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with service validation"""
    status = {
        "status": "healthy",
        "services": {}
    }
    
    # Check Qdrant (required)
    try:
        qdrant = get_qdrant_service()
        qdrant.ping()
        status["services"]["qdrant"] = "available"
    except Exception:
        status["services"]["qdrant"] = "unavailable"
    
    # Check memory service (optional)
    try:
        from app.services.memory_service import MemoryService
        memory_service = MemoryService()
        status["services"]["memory"] = "available" if memory_service._is_available() else "unavailable"
    except:
        status["services"]["memory"] = "unavailable"
    
    # Return 503 if critical services are down in production
    if not settings.DEBUG and status["services"].get("qdrant") != "available":
        return JSONResponse(
            content=status,
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    return status


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )

