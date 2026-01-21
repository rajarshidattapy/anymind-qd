from pydantic_settings import BaseSettings
from pydantic import field_validator, Field, ConfigDict
from typing import List, Union, Optional
import os
import json
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        env_parse_none_str=None,
        extra="ignore"  # Ignore extra fields like VITE_* variables
    )
    # App settings
    APP_NAME: str = "Mantlememo API"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    @field_validator("PORT")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if v < 1 or v > 65535:
            raise ValueError("PORT must be between 1 and 65535")
        return v
    
    # CORS - handle both JSON and comma-separated strings
    # Make it Optional to prevent pydantic-settings from trying to parse empty strings as JSON
    CORS_ORIGINS: Optional[List[str]] = Field(
        default=None,
        description="CORS allowed origins"
    )
    
    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str], None]) -> List[str]:
        # Handle None or empty values
        if v is None:
            return [
                "http://localhost:8080",
                "http://localhost:5173",
                "http://127.0.0.1:8080",
                "http://127.0.0.1:5173",
            ]
        
        if isinstance(v, str):
            # Handle empty string
            if not v.strip():
                return [
                    "http://localhost:8080",
                    "http://localhost:5173",
                    "http://127.0.0.1:8080",
                    "http://127.0.0.1:5173",
                ]
            
            # Try JSON first
            if v.strip().startswith("["):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            
            # Fall back to comma-separated
            if v.strip():
                origins = [origin.strip() for origin in v.split(",") if origin.strip()]
                if origins:
                    return origins
            
            # Return default if empty after processing
            return [
                "http://localhost:8080",
                "http://localhost:5173",
                "http://127.0.0.1:8080",
                "http://127.0.0.1:5173",
            ]
        
        # If it's already a list, return it
        if isinstance(v, list):
            return v
        
        # Fallback to default
        return [
            "http://localhost:8080",
            "http://localhost:5173",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:5173",
        ]
    
    # Qdrant (single persistence layer)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # Embeddings (used for Qdrant vectors; keep aligned with mem0 if used)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    QDRANT_MESSAGE_VECTOR_SIZE: int = int(os.getenv("QDRANT_MESSAGE_VECTOR_SIZE", "1536"))
    QDRANT_CAPSULE_VECTOR_SIZE: int = int(os.getenv("QDRANT_CAPSULE_VECTOR_SIZE", "1536"))

    # Secret encryption (agent api_key at rest)
    API_KEY_ENCRYPTION_SECRET: str = os.getenv("API_KEY_ENCRYPTION_SECRET", "")
    
    # LLM API Keys
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    
    # Mem0 (open-source). We do NOT use the hosted platform (Qdrant is the only persistence layer).
    MEM0_ENABLED: bool = os.getenv("MEM0_ENABLED", "True").lower() == "true"
    
    # Solana
    SOLANA_RPC_URL: str = os.getenv("SOLANA_RPC_URL", "https://api.devnet.solana.com")
    SOLANA_NETWORK: str = os.getenv("SOLANA_NETWORK", "devnet")
    
    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    def model_post_init(self, __context):
        # Ensure CORS_ORIGINS always has a value after initialization
        if self.CORS_ORIGINS is None or (isinstance(self.CORS_ORIGINS, list) and len(self.CORS_ORIGINS) == 0):
            self.CORS_ORIGINS = [
                "http://localhost:8080",
                "http://localhost:5173",
                "http://127.0.0.1:8080",
                "http://127.0.0.1:5173",
            ]


settings = Settings()

