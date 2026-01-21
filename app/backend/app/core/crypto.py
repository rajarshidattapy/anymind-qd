from __future__ import annotations

import base64
import hashlib
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import settings


def _derive_fernet_key(secret: str) -> bytes:
    """
    Fernet requires a 32-byte urlsafe base64-encoded key.
    We derive it deterministically from a secret string.
    """
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


def _fernet() -> Fernet:
    secret = settings.API_KEY_ENCRYPTION_SECRET or settings.SECRET_KEY
    if not secret:
        raise RuntimeError("API_KEY_ENCRYPTION_SECRET or SECRET_KEY must be set for encryption")
    return Fernet(_derive_fernet_key(secret))


def encrypt_secret(plaintext: Optional[str]) -> Optional[str]:
    if plaintext is None:
        return None
    if plaintext == "":
        return ""
    token = _fernet().encrypt(plaintext.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_secret(ciphertext: Optional[str]) -> Optional[str]:
    if ciphertext is None:
        return None
    if ciphertext == "":
        return ""
    try:
        plain = _fernet().decrypt(ciphertext.encode("utf-8"))
        return plain.decode("utf-8")
    except InvalidToken as e:
        # Fail loudly; corrupted/invalid secret should not be silently ignored.
        raise RuntimeError("Failed to decrypt secret (invalid token or wrong key)") from e

