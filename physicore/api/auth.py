"""
PhysiCore API Authentication
============================
API key authentication for the PhysiCore REST API.

Keys are stored as SHA-256 hashes — the raw key is never saved server-side.

Usage in server.py:
    from physicore.api.auth import require_api_key
    @app.get("/api/engine/step")
    async def step(api_key: str = Depends(require_api_key)):
        ...

Environment variables:
    PHYSICORE_API_KEYS   — comma-separated list of valid API keys
                           e.g. "pk_live_abc123,pk_live_xyz789"
    PHYSICORE_SKIP_AUTH  — set to "1" to disable auth (local dev only)

Generating a key (Python):
    import secrets; print("pk_live_" + secrets.token_urlsafe(32))
"""

from __future__ import annotations

import os
import hashlib
import secrets
from typing import Optional, Set
from fastapi import Header, HTTPException, status


# ── Key store ─────────────────────────────────────────────────────────────────
# In production: load from environment variable PHYSICORE_API_KEYS
# In local dev:  set PHYSICORE_SKIP_AUTH=1 to bypass

def _load_valid_keys() -> Set[str]:
    """Load SHA-256 hashes of valid API keys from environment."""
    raw = os.environ.get("PHYSICORE_API_KEYS", "")
    if not raw:
        return set()
    keys = set()
    for k in raw.split(","):
        k = k.strip()
        if k:
            keys.add(hashlib.sha256(k.encode()).hexdigest())
    return keys


def _is_local_dev() -> bool:
    return os.environ.get("PHYSICORE_SKIP_AUTH", "0") == "1"


# A-02: Cache hashes at module load time — not on every API request (was 60 reads/sec at 60Hz)
_VALID_KEY_HASHES: Set[str] = _load_valid_keys()


# ── FastAPI dependency ─────────────────────────────────────────────────────────

async def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None),
) -> str:
    """
    FastAPI dependency. Validates API key from:
      1. X-API-Key header   (preferred)
      2. Authorization: Bearer <key>

    Returns the raw API key on success.
    Raises HTTP 401 on failure.

    In local dev (PHYSICORE_SKIP_AUTH=1): always passes.
    """
    if _is_local_dev():
        return "dev"

    # Extract key
    raw_key: Optional[str] = None
    if x_api_key:
        raw_key = x_api_key.strip()
    elif authorization and authorization.lower().startswith("bearer "):
        raw_key = authorization[7:].strip()

    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Set X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate
    # A-02: use module-level cache — loaded once at startup, not per request
    valid_hashes = _VALID_KEY_HASHES
    if not valid_hashes:
        # A-01: fail CLOSED — no silent pass when keys not configured
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Server not configured: PHYSICORE_API_KEYS not set. "
                "Set PHYSICORE_SKIP_AUTH=1 for local dev, "
                "or set PHYSICORE_API_KEYS=your_key for production."
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    if key_hash not in valid_hashes:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return raw_key


# ── Key generation utility ────────────────────────────────────────────────────

def generate_api_key(prefix: str = "pk_live") -> str:
    """
    Generate a cryptographically secure API key.
    Call this once per team and store the result — it won't be shown again.

    Example:
        from physicore.api.auth import generate_api_key
        key = generate_api_key()
        print(key)   # pk_live_xxxxxxxxxxxx
        # Add to PHYSICORE_API_KEYS on your server
    """
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def hash_key(raw_key: str) -> str:
    """Return the SHA-256 hash of a key — store this, not the raw key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


if __name__ == "__main__":
    key = generate_api_key()
    print(f"\nNew PhysiCore API key:\n  {key}\n")
    print(f"Add to PHYSICORE_API_KEYS on your server (comma-separated for multiple).")
    print(f"SHA-256 hash (stored server-side): {hash_key(key)}\n")
