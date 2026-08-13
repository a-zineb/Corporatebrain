from __future__ import annotations

from functools import lru_cache
import os

import jwt
from fastapi import Header, HTTPException


@lru_cache(maxsize=1)
def _jwks_client():
    url = os.getenv("CLERK_JWKS_URL", "").strip()
    return jwt.PyJWKClient(url) if url else None


def current_user_id(authorization: str | None = Header(default=None)) -> str:
    """Verify Clerk session JWTs when Clerk is configured.

    Local mode remains available for the existing offline MVP when no JWKS URL
    is configured. The frontend clearly labels that state and does not invent a
    profile identity.
    """
    client = _jwks_client()
    if client is None:
        return "local-development"
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        key = client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["RS256"],
            options={"require": ["exp", "iat", "sub"]},
        )
        allowed = {item.strip() for item in os.getenv("CLERK_AUTHORIZED_PARTIES", "").split(",") if item.strip()}
        if allowed and claims.get("azp") not in allowed:
            raise jwt.InvalidTokenError("invalid authorized party")
        return str(claims["sub"])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired session") from exc
