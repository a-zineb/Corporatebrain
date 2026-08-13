from __future__ import annotations

from functools import lru_cache
import os

import jwt
from fastapi import Header


@lru_cache(maxsize=1)
def _jwks_client():
    url = os.getenv("CLERK_JWKS_URL", "").strip()
    return jwt.PyJWKClient(url) if url else None


def current_user_id(authorization: str | None = Header(default=None)) -> str:
    """Resolve Clerk identity when possible without coupling auth to RAG.

    Corporate Brain supports signed-out local use. Missing, invalid, expired or
    temporarily unverifiable tokens therefore fall back to the anonymous scope;
    they never prevent Direct Answer, Catalog, AI Answer, search or sources.
    """
    client = _jwks_client()
    if client is None or not authorization:
        return "anonymous"
    if not authorization or not authorization.startswith("Bearer "):
        return "anonymous"
    token = authorization.removeprefix("Bearer ").strip()
    if not token or token in {"null", "undefined"}:
        return "anonymous"
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
    except (jwt.PyJWTError, OSError):
        return "anonymous"
