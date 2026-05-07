"""Password hashing + JWT issuance / decoding.

Two responsibilities, one module:

* ``hash_password`` / ``verify_password`` — bcrypt via passlib.
* ``create_access_token`` / ``decode_token`` — HS256 via python-jose.

Token payload:

    {
        "sub": "<username>",
        "uid": <int>,
        "role": "admin" | "analyst",
        "exp": <unix_ts>,
        "iat": <unix_ts>,
    }

We carry ``role`` inside the token so route guards don't need a DB
round-trip per request — the trade-off is that role changes only take
effect after the token expires (max ``JWT_EXP_MINUTES``). Acceptable
for a thesis demo; would add a token-revocation table for prod.
"""
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt

from config.settings import JWT_ALGORITHM, JWT_EXP_MINUTES, JWT_SECRET


# Direct-bcrypt rather than passlib: passlib is unmaintained and breaks
# against bcrypt >=4 (its compatibility shims poke at private attributes
# that no longer exist). Cost factor 12 ≈ 250ms / hash on modern CPUs —
# fine interactively, infeasible to brute-force.
_BCRYPT_COST = 12


def _normalize(plain: str) -> bytes:
    # bcrypt's input is hard-capped at 72 bytes by the algorithm; longer
    # passwords are silently truncated. Pre-hashing with sha256 keeps
    # arbitrary-length passwords distinguishable while staying inside the
    # cap.
    return hashlib.sha256(plain.encode("utf-8")).digest()


def hash_password(plain: str) -> str:
    salt = bcrypt.gensalt(rounds=_BCRYPT_COST)
    return bcrypt.hashpw(_normalize(plain), salt).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_normalize(plain), hashed.encode("ascii"))
    except ValueError:
        # Malformed stored hash — treat as a non-match rather than 500.
        return False


def create_access_token(*, uid: int, username: str, role: str) -> str:
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub": username,
        "uid": uid,
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXP_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    """Return the decoded payload or raise ``ValueError`` on any failure.

    ``ValueError`` (not ``JWTError``) so callers in the API layer don't
    need to import jose just to handle "bad token".
    """
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError as exc:
        raise ValueError(f"invalid token: {exc}") from exc
