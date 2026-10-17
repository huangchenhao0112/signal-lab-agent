from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)

from jose import jwt
from passlib.context import CryptContext

from app.core.config import get_settings

settings = get_settings()

SECRET_KEY = settings.jwt_secret_key
ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = (
    settings.access_token_expire_minutes
)


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


def hash_password(
    password: str,
) -> str:
    return pwd_context.hash(
        password
    )


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    return pwd_context.verify(
        plain_password,
        hashed_password,
    )


def create_access_token(
    user_id: str,
) -> str:

    expire = (
        datetime.now(
            timezone.utc
        )
        + timedelta(
            minutes=(
                ACCESS_TOKEN_EXPIRE_MINUTES
            )
        )
    )

    payload = {
        "sub": user_id,
        "exp": expire,
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM,
    )