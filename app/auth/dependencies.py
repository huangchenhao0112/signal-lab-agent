from __future__ import annotations

from fastapi import (
    Depends,
    HTTPException,
)

from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from jose import (
    JWTError,
    jwt,
)

from app.auth.security import (
    ALGORITHM,
    SECRET_KEY,
)


security = HTTPBearer()


def get_current_user(
    credentials: (
        HTTPAuthorizationCredentials
    ) = Depends(
        security
    ),
) -> str:

    token = (
        credentials.credentials
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[
                ALGORITHM
            ],
        )

        user_id = payload.get(
            "sub"
        )

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="无效Token",
            )

        return str(
            user_id
        )

    except JWTError as error:
        raise HTTPException(
            status_code=401,
            detail="Token验证失败",
        ) from error