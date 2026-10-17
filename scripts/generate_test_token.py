from datetime import datetime, timedelta, timezone

from jose import jwt

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )

    payload = {
        "sub": "test-user",
        "exp": expire,
    }

    token = jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    print(token)


if __name__ == "__main__":
    main()