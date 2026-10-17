from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.auth.security import create_access_token

@pytest.fixture(scope="module")
def client():
    """
    FastAPI 测试客户端。
    module scope 表示一个测试模块共用一个 client。
    """
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="module")
def auth_headers():
    """
    生成合法 JWT 请求头。

    用于所有需要鉴权的接口测试。
    """

    token = create_access_token(
        user_id="pytest-user"
    )

    return {
        "Authorization": (
            f"Bearer {token}"
        )
    }