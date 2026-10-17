from __future__ import annotations

from contextlib import (
    asynccontextmanager,
)

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
)

from sqlalchemy.orm import (
    Session,
)

from app.agent_runtime import (
    AgentRuntime,
)

from app.database.mysql import (
    get_db,
)

from app.database.repository import (
    delete_agent_session,
    get_agent_session,
    update_agent_session_title,
)

from app.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentHealthResponse,
    AgentSessionResponse,
    AgentSessionUpdateRequest,
    AgentToolsResponse,
    SessionDeleteResponse,
)

from app.auth.dependencies import (
    get_current_user,
)

# ==========================================
# FastAPI生命周期
# ==========================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    runtime = (
        AgentRuntime()
    )

    try:
        await (
            runtime.startup()
        )

        app.state.agent_runtime = (
            runtime
        )

        print(
            "SignalLab Unified Agent "
            "启动成功。"
        )

        yield

    finally:
        await (
            runtime.shutdown()
        )

        print(
            "SignalLab Unified Agent "
            "已关闭。"
        )


app = FastAPI(
    title=(
        "SignalLab Research Agent"
    ),
    description=(
        "SignalLab科研智能体服务"
    ),
    version="1.1.0",
    lifespan=lifespan,
    root_path="/api",
)


def get_agent_runtime(
    request: Request,
) -> AgentRuntime:

    runtime = getattr(
        request.app.state,
        "agent_runtime",
        None,
    )

    if runtime is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "AgentRuntime尚未初始化"
            ),
        )

    return runtime


# ==========================================
# Root
# ==========================================


@app.get("/")
async def root():
    return {
        "name": (
            "SignalLab Research Agent"
        ),
        "version": "1.1.0",
        "docs": "/docs",
        "agent_chat": (
            "/agent/chat"
        ),
    }


# ==========================================
# Health
# ==========================================


@app.get(
    "/agent/health",
    response_model=(
        AgentHealthResponse
    ),
    tags=[
        "Unified Agent"
    ],
)
async def agent_health(
    request: Request,
):
    runtime = (
        get_agent_runtime(
            request
        )
    )

    return (
        await runtime.health()
    )


# ==========================================
# MCP Tools
# ==========================================


@app.get(
    "/agent/tools",
    response_model=(
        AgentToolsResponse
    ),
    tags=[
        "Unified Agent"
    ],
)
async def agent_tools(
    request: Request,
):
    runtime = (
        get_agent_runtime(
            request
        )
    )

    tools = (
        runtime.available_tools
    )

    return {
        "ok": True,
        "count": len(
            tools
        ),
        "tools": tools,
    }


# ==========================================
# Agent Chat
# ==========================================


@app.post(
    "/agent/chat",
    response_model=(
        AgentChatResponse
    ),
    tags=[
        "Unified Agent"
    ],
)
async def agent_chat(
    payload: AgentChatRequest,
    request: Request,

    # FastAPI自动注入MySQL Session
    db: Session = Depends(
        get_db
    ),
    current_user: str = Depends(
        get_current_user
    ),
):
    runtime = (
        get_agent_runtime(
            request
        )
    )

    try:
        return (
            await runtime.chat(
                message=(
                    payload.message
                ),
                session_id=(
                    payload.session_id
                ),
                db=db,
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(
                error
            ),
        ) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        ) from error


# ==========================================
# Session Read
# ==========================================


@app.get(
    "/agent/sessions/{session_id}",
    response_model=(
        AgentSessionResponse
    ),
    tags=[
        "Agent Sessions"
    ],
)
def read_agent_session(
    session_id: str,

    db: Session = Depends(
        get_db
    ),
):
    record = (
        get_agent_session(
            db=(
                db
            ),
            session_id=(
                session_id
            ),
        )
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session不存在"
            ),
        )

    return record


# ==========================================
# Session Update
# ==========================================


@app.patch(
    "/agent/sessions/{session_id}",
    response_model=(
        AgentSessionResponse
    ),
    tags=[
        "Agent Sessions"
    ],
)
def update_session(
    session_id: str,
    payload: (
        AgentSessionUpdateRequest
    ),

    db: Session = Depends(
        get_db
    ),
):
    record = (
        update_agent_session_title(
            db=(
                db
            ),
            session_id=(
                session_id
            ),
            title=(
                payload.title
            ),
        )
    )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session不存在"
            ),
        )

    return record


# ==========================================
# Session Delete
# ==========================================


@app.delete(
    "/agent/sessions/{session_id}",
    response_model=(
        SessionDeleteResponse
    ),
    tags=[
        "Agent Sessions"
    ],
)
async def delete_session(
    session_id: str,
    request: Request,

    db: Session = Depends(
        get_db
    ),
):
    runtime = (
        get_agent_runtime(
            request
        )
    )

    # 1. Redis
    await (
        runtime.clear_session(
            session_id
        )
    )

    # 2. MySQL
    deleted = (
        delete_agent_session(
            db=(
                db
            ),
            session_id=(
                session_id
            ),
        )
    )

    return {
        "ok": True,
        "session_id": (
            session_id
        ),
        "message": (
            "Session已清除"
            if deleted
            else
            "Redis Memory已清除，"
            "MySQL中未找到Session"
        ),
    }