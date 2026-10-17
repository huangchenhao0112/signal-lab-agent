from __future__ import annotations

import json
import logging
import sys
import time
import uuid
import os
from contextlib import (
    AsyncExitStack,
)

from pathlib import Path

from typing import Any

from sqlalchemy.orm import Session

from mcp import (
    Client,
    StdioServerParameters,
)

from app.core.config import (
    get_settings,
)

from app.database.mysql import (
    check_mysql_connection,
)

from app.database.repository import (
    get_or_create_agent_session,
    save_agent_message,
    touch_agent_session,
)

from app.core.llm_client import (
    LLMClient,
)

from agent.memory.redis_memory import (
    RedisConversationMemory,
)

from agent.memory.memory_service import (
    ConversationMemoryService,
)

from agent.workflow.graph import (
    build_agent_graph,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

MCP_SERVER_FILE = (
    PROJECT_ROOT
    / "mcp_server"
    / "server.py"
)


# ==========================================
# Logging
# ==========================================


LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_FILE = (
    LOG_DIR
    / "agent.log"
)


logger = logging.getLogger(
    "signal_lab_agent"
)

logger.setLevel(
    logging.INFO
)


if not logger.handlers:
    file_handler = (
        logging.FileHandler(
            LOG_FILE,
            encoding="utf-8",
        )
    )

    formatter = (
        logging.Formatter(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        )
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )


class AgentRuntime:
    """
    SignalLab Unified Agent运行时。

    负责统一管理：

    LLM
    Redis
    MemoryService
    MCP Client / Server
    Tool Discovery
    LangGraph
    """

    def __init__(
        self,
    ) -> None:

        self.settings = (
            get_settings()
        )

        self.llm: (
            LLMClient | None
        ) = None

        self.memory_store: (
            RedisConversationMemory
            | None
        ) = None

        self.memory_service: (
            ConversationMemoryService
            | None
        ) = None

        self.mcp_client: (
            Client | None
        ) = None

        self.graph = None

        self.available_tools: list[
            dict[str, Any]
        ] = []

        self._exit_stack: (
            AsyncExitStack | None
        ) = None

        self.started = False

    async def startup(
        self,
    ) -> None:
        """
        FastAPI启动时只执行一次。
        """

        if self.started:
            return

        # ==================================
        # 1. LLM
        # ==================================

        self.llm = (
            LLMClient(
                settings=(
                    self.settings
                )
            )
        )

        # ==================================
        # 2. Redis
        # ==================================

        self.memory_store = (
            RedisConversationMemory(
                redis_url=(
                    self.settings.redis_url
                ),
            )
        )

        redis_ok = (
            await self.memory_store
            .ping()
        )

        if not redis_ok:
            raise RuntimeError(
                "Redis连接失败"
            )

        self.memory_service = (
            ConversationMemoryService(
                memory=(
                    self.memory_store
                ),
                llm=(
                    self.llm
                ),
                summary_trigger_messages=14,
                recent_window_messages=8,
            )
        )

        # ==================================
        # 3. MCP Server
        # ==================================

        if not (
            MCP_SERVER_FILE.exists()
        ):
            raise FileNotFoundError(
                "找不到MCP Server文件："
                f"{MCP_SERVER_FILE}"
            )

        server_params = (
            StdioServerParameters(
                command=(
                    sys.executable
                ),
                args=[
                    str(
                        MCP_SERVER_FILE
                    )
            ]   ,
                env=os.environ.copy(),
            )
        )

        self._exit_stack = (
            AsyncExitStack()
        )

        await (
            self._exit_stack
            .__aenter__()
        )

        self.mcp_client = (
            await self._exit_stack
            .enter_async_context(
                Client(
                    server_params
                )
            )
        )

        # ==================================
        # 4. Tool Discovery
        # ==================================

        tools_result = (
            await self.mcp_client
            .list_tools()
        )

        tools = (
            tools_result.tools
        )

        self.available_tools = [
            {
                "name": (
                    tool.name
                ),
                "description": (
                    tool.description
                ),
                "input_schema": (
                    tool.input_schema
                ),
            }
            for tool in tools
        ]

        if not tools:
            raise RuntimeError(
                "MCP Server未发现任何Tool"
            )

        # ==================================
        # 5. LangGraph
        # ==================================

        self.graph = (
            build_agent_graph(
                llm=(
                    self.llm
                ),
                client=(
                    self.mcp_client
                ),
                tools=tools,
            )
        )

        self.started = True

    async def shutdown(
        self,
    ) -> None:

        self.started = False

        if self._exit_stack:
            try:
                await (
                    self._exit_stack
                    .aclose()
                )
            finally:
                self._exit_stack = None

        self.mcp_client = None

        if self.memory_store:
            try:
                await (
                    self.memory_store
                    .close()
                )
            finally:
                self.memory_store = None

        self.memory_service = None

        if self.llm:
            try:
                self.llm.close()
            finally:
                self.llm = None

        self.graph = None

        self.available_tools = []

    async def chat(
        self,
        message: str,
        db: Session,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        一轮完整Agent请求。

        当前顺序：

        1. 校验问题
        2. trace_id
        3. session_id
        4. MySQL查询/创建Session
        5. Redis读取Memory
        6. LangGraph执行
        7. Redis保存Memory
        8. MySQL更新Session
        9. 返回结果
        """

        if not self.started:
            raise RuntimeError(
                "AgentRuntime尚未启动"
            )

        if self.graph is None:
            raise RuntimeError(
                "LangGraph尚未初始化"
            )

        if (
            self.memory_service
            is None
        ):
            raise RuntimeError(
                "Memory Service尚未初始化"
            )

        clean_message = (
            message.strip()
        )

        if not clean_message:
            raise ValueError(
                "message不能为空"
            )

        # ==================================
        # Trace
        # ==================================

        trace_id = str(
            uuid.uuid4()
        )

        start_time = (
            time.perf_counter()
        )

        # ==================================
        # Session ID
        # ==================================

        actual_session_id = ""

        if session_id:
            actual_session_id = (
                session_id.strip()
            )

        if not actual_session_id:
            actual_session_id = (
                str(
                    uuid.uuid4()
                )
            )

        logger.info(
            json.dumps(
                {
                    "event": (
                        "agent_request_started"
                    ),
                    "trace_id": (
                        trace_id
                    ),
                    "session_id": (
                        actual_session_id
                    ),
                    "query": (
                        clean_message
                    ),
                },
                ensure_ascii=False,
            )
        )

        try:
            # ==============================
            # 1. MySQL Session Metadata
            # ==============================

            title = (
                clean_message[:100]
            )

            (
                _session_record,
                session_created,
            ) = (
                get_or_create_agent_session(
                    db=(
                        db
                    ),
                    session_id=(
                        actual_session_id
                    ),
                    title=(
                        title
                    ),
                )
            )

            logger.info(
                json.dumps(
                    {
                        "event": (
                            "mysql_session_checked"
                        ),
                        "trace_id": (
                            trace_id
                        ),
                        "session_id": (
                            actual_session_id
                        ),
                        "created": (
                            session_created
                        ),
                    },
                    ensure_ascii=False,
                )
            )

            # ==============================
            # 2. Redis读取上下文
            # ==============================

            (
                summary,
                history,
            ) = (
                await self.memory_service
                .load_context(
                    actual_session_id
                )
            )

            # ==============================
            # 3. LangGraph执行
            # ==============================

            result = (
                await self.graph.ainvoke(
                    {
                        "user_query": (
                            clean_message
                        ),
                        "session_id": (
                            actual_session_id
                        ),
                        "conversation_summary": (
                            summary
                        ),
                        "conversation_history": (
                            history
                        ),
                        "available_tools": (
                            self.available_tools
                        ),
                        "top_k": 3,
                    }
                )
            )

            answer = (
                result.get(
                    "final_answer",
                    "没有生成最终回答。",
                )
            )

            # ==============================
            # 4. MySQL保存完整聊天记录
            # ==============================
            save_agent_message(
                db=db,
                session_id=actual_session_id,
                role="user",
                content=clean_message,
                trace_id=trace_id,
            )

            save_agent_message(
                db=db,
                session_id=actual_session_id,
                role="assistant",
                content=answer,
                trace_id=trace_id,
            )

            # ==============================
            # 5. Redis保存Memory
            # ==============================
            await (
                self.memory_service
                .save_turn(
                    session_id=(
                        actual_session_id
                    ),
                    user_query=(
                        clean_message
                    ),
                    assistant_answer=(
                        answer
                    ),
                )
            )

            await (
                self.memory_service
                .compact_if_needed(
                    actual_session_id
                )
            )

            # ==============================
            # 6. 更新MySQL Session
            # ==============================

            touch_agent_session(
                db=(
                    db
                ),
                session_id=(
                    actual_session_id
                ),
            )

            # ==============================
            # 7. Latency
            # ==============================

            latency_ms = (
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            )

            tools_used = (
                result.get(
                    "tools_used",
                    [],
                )
            )

            # 去重并保持顺序
            tools_used = list(
                dict.fromkeys(
                    tools_used
                )
            )

            logger.info(
                json.dumps(
                    {
                        "event": (
                            "agent_request_completed"
                        ),
                        "trace_id": (
                            trace_id
                        ),
                        "session_id": (
                            actual_session_id
                        ),
                        "route": (
                            result.get(
                                "route"
                            )
                        ),
                        "tools_used": (
                            tools_used
                        ),
                        "latency_ms": (
                            round(
                                latency_ms,
                                2,
                            )
                        ),
                        "error": (
                            result.get(
                                "error"
                            )
                        ),
                    },
                    ensure_ascii=False,
                )
            )

            return {
                "ok": True,
                "session_id": (
                    actual_session_id
                ),
                "answer": (
                    answer
                ),
                "route": (
                    result.get(
                        "route"
                    )
                ),
                "route_reason": (
                    result.get(
                        "route_reason"
                    )
                ),
                "tools_used": (
                    tools_used
                ),
                "tool_plan": (
                    result.get(
                        "tool_plan",
                        [],
                    )
                ),
                "trace_id": (
                    trace_id
                ),
                "latency_ms": (
                    round(
                        latency_ms,
                        2,
                    )
                ),
            }

        except Exception as error:
            db.rollback()

            latency_ms = (
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            )

            logger.exception(
                json.dumps(
                    {
                        "event": (
                            "agent_request_failed"
                        ),
                        "trace_id": (
                            trace_id
                        ),
                        "session_id": (
                            actual_session_id
                        ),
                        "query": (
                            clean_message
                        ),
                        "error_type": (
                            type(
                                error
                            ).__name__
                        ),
                        "error_message": (
                            str(
                                error
                            )
                        ),
                        "latency_ms": (
                            round(
                                latency_ms,
                                2,
                            )
                        ),
                    },
                    ensure_ascii=False,
                )
            )

            raise

    async def clear_session(
        self,
        session_id: str,
    ) -> None:
        """
        清除Redis Memory。

        MySQL删除放在FastAPI业务层处理。
        """

        if (
            self.memory_service
            is None
        ):
            raise RuntimeError(
                "Memory Service尚未初始化"
            )

        clean_session_id = (
            session_id.strip()
        )

        if not clean_session_id:
            raise ValueError(
                "session_id不能为空"
            )

        await (
            self.memory_service
            .clear(
                clean_session_id
            )
        )

    async def health(
        self,
    ) -> dict[str, Any]:

        redis_ok = False

        if self.memory_store:
            try:
                redis_ok = bool(
                    await (
                        self.memory_store
                        .ping()
                    )
                )

            except Exception:
                redis_ok = False

        mysql_ok = (
            check_mysql_connection()
        )

        mcp_ok = (
            self.mcp_client
            is not None
            and len(
                self.available_tools
            ) > 0
        )

        graph_ok = (
            self.graph
            is not None
        )

        return {
            "ok": (
                self.started
                and mysql_ok
                and redis_ok
                and mcp_ok
                and graph_ok
            ),
            "mysql": (
                mysql_ok
            ),
            "redis": (
                redis_ok
            ),
            "mcp": (
                mcp_ok
            ),
            "graph": (
                graph_ok
            ),
            "tool_count": len(
                self.available_tools
            ),
        }