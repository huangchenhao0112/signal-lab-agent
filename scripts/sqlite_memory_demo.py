import argparse
import operator
import sqlite3
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "data"
    / "checkpoints"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "sqlite_demo.sqlite"
)


class DemoState(TypedDict):
    messages: Annotated[
        list[dict[str, str]],
        operator.add,
    ]


def generate_reply(
    state: DemoState,
) -> dict[str, list[dict[str, str]]]:
    """生成一个不调用LLM的演示回答。"""

    latest_message = state["messages"][-1]

    user_content = latest_message["content"]

    answer = (
        "我已经记录了你的消息："
        f"{user_content}"
    )

    return {
        "messages": [
            {
                "role": "assistant",
                "content": answer,
            }
        ]
    }


def build_graph(
    checkpointer: SqliteSaver,
):
    """创建最小LangGraph演示工作流。"""

    workflow = StateGraph(DemoState)

    workflow.add_node(
        "generate_reply",
        generate_reply,
    )

    workflow.add_edge(
        START,
        "generate_reply",
    )

    workflow.add_edge(
        "generate_reply",
        END,
    )

    return workflow.compile(
        checkpointer=checkpointer,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "演示LangGraph SQLite持久化记忆"
        )
    )

    parser.add_argument(
        "--session-id",
        required=True,
        help="会话ID",
    )

    parser.add_argument(
        "--message",
        required=True,
        help="本轮用户消息",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        str(CHECKPOINT_PATH),
        check_same_thread=False,
    )

    try:
        checkpointer = SqliteSaver(
            connection,
        )

        graph = build_graph(
            checkpointer,
        )

        config = {
            "configurable": {
                "thread_id": args.session_id,
            }
        }

        final_state = graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": args.message,
                    }
                ]
            },
            config=config,
        )

        print("=" * 60)
        print(f"Session ID：{args.session_id}")
        print(f"数据库文件：{CHECKPOINT_PATH}")
        print("=" * 60)

        for message in final_state["messages"]:
            role = message.get(
                "role",
                "unknown",
            )

            content = message.get(
                "content",
                "",
            )

            print(f"{role}：{content}")

    finally:
        connection.close()


if __name__ == "__main__":
    main()