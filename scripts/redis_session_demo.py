import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import redis
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")


class RedisSessionStore:
    """
    使用Redis保存活跃会话消息。

    Redis Key格式：
    signal-agent:session:{session_id}:messages
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 1800,
    ) -> None:
        self.ttl_seconds = ttl_seconds

        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )

    def ping(self) -> bool:
        """检查Redis连接。"""
        return bool(self.client.ping())

    @staticmethod
    def build_key(
        session_id: str,
    ) -> str:
        """根据session_id生成Redis Key。"""
        cleaned_session_id = session_id.strip()

        if not cleaned_session_id:
            raise ValueError(
                "session_id不能为空"
            )

        return (
            "signal-agent:"
            f"session:{cleaned_session_id}:messages"
        )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> dict:
        """
        将消息加入Redis List，
        并刷新会话的过期时间。
        """
        cleaned_content = content.strip()

        if not cleaned_content:
            raise ValueError(
                "消息内容不能为空"
            )

        key = self.build_key(session_id)

        message = {
            "role": role,
            "content": cleaned_content,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        serialized_message = json.dumps(
            message,
            ensure_ascii=False,
        )

        pipeline = self.client.pipeline()

        pipeline.rpush(
            key,
            serialized_message,
        )

        pipeline.expire(
            key,
            self.ttl_seconds,
        )

        pipeline.execute()

        return message

    def get_messages(
        self,
        session_id: str,
    ) -> list[dict]:
        """读取指定会话中的全部消息。"""
        key = self.build_key(session_id)

        serialized_messages = self.client.lrange(
            key,
            0,
            -1,
        )

        return [
            json.loads(message)
            for message in serialized_messages
        ]

    def get_ttl(
        self,
        session_id: str,
    ) -> int:
        """获取会话剩余生存时间。"""
        key = self.build_key(session_id)

        return int(
            self.client.ttl(key)
        )

    def delete_session(
        self,
        session_id: str,
    ) -> bool:
        """删除指定会话。"""
        key = self.build_key(session_id)

        deleted_count = self.client.delete(key)

        return deleted_count > 0

    def close(self) -> None:
        """关闭Redis连接。"""
        self.client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redis会话存储演示",
    )

    parser.add_argument(
        "command",
        choices=[
            "add",
            "show",
            "ttl",
            "delete",
        ],
        help="要执行的操作",
    )

    parser.add_argument(
        "session_id",
        help="会话ID",
    )

    parser.add_argument(
        "role",
        nargs="?",
        choices=[
            "system",
            "user",
            "assistant",
            "tool",
        ],
        help="消息角色，add命令必须填写",
    )

    parser.add_argument(
        "content",
        nargs="?",
        help="消息内容，add命令必须填写",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    redis_url = os.getenv(
        "REDIS_URL",
        "redis://127.0.0.1:6379/0",
    )

    ttl_seconds = int(
        os.getenv(
            "REDIS_SESSION_TTL_SECONDS",
            "1800",
        )
    )

    store = RedisSessionStore(
        redis_url=redis_url,
        ttl_seconds=ttl_seconds,
    )

    try:
        store.ping()

        if args.command == "add":
            if args.role is None:
                parser.error(
                    "add命令必须提供role"
                )

            if args.content is None:
                parser.error(
                    "add命令必须提供content"
                )

            message = store.add_message(
                session_id=args.session_id,
                role=args.role,
                content=args.content,
            )

            print("消息写入成功：")
            print(
                json.dumps(
                    message,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        elif args.command == "show":
            messages = store.get_messages(
                session_id=args.session_id,
            )

            print(
                f"session_id：{args.session_id}"
            )

            print(
                f"消息数量：{len(messages)}"
            )

            print(
                json.dumps(
                    messages,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        elif args.command == "ttl":
            ttl = store.get_ttl(
                session_id=args.session_id,
            )

            if ttl == -2:
                print("该会话不存在")

            elif ttl == -1:
                print("该会话没有设置过期时间")

            else:
                print(
                    f"剩余TTL：{ttl}秒"
                )

        elif args.command == "delete":
            deleted = store.delete_session(
                session_id=args.session_id,
            )

            if deleted:
                print("会话删除成功")
            else:
                print("会话不存在")

    except redis.RedisError as error:
        raise SystemExit(
            f"Redis操作失败：{error}"
        ) from error

    finally:
        store.close()


if __name__ == "__main__":
    main()