import re
from pathlib import Path

from app.rag.schemas import SourceDocument


SUPPORTED_SUFFIXES = {".txt", ".md"}


def clean_text(text: str) -> str:
    """对读取到的文本进行简单清洗。"""

    # 把Windows换行符统一成 \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 去除每一行开头和结尾的多余空格
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 三个及以上连续换行统一成两个换行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_documents(directory: Path) -> list[SourceDocument]:
    """递归读取目录内所有txt和md文档。"""

    if not directory.exists():
        raise FileNotFoundError(f"知识库目录不存在：{directory}")

    documents: list[SourceDocument] = []

    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        content = path.read_text(encoding="utf-8-sig")
        cleaned_content = clean_text(content)

        if not cleaned_content:
            continue

        documents.append(
            SourceDocument(
                source=path.relative_to(directory).as_posix(),
                content=cleaned_content,
            )
        )

    return documents