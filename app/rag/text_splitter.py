import hashlib

from app.rag.schemas import DocumentChunk, SourceDocument


SPLIT_MARKS = ("\n", "。", "！", "？", "；")


def find_split_position(
    text: str,
    start: int,
    expected_end: int,
    chunk_size: int,
) -> int:
    """尽量在换行或中文标点处结束当前文本块。"""

    search_start = start + chunk_size // 2
    best_position = -1

    for mark in SPLIT_MARKS:
        position = text.rfind(mark, search_start, expected_end)

        if position > best_position:
            best_position = position + len(mark)

    if best_position > start:
        return best_position

    return expected_end


def split_document(
    document: SourceDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[DocumentChunk]:
    """把一篇完整文档切分成多个相互重叠的文本块。"""

    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap不能小于0")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap必须小于chunk_size")

    text = document.content
    chunks: list[DocumentChunk] = []

    start = 0
    chunk_index = 0

    while start < len(text):
        expected_end = min(start + chunk_size, len(text))
        end = expected_end

        if expected_end < len(text):
            end = find_split_position(
                text=text,
                start=start,
                expected_end=expected_end,
                chunk_size=chunk_size,
            )

        chunk_content = text[start:end].strip()

        if chunk_content:
            raw_id = (
                f"{document.source}:{chunk_index}:{chunk_content}"
            )
            chunk_id = hashlib.sha1(
                raw_id.encode("utf-8")
            ).hexdigest()[:12]

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=document.source,
                    chunk_index=chunk_index,
                    content=chunk_content,
                )
            )

            chunk_index += 1

        if end >= len(text):
            break

        # 下一段向前回退一部分，形成文本重叠
        start = max(end - chunk_overlap, start + 1)

    return chunks