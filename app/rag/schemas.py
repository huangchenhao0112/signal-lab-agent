from pydantic import BaseModel, Field


class SourceDocument(BaseModel):
    """从原始文件中读取出来的一篇完整文档。"""

    source: str = Field(description="原始文件的相对路径")
    content: str = Field(description="文档清洗后的完整内容")


class DocumentChunk(BaseModel):
    """原始文档经过切分后得到的一个文本块。"""

    chunk_id: str = Field(description="文本块的唯一编号")
    source: str = Field(description="文本块来自哪个文件")
    chunk_index: int = Field(description="文本块在原文档中的顺序")
    content: str = Field(description="文本块正文")

class SearchResult(BaseModel):
    """向量数据库返回的一条检索结果。"""

    chunk_id: str
    source: str
    chunk_index: int
    content: str
    distance: float