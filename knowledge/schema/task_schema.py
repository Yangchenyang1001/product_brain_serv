"""
任务状态模型

定义任务状态相关的数据模型:
    - TaskStatusResponse    任务状态响应模型，包含任务ID、状态、进度等信息
"""

from typing import List

from pydantic import BaseModel, Field, field_validator


class ChunkWriteRequest(BaseModel):
    """切片新增或修改请求。"""

    chunk_order: int | None = Field(default=None, gt=0, description="切片展示顺序")
    title: str | None = Field(default=None, max_length=500, description="切片标题")
    parent_title: str | None = Field(default=None, max_length=500, description="上级标题")
    content: str = Field(..., max_length=65535, description="切片正文")

    @field_validator("content")
    @classmethod
    def validate_content(cls, content: str) -> str:
        """去除正文首尾空格，避免写入空切片。"""
        normalized_content = content.strip()
        # 业务上空白字符串不构成有效知识内容。
        if not normalized_content:
            raise ValueError("切片正文不能为空")
        return normalized_content


class KnowledgeChunkRow(BaseModel):
    """知识单元切片行数据。"""

    chunk_id: str = Field(..., description="切片ID")
    chunk_order: int = Field(..., gt=0, description="切片次序")
    title: str | None = Field(default=None, description="切片标题")
    parent_title: str | None = Field(default=None, description="上级标题")
    content: str = Field(..., description="切片正文")
    content_length: int = Field(..., ge=0, description="正文长度")


class KnowledgeChunkPageResponse(BaseModel):
    """知识单元切片分页响应。"""

    document_id: str = Field(..., description="外部知识库文档ID")
    total: int = Field(..., ge=0, description="切片总数")
    chunk_count: int = Field(..., ge=0, description="切片总数")
    total_text_length: int = Field(..., ge=0, description="正文总长度")
    rows: List[KnowledgeChunkRow] = Field(..., description="当前页切片")


class KnowledgeChunkMutationResponse(KnowledgeChunkRow):
    """切片新增或修改响应。"""

    document_id: str = Field(..., description="外部知识库文档ID")
    chunk_count: int = Field(..., ge=0, description="切片总数")
    total_text_length: int = Field(..., ge=0, description="正文总长度")


class DeleteChunkResponse(BaseModel):
    """切片删除响应。"""

    chunk_count: int = Field(..., ge=0, description="删除后切片数量")
    total_text_length: int = Field(..., ge=0, description="删除后正文总长度")


class DeleteDocumentChunksResponse(BaseModel):
    """按文档物理删除全部切片的响应。"""

    document_id: str = Field(..., description="外部知识库文档ID")
    deleted_count: int = Field(..., ge=0, description="本次删除的切片数量")
