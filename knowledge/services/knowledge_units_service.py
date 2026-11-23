import logging
import uuid
from typing import Any, Dict

from fastapi import HTTPException

from knowledge.schema.task_schema import ChunkWriteRequest
from knowledge.utils import milvus_utils
from knowledge.utils.embedding_utils import generate_embeddings
from knowledge.utils.id_util import next_chunk_id

logger = logging.getLogger(__name__)


class KnowledgeUnitsService:

    def page_chunks(self, document_id: str, page_num: int, page_size: int) -> dict:
        """按切片次序查询指定文档的主切片。"""
        normalized_document_id = self._validate_document_id(document_id)
        # 页码和页大小必须为正数，避免产生无意义或反向切片区间。
        if page_num <= 0 or page_size <= 0:
            raise HTTPException(status_code=400, detail="分页参数必须为正整数")

        chunk_list = self._list_chunks(normalized_document_id)
        # 本方案没有独立文档元数据，无主切片即视为文档不存在。
        if not chunk_list:
            return {
                "document_id": normalized_document_id,
                "total": 0,
                "chunk_count": 0,
                "total_text_length": 0,
                "rows": [],
            }
        # chunk_order是展示主次序，chunk_id仅在次序相同时保证结果稳定。
        sorted_chunks = sorted(
            chunk_list,
            key=lambda chunk: (int(chunk.get("chunk_order", 0)), int(chunk.get("chunk_id", 0))),
        )
        start_index = (page_num - 1) * page_size
        page_chunk_list = sorted_chunks[start_index:start_index + page_size]
        chunk_count, total_text_length = self._calculate_chunk_statistics(sorted_chunks)
        return {
            "document_id": normalized_document_id,
            "total": chunk_count,
            "chunk_count": chunk_count,
            "total_text_length": total_text_length,
            "rows": [self._to_chunk_row(chunk) for chunk in page_chunk_list],
        }

    def add_chunk(self, document_id: str, request: ChunkWriteRequest) -> dict:
        """在指定文档末尾新增一个主切片。"""
        normalized_document_id = self._validate_document_id(document_id)
        chunk_list = self._list_chunks(normalized_document_id)
        # 新增需要从现有切片继承文档元数据，全部删除后无法重建首片。
        if not chunk_list:
            raise HTTPException(status_code=404, detail="文档不存在或已删除全部切片")

        source_chunk = chunk_list[0]
        dense_vector, sparse_vector = self._generate_chunk_vectors(source_chunk.get("item_name", ""), request.content)
        new_chunk = {
            "document_id": normalized_document_id,
            # 主键由应用生成，新增后即可作为稳定资源标识返回给平台
            "chunk_id": next_chunk_id(),
            "chunk_order": request.chunk_order or max(
                int(chunk.get("chunk_order", 0)) for chunk in chunk_list
            ) + 1,
            "title": request.title or "",
            "parent_title": request.parent_title or "",
            "content": request.content,
            "item_name": source_chunk.get("item_name", ""),
            "file_title": source_chunk.get("file_title", ""),
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }
        try:
            milvus_utils.insert_chunk(new_chunk)
        except milvus_utils.ChunkIdNotWritableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Milvus新增切片失败") from exc

        logger.info("新增知识切片完成，document_id=%s, chunk_id=%s",
                    normalized_document_id, new_chunk["chunk_id"])
        return self._build_mutation_response(normalized_document_id, new_chunk)

    def update_chunk(self, document_id: str, chunk_id: str, request: ChunkWriteRequest) -> dict:
        """修改指定主切片并重新生成混合向量。"""
        normalized_document_id = self._validate_document_id(document_id)
        normalized_chunk_id = self._validate_chunk_id(chunk_id)
        old_chunk = self._get_chunk(normalized_document_id, normalized_chunk_id)
        # 联合查询为空表示切片不存在或不属于当前文档。
        if old_chunk is None:
            raise HTTPException(status_code=404, detail="切片不存在或不属于当前文档")

        dense_vector, sparse_vector = self._generate_chunk_vectors(old_chunk.get("item_name", ""), request.content)
        updated_chunk = dict(old_chunk)
        updated_chunk.update({
            "document_id": normalized_document_id,
            "chunk_id": normalized_chunk_id,
            "chunk_order": request.chunk_order or int(old_chunk["chunk_order"]),
            "title": request.title or "",
            "parent_title": request.parent_title or "",
            "content": request.content,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        })
        try:
            milvus_utils.update_chunk(updated_chunk)
        except milvus_utils.ChunkIdNotWritableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Milvus修改切片失败") from exc

        logger.info("修改知识切片完成，document_id=%s, chunk_id=%s",
                    normalized_document_id, normalized_chunk_id)
        return self._build_mutation_response(normalized_document_id, updated_chunk)

    def delete_chunk(self, document_id: str, chunk_id: str) -> dict:
        """删除指定主切片，保留其余切片的原次序。"""
        normalized_document_id = self._validate_document_id(document_id)
        normalized_chunk_id = self._validate_chunk_id(chunk_id)
        old_chunk = self._get_chunk(normalized_document_id, normalized_chunk_id)
        # 删除前确认文档边界，防止同一chunk_id误操作其他文档。
        if old_chunk is None:
            raise HTTPException(status_code=404, detail="切片不存在或不属于当前文档")

        try:
            milvus_utils.delete_chunk(normalized_document_id, normalized_chunk_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Milvus删除切片失败") from exc

        remaining_chunks = self._list_chunks(normalized_document_id)
        chunk_count, total_text_length = self._calculate_chunk_statistics(remaining_chunks)
        logger.info("删除知识切片完成，document_id=%s, chunk_id=%s",
                    normalized_document_id, normalized_chunk_id)
        return {"chunk_count": chunk_count, "total_text_length": total_text_length}

    def delete_document_chunks(self, document_id: str) -> dict:
        """按文档ID物理删除全部主切片，重复调用保持成功。"""
        normalized_document_id = self._validate_document_id(document_id)
        chunk_list = self._list_chunks(normalized_document_id)
        # 文档元数据没有独立存储；无切片时按幂等成功处理，支持平台删除重试。
        if not chunk_list:
            logger.info("知识文档切片已为空，document_id=%s", normalized_document_id)
            return {
                "document_id": normalized_document_id,
                "deleted_count": 0,
            }

        try:
            milvus_utils.delete_chunks_by_document_id(normalized_document_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Milvus删除文档切片失败") from exc

        logger.info("删除知识文档全部切片完成，document_id=%s, deleted_count=%s",
                    normalized_document_id, len(chunk_list))
        return {
            "document_id": normalized_document_id,
            "deleted_count": len(chunk_list),
        }

    @staticmethod
    def _validate_document_id(document_id: str) -> str:
        """校验并标准化文档UUID。"""
        try:
            return str(uuid.UUID(document_id))
        except (ValueError, TypeError, AttributeError) as exc:
            raise HTTPException(status_code=400, detail="document_id必须为UUID") from exc

    @staticmethod
    def _validate_chunk_id(chunk_id: str) -> str:
        """校验对外使用的切片ID，JSON中始终按字符串传递。"""
        try:
            parsed_chunk_id = int(chunk_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="chunk_id必须为正整数") from exc
        # Milvus主键只接受正整数。
        if parsed_chunk_id <= 0:
            raise HTTPException(status_code=400, detail="chunk_id必须为正整数")
        return str(parsed_chunk_id)

    @staticmethod
    def _calculate_chunk_statistics(chunk_list: list) -> tuple[int, int]:
        """计算切片数量和正文总长度。"""
        return len(chunk_list), sum(len(chunk.get("content", "")) for chunk in chunk_list)

    @staticmethod
    def _to_chunk_row(chunk: Dict[str, Any]) -> dict:
        """仅返回平台管理所需字段，隔离向量和内部路径。"""
        content = chunk.get("content", "")
        return {
            "chunk_id": str(chunk["chunk_id"]),
            "chunk_order": int(chunk["chunk_order"]),
            "title": chunk.get("title") or None,
            "parent_title": chunk.get("parent_title") or None,
            "content": content,
            "content_length": len(content),
        }

    def _list_chunks(self, document_id: str) -> list:
        """调用Milvus查询主切片并统一转换上游错误。"""
        try:
            return milvus_utils.list_chunks_by_document_id(document_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Milvus查询切片失败") from exc

    @staticmethod
    def _get_chunk(document_id: str, chunk_id: str) -> dict | None:
        """调用Milvus查询单个主切片并统一转换上游错误。"""
        try:
            return milvus_utils.get_chunk(document_id, chunk_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Milvus查询切片失败") from exc

    @staticmethod
    def _generate_chunk_vectors(item_name: str, content: str) -> tuple[list, dict]:
        """按导入流程的相同规则为单个切片生成稠密和稀疏向量。"""
        try:
            embedding_result = generate_embeddings([f"{item_name}\n{content}"])
            # 新增和修改必须同时获得稠密、稀疏向量才能写入主切片。
            if not embedding_result or not embedding_result.get("dense") or not embedding_result.get("sparse"):
                raise RuntimeError("BGE-M3未返回完整混合向量")
            return embedding_result["dense"][0], embedding_result["sparse"][0]
        except Exception as exc:
            raise HTTPException(status_code=502, detail="BGE-M3向量生成失败") from exc

    def _build_mutation_response(self, document_id: str, chunk: Dict[str, Any]) -> dict:
        """构建新增或修改响应，并附带实时文档统计。"""
        current_chunks = self._list_chunks(document_id)
        chunk_count, total_text_length = self._calculate_chunk_statistics(current_chunks)
        response = self._to_chunk_row(chunk)
        response.update({
            "document_id": document_id,
            "chunk_count": chunk_count,
            "total_text_length": total_text_length,
        })
        return response
