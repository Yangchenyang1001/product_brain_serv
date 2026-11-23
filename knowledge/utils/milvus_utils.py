import logging
from typing import Dict, List, Any, Optional

from pymilvus import MilvusClient

from knowledge.processor.import_process.config import get_config
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)

# 管理端读写使用强一致：保证删除后立即查询能看到最新切片列表。
_CHUNK_ADMIN_CONSISTENCY_LEVEL = "Strong"

# 集合主键是否允许应用侧显式写入，按集合名缓存，避免每次写请求都 describe。
_CHUNK_ID_WRITABLE_CACHE: Dict[str, bool] = {}


class ChunkIdNotWritableError(RuntimeError):
    """切片集合使用 AutoID 主键，无法稳定持有 chunk_id，必须重建集合。"""

_CHUNK_OUTPUT_FIELDS = [
    "chunk_id",
    "document_id",
    "chunk_order",
    "title",
    "parent_title",
    "content",
    "item_name",
    "file_title",
]


def get_milvus_client() -> MilvusClient:
    """
    获取Milvus客户端对象
    :return:
    """
    return StorageClients.get_milvus_client()


def get_chunks_collection_name() -> str:
    """获取主切片集合名。"""
    return get_config().chunks_collection


def escape_milvus_string(value: str) -> str:
    """
    Milvus数据库过滤表达式中字符串的安全转义函数（防止解析失败）
    作用： 转义特殊字符（反斜杠、双引号），避免Milvus解析filter时报错
    参数 value: 需要转义的原始字符串
    返回 str: 转义后的安全字符串
    """
    # 转义反斜杠（\ → \\） 双引号（" → \"） 单引号（' → \'）
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value


def _parse_chunk_id(chunk_id: str) -> int:
    """将对外使用的字符串切片ID转换为Milvus数字主键。"""
    try:
        parsed_chunk_id = int(chunk_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("chunk_id必须为正整数") from exc

    # Milvus主键只接受正整数。
    if parsed_chunk_id <= 0:
        raise ValueError("chunk_id必须为正整数")
    return parsed_chunk_id


def assert_chunk_id_writable(client: MilvusClient, collection_name: str) -> None:
    """校验集合主键允许应用侧写入：AutoID 集合无法保持 chunk_id 稳定，必须重建。"""
    # 只缓存通过校验的集合，集合被重建后无需重启即可重新识别。
    if _CHUNK_ID_WRITABLE_CACHE.get(collection_name):
        return

    fields = client.describe_collection(collection_name=collection_name).get("fields", [])
    primary_key = next((field for field in fields if field.get("name") == "chunk_id"), None)
    if primary_key is None:
        raise ChunkIdNotWritableError(f"集合 {collection_name} 缺少 chunk_id 主键字段，请重新导入")
    if primary_key.get("auto_id", False):
        raise ChunkIdNotWritableError(
            f"集合 {collection_name} 的 chunk_id 仍是 AutoID 结构，无法稳定更新主键，请删除该集合后重新导入"
        )
    _CHUNK_ID_WRITABLE_CACHE[collection_name] = True


def list_chunks_by_document_id(document_id: str) -> List[Dict[str, Any]]:
    """查询指定文档的全部主切片。"""
    client = get_milvus_client()
    collection_name = get_chunks_collection_name()
    if not client.has_collection(collection_name):
        logger.warning("切片集合不存在，按空结果返回：collection=%s", collection_name)
        return []
    document_filter = f'document_id=="{escape_milvus_string(document_id)}"'
    return client.query(
        collection_name=collection_name,
        filter=document_filter,
        output_fields=_CHUNK_OUTPUT_FIELDS,
        limit=16384,
        # 等待QueryNode追平最新时间戳，避免读到删除前的旧快照。
        consistency_level=_CHUNK_ADMIN_CONSISTENCY_LEVEL,
    )


def get_chunk(document_id: str, chunk_id: str) -> Optional[Dict[str, Any]]:
    """按文档ID和切片ID查询唯一主切片。"""
    client = get_milvus_client()
    collection_name = get_chunks_collection_name()
    if not client.has_collection(collection_name):
        logger.warning("切片集合不存在，按空结果返回：collection=%s", collection_name)
        return None
    parsed_chunk_id = _parse_chunk_id(chunk_id)
    document_filter = f'document_id=="{escape_milvus_string(document_id)}"'
    chunk_filter = f"{document_filter} and chunk_id == {parsed_chunk_id}"
    chunk_list = client.query(
        collection_name=collection_name,
        filter=chunk_filter,
        output_fields=_CHUNK_OUTPUT_FIELDS,
        limit=1,
        # 等待QueryNode追平最新时间戳，避免读到删除前的旧快照。
        consistency_level=_CHUNK_ADMIN_CONSISTENCY_LEVEL,
    )
    return chunk_list[0] if chunk_list else None


def insert_chunk(chunk: Dict[str, Any]) -> str:
    """新增主切片，chunk_id 由调用方生成并随数据写入。"""
    client = get_milvus_client()
    collection_name = get_chunks_collection_name()
    parsed_chunk_id = _parse_chunk_id(str(chunk.get("chunk_id", "")))
    assert_chunk_id_writable(client, collection_name)

    inserted_result = client.insert(
        collection_name=collection_name,
        data=[chunk],
        # 写请求同步使用强一致，保证写成功后立即可查，与读端形成写后读闭环。
        consistency_level=_CHUNK_ADMIN_CONSISTENCY_LEVEL,
    )
    # 集合主键必须由应用生成：返回的主键与写入值不一致说明集合仍是 AutoID 结构。
    inserted_ids = inserted_result.get("ids", [])
    if inserted_ids and int(inserted_ids[0]) != parsed_chunk_id:
        raise ChunkIdNotWritableError(
            f"集合 {collection_name} 的 chunk_id 由 Milvus 重新生成，请删除该集合后重新导入"
        )
    return str(parsed_chunk_id)


def update_chunk(chunk: Dict[str, Any]) -> str:
    """覆盖更新主切片的正文、标题和向量，主键与次序保持不变。"""
    client = get_milvus_client()
    collection_name = get_chunks_collection_name()
    updated_chunk = dict(chunk)
    parsed_chunk_id = _parse_chunk_id(str(chunk.get("chunk_id", "")))
    updated_chunk["chunk_id"] = parsed_chunk_id
    assert_chunk_id_writable(client, collection_name)

    # 主键由应用生成，upsert 才能原地覆盖同一条切片而不产生新主键。
    client.upsert(
        collection_name=collection_name,
        data=[updated_chunk],
        # upsert在Milvus内部是delete+insert，同样需要强一致才能立即覆盖旧内容。
        consistency_level=_CHUNK_ADMIN_CONSISTENCY_LEVEL,
    )
    return str(parsed_chunk_id)


def delete_chunk(document_id: str, chunk_id: str) -> None:
    """按文档ID和切片ID删除主切片，不影响其他文档。"""
    client = get_milvus_client()
    collection_name = get_chunks_collection_name()
    parsed_chunk_id = _parse_chunk_id(chunk_id)
    document_filter = f'document_id=="{escape_milvus_string(document_id)}"'
    chunk_filter = f"{document_filter} and chunk_id == {parsed_chunk_id}"
    client.delete(
        collection_name=collection_name,
        filter=chunk_filter,
        # 等待QueryNode追平最新时间戳，避免读到删除前的旧快照。
        consistency_level=_CHUNK_ADMIN_CONSISTENCY_LEVEL,
    )


def delete_chunks_by_document_id(document_id: str) -> None:
    """按文档ID物理删除全部主切片，不影响其他文档。"""
    client = get_milvus_client()
    collection_name = get_chunks_collection_name()
    document_filter = f'document_id=="{escape_milvus_string(document_id)}"'
    client.delete(
        collection_name=collection_name,
        filter=document_filter,
        # 等待QueryNode追平最新时间戳，避免读到删除前的旧快照。
        consistency_level=_CHUNK_ADMIN_CONSISTENCY_LEVEL,
    )
