"""
Milvus 存储节点

将向量嵌入结果及元数据（商品名、原文、来源等）写入 Milvus 向量库，完成导入流程。
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional, Dict, Any, List

from pymilvus import MilvusClient, DataType, CollectionSchema

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import get_config, ImportConfig
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
import logging

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScalarFieldSpec:
    """标量字段规范"""
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None


# 预定义的标量字段（复用）
_SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
    ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
)


class _MilvusInserter:
    """职责：将数据插入到Milvus 以及 回填chunk_id"""

    def __init__(self, client: MilvusClient, collection_name: str):
        self._client = client
        self._collection_name = collection_name

    def insert(self, chunks: List[Dict[str, Any]]) -> List[dict[str, Any]]:
        logger.info(f"开始插入{len(chunks)}块到Milvus...")

        # 1. 插入
        inserted_result = self._client.insert(
            collection_name=self._collection_name,
            data=chunks
        )
        inserted_count = inserted_result.get('insert_count')
        ids = inserted_result.get('ids')

        # 2. 回填id
        self._fill_chunk_ids(chunks, ids)
        logger.info(f"完成插入{inserted_count}记录,并且回填chunk_id到chunk中")
        return chunks

    def _fill_chunk_ids(self, chunks: List[Dict[str, Any]], ids: List[Any]):
        for chunk, id in zip(chunks, ids):
            chunk["chunk_id"] = id


class _MilvusSchemaBuilder:
    """职责：专门负责构建约束"""

    @staticmethod
    def build(client: MilvusClient, dim: int) -> CollectionSchema:

        logger.info("开始构建约束(schema)...")

        # 1. 构建约束对象(动态映射)
        schema = client.create_schema(enable_dynamic_field=True)

        # 2. 构建主键字段约束
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.INT64,
            is_primary=True,
            auto_id=True
        )

        # 3. 构建向量字段约束
        # 3.1 稠密向量字段
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=dim
        )
        # 3.2 稀疏向量字段
        schema.add_field(
            field_name="sparse_vector",
            datatype=DataType.SPARSE_FLOAT_VECTOR,
        )

        # 4. 构建标量字段约束（复用规范）
        for scalar_field in _SCALAR_FIELDS:
            kwargs: Dict[str, Any] = {
                "field_name": scalar_field.field_name,
                "datatype": scalar_field.datatype
            }
            if scalar_field.max_length is not None:
                kwargs['max_length'] = scalar_field.max_length
            schema.add_field(**kwargs)

        logger.info(f"构建约束(schema)完成...")
        return schema


class _MilvusIndexBuilder:
    """职责：负责处理Milvus的索引"""

    @staticmethod
    def build(client: MilvusClient, collection_name: str):
        logger.info(f"开始构建集合 {collection_name} 索引...")

        # 1. 创建索引对象
        index = client.prepare_index_params(collection_name=collection_name)

        # 2. 给向量字段添加索引
        # 2.1 稠密向量字段添加索引
        index.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE"
        )

        # 2.2 稀疏向量字段添加索引
        index.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )

        logger.info(f"构建集合 {collection_name} 索引完成...")
        return index


class ImportMilvusNode(BaseNode):
    name: str = "import_milvus_node"

    def _validate_get_inputs(self, state: ImportGraphState) -> tuple[list[Any], int, ImportConfig]:
        self.log_step("step1", "参数校验")

        # 1. 获取chunks
        chunks = state.get('chunks')
        config = get_config()

        # 2. 校验是否为空
        if not chunks:
            raise ValidationError("待入库的切块chunk不存在", self.name)

        # 3. 校验是否有混合向量
        validated_chunks = []
        for chunk in chunks:
            if chunk.get('dense_vector') and chunk.get('sparse_vector'):
                validated_chunks.append(chunk)
            else:
                self.logger.error("待入库的切块chunk的混合向量不存在")

        # 4. 判断有效集合
        if not validated_chunks:
            raise ValidationError("入库的chunk都无效", self.name)

        # 5. 获取向量维度
        dim = config.embedding_dim
        self.logger.info(f"导入Milvus向量数据库的有效块：{len(validated_chunks)},且chunk的向量维度{dim}")

        return validated_chunks, dim, config

    def _ensure_has_collection(self, milvus_client: MilvusClient, collection_name: str,
                               dim: int, delete_flag: bool = False):

        self.log_step("step2", f"准备集合 {collection_name} 创建")

        # 1. 判断是否要删除集合
        if delete_flag and milvus_client.has_collection(collection_name=collection_name):
            self.logger.info(f"Milvus中的集合 {collection_name}已被删除")
            milvus_client.drop_collection(collection_name=collection_name)

        # 2. 判断集合是否存在
        if milvus_client.has_collection(collection_name=collection_name):
            self.logger.info(f"{collection_name}集合已经存在")
            return

        # 3. 创建约束
        schema = _MilvusSchemaBuilder.build(milvus_client, dim)

        # 4. 创建索引
        index = _MilvusIndexBuilder.build(milvus_client, collection_name)

        # 5. 创建集合
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index
        )

    @dataclass(frozen=True)
    class ScalarFieldSpec:
        field_name: str
        datatype: DataType
        max_length: Optional[int] = None

    # 标量字段规范（复用）
    _SCALAR_FIELDS: Sequence[ScalarFieldSpec] = (
        ScalarFieldSpec(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
        ScalarFieldSpec(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
        ScalarFieldSpec(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
        ScalarFieldSpec(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
        ScalarFieldSpec(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
    )

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """

          a. step1 参数校验
            ⅰ. 获取chunks验证混合向量, 并返回
          b. 获取milvus客户端
          c. step2 集合准备
            ⅰ. 集合存在检查
            ⅱ. 构建scheme
            ⅲ. 构建索引
            ⅳ. 创建集合
          d. step3 数据插入
            ⅰ. 批量插入数据, 回去id并回填到chunk_id
          e. 更新state并返回
        :param state:
        :return:
        """
        # 1. 参数校验
        validated_chunks, dim, config = self._validate_get_inputs(state)

        # 2. 获取milvus客户端
        milvus_client = StorageClients.get_milvus_client()

        if milvus_client is None:
            return state

        # 3. 获取集合名字
        collection = getattr(config, 'chunks_collection')

        # 4. 确保集合存在
        self._ensure_has_collection(milvus_client, collection, dim)

        # 5. 插入
        inserter = _MilvusInserter(client=milvus_client, collection_name=collection)
        final_chunks = inserter.insert(chunks=validated_chunks)

        # 6. 更新state
        state['chunks'] = final_chunks

        return state


if __name__ == "__main__":
    setup_logging()

    temp_dir = Path(
        r"C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain\knowledge\test\output\万用表RS-12的使用\auto")

    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        logger.error(f"找不到输入文件: {input_path}")
        exit(1)

    with open(input_path, "r", encoding="utf-8") as fh:
        content = json.load(fh)

        state: ImportGraphState = {
            "chunks": content.get("chunks", [])
        }

        import_milvus = ImportMilvusNode()
        result_state = import_milvus(state)

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(result_state, fh, ensure_ascii=False, indent=4)

            logger.info(f"备份临时文件{output_path}成功")
