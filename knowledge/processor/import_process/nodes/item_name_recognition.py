"""
商品识别节点

利用 LLM 从文档切分块中识别商品名称，为后续的向量检索和查询提供商品维度信息。
"""
import json
import os
from typing import List, Dict, Any, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import DataType

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import StateFieldError, ValidationError
from knowledge.processor.import_process.nodes.document_split import DocumentSplitNode
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.prompt.import_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, ITEM_NAME_SYSTEM_PROMPT
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class ItemNameRecognitionNode(BaseNode):
    name: str = "item_name_rec_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
        1. 输入state, 包含file_title文件名, chunks[]切片
        2. step1 参数校验 _validdate_date()
          a. 失败抛出异常StateFieldError
        3. step2 构建数据识别上下文_prepare_context()
          a. 从配置的前k个切片确定上下文文本, 输出【切片】- N - content
        4. step3 llm识别商品名_recognition_name()
          a. 成功则返回商品名称
          b. 失败则降级操作
        5. step4 BGE-M3向量化_embedding_item_name()
          a. 生成稠密与稀疏向量(dense,sparse)
        6. step5 存储到Milvus   _insert_milvus()
        7. step6 回填item_name到state和chunks _fill_item_name
        :param state:
        :return:
        """
        # step1 参数校验 _validdate_date()
        file_title, chunks, item_name_chunks_k, item_name_chunk_size = self._validdate_date(state)
        # step2 构建数据识别上下文_prepare_context()
        contexts: str = self._prepare_context(chunks, item_name_chunks_k, item_name_chunk_size)
        # step3 llm识别商品名_recognition_name()
        item_name: str = self._recognition_name(file_title, contexts)
        # step4 BGE-M3向量化_embedding_item_name()
        dense, sparse = self._embedding_item_name(item_name)
        # step5 存储到Milvus   _insert_milvus()
        self._insert_milvus(item_name, file_title, dense, sparse)
        # step6 回填item_name到state和chunks _fill_item_name
        self._fill_item_name(item_name, state, chunks)

        self._backup_chunks(state, chunks)

        return state

    def _validdate_date(self, state: ImportGraphState):
        """
        1. 参数校验
        :param state:
        :return:
        """
        self.logger.info(f"step1 参数校验 _validdate_date()")
        chunks: List[Dict[str, Any]] = state['chunks']
        file_title: str = state['file_title']

        if not file_title:
            raise StateFieldError(self.name, "file_title", message="File title is missing")
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(self.name, "chunks", expected_type=list, message="Chunks is error")

        # 5. 获取商品名识别需要的chunk块数
        item_name_chunks_k = self.config.item_name_chunk_k
        if not item_name_chunks_k or item_name_chunks_k <= 0:
            raise ValidationError(message="item_name_chunk_k为空或者无效", node_name=self.name)

        # 6. 获取商品名识别上下文的最大字符数
        item_name_chunk_size = self.config.item_name_chunk_size
        if not item_name_chunk_size or item_name_chunk_size <= 0:
            raise ValidationError(message="item_name_chunk_size为空或者无效", node_name=self.name)

        config = get_config()
        item_name_chunk_k = config.item_name_chunk_k
        item_name_chunk_size = config.item_name_chunk_size
        return file_title, chunks, item_name_chunk_k, item_name_chunk_size

    def _prepare_context(self, chunks: List[Dict[str, Any]], item_name_chunks_k: int, item_name_chunk_size: int) -> str:
        """
        1. 构建数据识别上下文
        :param chunks:
        :param item_name_chunks_k:
        :param item_name_chunk_size:
        :return:
        """

        self.logger.info(f"step2 构建数据识别上下文_prepare_context()")
        contexts: List[str] = []
        total: int = 0
        for index, chunk in enumerate(chunks[:item_name_chunks_k]):
            if not isinstance(chunk, dict):
                continue
            content_ = chunk.get('content', '')
            content_ = f"【切片】-{index}-{content_}"

            if total + len(contexts) > item_name_chunk_size:
                break

            contexts.append(content_)
            total += len(content_)
        return "\n".join(contexts)

    def _recognition_name(self, file_title: str, item_name_recognition_context: str):
        """
        llm识别商品名
        :param file_title:
        :param item_name_recognition_context:
        :return:
        """
        self.logger.info(f"step3 llm识别商品名_recognition_name()")
        try:

            # 1. 获取LLM客户端（不需要JSON格式）
            llm_client = AIClients.get_llm_openai(response_format=False)

            # 构建提示词工程
            user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(
                file_title=file_title,
                context=item_name_recognition_context
            )

            llm_response = llm_client.invoke(
                [HumanMessage(content=user_prompt), SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT)])

            llm_result = llm_response.content.strip()

            if not llm_result or llm_result == "UNKNOWN":
                self.logger.info(f"LLM未识别出商品名，降级使用标题: {file_title}")
                # 降级操作
                return file_title
            self.logger.info(f"LLM提取到商品名: {llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM调用失败，降级使用标题: {file_title}，异常: {e}")
            return file_title

    def _embedding_item_name(self, item_name: str) -> Tuple[List[float], Dict[int, float]]:
        """
        向量化处理
        :param item_name:
        :return:
        """
        self.logger.info(f"step4 BGE-M3向量化_embedding_item_name()")
        try:
            bge_m3_client = AIClients.get_bge_m3_client()
            vector_result = bge_m3_client.encode_documents([item_name])
            """
            vector_result['sparse'] 是一个 scipy.sparse.csr_matrix（压缩稀疏行矩阵）对象。
            csr对象包含
                indices：存放了所有非零值的位置（对应 Token ID）。
                data：存放了所有非零值的大小（对应权重）。
                indptr：一个“索引指针”，用来告诉你第几个句子的开头和结尾分别在哪。
            由于[item_name]只有一个对象 所以indptr[0]是开始位置，indptr[1]是结束位置
            
            """
            dense_vector = vector_result['dense'][0].tolist()
            start_index = vector_result['sparse'].indptr[0]
            end_index = vector_result['sparse'].indptr[1]
            token_id = vector_result['sparse'].indices[start_index:end_index].tolist()
            weight = vector_result['sparse'].data[start_index:end_index].tolist()
            # 从这个压缩包里，精准切出属于‘第一个句子’的那部分 Token ID 和权重
            sparse_vector = dict(zip(token_id, weight))

            return dense_vector, sparse_vector
        except ConnectionError as e:
            self.logger.error(f"BGE-M3 客户端获取失败: {e}")
            return None, None
        except Exception as e:
            self.logger.error(f"商品名 [{item_name}] 向量化处理失败: {e}")
            return None, None

    def _insert_milvus(self, item_name: str, file_title: str, dense_vector: List[float],
                       sparse_vector: Dict[int, float]):

        self.logger.info(f"step5 存储到Milvus   _insert_milvus()")
        if not dense_vector or not sparse_vector:
            self.logger.error(f"文档{file_title} 对应的商品名{item_name} 向量生成不完整")
            return
        item_name_collection = self.config.item_name_collection
        try:
            milvus_client = StorageClients.get_milvus_client()
        except Exception as e:
            self.logger.error(f"Milvus 客户端创建失败: {e}")
            return

        try:
            if not milvus_client.has_collection(item_name_collection):
                self._create_item_name_collection(item_name_collection, milvus_client)

            data = {
                "file_title": file_title,
                "item_name": item_name,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector
            }
            result = milvus_client.insert(collection_name=item_name_collection, data=[data])
            self.logger.info(f"已成功保存到 Milvus，ID: {result['ids'][0]}")
        except Exception as e:
            self.logger.error(f"Milvus 数据操作失败: {e}")

    def _create_item_name_collection(self, collection_name, milvus_client):
        self.logger.info(f"step5 创建集合   _create_item_name_collection()")
        schema = milvus_client.create_schema()
        schema.add_field(field_name="pk", datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_param = milvus_client.prepare_index_params()
        index_param.add_index(field_name="dense_vector", index_name="dense_vector_index",
                              index_type="AUTOINDEX", metric_type="COSINE")
        index_param.add_index(field_name="sparse_vector", index_name="sparse_vector_index",
                              index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

        milvus_client.create_collection(collection_name=collection_name,
                                        schema=schema, index_params=index_param)
        self.logger.info(f"集合 {collection_name} 创建成功并构建了索引")

    def _fill_item_name(self, item_name: str, state: ImportGraphState, chunks: List[Dict[str, Any]]):
        self.logger.info(f"step6 回填商品名   _fill_item_name()")
        for chunk in chunks:
            chunk['item_name'] = item_name

        state['item_name'] = item_name

    def _backup_chunks(self, state, chunks):
        """
        将回填item_name的切片列表结果备份到json文件，给下个节点单元测试使用。
        :param state:
        :param chunks:
        :return:
        """
        self.logger.info(f"step7 备份切片   _backup_chunks()")
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return

        os.makedirs(local_dir, exist_ok=True)  # exist_ok=True：如果目录已存在，不会抛出异常，直接跳过
        output_path = os.path.join(local_dir, "chunks_item_name.json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.logger.warning(f"备份失败：{e}")


if __name__ == '__main__':
    setup_logging()
    document_node = DocumentSplitNode()
    # 构造状态字典
    file_path = r"C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain\knowledge\test\output\万用表RS-12的使用\auto\万用表RS-12的使用_new.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    state = {
        "file_title": "hak180产品安全千册",
        "md_content": content,
        "file_dir": r"C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain\knowledge\test\output\万用表RS-12的使用\auto"}
    process = document_node.process(state)

    node = ItemNameRecognitionNode()
    node_process = node.process(process)
    print(node_process)
