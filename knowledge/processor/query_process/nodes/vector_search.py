"""
向量检索节点

基于商品名和用户查询在 Milvus 向量库中进行相似度检索，召回相关文档块。
"""
import json
from typing import Tuple, List

from pymilvus import AnnSearchRequest

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors

from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class VectorSearchNode(BaseNode):
    name: str = "vector_search_node"

    def _validate_query_inputs(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验输入参数"""

        # 1. 获取 state 的 rewritten_query
        rewritten_query = state.get('rewritten_query', "")

        # 2. 获取 state 的 item_names
        item_names = state.get('item_names', "")

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="rewritten_query",
                expected_type=str
            )

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(
                node_name=self.name,
                field_name="item_names",
                expected_type=list
            )

        # 4. 返回
        return rewritten_query, item_names

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        1. step1 参数校验

        2. step2 获取依赖
        3. step3 查询向量化
        4. step4 构建过滤条件
        5. step5 创建混合搜索请求
        6. step6 执行混合检索
        7. step7 结束返回
        :param state:
        :return:
        """
        # step1 参数校验
        rewritten_query, item_names = self._validate_query_inputs(state)
        # step2 获取依赖
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return {"embedding_chunks": []}

        try:
            embedding_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            return {"embedding_chunks": []}

        # step3 查询向量化
        try:
            embed_query = generate_bge_m3_hybrid_vectors(embedding_client, embedding_documents=[rewritten_query])
        except Exception as e:
            self.logger.error(f"问题{rewritten_query}嵌入失败")
            return {"embedding_chunks": []}

        # step4 构建过滤条件
        # 使用过滤表达式模板，将动态值从表达式中分离，优化中日韩字符的查询性能
        expr = "item_name in {item_names}"
        filter_params = {"item_names": item_names}
        try:
            # step5 创建混合搜索请求
            hybrid_requests: List[AnnSearchRequest] = create_hybrid_search_requests(
                dense_vector=embed_query['dense'][0],
                sparse_vector=embed_query['sparse'][0],
                expr=expr,
                expr_params=filter_params,
                limit=5
            )
            # step6 执行混合检索
            reps = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_requests,
                norm_score=True,
                output_fields=["chunk_id", "content", "item_name", 'title']
            )

            if not reps or not reps[0]:
                return {"embedding_chunks": []}
            return {"embedding_chunks": reps[0]}
        except Exception as e:
            self.logger.error(f"混合检索失败 原因:{str(e)}")
            return {"embedding_chunks": []}


if __name__ == '__main__':
    state = {
        "rewritten_query": "华为擎云 W525需要注意什么",
        "item_names": ["华为擎云 W525"]
    }

    vector_search = VectorSearchNode()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))
