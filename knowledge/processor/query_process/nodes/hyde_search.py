"""
HyDE 检索节点

利用 LLM 根据用户查询生成假设性文档（HyDE），再以假设文档作为查询向量进行检索，提升召回质量。
"""
import json
from typing import Tuple, List

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import AnnSearchRequest

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.exceptions import StateFieldError
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.hyde_prompt import USER_HYDE_PROMPT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors

from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query


class HyDeSearchNode(BaseNode):
    name: str = "hyde_search_node"

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

    def _generate_hy_document(self, validated_query: str, validate_item_names: List[str]) -> str:
        """使用 LLM 生成假设性文档"""

        # 1. 获取 LLM 客户端
        llm_client = AIClients.get_llm_openai(False)

        # 2. 判断
        if llm_client is None:
            return ""

        # 3. 获取系统提示词以及用户提示词
        user_prompt = USER_HYDE_PROMPT_TEMPLATE.format(
            item_hint=validate_item_names,
            rewritten_query=validated_query
        )
        system_prompt = f"您是一位{validate_item_names}的技术文档领域的专家，主要擅长编写技术文档、操作手册、文档规格说明"

        try:
            # 4. 获取 AIMessage
            llm_response = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # 5. 获取内容
            llm_response_content = getattr(llm_response, 'content', "").strip()

            # 6. 判断是否存在
            if not llm_response_content:
                return ""

            return llm_response_content

        except Exception as e:
            self.logger.error(f"LLM调用失败:{str(e)}")
            return ""

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        1. step1 参数校验
        2. step2 大模型生成假设性文档
        3. step3 获取依赖+文档拼接+向量化
        4. step4 构建过滤条件
        5. step5 创建搜索请求+执行检索
        6. step6 返回结果
        :param state:
        :return:
        """
        # step1 参数校验

        rewritten_query, item_names = self._validate_query_inputs(state)
        # step2 大模型生成假设性文档
        hyde_document = self._generate_hy_document(rewritten_query, item_names)
        # step3 获取依赖+文档拼接+向量化
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端获取失败 原因:{str(e)}")
            return {"hyde_embedding_chunks": []}

        try:
            embedding_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型获取失败 原因:{str(e)}")
            return {"hyde_embedding_chunks": []}

        if not embedding_client or not milvus_client:
            return {"hyde_embedding_chunks": []}

        try:
            embed_query = generate_bge_m3_hybrid_vectors(embedding_client,
                                                         embedding_documents=[f"{rewritten_query}\n{hyde_document}"],
                                                         is_query=False)
        except Exception as e:
            self.logger.error(f"问题{rewritten_query}嵌入失败")
            return {"hyde_embedding_chunks": []}
        if not embed_query:
            return {"hyde_embedding_chunks": []}

        """构建商品名过滤表达式"""
        quoted = ", ".join(f'"{v}"' for v in item_names)
        expr = f" item_name in [{quoted}]"
        try:
            hybrid_requests: List[AnnSearchRequest] = create_hybrid_search_requests(
                dense_vector=embed_query['dense'][0],
                sparse_vector=embed_query['sparse'][0],
                expr=expr,
                limit=5
            )
            reps = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_requests,
                norm_score=True,
                output_fields=["chunk_id", "content", "item_name", 'title']
            )

            if not reps or not reps[0]:
                return {"hyde_embedding_chunks": []}
            return {"hyde_embedding_chunks": reps[0]}
        except Exception as e:
            self.logger.error(f"混合检索失败 原因:{str(e)}")
            return {"hyde_embedding_chunks": []}


if __name__ == '__main__':
    state = {
        "rewritten_query": "华为擎云 W525需要注意什么",
        "item_names": ["华为擎云 W525"]
    }

    hyde_nodoe = HyDeSearchNode()
    result = hyde_nodoe.process(state)

    for r in result.get('hyde_embedding_chunks', []):
        print(json.dumps(r, ensure_ascii=False, indent=2))
