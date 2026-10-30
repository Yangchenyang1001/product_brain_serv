"""
商品确认节点

利用 LLM 从用户查询中识别并确认商品名称，作为后续向量检索和 Web 搜索的过滤维度。
"""
import json
import logging
import re
from json import JSONDecodeError
from typing import Dict, Any, List, Tuple, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import AnnSearchRequest

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.config import QueryConfig
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.prompt.query_prompt import ITEM_NAME_EXTRACT_TEMPLATE
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.base import logger
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.utils.mongo_history_util import get_recent_messages


class ItemNameExtractor:
    """
    商品名称提取器
    用于从用户查询中提取商品名称，并生成重写查询。
    """

    def extract_item_name(self, original_query: str, history_text: str) -> Dict[str, Any]:
        """
        商品名称提取  根据用户查询和历史会话，提取商品名称并生成重写查询。
        :param original_query: 原查询
        :param history_text: 历史会话
        :return: 商品名称列表和重写查询结果
                {
                    "item_names": List[str],  # 商品名称列表
                    "rewritten_query": str  # 重写查询
                }

        """

        result = {"item_names": [], "rewritten_query": original_query}
        # 1. 获取llm客户端
        llm = AIClients.get_llm_openai(response_format=True)

        if llm is None:
            logger.error(f"获取llm客户端失败")
            return result

        # 2. 组装提示词
        human_prompt = ITEM_NAME_EXTRACT_TEMPLATE.format(
            history_text=history_text if history_text else "暂无上下文",
            query=original_query
        )
        system_prompt = "你是一个专业的客服助手，擅长理解用户意图和提取关键信息。"

        # 3. LLM 调用
        logging.info("LLM 调用")
        llm_response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        llm_content = llm_response.content.strip()

        # 4. 判断 LLM 的输出
        if not llm_content:
            return result

        try:
            # 5. 清洗和解析
            parsed_result = self._clean_parse(llm_content)
            result["rewritten_query"] = parsed_result.get("rewritten_query") or original_query
            result["item_names"] = parsed_result.get("item_names") or ""
            logging.info(f"清洗以及解析LLM的输出结果: {result}")

        except Exception as e:
            logging.error(f"清洗以及解析LLM的输出失败: {str(e)}")

        return result

    def _clean_parse(self, llm_content: str) -> Dict[str, Any]:
        """清洗并解析 LLM 响应"""
        # 1. 清洗 json 代码块围栏
        cleaned = re.sub(r"^```(?:json)?\s*", "", llm_content.strip())
        content = re.sub(r"\s*```$", "", cleaned)

        # 2. 反序列化
        logging.info(f"清洗以及解析LLM的输出: {content}")
        try:
            parsed_llm_result: Dict[str, Any] = json.loads(content)
            # 2.1 清洗 item_names
            rwa_item_names = parsed_llm_result.get('item_names')
            if not isinstance(rwa_item_names, list):
                clean_item_names = []
            else:
                clean_item_names = [raw_item for raw_item in rwa_item_names if raw_item.strip()]

            # 2.2 清洗 rewritten_query
            raw_rewritten_query = parsed_llm_result.get('rewritten_query')
            clean_rewritten_query = "" if not isinstance(raw_rewritten_query, str) else raw_rewritten_query.strip()

            return {"item_names": clean_item_names, "rewritten_query": clean_rewritten_query}
        except JSONDecodeError as e:
            raise ValueError(f"JSON反序列LLM的输出失败：{str(e)}")


class ItemNameAligner:
    def _match_vector(self, item_names: List[str] | Any, collection_name: str) -> List[Dict[str, Any]]:
        """
        向量匹配  根据商品名称列表，匹配向量数据库，返回匹配结果。
        :param item_names: 商品名称列表
        :return: 匹配结果列表
        [
            {
                "extracted_name": str,  # 提取的商品名
                "matches": [
                    {"item_name": str, "score": float}  # 匹配的商品名和分数
                ]
            }
        ]
        """

        logger.info(f"item_names: {item_names}")
        if not item_names:
            return []

        # 1. 定义最终收缩结果
        search_results: List[Dict[str, Any]] = []
        # 2. 获取milvus客户端
        client = StorageClients.get_milvus_client()
        if client is None:
            logger.error(f"获取milvus客户端失败")
            return search_results
        # 3. 获取嵌入模型
        embedding_model = AIClients.get_bge_m3_client()
        if embedding_model is None:
            logger.error(f"获取embeddingModel嵌入模型失败")
            return search_results
        # 4. 嵌入item_names 获取稠密 稀疏向量
        hybrid_embedding_result = generate_bge_m3_hybrid_vectors(embedding_model, item_names)

        # 5. 遍历LLM提取所有商品名
        for index, extract_item_name in enumerate(item_names):
            # 5.1 创建混合检索的请求
            hybrid_search_requests: List[AnnSearchRequest] = create_hybrid_search_requests(
                dense_vector=hybrid_embedding_result['dense'][index],
                sparse_vector=hybrid_embedding_result['sparse'][index],
            )
            # 5.2 执行混合检索
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=client,
                collection_name=collection_name,
                search_requests=hybrid_search_requests,
                ranker_weights=(0.5, 0.5),
                norm_score=True,
                output_fields=["item_name"]
            )
            # 5.3 解析混合检索请求的结果对象
            item_name_search_result = {
                "extracted_name": extract_item_name,
                "matches": [
                    {"item_name": h["entity"]["item_name"], "score": h["distance"]}
                    for h in (hybrid_search_result[0] if hybrid_search_result else [])
                ]
            }

            # 5.4 将构建好的查询结果放入最终搜索结果中
            search_results.append(item_name_search_result)

        return search_results

    def _item_name_score_align(self, search_results: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        # 1. 定义两个容器
        confirmed = []
        options = []  # 最多只留下 3 个
        # 2. 遍历向量数据库查询到所有的结果
        for item_name_search_result in search_results:
            # 2.0 获取提取的商品名
            extracted_name = item_name_search_result.get("extracted_name")
            # 2.1 对某一商品名下找到相似的item_name按照分数降序
            matches = sorted(
                item_name_search_result.get("matches", []),
                key=lambda x: x["score"],
                reverse=True
            )
            # 2.2 获取 matches 中分数值能进入 confirmed 容器阈值的对象
            high = [match for match in matches if match["score"] > 0.7]

            # 3. 将 high 中的对象放入 confirmed 容器中
            if high:
                # 3.1 准备找最精准的那一个
                extract = next(
                    (h for h in high if str(h['item_name']) == extracted_name),
                    None
                )

                # 场景 A: 找到了精确匹配
                if extract:
                    picked = extract["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 场景 B: 只有一条高置信结果
                elif len(high) == 1:
                    picked = high[0]["item_name"]
                    if picked not in confirmed:
                        confirmed.append(picked)
                # 场景 C: 多条相似，加入 options 让用户选择
                else:
                    for h in high[:3]:
                        picked = h.get('item_name')
                        if picked not in options and picked not in confirmed:
                            options.append(picked)

            # 4. 询问是否能进入 options 中
            else:
                mid = [
                    m for m in matches
                    if m['score'] >= 0.6
                       and m.get('item_name') not in options
                       and m.get('item_name') not in confirmed
                ]
                if mid:
                    for m in mid[:3]:
                        picked = m.get('item_name')
                        options.append(picked)

        return confirmed, options[:3]

    def _item_name_score_filter(self, confirmed: List[str], search_results: List[Dict[str, Any]]) -> List[str]:
        if not confirmed or not search_results:
            return confirmed
        item_name_score = {}
        for search_result in search_results:
            matches: List = search_result.get("matches")
            for m in matches:
                item_name = m.get("item_name")
                score = m.get("score")
                if item_name in confirmed:
                    item_name_score[item_name] = max(item_name_score.get(item_name) or 0, score)
        if not item_name_score:
            return confirmed

        sorted_item_name_score = sorted(item_name_score.items(), key=lambda x: x[1], reverse=True)
        # 3. 取出分数值最大的（作为基准）
        max_item_name_score = sorted_item_name_score[0][1]
        # 4. 保留分数差在阈值内的商品名
        return [name for name, score in item_name_score.items() if max_item_name_score - score <= 0.15]


class ItemNameConfirmNode(BaseNode):
    name: str = "item_name_confirm_node"

    def __init__(self, config: Optional[QueryConfig] = None):
        super().__init__(config)
        self.item_name_aligner = ItemNameAligner()
        self.item_name_extractor = ItemNameExtractor()

    def _decide(self, state: QueryGraphState, item_names: List[str], confirmed: List[str], options: List[str],
                rewritten_query: str):
        """根据对齐结果更新 state"""

        if confirmed:
            # 确认成功：更新状态，继续检索流程
            state['rewritten_query'] = rewritten_query
            state['item_names'] = confirmed

        elif options:
            # 有候选：设置选择提示，流程中断
            state['answer'] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            # 无法识别：设置提示，流程中断
            state['answer'] = "抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        1. step1 获取历史对话 get_recent_messages()
          a. 访问mongodb获取集合列表   chat_message 获取历史对话
          b. 构建history_text
        2. step2 LLM提取商品名称
          a. 输入original_query + history_text
          b. 输出item_names + rewritten_query
          c. 实现商品名提取和指代消除
        3. step3 向量匹配
          a. 输入items_names   执行混合向量检索  输出search_result
            ⅰ. BGEM3生成混合向量, Milvus查询结果
        4. step4 评分对齐
          a. search_result 按照score降序排序通过置信度确定confirmed和options, 判断用户提问的商品是已确认商品, 还是提示选择的商品
        5. step5 分数过滤差异
          a. 计算每个商品的最高分, 保留max_score- score < 0.15的商品
        6. step6 决策分支
          a. confirmed有值
            ⅰ. 确认成功, 继续检索执行历史回填
          b. options有值
            ⅰ. 选择提示直接返回结果
          c. 都无值
            ⅰ. 无法识别直接返回结果
        7. step7 历史回填
          a. confirmed有值时, 将历史对话中item_names为空的消息批量更新 update_message_item_names()
        :param state:
        :return:
        """

        # step1 获取历史对话
        sessionId = state.get("session_id")
        original_query = state.get("original_query")
        # 从mongodb中获取历史的会话列表    10轮对话
        chat_history = get_recent_messages(sessionId, limit=10)
        # 规范化：ObjectId 转 str，避免后续 json 序列化报错
        chat_history = [
            {**msg, "_id": str(msg.get("_id", ""))} for msg in chat_history
        ]

        # step1.1  构建历史对话文本
        history_text = ""
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("text", "")
            history_text += f"{role}: {content}\n"
        logging.info(f"step1 获取历史对话: {history_text}")

        # step2 LLM提取商品名称
        extractor_item_names_dict: Dict[str, Any] = self.item_name_extractor.extract_item_name(original_query,
                                                                                               history_text)
        if extractor_item_names_dict is None:
            logging.info(f"step2 LLM无法提取商品名称")
            return state
        item_names = extractor_item_names_dict.get("item_names")
        rewritten_query = extractor_item_names_dict.get("rewritten_query")
        # step3 向量匹配
        result: List[Dict[str, Any]] = self.item_name_aligner._match_vector(item_names,
                                                                            self.config.item_name_collection)
        logging.info(f"向量匹配: {json.dumps(result, ensure_ascii=False, indent=2)}\n")
        # step4 评分对齐
        confirmed, options = self.item_name_aligner._item_name_score_align(result)
        logging.info(f"评分对齐确认: {json.dumps(confirmed, ensure_ascii=False, indent=2)}\n")
        logging.info(f"评分对齐选择: {json.dumps(options, ensure_ascii=False, indent=2)}\n")
        #
        # # step5 分数过滤差异
        confirmed = self.item_name_aligner._item_name_score_filter(confirmed, result)
        logging.info(f"分数过滤差异: {json.dumps(confirmed, ensure_ascii=False, indent=2)}\n")
        #
        # # step6 决策分支
        self._decide(state, item_names, confirmed, options, rewritten_query)
        # step7 历史回填
        state["history"] = chat_history
        logging.info(f"最终结果: {json.dumps(state, ensure_ascii=False, indent=2, default=str)}\n")
        return state


if __name__ == '__main__':
    setup_logging()
    test_state: QueryGraphState = {
        "session_id": "test_session_id",
        # "original_query": "RS-12 数字万用表怎么测试电阻？以及华为擎云L420 用户手册 中包含操作环境嘛？"
        "original_query": "RS-12数字万用表怎么测试电阻？以及H3C LA2608中包含操作环境嘛？"
        # "original_query": "今天天气怎么样?"
    }
    print(f"输入: {json.dumps(test_state, ensure_ascii=False, indent=2)}\n")

    node_item_name_confirm = ItemNameConfirmNode()
    result = node_item_name_confirm.process(test_state)
