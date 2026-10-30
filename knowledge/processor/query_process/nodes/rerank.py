"""
重排序节点

使用 Reranker 模型对 RRF 融合后的候选文档进行二次精排，提高最终送入答案生成的文档相关性。
"""
from typing import Dict, Any, List

from knowledge.processor.query_process.base import BaseNode, setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankNode(BaseNode):
    name: str = "rerank_node"

    def _cliff_catoff(self, reranked_docs: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        断崖检测动态截断  第一断崖处理
        扫描所有相邻文档分数差，找到最大落差位置进行截断，
        扫描所有相邻文档分数差，找到最大落差位置进行截断，
        同时保证至少返回 lower_bound 个文档。

        参数:
            reranked_docs: 按分数降序排列的文档列表
            rerank_min_top_k: 最少返回文档数
            rerank_max_top_k: 最多返回文档数

        返回值:
            截断后的文档列表
        """
        upper_bound = min(self.config.rerank_max_top_k, len(reranked_docs))
        lower_bound = min(self.config.rerank_min_top_k, upper_bound)

        if upper_bound <= 1:
            return reranked_docs[:upper_bound]

        cut_off = upper_bound
        max_gap = 0.0

        for i in range(0, upper_bound - 1):
            current_score = reranked_docs[i].get("score")
            next_score = reranked_docs[i + 1].get("score")

            if current_score is None or next_score is None:
                continue

            # 分数差值
            gap = current_score - next_score

            if gap >= self.config.rerank_gap_abs:
                cut_off = i + 1
                self.logger.info(f"位置{cut_off}发生断崖，gap={gap:.4f}")
                break
        # 兜底：不管断崖在哪，至少保留lower_bound个
        cut_off = max(cut_off, lower_bound)
        return reranked_docs[:cut_off]  # 左闭开区间：包含起始位置，不包含结束位置

    def _format_rrf_docs(self, content: str, title: str = "", chunk_id=None,
                         url: str = "", source: str = "") -> Dict[str, Any]:
        """构建统一的文档结构"""
        return {
            "content": content,  # 文档内容（必填）
            "title": title,  # 标题
            "chunk_id": chunk_id,  # 本地文档 ID
            "url": url,  # 网页链接（仅 web）
            "source": source  # 来源: local / web
        }

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        重排序节点
        1. step1 获取查询文本
        2. step2 合并本地RRF结果
        3. step3 合并网络搜索结果
        4. step4 构建query-document
        5. step5 Reranker计算得分
        6. step6 断崖检测动态截断
        7. step7 更新状态并返回
        :param state:
        :return:
        """
        # step1 获取查询文本
        user_query = state.get('rewritten_query', '') or state.get('original_query', '')
        # step2 合并本地RRF结果
        rrf_web_docs = []

        for rrf_doc in (state.get("rrf_chunks") or []):
            if not isinstance(rrf_doc, dict):
                continue
            content = rrf_doc.get('content', '').strip()
            if not content:
                continue
            title = rrf_doc.get("title", "").strip()
            chunk_id = rrf_doc.get("chunk_id", "")

            format_rrf_doc = self._format_rrf_docs(
                content=content, title=title, chunk_id=chunk_id, source="local"
            )
            rrf_web_docs.append(format_rrf_doc)
        # step3 合并网络搜索结果
        for web_doc in (state.get('web_search_docs') or []):
            if not isinstance(web_doc, dict):
                continue

            content = web_doc.get('content', '') or web_doc.get('snippet', '').strip()
            if not content:
                continue

            title = web_doc.get('title', '').strip()
            url = web_doc.get('url', '').strip()

            format_web_doc = self._format_rrf_docs(
                content=content, title=title, url=url, source="web"
            )
            rrf_web_docs.append(format_web_doc)
        self.logger.info(f"收集到准备进行 Rerank 精排的文档 {len(rrf_web_docs)}")
        # step4 构建query-document
        if not rrf_web_docs:
            return state
        # 获取模型客户端
        rerank_client = AIClients.get_bge_m3_rerank_client()
        if rerank_client is None:
            self.logger.warning("Rerank client is None")
            return state
        # 构建Q->D的pair对
        query_doc_content_pairs = [(user_query, doc.get('content')) for doc in rrf_web_docs]
        # 计算分数并归一化
        # step5 Reranker计算得分
        try:
            reranker_scores = rerank_client.compute_score(sentence_pairs=query_doc_content_pairs, normalize=True)
            # 单文档防护  由于reranker_scores的长度为1时，会返回浮点数, 其他情况返回列表
            if isinstance(reranker_scores, (float, int)):
                reranker_scores = [reranker_scores]

            # 映射分数和文档
            final_score_doc = [{**doc, "score": score} for doc, score in zip(rrf_web_docs, reranker_scores)]

            # 排序并返回
            final_score_doc.sort(key=lambda x: x['score'], reverse=True)
        except Exception as e:
            self.logger.error(f"Rerank 重排序失败: {str(e)}")
            final_score_doc = [{**doc, "score": None} for doc in rrf_web_docs]

        cutoff_docs = self._cliff_catoff(final_score_doc)

        state['reranked_docs'] = cutoff_docs

        return state


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankNode()
    result = node.process(mock_state)

    print("\n【重排序结果】:")
    for i, doc in enumerate(result["reranked_docs"], 1):
        score = doc.get('score')
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"[{i}] score={score_str} | {doc['source']:5} | {doc['content'][:50]}...")

    print("-" * 60)
    print("测试完成")
