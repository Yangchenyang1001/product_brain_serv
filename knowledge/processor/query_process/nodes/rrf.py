"""
RRF 融合节点

使用倒数排名融合（Reciprocal Rank Fusion）算法，将向量检索、HyDE 检索和 Web 搜索的多路召回结果进行融合排序。
"""
from typing import List, Dict, Any, Tuple, Optional

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState


class RrfNode(BaseNode):
    name: str = "rrf_node"

    def __init__(self):
        super().__init__()
        self._top_k = self.config.rrf_max_results
        self._rrf_k = self.config.rrf_k

    def _normalize_input(self, rrf_input: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        统一处理各路检索到的结果

        Args:
            rrf_input: 各路不同数据结构的检索结果

        Returns:
            统一处理后的标准数据结构的检索结果
        """
        diff_path_result = []

        # 1. 遍历各路搜索结果
        if not rrf_input:
            return []

        # 2. 遍历该路的所有结果
        for doc in rrf_input:
            # 2.1 判断是否有效
            if not isinstance(doc, dict):
                continue

            # 2.2 获取 entity
            entity = doc.get('entity')
            if not entity:
                continue

            diff_path_result.append(entity)

        return diff_path_result

    def _rrf_merge(self, rrf_inputs, _rrf_k: int, _top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        chunk_scores = {}  # 存放所有的chunk的rrf计算后的分数
        chunk_data = {}  # 存放所有的chunk的数据

        for rrf_input, weight in rrf_inputs:
            for i, doc in enumerate(rrf_input, 1):
                chunk_id = doc.get('chunk_id')
                if not chunk_id:
                    continue
                # RRF 公式: score += weight / (k + rank)
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / _rrf_k * i
                # 使用 setdefault 保留首次遇到的文档版本
                chunk_data.setdefault(chunk_id, doc)
        # 按得分降序排序，截取前 top_k 条
        sorted_results = sorted(
            [(chunk_data[cid], score) for cid, score in chunk_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        # step5 排序截断
        return sorted_results[:_top_k] if _top_k else sorted_results

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        1. step 1获取各路检索结果
        2. step2 格式规整化
        3. step3 设置权重
        4. step4 rrf计算
        5. step5 排序截断
        6. step6 更新状态并返回
        :param state:
        :return:
        """
        # step 1 获取各路检索结果
        vector_search_results = state.get("embedding_chunks", [])
        hyde_search_results = state.get("hyde_embedding_chunks", [])
        # step 2 格式规整化

        rrf_pre_vectors = self._normalize_input(vector_search_results)
        rrf_pre_hyde_vectors = self._normalize_input(hyde_search_results)

        # step 3 设置权重
        search_source = {
            "vector_search_results": (rrf_pre_vectors, 1.0),
            "hyde_search_results": (rrf_pre_hyde_vectors, 1.0),
        }
        #  构建 rrf_inputs
        rrf_inputs = list(search_source.values())
        # step4 rrf计算
        merge = self._rrf_merge(rrf_inputs, self._rrf_k, self._top_k)

        rrf_chunks = [doc for doc, _ in merge]
        self.logger.info(f"RRF 融合完成，返回 {len(rrf_chunks)} 条结果")

        if merge:
            scores = [s for _, s in merge]
            self.logger.info(f"分数范围: [{min(scores):.6f}, {max(scores):.6f}]")

        # 7. 更新 state
        state['rrf_chunks'] = rrf_chunks

        # 8. 返回 state
        return state



if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: RRF 融合节点")
    print("=" * 60)

    # 模拟两路检索结果
    # chunk_1 命中 2 路（预期最高分）
    # chunk_2 命中 2 路
    # chunk_3, chunk_4 各命中 1 路
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": "chunk_1", "content": "向量搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_2", "content": "向量搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_3", "content": "向量搜索结果#4"}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": "chunk_2", "content": "HyDE搜索结果#1"}},
            {"entity": {"chunk_id": "chunk_1", "content": "HyDE搜索结果#2"}},
            {"entity": {"chunk_id": "chunk_4", "content": "HyDE搜索结果#3"}},
        ],
    }

    print("【输入状态】:")
    print(f"  embedding_chunks: {len(mock_state['embedding_chunks'])} 条")
    print(f"  hyde_embedding_chunks: {len(mock_state['hyde_embedding_chunks'])} 条")
    print("-" * 60)

    rrf_node = RrfNode()
    result = rrf_node.process(mock_state)

    print("\n【融合结果】:")
    for i, chunk in enumerate(result["rrf_chunks"], 1):
        print(f"[{i}] {chunk.get('chunk_id')} - {chunk.get('content')}")

    print("-" * 60)
    print("测试完成")