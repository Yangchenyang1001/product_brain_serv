"""
检索服务

封装查询检索的完整业务逻辑，协调查询流程图的执行、会话管理和 SSE 流式推送。
"""
import uuid
import logging
from typing import List, Dict, Any
from knowledge.processor.query_process.main_graph import query_app
from knowledge.utils.task_util import update_task_status,get_task_result, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED,TASK_STATUS_FAILED
from knowledge.utils.sse_util import push_sse_event, SSEEvent


class QueryService:

    def generate_session_id(self):
        return str(uuid.uuid4())

    def generate_task_id(self):
        return str(uuid.uuid4())

    def run_query_graph(self, query, session_id, task_id, is_stream):

        """
            启动查询流程图
        """
        final_state = None
        try:
            init_state = {
                "original_query": query,
                "session_id": session_id,
                "task_id": task_id,
                "is_stream": is_stream
            }

            update_task_status(task_id, TASK_STATUS_PROCESSING)
            # 虽然langgraph流程是非流式输出, 但是答案生成节点输出流式, 通过sse生成器 将输出从队列中取走
            final_state = query_app.invoke(init_state)

            update_task_status(task_id, TASK_STATUS_COMPLETED)

        except Exception as e:
            logging.exception(f"查询流程执行失败: {e}")
            update_task_status(task_id, TASK_STATUS_FAILED)
            # 流式场景：推一个 FINAL 终止事件，避免前端 SSE 生成器死循环等待
            if is_stream:
                push_sse_event(task_id, SSEEvent.FINAL, {"answer": "查询处理失败，请稍后重试。"})

        return final_state

    def get_task_result(self, task_id, param):
        return get_task_result(task_id, param)

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        from knowledge.utils.mongo_history_util import get_recent_messages
        records = get_recent_messages(session_id, limit=limit)
        records.reverse()
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id", ""),
                "role": r.get("role", ""),
                "text": r.get("text", ""),
                "rewritten_query": r.get("rewritten_query", ""),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]

    def clear_history(self, session_id: str) -> int:
        from knowledge.utils.mongo_history_util import clear_history
        return clear_history(session_id)

