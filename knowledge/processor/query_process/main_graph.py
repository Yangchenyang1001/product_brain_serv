"""
查询流程图定义

使用 LangGraph 定义查询的整体处理流程图，串联各处理节点完成从用户查询到答案生成的完整链路。

流程节点顺序:
    item_name_confirm -> vector_search / hyde_search / web_search_mcp -> rrf -> rerank -> answer_output
"""
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from dotenv import load_dotenv

from knowledge.processor.query_process.base import setup_logging
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.nodes.answer_output import AnswerOutputNode
from knowledge.processor.query_process.nodes.item_name_confirm import ItemNameConfirmNode
from knowledge.processor.query_process.nodes.vector_search import VectorSearchNode
from knowledge.processor.query_process.nodes.hyde_search import HyDeSearchNode
from knowledge.processor.query_process.nodes.web_search_mcp import McpSearchNode
from knowledge.processor.query_process.nodes.rrf import RrfNode
from knowledge.processor.query_process.nodes.rerank import RerankNode

# 加载环境变量
load_dotenv()


def route_after_item_confirm(state: QueryGraphState) -> bool:
    """商品名称确认后的路由逻辑。

    根据是否已有答案决定是否跳过搜索直接输出。

    Args:
        state: 查询图状态。

    Returns:
        True 表示已有答案需要跳过搜索，False 表示继续搜索流程。
    """
    if state.get("answer"):
        return True
    return False


def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图。

    Returns:
        编译后的 StateGraph 实例。
    """
    # 1. 定义LangGraph工作流
    workflow = StateGraph(QueryGraphState)

    # 2. 实例化节点
    nodes = {
        "item_name_confirm": ItemNameConfirmNode(),
        "multi_search": lambda x: x,   # 虚拟节点（分发）
        "search_embedding": VectorSearchNode(),
        "search_embedding_hyde": HyDeSearchNode(),
        "web_search_mcp": McpSearchNode(),
        "join": lambda x: {},  # 虚拟节点（汇合）
        "rrf": RrfNode(),
        "rerank": RerankNode(),
        "answer_output": AnswerOutputNode()
    }

    # 3. 添加节点
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 4. 设置入口点
    workflow.set_entry_point("item_name_confirm")

    # 5. 添加条件边：商品名称确认后根据是否有答案路由
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {
            False: "multi_search",
            True: "answer_output"
        }
    )

    # 6. 多路搜索分发（并行执行）
    workflow.add_edge("multi_search", "search_embedding")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "web_search_mcp")

    # 7. 多路搜索汇合
    workflow.add_edge("search_embedding", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("web_search_mcp", "join")

    # 8. 顺序边
    workflow.add_edge("join", "rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge("rerank", "answer_output")
    workflow.add_edge("answer_output", END)

    # 9. 返回可运行的状态
    return workflow.compile()


# 创建全局图实例
query_app = create_query_graph()



if __name__ == '__main__':
    setup_logging()
    # 打印图结构（ASCII 可视化）

    print("图结构:")
    query_app.get_graph().print_ascii()
    final_state = None
    for event in query_app.stream({"original_query":"测试"}, stream_mode="updates"):
        for node_name, process_state in event.items():
            final_state = process_state

