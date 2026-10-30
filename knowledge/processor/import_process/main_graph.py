"""
导入流程图定义

使用 LangGraph 定义文档导入的整体处理流程图，串联各处理节点完成从文件上传到向量入库的完整链路。

流程节点顺序:
    entry -> pdf_to_md -> md_img -> document_split -> item_name_recognition -> bge_embedding -> import_milvus
"""
import json
import logging

from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.pdf_to_md import PdfToMdNode
from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.md_img import MarkDownImageNode
from knowledge.processor.import_process.nodes.document_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode

from knowledge.processor.import_process.state import ImportGraphState, create_default_state


def load_file_router(state: ImportGraphState):
    """
    入口节点后的路由逻辑

    根据文件类型决定走 PDF 转换分支还是直接处理 MD 分支

    Args:
        state: 当前图状态

    Returns:
        下一个节点名称
    """
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    if state.get("is_md_read_enabled"):
        return "md_img_node"
    return "END"


def create_import_graph() -> CompiledStateGraph:
    """
    创建导入流程图

    Returns:
        编译后的 StateGraph 实例

    流程结构:
        entry_node
              │
              ├── (PDF) ──> pdf_to_md_node ──┐
              │                              │
              └── (MD) ─────────────────────>├──> md_img_node
                                              │
                                              v
                                      document_split_node
                                              │
                                              v
                                      item_name_rec_node
                                              │
                                              v
                                        bge_embedding_node
                                              │
                                              v
                                        import_milvus_node
                                              │
                                              v
                                             END
    """
    # 1. 定义状态, 已经定义过 state.py
    # 2. 定义节点  参考nodes目录
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_rec_node": ItemNameRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    # 3. 创建图实例
    import_graph = StateGraph(ImportGraphState)

    for key, node in nodes.items():
        import_graph.add_node(key, node)
        print(f"key {key}, value {node}")

    # 4.  添加边
    import_graph.set_entry_point("entry_node")
    # 4.1 条件边设置
    import_graph.add_conditional_edges(
        "entry_node",
        load_file_router,
        {
            "pdf_to_md_node": "pdf_to_md_node",
            "md_img_node": "md_img_node",
            "END": END
        }
    )

    import_graph.add_edge("pdf_to_md_node", "md_img_node")
    import_graph.add_edge("md_img_node", "document_split_node")
    import_graph.add_edge("document_split_node", "item_name_rec_node")
    import_graph.add_edge("item_name_rec_node", "bge_embedding_node")
    import_graph.add_edge("bge_embedding_node", "import_milvus_node")
    import_graph.add_edge("import_milvus_node", END)

    # 5. 编译节点
    graph_compiled = import_graph.compile()

    return graph_compiled


# 启动函数
def run_import_graph(inputState: ImportGraphState) -> ImportGraphState | None:
    """
    inputstate  仅仅需要输入
        import_file_path: str  # 导入文件路径（原始输入）
        file_dir: str  # 导入(出)文件目录
    :param inputState:
    :return:
    """
    # 获取编译后的图对象
    compiled_graph = create_import_graph()

    # 打印图结构（ASCII 可视化）
    print("图结构:")
    compiled_graph.get_graph().print_ascii()

    initState = create_default_state(**inputState)

    final_state = None
    for event in compiled_graph.stream(initState, stream_mode="updates"):
        for node_name, process_state in event.items():
            final_state = process_state

    return final_state


if __name__ == '__main__':
    setup_logging(level=logging.DEBUG)  # 配置日志，否则节点里的 logger.info/debug 不会输出

    import_file_path = r"C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain_docs\doc\华为擎云 W525 用户指南-(PGUW-WBX0,KOS&UOS_02,zh-cn).pdf"
    file_dir = r"C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain\knowledge\test\output"
    # input_state = {"import_file_path": import_file_path, "file_dir": file_dir}

    graph = run_import_graph(
        inputState={
            "import_file_path": import_file_path,
            "file_dir": file_dir,
        }
    )

    # print(json.dumps(graph, indent=2, ensure_ascii=False))
