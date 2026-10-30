"""
向量嵌入节点

使用 BGE-M3 模型将切分后的文本块转换为向量表示，为存入 Milvus 向量库做准备。
"""
import json
from pathlib import Path
from typing import Dict, List, Any

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.client.ai_clients import AIClients


class BgeEmbeddingChunksNode(BaseNode):
    name: str = "bge_embedding_node"

    def process(self, state: ImportGraphState) -> ImportGraphState | dict:
        """
          a. step1 参数校验 _validate_get_inputs()
            ⅰ. 获取chunks并校验有效性
            ⅱ. 获取批次大小 embedding_batch_size
          b. step2 分批循环处理
            ⅰ. step2.1 构造嵌入内容 item_name+content
            ⅱ. step2.2调用模型向量化并提取稠密和稀疏向量
            ⅲ. 注入到chunks
          c. step3 合并所有批次 更新state
        :param state:
        :return:
        """
        chunks, embedding_batch_size = self._validate_get_inputs(state)
        final_chunks = []  # 最终返回结果
        total_len = len(chunks)  # 切片总数
        self.logger.info(f"step2 分批循环处理:总数{total_len}")
        for i in range(0, total_len, embedding_batch_size):
            self.logger.info(f"step2 第{i}批:总数{total_len}，开始处理")
            # 子切片集合 长度是 embedding_batch_size
            chunkSub = chunks[i:i + embedding_batch_size]
            # 列表推导式 构造嵌入内容 item_name+content
            chunk_full = [f"{chunk.get('item_name')}\n{chunk.get('content')}"
                          for chunk in chunkSub]

            try:
                bge_m3_client = AIClients.get_bge_m3_client()
                # 生成向量
                vector_result = bge_m3_client.encode_documents(documents=chunk_full)

                # 如果向量结果不存在，则跳过
                if not vector_result:
                    self.logger.warning(f"嵌入后的结果不存在...")
                    continue
            except Exception as e:
                self.logger.warning(f"嵌入向量嵌入失败...{str(e)}")
                continue

            # 提取稀疏向量
            sparse_vector = []
            for index, chunk in enumerate(chunkSub):
                dense_vector = vector_result['dense'][index].tolist()  # 稠密向量
                sparse_result = vector_result['sparse']  # 稀疏向量结果
                start_index = sparse_result.indptr[index]  # 稀疏向量的起始索引
                end_index = sparse_result.indptr[index + 1]  # 稀疏向量的结束索引
                token_id = sparse_result.indices[start_index:end_index].tolist()  # 稀疏向量的token_id
                weight = sparse_result.data[start_index:end_index].tolist()  # 稀疏向量的权重
                sparse_vector.append(dict(zip(token_id, weight)))

                chunkSub[index]["dense_vector"] = dense_vector
                chunkSub[index]["sparse_vector"] = sparse_vector[index]

                self.logger.info(f"step2 初始下标为{i}的第{index}个chunk处理完成")

            # 扩展最终结果
            final_chunks.extend(chunkSub)

        state["chunks"] = final_chunks
        return state;

    def _validate_get_inputs(self, state) -> tuple[list[dict], int]:
        self.logger.info("step1 参数校验 _validate_get_inputs()")
        chunks = state.get("chunks", [])
        if not chunks or not isinstance(chunks, list):
            raise ValidationError(f"chunks为空或者无效", self.name)
        size = get_config().embedding_batch_size

        return chunks, size
if __name__ == '__main__':
    setup_logging()
    base_temp_dir = Path(r"C:\Users\43779\Desktop\AIlearnrecord\shopkeeper_brain\knowledge\test\output\万用表RS-12的使用\auto")

    input_path = base_temp_dir / "chunks_item_name.json"
    output_path = base_temp_dir / "chunks_vector.json"

    # 1. 读取上游状态
    if not input_path.exists():
        print(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    # 2. 构建模拟的图状态
    state = {
        "chunks": content
    }

    # 3. 触发节点执行
    node_bge_embedding = BgeEmbeddingChunksNode()
    proceed_result = node_bge_embedding.process(state)

    # 4. 结果落盘
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(proceed_result, f, ensure_ascii=False, indent=4)

    print(f"向量生成测试完成！结果已成功备份至:\n{output_path}")