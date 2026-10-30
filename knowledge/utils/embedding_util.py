import math
from typing import List, Dict, Any
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


def generate_bge_m3_hybrid_vectors(model: BGEM3EmbeddingFunction, embedding_documents: List[str],
                                   is_query: bool = True):
    """
    为文本生成混合向量嵌入（稠密 + 稀疏）
    Args:
        embedding_model: BGE-M3嵌入模型
        embedding_documents: 要生成嵌入的文本列表
    Returns:
        {"dense": [...], "sparse": [...]}
    Raises:
        ValueError: 输入参数无效
        RuntimeError: 嵌入生成失败
    """
    # 1. 参数校验
    if not embedding_documents:
        raise ValueError("embedding_documents 不能为空")

    if not all(isinstance(doc, str) and doc.strip() for doc in embedding_documents):
        raise ValueError("embedding_documents 中存在无效元素（空字符串或非字符串类型）")

    # 2. 生成嵌入
    """
    encode_queries vs encode_documents 区别
    BGE-M3 采用非对称检索策略，两个方法会在文本前添加不同的指令前缀（instruction prefix），让模型知道当前编码的是"问题"还是"文档"，从而生成更适合匹配的向量。
    方法               用途          适用场景         输入参数名
    ----------------------------------------------------------
    encode_queries   编码查询文本    用户搜索问题      queries
    encode_documents 编码文档内容    入库存储的文档    documents
    """
    try:
        if is_query:
            embedding_result = model.encode_queries(embedding_documents)
        else:
            embedding_result = model.encode_documents(embedding_documents)
    except Exception as e:
        raise RuntimeError(f"BGE-M3 嵌入生成失败: {e}") from e

    # 3. 校验嵌入结果
    if 'dense' not in embedding_result or 'sparse' not in embedding_result:
        raise RuntimeError(f"嵌入结果缺少必要字段，实际返回: {list(embedding_result.keys())}")

    # 5. 解析稀疏向量（CSR 矩阵 → dict）
    try:
        processed_sparse = []
        csr_array = embedding_result['sparse']

        for index in range(len(embedding_documents)):
            start = csr_array.indptr[index]
            end = csr_array.indptr[index + 1]
            token_ids = csr_array.indices[start:end].tolist()
            weights = csr_array.data[start:end].tolist()
            processed_sparse.append(dict(zip(token_ids, weights)))
    except (IndexError, AttributeError) as e:
        raise RuntimeError(f"稀疏向量解析失败（CSR 矩阵结构异常）: {e}") from e

    # 6. 返回
    return {
        "dense": [den.tolist() for den in embedding_result["dense"]],
        "sparse": processed_sparse
    }


# def generate_hybrid_embeddings(
#         embedding_model,
#         embedding_documents: List[str]
# ) -> Dict[str, Any]:
#     """
#     为一组文本生成混合嵌入（稠密向量 + 稀疏向量）
#
#     Args:
#         embedding_model: BGE-M3 模型实例
#         embedding_documents: 待嵌入的文本列表
#
#     Returns:
#         {
#             "dense": [[...], ...],      # 稠密向量列表
#             "sparse": [{...}, ...]      # 稀疏向量字典列表
#         }
#     """
#     # 1. 调用模型推理
#     raw_embeddings = embedding_model(embedding_documents)
#
#     # 2. 转换稠密向量
#     dense_vectors = [emb.tolist() for emb in raw_embeddings["dense"]]
#
#     # 3. 提取并归一化稀疏向量
#     sparse_vectors = _extract_sparse_vectors(raw_embeddings, len(embedding_documents))
#
#     return {
#         "dense": dense_vectors,
#         "sparse": sparse_vectors,
#     }

# def _extract_sparse_vectors(raw_embeddings, text_count) -> List[Dict[int, float]]:
#     """从 CSR 稀疏矩阵中提取稀疏向量"""
#
#     sparse_matrix = raw_embeddings["sparse"]
#     sparse_vectors = []
#
#     for i in range(text_count):
#         # CSR 矩阵按行存储，通过 indptr 定位每行的起止位置
#         row_start = sparse_matrix.indptr[i]
#         row_end = sparse_matrix.indptr[i + 1]
#
#         # 提取该行的非零元素：{token_id: weight}
#         sparse_dict = dict(zip(
#             sparse_matrix.indices[row_start:row_end].tolist(),
#             sparse_matrix.data[row_start:row_end].tolist(),
#         ))
#
#         # 归一化
#         sparse_vectors.append(normalize_sparse_vector(sparse_dict))
#
#     return sparse_vectors
#
#
# def normalize_sparse_vector(sparse_vector: Dict[int, float]) -> Dict[int, float]:
#     """
#     对稀疏向量进行 L2 归一化，使向量模长为 1
#
#     Args:
#         sparse_vector: 原始稀疏向量 {token_id: weight}
#
#     Returns:
#         归一化后的稀疏向量 {token_id: weight}；
#         当输入为空或全零（模长为 0）时，原样返回，避免除零错误
#     """
#     # 1. 空向量直接返回
#     if not sparse_vector:
#         return {}
#
#     # 2. 计算 L2 范数：sqrt(Σ weight^2)
#     l2_norm = math.sqrt(sum(weight * weight for weight in sparse_vector.values()))
#
#     # 3. 范数为 0 时无法归一化，原样返回
#     if l2_norm == 0.0:
#         return sparse_vector
#
#     # 4. 每个权重除以范数
#     return {token_id: weight / l2_norm for token_id, weight in sparse_vector.items()}
