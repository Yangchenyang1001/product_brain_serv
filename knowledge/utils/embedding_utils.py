import logging
from typing import Dict, List

import numpy as np
from pymilvus.model.hybrid import BGEM3EmbeddingFunction

from knowledge.utils.client.ai_clients import AIClients

logger = logging.getLogger(__name__)


def get_bge_m3_ef() -> BGEM3EmbeddingFunction:
    """
    获取BGE-M3向量模型单例
    :return: BGE-M3 embedding function
    """
    # 模型实例统一由客户端管理器缓存，避免重复加载占用显存。
    return AIClients.get_bge_m3_client()


def generate_embeddings(texts: List[str]) -> Dict[str, list]:
    """
    生成嵌入式向量（稠密 + 稀疏）
    :param texts: 待向量化的文本列表
    :return: {"dense": [...], "sparse": [...]}
    """
    model = get_bge_m3_ef()
    docs_embeddings = model.encode_documents(texts)

    # scipy稀疏行必须先生成dict，才能按非零维度做L2归一化。
    sparse_vectors = [
        {int(k): float(v) for k, v in zip(row.indices, row.data)}
        for row in docs_embeddings["sparse"]
    ]

    return {
        "dense": [emb.tolist() for emb in docs_embeddings["dense"]],
        # 稀疏向量在此统一归一化，保证写入和检索两侧的向量尺度一致。
        "sparse": [normalize_sparse_vector(vector) for vector in sparse_vectors],
    }


def normalize_sparse_vector(sparse_vec):
    """
    对稀疏向量做 L2 归一化（仅处理非零维度，不影响零维度）
    :param sparse_vec: 原始稀疏向量（dict 格式：{维度: 数值}）
    :return: 归一化后的稀疏向量
    """
    if not sparse_vec:  # 空向量直接返回
        return sparse_vec

    # 提取非零维度的数值
    values = np.array(list(sparse_vec.values()), dtype=np.float64)
    # 计算 L2 范数（避免除以 0）
    l2_norm = np.linalg.norm(values)
    if l2_norm < 1e-9:  # 范数接近 0 时，直接返回原向量（避免除零错误）
        return sparse_vec

    # 归一化：每个数值除以 L2 范数
    normalized_values = values / l2_norm
    # 重建稀疏向量 dict
    return dict(zip(sparse_vec.keys(), normalized_values))
