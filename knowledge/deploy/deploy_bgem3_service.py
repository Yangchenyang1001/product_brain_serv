"""
BGE-M3 远程向量化服务（FastAPI + Uvicorn）

背景
----
本项目本地调用 BGE-M3 的入口是 `knowledge/utils/embedding_util.py` 中的
`generate_bge_m3_hybrid_vectors(model, embedding_documents, is_query)`，
它返回：

    {
        "dense":  [[float, ...], ...],        # 稠密向量，每条文本一个 1024 维向量
        "sparse": [{token_id: weight, ...}]   # 稀疏向量，每条文本一个 dict
    }

本服务把同样的能力发布为 HTTP 接口，**请求/响应结构与上述本地调用完全一致**，
客户端只需把本地函数调用替换为一次 HTTP 请求即可，无需改动下游解析逻辑。

接口一览
--------
GET  /health   健康检查（模型是否加载成功、运行设备）
GET  /info     服务与模型信息（向量维度、设备、批次配置）
POST /embed    向量化（稠密 + 稀疏），请求体：

    {
        "embedding_documents": ["文本1", "文本2"],   # 亦兼容字段名 "texts"
        "is_query": true                            # true=查询编码 / false=文档编码
    }

    响应体：

    {
        "dense":  [[...], [...]],
        "sparse": [{"100": 0.12, ...}, {...}],       # 注意：JSON 的 key 必须是字符串
        "dim": 1024,
        "count": 2,
        "is_query": true,
        "cost_ms": 12.3
    }

依赖安装
--------
    pip install "fastapi>=0.110" "uvicorn[standard]>=0.27" "pymilvus[model]>=2.4"

启动方式
--------
    # 方式一：直接运行（推荐，模型与 worker 一一对应）
    python deploy_bgem3_service.py

    # 方式二：uvicorn 命令行
    uvicorn deploy_bgem3_service:app --host 0.0.0.0 --port 6006 --workers 1

    # 常用环境变量
    BGE_M3_PATH=/usr-data/models/BAAI/bge-m3   模型目录
    BGE_DEVICE=cuda                            推理设备（cuda / cpu）
    BGE_FP16=true                              是否半精度
    BGE_BATCH_SIZE=16                          单次前向的批大小
    BGE_MAX_INPUT_TEXTS=128                    单请求最多文本条数

客户端调用示例（与本地返回值对齐）
----------------------------------
    import requests

    def remote_generate_bge_m3_hybrid_vectors(texts, is_query=True,
                                              url="http://127.0.0.1:6006/embed"):
        resp = requests.post(url, json={"embedding_documents": texts,
                                        "is_query": is_query}, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        return {
            "dense": body["dense"],
            # JSON 只能承载字符串 key，这里还原成 int token_id，保持与本地一致
            "sparse": [{int(k): v for k, v in item.items()} for item in body["sparse"]],
        }

注意
----
1. GPU 模型**必须单 worker** 运行，多 worker 会让每张卡重复加载一份模型。
2. 稀疏向量的 key 是 token_id，JSON 序列化后变为字符串，客户端需按上面示例转回 int。
3. 本服务默认开启 normalize_embeddings（pymilvus 默认值 True），稠密向量已 L2 归一化，
   下游可直接用内积做余弦相似度计算。
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

try:
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction
except ImportError as exc:  # 依赖缺失时给出明确指引，而不是抛出难以定位的 ImportError
    raise RuntimeError(
        "缺少 BGE-M3 依赖，请先执行: pip install -U 'pymilvus[model]'"
    ) from exc

# --------------------------------------------------------------------------- #
# 日志配置
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("bgem3-service")


# --------------------------------------------------------------------------- #
# 服务配置：全部支持环境变量覆盖，便于容器化部署
# --------------------------------------------------------------------------- #
class ServiceConfig:
    """服务运行期配置，从环境变量读取，缺省值适配 GPU 服务器现状"""

    # 模型目录：已将 BGE-M3 下载到该路径
    MODEL_PATH: str = os.getenv("BGE_M3_PATH", "/usr-data/models/BAAI/bge-m3")
    # 推理设备：cuda 优先，无 GPU 时可显式指定 cpu
    DEVICE: str = os.getenv("BGE_DEVICE", "cuda")
    # 半精度推理，显存占用约减半，精度损失可忽略
    USE_FP16: bool = os.getenv("BGE_FP16", "true").lower() in ("1", "true", "yes", "on")
    # 单次前向批大小，显存不足时调小
    BATCH_SIZE: int = int(os.getenv("BGE_BATCH_SIZE", "16"))
    # 单请求允许的最大文本条数，防止超大请求打爆显存
    MAX_INPUT_TEXTS: int = int(os.getenv("BGE_MAX_INPUT_TEXTS", "128"))
    # 服务监听地址与端口
    HOST: str = os.getenv("BGE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("BGE_PORT", "6006"))


# --------------------------------------------------------------------------- #
# 模型全局状态：进程级单例 + 串行锁
# --------------------------------------------------------------------------- #
class ModelState:
    """保存已加载的模型实例及元信息"""

    model: Optional[BGEM3EmbeddingFunction] = None
    dense_dim: int = 0
    load_error: Optional[str] = None
    # 编码锁：GPU 前向不适合并发执行，用锁把请求串行化，避免显存峰值叠加
    infer_lock: asyncio.Lock = asyncio.Lock()


# --------------------------------------------------------------------------- #
# 请求 / 响应模型：字段名与本地函数参数对齐
# --------------------------------------------------------------------------- #
class EmbedRequest(BaseModel):
    """向量化请求体

    - embedding_documents: 待向量化文本列表，对应本地函数同名参数
    - texts: 同义字段，方便习惯 OpenAI 风格（input/texts）的调用方
    - is_query: True 走 encode_queries（查询），False 走 encode_documents（入库文档）
    """

    embedding_documents: Optional[List[str]] = Field(
        default=None, description="待向量化的文本列表（与本地调用参数同名）"
    )
    texts: Optional[List[str]] = Field(
        default=None, description="embedding_documents 的同义字段"
    )
    is_query: bool = Field(
        default=True, description="True=查询编码；False=文档编码"
    )

    def resolve_texts(self) -> List[str]:
        """取出实际待编码文本：优先 embedding_documents，其次 texts"""
        return self.embedding_documents if self.embedding_documents is not None else (self.texts or [])


class EmbedResponse(BaseModel):
    """向量化响应体，dense/sparse 结构与本地函数返回值一致"""

    dense: List[List[float]] = Field(description="稠密向量，每条文本一个向量")
    sparse: List[Dict[str, float]] = Field(
        description="稀疏向量，每条文本一个 {token_id: weight}（JSON key 为字符串）"
    )
    dim: int = Field(description="稠密向量维度")
    count: int = Field(description="本次处理的文本条数")
    is_query: bool = Field(description="本次使用的编码模式")
    cost_ms: float = Field(description="服务端推理耗时（毫秒）")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str
    model_loaded: bool
    device: str
    model_path: str
    error: Optional[str] = None


class InfoResponse(BaseModel):
    """服务信息响应"""

    model_name: str
    model_path: str
    device: str
    fp16: bool
    batch_size: int
    max_input_texts: int
    dense_dim: int
    support_sparse: bool = True
    support_colbert: bool = False


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def load_model() -> BGEM3EmbeddingFunction:
    """加载 BGE-M3 模型（进程内只执行一次）"""
    logger.info(
        "开始加载 BGE-M3 | path=%s device=%s fp16=%s batch_size=%s",
        ServiceConfig.MODEL_PATH,
        ServiceConfig.DEVICE,
        ServiceConfig.USE_FP16,
        ServiceConfig.BATCH_SIZE,
    )
    started = time.time()
    model = BGEM3EmbeddingFunction(
        model_name=ServiceConfig.MODEL_PATH,
        batch_size=ServiceConfig.BATCH_SIZE,
        device=ServiceConfig.DEVICE,
        normalize_embeddings=True,
        use_fp16=ServiceConfig.USE_FP16,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    logger.info("BGE-M3 加载完成，耗时 %.2fs", time.time() - started)
    return model


def encode_sync(
    texts: List[str], is_query: bool
) -> Tuple[List[List[float]], List[Dict[str, float]]]:
    """同步执行编码，返回 (dense, sparse)，异常向上抛出由路由层统一转 HTTP 状态码

    该函数在独立线程中运行（run_in_threadpool），避免阻塞事件循环。
    """
    model = ModelState.model
    if model is None:
        raise RuntimeError("模型尚未加载完成")

    # 查询编码与文档编码走不同方法（BGE-M3 非对称检索语义）
    result = model.encode_queries(texts) if is_query else model.encode_documents(texts)

    # 1. 稠密向量：numpy.ndarray -> list
    dense = [vec.tolist() for vec in result["dense"]]

    # 2. 稀疏向量：CSR 矩阵按行还原为 {token_id: weight}
    csr = result["sparse"]
    sparse: List[Dict[str, float]] = []
    for index in range(len(texts)):
        start = csr.indptr[index]
        end = csr.indptr[index + 1]
        token_ids = csr.indices[start:end].tolist()
        weights = csr.data[start:end].tolist()
        # JSON 的 object key 必须是字符串，这里直接转 str，客户端按示例转回 int
        sparse.append({str(tid): float(weight) for tid, weight in zip(token_ids, weights)})

    return dense, sparse


def validate_texts(texts: List[str]) -> None:
    """入参校验：非空、元素为字符串且有实际内容、条数不超上限"""
    if not texts:
        raise HTTPException(status_code=400, detail="embedding_documents 不能为空")

    if len(texts) > ServiceConfig.MAX_INPUT_TEXTS:
        raise HTTPException(
            status_code=400,
            detail=f"单次请求文本条数超限：{len(texts)} > {ServiceConfig.MAX_INPUT_TEXTS}",
        )

    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"第 {index} 条文本无效（空字符串或非字符串类型）",
            )


# --------------------------------------------------------------------------- #
# 生命周期：启动时加载模型并预热
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期钩子：启动加载模型，关闭时释放显存"""
    try:
        ModelState.model = await run_in_threadpool(load_model)
        # 预热：触发 CUDA kernel 与显存分配，同时探测稠密向量维度
        probe_dense, _ = await run_in_threadpool(encode_sync, ["预热文本"], True)
        ModelState.dense_dim = len(probe_dense[0])
        ModelState.load_error = None
        logger.info("模型预热完成，dense_dim=%s", ModelState.dense_dim)
    except Exception as exc:  # 加载失败不直接退出，交由 /health 暴露，便于运维排查
        ModelState.model = None
        ModelState.load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("模型加载失败，服务将以降级状态运行")

    yield

    if ModelState.model is not None:
        ModelState.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("模型已释放")


app = FastAPI(
    title="BGE-M3 Embedding Service",
    description="BGE-M3 混合向量（稠密 + 稀疏）远程化服务",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# 路由
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, summary="健康检查")
async def health() -> HealthResponse:
    """返回模型加载状态，供 K8s / 负载均衡探活使用"""
    loaded = ModelState.model is not None
    return HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        device=ServiceConfig.DEVICE,
        model_path=ServiceConfig.MODEL_PATH,
        error=ModelState.load_error,
    )


@app.get("/info", response_model=InfoResponse, summary="服务与模型信息")
async def info() -> InfoResponse:
    """返回模型配置与向量维度，便于客户端做参数自检"""
    return InfoResponse(
        model_name="BAAI/bge-m3",
        model_path=ServiceConfig.MODEL_PATH,
        device=ServiceConfig.DEVICE,
        fp16=ServiceConfig.USE_FP16,
        batch_size=ServiceConfig.BATCH_SIZE,
        max_input_texts=ServiceConfig.MAX_INPUT_TEXTS,
        dense_dim=ModelState.dense_dim,
    )


@app.post("/embed", response_model=EmbedResponse, summary="文本向量化（稠密 + 稀疏）")
async def embed(request: EmbedRequest) -> EmbedResponse:
    """与本地 generate_bge_m3_hybrid_vectors 对齐的向量化接口"""
    if ModelState.model is None:
        raise HTTPException(
            status_code=503,
            detail=f"模型不可用：{ModelState.load_error or '正在加载中，请稍后重试'}",
        )

    texts = request.resolve_texts()
    validate_texts(texts)

    started = time.time()
    try:
        # 串行化 GPU 推理，避免并发请求叠加显存峰值
        async with ModelState.infer_lock:
            dense, sparse = await run_in_threadpool(encode_sync, texts, request.is_query)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("向量化失败")
        raise HTTPException(status_code=500, detail=f"向量化失败: {exc}") from exc

    cost_ms = round((time.time() - started) * 1000, 2)
    logger.info(
        "embed 完成 | count=%s is_query=%s dim=%s cost=%.2fms",
        len(texts),
        request.is_query,
        len(dense[0]) if dense else 0,
        cost_ms,
    )

    return EmbedResponse(
        dense=dense,
        sparse=sparse,
        dim=len(dense[0]) if dense else 0,
        count=len(texts),
        is_query=request.is_query,
        cost_ms=cost_ms,
    )


if __name__ == "__main__":
    # 注意：workers 必须为 1，GPU 模型在多 worker 下会被重复加载
    uvicorn.run(
        app,
        host=ServiceConfig.HOST,
        port=ServiceConfig.PORT,
        workers=1,
        log_level="info",
    )
