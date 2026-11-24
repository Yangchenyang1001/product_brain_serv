# knowledge/processor/import_process/config.py

"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from typing import Set, Optional
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000  # 切片最大长度
    img_content_length: int = 200  # 图片上下文最大长度
    min_content_length: int = 500  # 合并短内容的最小长度
    overlap_sentences: int = 1  # 句子级切分时重叠句数
    item_name_chunk_k: int = 3  # 商品名识别时使用的切片数量
    item_name_chunk_size: int = 2500  # 商品名识别时使用的切片内容长度

    # 支持的图片扩展名
    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )

    # ==================== MinerU 配置 ====================
    # 网络版 PDF 解析使用（pdf_to_md_net 节点）
    mineru_base_url: str = field(
        default_factory=lambda: os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4")
    )
    mineru_api_token: str = field(
        default_factory=lambda: os.getenv("MINERU_API_TOKEN", "")
    )
    mineru_model_version: str = field(
        default_factory=lambda: os.getenv("MINERU_MODEL_VERSION", "pipeline")
    )
    mineru_language: str = field(
        default_factory=lambda: os.getenv("MINERU_LANGUAGE", "ch")
    )
    mineru_enable_formula: bool = field(
        default_factory=lambda: os.getenv("MINERU_ENABLE_FORMULA", "true").lower() in ("1", "true", "yes")
    )
    mineru_enable_table: bool = field(
        default_factory=lambda: os.getenv("MINERU_ENABLE_TABLE", "true").lower() in ("1", "true", "yes")
    )
    mineru_poll_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("MINERU_POLL_INTERVAL_SECONDS", "5"))
    )
    mineru_max_wait_seconds: int = field(
        default_factory=lambda: int(os.getenv("MINERU_MAX_WAIT_SECONDS", "1800"))
    )
    mineru_request_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("MINERU_REQUEST_TIMEOUT_SECONDS", "120"))
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )

    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = False
    # 预签名下载链接有效期（天）
    minio_presign_expiry_days: int = field(
        default_factory=lambda: int(os.getenv("MINIO_PRESIGN_EXPIRY_DAYS", "7"))
    )
    # 对外访问地址（如 http://192.168.1.10:9000）：MINIO_ENDPOINT 是容器内/本机地址时，
    # 预签名链接里的 host 会是 127.0.0.1，Java 侧打不开；配置此项后改用该地址签名。
    # 注意 SigV4 签名包含 Host，不能签完再替换域名，必须用该地址的客户端来签。
    minio_public_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_PUBLIC_ENDPOINT", "")
    )

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 8

    # ==================== 速率限制 ====================
    requests_per_minute: int = 15  # 图片总结 API 速率限制 时间滑动窗口

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()

    def get_minio_base_url(self) -> str:
        """获取 MinIO 基础 URL"""
        protocol = "https://" if self.minio_secure else "http://"
        return protocol + f"{self.minio_endpoint}"


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
