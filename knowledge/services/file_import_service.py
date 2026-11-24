"""
文件导入服务

封装文件导入的完整业务逻辑，协调导入流程图的执行、任务状态管理和异常处理。
"""
import datetime
import hashlib
import logging
import os
import uuid
from typing import Dict, Optional, Tuple

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError
from knowledge.processor.import_process.main_graph import run_import_graph
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import update_task_status, add_running_task, add_done_task, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, set_task_result

logger = logging.getLogger(__name__)


class ImportFileService:
    """
    处理上传文件的业务层类
    """

    def __init__(self):
        pass

    """
    处理上传文件的业务逻辑
    """

    def process_upload_file(self, file: UploadFile) -> Dict[str, str]:
        # 1. 生成任务ID：仅用于流程追踪与 /status 轮询
        task_id = self.generate_task_id()
        # 2. 生成文档ID：主切片的归属边界，后续查询/编辑/删除切片都以它为入口
        document_id = self.generate_document_id()
        # 文件上传存储路径
        date_path = self._get_date_path(get_local_base_dir())
        file_dir = os.path.join(date_path, task_id)

        os.makedirs(file_dir, exist_ok=True)

        add_running_task(task_id,"upload_file")
        # 双写
        # 保存到本地（同一次写入顺带算出 SHA-256，避免二次读盘）
        import_file_path, file_sha256 = self._upload_file_to_local(file, file_dir)
        # 保存到minIO，返回对象名用于生成下载链接
        object_name = self._upload_file_to_minio(import_file_path, file.filename, task_id)
        original_file_download_url = self._build_download_url(object_name)

        add_done_task(task_id,"upload_file")

        # 上传阶段就能确定的产出：摘要 + 原始文件下载链接（Java 侧 fileSha256 / originalFileDownloadUrl）
        set_task_result(task_id, "file_sha256", file_sha256)
        set_task_result(task_id, "original_file_download_url", original_file_download_url or None)

        # 3. 上传上下文：document_id 随图状态一路带到入库节点
        return {
            "task_id": task_id,
            "document_id": document_id,
            "file_dir": file_dir,
            "import_file_path": import_file_path,
        }

    def generate_task_id(self):
        return uuid.uuid4().hex[:8]

    def generate_document_id(self) -> str:
        """生成标准 UUID 字符串，切片管理接口按 UUID 校验，不能使用截断 ID。"""
        return str(uuid.uuid4())

    """
    运行导入流程图
    """

    def run_import_graph(self, upload_context: Dict[str, str]):
        task_id = upload_context["task_id"]
        document_id = upload_context["document_id"]
        # 文档ID先落库，导入失败时前端也能拿到该文档标识用于排障
        set_task_result(task_id, "document_id", document_id)

        try:
            inputState = {
                "import_file_path": upload_context["import_file_path"],
                "file_dir": upload_context["file_dir"],
                "task_id": task_id,
                "document_id": document_id,
            }
            update_task_status(task_id, TASK_STATUS_PROCESSING)
            final_state = run_import_graph(inputState)
            # 回写入库结果：切片数与文档ID构成该任务的产出
            final_chunks = (final_state or {}).get("chunks") or []
            set_task_result(task_id, "chunk_count", len(final_chunks))
            set_task_result(task_id, "item_name", (final_state or {}).get("item_name") or "")
            set_task_result(task_id, "total_text_length", sum(len(c.get("content") or "") for c in final_chunks))
            logger.info(f"{task_id}导入流程完成，document_id={document_id}，切片数={len(final_chunks)}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger.error(f"{task_id}导入流程失败{e}")


    def _get_date_path(self, local_base_dir) -> str:
        return os.path.join(local_base_dir, datetime.datetime.now().strftime("%Y%m%d"))

    def _upload_file_to_local(self, file: UploadFile, file_dir:str) -> Tuple[str, str]:
        """
        上传的文件保存到本地，同时流式计算 SHA-256
        :param file:
        :param file_dir:
        :return: (本地文件路径, 文件内容的 sha256 小写十六进制摘要)
        """
        # 确保文件目录存在
        os.makedirs(file_dir, exist_ok=True)
        # 保存文件
        digest = hashlib.sha256()
        try:
            # 构建文件的路径
            file_path = os.path.join(file_dir, file.filename)
            with open(file_path, "wb") as f:
                # 分块读写（1MB），一边落盘一边喂给摘要器，64位十六进制结果与小写约定一致
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    f.write(chunk)
        except IOError as e:
            logger.error(f"文件保存失败: {e}")
            raise FileProcessingError(f"{file.filename}文件保存失败: {e}")

        return file_path, digest.hexdigest()

    def _upload_file_to_minio(self, import_file_path: str, file_name: str, task_id: str) -> Optional[str]:
        """
        文件保存到minIO
        :param import_file_path: 上传的文件路径
        :param file_name: 文件名
        :param task_id: 任务ID，作为对象路径的一段，避免同名文件互相覆盖
        :return: MinIO 对象名；上传失败返回 None
        """
        try:
            client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.warning(f"连接MinIO失败: {e}")
            # 获取客户端失败不影响后续流程
            return None

        try:
            config = get_config()
            date_path = datetime.datetime.now().strftime('%Y%m%d')
            # 路径带上任务ID：同一天重复上传同名文件不会覆盖前一份
            object_name = f"origin_files/{date_path}/{task_id}/{file_name}"
            client.fput_object(bucket_name=config.minio_bucket, object_name=object_name,
                                             file_path=import_file_path)
        except Exception as e:
            logger.warning(f"MinIO 文件上传失败: {e}")
            return None

        return object_name

    def _build_download_url(self, object_name: Optional[str]) -> str:
        """
        为已上传对象生成预签名下载链接（Java 侧 originalFileDownloadUrl）
        :param object_name: MinIO 对象名
        :return: 预签名 URL；对象名为空或签名失败返回空串
        """
        if not object_name:
            return ""
        try:
            config = get_config()
            client = StorageClients.get_minio_presign_client()
            return client.presigned_get_object(
                bucket_name=config.minio_bucket,
                object_name=object_name,
                expires=datetime.timedelta(days=config.minio_presign_expiry_days),
            )
        except Exception as e:
            logger.warning(f"生成 MinIO 下载链接失败: {e}")
            return ""
