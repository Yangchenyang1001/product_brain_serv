"""
文件导入服务

封装文件导入的完整业务逻辑，协调导入流程图的执行、任务状态管理和异常处理。
"""
import datetime
import logging
import os
import shutil
import uuid

from fastapi import UploadFile

from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import FileProcessingError, MinioError
from knowledge.processor.import_process.main_graph import run_import_graph
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.task_util import update_task_status, add_running_task, add_done_task, TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED

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

    def process_upload_file(self, file: UploadFile):
        # 生成任务ID
        task_id = self.generate_task_id()
        # 文件上传存储路径
        date_path = self._get_date_path(get_local_base_dir())
        file_dir = os.path.join(date_path, task_id)

        os.makedirs(file_dir, exist_ok=True)

        add_running_task(task_id,"upload_file")
        # 双写
        # 保存到本地
        import_file_path = self._upload_file_to_local(file, file_dir)
        # 保存到minIO   file.name用于存储的子路径
        self._upload_file_to_minio(import_file_path, file.filename)

        add_done_task(task_id,"upload_file")

        return task_id, file_dir, import_file_path

    def generate_task_id(self):
        return uuid.uuid4().hex[:8]

    """
    运行导入流程图
    """

    def run_import_graph(self, task_id, file_dir, import_file_path):

        try:
            inputState = {
                "import_file_path": import_file_path,
                "file_dir": file_dir,
                "task_id": task_id
            }
            update_task_status(task_id, TASK_STATUS_PROCESSING)
            run_import_graph(inputState)
            logger.info(f"{task_id}导入流程完成")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
        except Exception as e:
            update_task_status(task_id, TASK_STATUS_FAILED)
            logger.error(f"{task_id}导入流程失败{e}")
        logger.info(f"{task_id}导入流程完成")


    def _get_date_path(self, local_base_dir) -> str:
        return os.path.join(local_base_dir, datetime.datetime.now().strftime("%Y%m%d"))

    def _upload_file_to_local(self, file: UploadFile, file_dir:str) -> str:
        """
        上传的文件保存到本地
        :param file:
        :param file_dir:
        :return:
        """
        # 确保文件目录存在
        os.makedirs(file_dir, exist_ok=True)
        # 保存文件
        try:
            # 构建文件的路径
            file_path = os.path.join(file_dir, file.filename)
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)  # 在缓冲区的写操作, 效率更高1MB
        except IOError as e:
            logger.error(f"文件保存失败: {e}")
            raise FileProcessingError(f"{file.filename}文件保存失败: {e}")

        return file_path

    def _upload_file_to_minio(self, import_file_path: str, file_name: str) -> str:
        """
        文件保存到minIO
        :param import_file_path: 上传的文件路径
        :param file_name: 文件名
        :return: 保存的文件路径
        """
        try:
            client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.warning(f"连接MinIO失败: {e}")
            # 获取客户端失败不影响后续流程
            return

        try:
            config = get_config()
            object_name = f"origin_files/{datetime.datetime.now().strftime('%Y%m%d')}/{file_name}  "
            client.fput_object(bucket_name=config.minio_bucket, object_name=object_name,
                                             file_path=import_file_path)
        except MinioError as e:
            logger.warning(f"MinIO 文件上传失败: {e}")
            return

        return ""
