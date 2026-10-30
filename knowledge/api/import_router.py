"""
导入服务路由 (port 8000)

路由接口:
    - POST /upload             上传文件
    - GET  /status/{task_id}    查询任务状态
"""
import logging
import os

import uvicorn
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Depends
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_import_file_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.import_process.base import setup_logging
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.services.file_import_service import ImportFileService
from knowledge.utils.task_util import get_task_info


def register_router(app: FastAPI):
    @app.get("/import")
    async def import_root():
        return FileResponse(path=os.path.join(get_front_page_dir(), "import.html"))
    """
        1. 文件上传处理由业务层处理, 需要创建业务层类. 采用依赖注入的方式创建业务层对象, 并单例创建,并且缓存重复利用
        2. 异步启动langgraph流程
    """

    @app.post("/upload", response_model=UploadResponse)
    async def upload_file(
            background_tasks: BackgroundTasks,
            service: ImportFileService = Depends(get_import_file_service),  # 依赖注入
            file: UploadFile = File(...),

    ):
        # 1. 上传文件（本地/minio）
        task_id, file_dir, import_file_path = service.process_upload_file(file)

        # 同步启动
        # service.run_import_graph(task_id, file_dir, import_file_path)
        # 2. 运行后台任务（跑graph的整个流程）

        background_tasks.add_task(service.run_import_graph, task_id, file_dir, import_file_path)

        # 3. 返回
        return UploadResponse(message="文件上传成功", task_id=task_id)

    @app.get("/status/{task_id}", response_model=TaskStatusResponse)
    async def get_status(task_id: str):
        """
        1. 在文件上传与启动langgraph时, 需要记录各个任务的状态
        2. 根据任务的taskid, 查询任务状态
        3. 状态维护容器 task_util.py

        :param task_id:
        :return:
        """
        info = get_task_info(task_id)
        return TaskStatusResponse(**info)


def create_app():
    app = FastAPI(description="知识库导入", version="v1.0")
    app.add_middleware(
        CORSMiddleware,  # 跨域设置
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,  # 允许凭证
        allow_methods=["*"],  # 允许所有方法
        allow_headers=["*"],  # 允许所有头
    )
    # 前端静态资源挂载
    front_page_dir = get_front_page_dir()
    if front_page_dir and not os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app


if __name__ == '__main__':
    setup_logging(level=logging.DEBUG)
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000)
