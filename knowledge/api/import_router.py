"""
导入服务路由 (port 8000)

路由接口:
    - POST /upload             上传文件
    - GET  /status/{task_id}    查询任务状态
    - GET    /knowledge-units/{document_id}/chunks                 分页查询知识切片
    - POST   /knowledge-units/{document_id}/chunks                 新增知识切片
    - PUT    /knowledge-units/{document_id}/chunks/{chunk_id}      修改知识切片
    - DELETE /knowledge-units/{document_id}/chunks/{chunk_id}      删除知识切片
    - DELETE /knowledge-units/{document_id}/chunks                 删除文档全部切片
"""
import logging
import os

import uvicorn
from fastapi import FastAPI, BackgroundTasks, UploadFile, File, Depends, status
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from knowledge.core.deps import get_import_file_service, get_knowledge_units_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.import_process.base import setup_logging
from knowledge.schema.upload_schema import UploadResponse, TaskStatusResponse
from knowledge.schema.task_schema import KnowledgeChunkPageResponse, DeleteDocumentChunksResponse, \
    KnowledgeChunkMutationResponse, ChunkWriteRequest, DeleteChunkResponse
from knowledge.services.file_import_service import ImportFileService
from knowledge.services.knowledge_units_service import KnowledgeUnitsService
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
        # 1. 上传文件（本地/minio），上传阶段生成 task_id 与 document_id
        upload_context = service.process_upload_file(file)

        # 2. 运行后台任务（跑graph的整个流程），document_id 随图状态传递到入库节点
        background_tasks.add_task(service.run_import_graph, upload_context)

        # 3. 返回：document_id 是后续切片管理接口的入口标识
        return UploadResponse(
            message="文件上传成功",
            task_id=upload_context["task_id"],
            document_id=upload_context["document_id"],
        )

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

    @app.get("/knowledge-units/{document_id}/chunks", response_model=KnowledgeChunkPageResponse)
    async def page_chunks_endpoint(
            document_id: str,
            page_num: int = 1,
            page_size: int = 10,
            service: KnowledgeUnitsService = Depends(get_knowledge_units_service),
    ):
        """分页查询知识单元的主切片。"""
        return service.page_chunks(document_id, page_num, page_size)

    @app.post(
        "/knowledge-units/{document_id}/chunks",
        response_model=KnowledgeChunkMutationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def add_chunk_endpoint(
            document_id: str,
            request: ChunkWriteRequest,
            service: KnowledgeUnitsService = Depends(get_knowledge_units_service),
    ):
        """在知识单元末尾新增主切片。"""
        return service.add_chunk(document_id, request)

    @app.delete(
        "/knowledge-units/{document_id}/chunks",
        response_model=DeleteDocumentChunksResponse,
    )
    async def delete_document_chunks_endpoint(
            document_id: str,
            service: KnowledgeUnitsService = Depends(get_knowledge_units_service),
    ):
        """按文档ID物理删除知识单元的全部主切片。"""
        return service.delete_document_chunks(document_id)

    @app.put(
        "/knowledge-units/{document_id}/chunks/{chunk_id}",
        response_model=KnowledgeChunkMutationResponse,
    )
    async def update_chunk_endpoint(
            document_id: str,
            chunk_id: str,
            request: ChunkWriteRequest,
            service: KnowledgeUnitsService = Depends(get_knowledge_units_service),
    ):
        """修改指定主切片。"""
        return service.update_chunk(document_id, chunk_id, request)

    @app.delete("/knowledge-units/{document_id}/chunks/{chunk_id}", response_model=DeleteChunkResponse)
    async def delete_chunk_endpoint(
            document_id: str,
            chunk_id: str,
            service: KnowledgeUnitsService = Depends(get_knowledge_units_service),
    ):
        """删除指定主切片。"""
        return service.delete_chunk(document_id, chunk_id)


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
