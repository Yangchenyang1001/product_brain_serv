"""
统一服务路由 (port 8000)

由 import_router.py (原 port 8000) 与 query_router.py (原 port 8001) 合并而来：
直接复用两个原有路由模块的 register_router，在同一个 FastAPI 应用、同一个端口 8000 上提供全部接口。

路由接口:
    导入/知识切片: 见 knowledge.api.import_router
    查询:         见 knowledge.api.query_router
"""
import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from knowledge.core.paths import get_front_page_dir
from knowledge.processor.import_process.base import setup_logging

from knowledge.api.import_router import register_router as register_import_router
from knowledge.api.query_router import register_router as register_query_router


def register_router(app: FastAPI):
    """注册全部路由（导入 + 查询）。"""
    register_import_router(app)
    register_query_router(app)


def create_app():
    app = FastAPI(description="知识库服务（导入 + 查询）", version="v1.0")
    app.add_middleware(
        CORSMiddleware,  # 跨域设置
        allow_origins=["*"],  # 允许所有来源
        allow_credentials=True,  # 允许凭证
        allow_methods=["*"],  # 允许所有方法
        allow_headers=["*"],  # 允许所有头
    )
    # 前端静态资源挂载
    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app


if __name__ == '__main__':
    setup_logging(level=logging.DEBUG)
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000)
