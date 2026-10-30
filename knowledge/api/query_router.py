"""
查询服务路由 (port 8001)

路由接口:
    - POST /query              发起查询
    - GET  /stream/{session_id}  SSE 流式获取
    - GET  /history/{session_id} 获取历史
    - DELETE /history/...      清除历史
"""
import asyncio
import logging
import os

import uvicorn
from fastapi import FastAPI, BackgroundTasks, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from knowledge.core.deps import get_query_service
from knowledge.core.paths import get_front_page_dir
from knowledge.processor.query_process.base import setup_logging
from knowledge.schema.query_schema import QueryResponse, StreamSubmitResponse
from knowledge.schema.query_schema import QueryRequest
from knowledge.services.query_service import QueryService
from knowledge.utils.sse_util import create_sse_queue, sse_generator
from knowledge.utils.task_util import get_task_result


def register_router(app):
    @app.get("/chat")
    def chat():
        return FileResponse(path=os.path.join(get_front_page_dir(), "chat.html"))

    @app.post("/query", response_model=QueryResponse | StreamSubmitResponse)
    async def query(query_request: QueryRequest,
                    background_tasks: BackgroundTasks,
                    service: QueryService = Depends(get_query_service)):
        session_id = query_request.session_id or service.generate_session_id()
        task_id = service.generate_task_id()
        is_stream = query_request.is_stream  # 是否是流式输出 默认False
        original_query = query_request.query
        if is_stream:
            # 创建SSE队列   7个节点的状态/ llm大模型结果
            create_sse_queue(task_id)
            # 异步流式输出
            background_tasks.add_task(service.run_query_graph, original_query, session_id, task_id, is_stream)
            return StreamSubmitResponse(message="查询流程已启动", task_id=task_id, session_id=session_id)

        else:
            # 非流式输出 同步
            """
            async def 方式 协程里调同步阻塞函数，会卡住这条线程，连累所有其他协程
            所以这里使用 asyncio.get_running_loop() 获取当前协程的事件循环，然后使用 loop.run_in_executor() 方法来执行同步阻塞函数
            
            
            为什么 Java 转过来容易踩坑?
            Java 里你写 servlet 或者 Spring MVC 的 controller，不管同步异步，Tomcat 都是借一条线程给你跑，所以你在 controller 里放心调 restTemplate.getForObject()（同步阻塞），无非是这条线程等 I/O，其他线程照常服务——你习惯了"阻塞无所谓，反正有线程池兜底"。
            到了 FastAPI，如果你写成 async def 然后照样调同步阻塞方法，没有那个兜底线程池了（因为你自己声明了 async，框架以为你会规矩地 await，就把你放在宝贵的单线程事件循环上）。这一脚踩下去不是"一个线程阻塞"而是"整条循环阻塞"。这就是为什么 async def 里乱调同步代码，在 Python 里比 Java 里后果严重得多。

            """

            loop = asyncio.get_running_loop()
            # executor=None 表示默认采用线程池处理任务
            await loop.run_in_executor(None, service.run_query_graph, original_query, session_id, task_id, is_stream)
            # service.run_query_graph(original_query, session_id, task_id, is_stream)
            answer = service.get_task_result(task_id, "answer")
        return QueryResponse(message="最终答案生成", session_id=session_id, answer=answer)

    @app.get("/stream/{task_id}")
    async def stream(task_id: str, request: Request
                     ) -> StreamingResponse:
        return StreamingResponse(sse_generator(task_id, request), media_type="text/event-stream")

    @app.get("/history/{session_id}")
    async def get_history(
            session_id: str, limit: int = 50,
            service: QueryService = Depends(get_query_service),
    ):
        try:
            items = service.get_history(session_id, limit)
            return {"session_id": session_id, "items": items}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"history error: {e}")

    @app.delete("/history/{session_id}")
    async def clear_chat_history(
            session_id: str,
            service: QueryService = Depends(get_query_service),
    ):
        count = service.clear_history(session_id)
        return {"message": "History cleared", "deleted_count": count}


def create_app():
    app = FastAPI(description="导入流程", version=1.0)

    app.add_middleware(CORSMiddleware,
                       allow_origins=["*"],
                       allow_credentials=False,
                       allow_methods=["*"],
                       allow_headers=["*"])

    front_page_dir = get_front_page_dir()
    if front_page_dir and os.path.exists(front_page_dir):
        app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)
    return app


if __name__ == '__main__':
    setup_logging(level=logging.INFO)
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8001)
