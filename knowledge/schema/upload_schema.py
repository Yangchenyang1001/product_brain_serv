from typing import Any, Dict, List

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """文件上传响应 POST   /upload 返回"""
    message: str = Field(..., description="响应消息")
    task_id: str = Field(..., description="任务ID")
    document_id: str = Field(..., description="文档ID（标准UUID），切片管理接口的入口标识")


class TaskStatusResponse(BaseModel):
    """任务状态响应  GET   /status/{task_id} 返回"""
    status: str = Field(...,description="任务状态")
    done_list: List[str] = Field(...,description="已完成节点列表")
    running_list: List[str] = Field(...,description="正在运行节点列表")
    durations: Dict[str,float] = Field(default={},description="各个节点耗时(秒)")
    result: Dict[str, Any] = Field(default_factory=dict, description="任务产出（document_id、file_sha256、original_file_download_url、chunk_count、item_name、total_text_length 等）")
