"""
路径常量配置

集中管理项目中所有文件路径常量，包括数据目录、临时目录、前端页面路径等。
"""
import os.path

KNOWLEDGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")
FRONT_PAGE_DIR = os.path.join(KNOWLEDGE_ROOT, "front")


def get_local_base_dir() -> str:
    return LOCAL_BASE_DIR


def get_front_page_dir() -> str:
    return FRONT_PAGE_DIR
