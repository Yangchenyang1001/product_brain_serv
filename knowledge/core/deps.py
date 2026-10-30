"""
依赖注入（单例管理）

提供全局单例对象的依赖注入管理，确保服务实例在应用生命周期内只创建一次。
"""
from functools import cache, lru_cache

from knowledge.services.file_import_service import ImportFileService
from knowledge.services.query_service import QueryService


@cache  # 缓存重复利用 长期有效
# @lru_cache  # 缓存重复利用 短期有效根据lru算法清理缓存
def get_import_file_service():
    """
    获取导入文件服务实例  放入缓存取重复利用
    :return:
    """
    return ImportFileService()


@lru_cache
def get_query_service() -> QueryService:
    return QueryService()
