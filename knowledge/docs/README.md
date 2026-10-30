# 掌柜大脑 (shopkeeper_brain)

## 项目简介

掌柜大脑是一个基于 LangGraph 的智能知识库系统，支持文档导入和智能查询两大核心功能。

## 架构概览

```
knowledge/
├── api/          # API 路由层
├── core/         # 核心配置
├── processor/    # 业务处理流程（LangGraph）
│   ├── import_process/   # 导入流程
│   └── query_process/    # 查询流程
├── schema/       # 数据模型定义
├── services/     # 业务服务层
├── utils/        # 工具函数库
├── prompt/       # 提示词
├── front/        # 前端页面
├── test/         # 测试代码
├── docs/         # 文档目录
├── temp_data/    # 临时数据目录
├── .env          # 环境配置文件
└── requirements.txt  # Python 依赖声明
```

## 核心功能

### 1. 导入流程 (port 8000)

文档上传 -> PDF转MD -> 图片处理 -> 文档切分 -> 商品识别 -> 向量嵌入 -> Milvus存储

### 2. 查询流程 (port 8001)

用户查询 -> 商品确认 -> 多路检索(向量/HyDE/Web) -> RRF融合 -> 重排序 -> 答案生成(SSE)
