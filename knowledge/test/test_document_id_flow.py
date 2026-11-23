"""
document_id 业务闭环测试

覆盖四个环节:
    Part 1  上传阶段: 生成标准 UUID、返回上传上下文
    Part 2  任务阶段: document_id 注入图状态、产出回写任务结果、响应契约暴露 document_id
    Part 3  入库阶段: 写入 document_id/chunk_order、非法与缺失场景显式失败
    Part 4  查询阶段: 通过 /knowledge-units/{document_id}/chunks 读回刚导入的切片

全部使用临时集合, 结束后自动 drop, 不改动主集合数据。

运行方式:
    .\\knowledge\\.venv\\Scripts\\python.exe knowledge\\test\\test_document_id_flow.py
"""
import io
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

# Windows 控制台默认 GBK, 统一改用 UTF-8 避免中文输出报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.getLogger("httpx").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(KNOWLEDGE_DIR / ".env")

import random

from fastapi.testclient import TestClient
from pymilvus import DataType
from starlette.datastructures import UploadFile

import knowledge.services.file_import_service as file_import_service
from knowledge.api.import_router import create_app
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ImportProcessError
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
from knowledge.schema.upload_schema import UploadResponse
from knowledge.utils import task_util
from knowledge.utils.client.storage_clients import StorageClients

TEMP_COLLECTION = f"kb_chunks_docid_test_{uuid.uuid4().hex[:8]}"
LEGACY_COLLECTION = f"kb_chunks_legacy_{uuid.uuid4().hex[:8]}"

_results = []


def check(name: str, passed: bool, detail: str = ""):
    """记录单条断言结果。"""
    _results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _fake_chunks(count: int, dim: int) -> list[dict]:
    """构造满足入库节点校验的切片(含混合向量), 不依赖 BGE 模型。"""
    return [
        {
            "content": f"第{i}段正文：设备上电前检查电源指示灯。",
            "title": f"标题{i}",
            "parent_title": "整本手册",
            "file_title": "document_id闭环测试手册",
            "item_name": "测试商品",
            "dense_vector": [random.uniform(-1, 1) for _ in range(dim)],
            "sparse_vector": {1: 0.5, 7: 0.25},
        }
        for i in range(1, count + 1)
    ]


def _create_legacy_collection(client, dim: int):
    """创建旧结构集合: 缺少 document_id / chunk_order, 用于验证结构升级保护。"""
    if client.has_collection(LEGACY_COLLECTION):
        client.drop_collection(LEGACY_COLLECTION)

    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    for field_name in ("content", "title", "parent_title", "file_title", "item_name"):
        schema.add_field(field_name=field_name, datatype=DataType.VARCHAR, max_length=65535)

    index = client.prepare_index_params()
    index.add_index(field_name="dense_vector", index_name="dense_idx",
                    index_type="AUTOINDEX", metric_type="COSINE")
    index.add_index(field_name="sparse_vector", index_name="sparse_idx",
                    index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    client.create_collection(collection_name=LEGACY_COLLECTION, schema=schema, index_params=index)


# ──────────────────────────── Part 1: 上传阶段 ────────────────────────────

def part1_upload_stage() -> dict:
    section("Part 1  上传阶段: document_id 生成")

    service = file_import_service.ImportFileService()
    tmp_dir = tempfile.mkdtemp(prefix="doc_id_upload_")
    origin_base_dir = file_import_service.get_local_base_dir
    # 隔离本地落盘目录, 避免污染真实上传目录
    file_import_service.get_local_base_dir = lambda: tmp_dir
    try:
        fake_file = UploadFile(file=io.BytesIO("测试文档内容".encode("utf-8")), filename="闭环测试.md")
        upload_context = service.process_upload_file(fake_file)
    finally:
        file_import_service.get_local_base_dir = origin_base_dir

    check("上传返回上传上下文(dict)", isinstance(upload_context, dict),
          f"type={type(upload_context).__name__}")
    required_keys = {"task_id", "document_id", "file_dir", "import_file_path"}
    check("上传上下文含 task_id/document_id/路径",
          required_keys <= set(upload_context), f"keys={sorted(upload_context)}")

    document_id = upload_context.get("document_id", "")
    parsed_ok = True
    try:
        parsed_ok = str(uuid.UUID(document_id)) == document_id
    except (ValueError, TypeError, AttributeError):
        parsed_ok = False
    check("document_id 是标准 UUID 字符串", parsed_ok, f"document_id={document_id}")

    check("document_id 与 task_id 不复用",
          document_id != upload_context.get("task_id"), f"task_id={upload_context.get('task_id')}")

    # 切片接口按 UUID 解析, 生成的 ID 必须能通过同一套校验
    from knowledge.services.knowledge_units_service import KnowledgeUnitsService
    normalized_ok = True
    try:
        normalized_ok = KnowledgeUnitsService._validate_document_id(document_id) == document_id
    except Exception as exc:  # noqa: BLE001 - 这里只需判定能否通过校验
        normalized_ok = False
        print(f"        校验异常: {exc}")
    check("document_id 能通过切片接口的 UUID 校验", normalized_ok, f"document_id={document_id}")

    return upload_context


# ──────────────────────────── Part 2: 任务阶段 ────────────────────────────

def part2_task_stage(upload_context: dict):
    section("Part 2  任务阶段: document_id 注入图状态与结果回传")

    service = file_import_service.ImportFileService()
    document_id = upload_context["document_id"]
    task_id = upload_context["task_id"]

    captured = {}
    origin_run_graph = file_import_service.run_import_graph

    def fake_run_import_graph(input_state):
        """替代真实图执行, 只验证状态注入与结果回写。"""
        captured.update(input_state)
        return {"chunks": [{"chunk_id": 1, "document_id": input_state.get("document_id"), "chunk_order": 1}]}

    file_import_service.run_import_graph = fake_run_import_graph
    try:
        service.run_import_graph(upload_context)
    finally:
        file_import_service.run_import_graph = origin_run_graph

    check("图状态注入 document_id", captured.get("document_id") == document_id,
          f"state.document_id={captured.get('document_id')}")
    check("图状态同时保留 task_id/路径",
          captured.get("task_id") == task_id
          and captured.get("file_dir") == upload_context["file_dir"]
          and captured.get("import_file_path") == upload_context["import_file_path"],
          f"keys={sorted(captured)}")

    task_info = task_util.get_task_info(task_id)
    check("任务状态为 completed", task_info.get("status") == task_util.TASK_STATUS_COMPLETED,
          f"status={task_info.get('status')}")
    check("任务产出回写 document_id", task_info.get("result", {}).get("document_id") == document_id,
          f"result={task_info.get('result')}")
    check("任务产出回写 chunk_count", task_info.get("result", {}).get("chunk_count") == 1,
          f"chunk_count={task_info.get('result', {}).get('chunk_count')}")

    # 响应契约: 前端必须能从 /upload 与 /status 拿到 document_id
    check("UploadResponse 暴露 document_id", "document_id" in UploadResponse.model_fields,
          f"fields={sorted(UploadResponse.model_fields)}")
    openapi_schemas = create_app().openapi().get("components", {}).get("schemas", {})
    check("openapi 的 UploadResponse 含 document_id",
          "document_id" in openapi_schemas.get("UploadResponse", {}).get("properties", {}),
          f"props={sorted(openapi_schemas.get('UploadResponse', {}).get('properties', {}))}")
    check("openapi 的 TaskStatusResponse 含 result",
          "result" in openapi_schemas.get("TaskStatusResponse", {}).get("properties", {}),
          f"props={sorted(openapi_schemas.get('TaskStatusResponse', {}).get('properties', {}))}")

    task_util.clear_task(task_id)


# ──────────────────────────── Part 3: 入库阶段 ────────────────────────────

def part3_import_stage(config) -> str:
    section(f"Part 3  入库阶段: 写入 document_id/chunk_order  collection={TEMP_COLLECTION}")

    client = StorageClients.get_milvus_client()
    if client.has_collection(TEMP_COLLECTION):
        client.drop_collection(TEMP_COLLECTION)

    dim = config.embedding_dim
    document_id = str(uuid.uuid4())
    node = ImportMilvusNode()

    # 1. 集合由节点按新 schema 创建, 验证建表路径也带上了两个字段
    result_state = node({"document_id": document_id, "chunks": _fake_chunks(3, dim)})
    fields = {field["name"] for field in client.describe_collection(TEMP_COLLECTION).get("fields", [])}
    check("新建集合包含 document_id/chunk_order",
          {"document_id", "chunk_order"} <= fields, f"fields={sorted(fields)}")

    chunks = (result_state or {}).get("chunks") or []
    check("state 回传 document_id", (result_state or {}).get("document_id") == document_id,
          f"document_id={(result_state or {}).get('document_id')}")
    check("每条切片写入同一 document_id",
          bool(chunks) and all(chunk.get("document_id") == document_id for chunk in chunks),
          f"count={len(chunks)}")
    check("chunk_order 从1开始连续编号",
          [chunk.get("chunk_order") for chunk in chunks] == [1, 2, 3],
          f"orders={[chunk.get('chunk_order') for chunk in chunks]}")
    check("插入后 chunk_id 由应用生成并落库",
          bool(chunks) and all(chunk.get("chunk_id") for chunk in chunks),
          f"ids={[chunk.get('chunk_id') for chunk in chunks]}")

    # 2. 落库结果与内存状态一致
    client.flush(TEMP_COLLECTION)
    rows = client.query(
        collection_name=TEMP_COLLECTION,
        filter=f'document_id=="{document_id}"',
        output_fields=["chunk_id", "document_id", "chunk_order", "content"],
        limit=10,
    )
    rows = sorted(rows, key=lambda row: row["chunk_order"])
    check("Milvus 可按 document_id 过滤到3条切片", len(rows) == 3, f"rows={len(rows)}")
    check("落库 chunk_order 与切片顺序一致",
          [row["chunk_order"] for row in rows] == [1, 2, 3],
          f"orders={[row['chunk_order'] for row in rows]}")

    # 3. 其他文档的切片不应被计入
    other_rows = client.query(
        collection_name=TEMP_COLLECTION,
        filter=f'document_id=="{uuid.uuid4()}"',
        output_fields=["chunk_id"],
        limit=10,
    )
    check("其他 document_id 查询结果为空(文档隔离)", other_rows == [], f"rows={other_rows}")

    # 4. 缺少 document_id -> 显式失败
    try:
        node({"chunks": _fake_chunks(1, dim)})
        check("缺少 document_id 时导入失败", False, "未抛出异常")
    except ImportProcessError as exc:
        check("缺少 document_id 时导入失败", "document_id" in str(exc), f"msg={str(exc)[:80]}")

    # 5. 非法 document_id -> 显式失败
    try:
        node({"document_id": "not-a-uuid", "chunks": _fake_chunks(1, dim)})
        check("非法 document_id 时导入失败", False, "未抛出异常")
    except ImportProcessError as exc:
        check("非法 document_id 时导入失败", "UUID" in str(exc), f"msg={str(exc)[:80]}")

    return document_id


def part3b_legacy_collection(config):
    section(f"Part 3b  旧结构集合保护  collection={LEGACY_COLLECTION}")

    client = StorageClients.get_milvus_client()
    _create_legacy_collection(client, config.embedding_dim)

    origin_collection = config.chunks_collection
    config.chunks_collection = LEGACY_COLLECTION
    try:
        ImportMilvusNode()({"document_id": str(uuid.uuid4()), "chunks": _fake_chunks(1, config.embedding_dim)})
        check("旧结构集合导入时显式失败", False, "未抛出异常")
    except ImportProcessError as exc:
        message = str(exc)
        check("旧结构集合导入时显式失败",
              "缺少字段" in message and "document_id" in message, f"msg={message[:100]}")
    finally:
        config.chunks_collection = origin_collection


# ──────────────────────────── Part 4: 查询闭环 ────────────────────────────

def part4_query_closed_loop(config, document_id: str):
    section("Part 4  查询阶段: 导入的切片可被 /knowledge-units 读回")

    test_client = TestClient(create_app())
    resp = test_client.get(f"/knowledge-units/{document_id}/chunks",
                           params={"page_num": 1, "page_size": 10})
    body = resp.json() if resp.status_code == 200 else {}
    check("GET 分页 返回200", resp.status_code == 200,
          f"status={resp.status_code}, body={resp.text[:120]}")
    check("GET 分页 total=3", body.get("total") == 3, f"total={body.get('total')}")
    check("GET 分页 按 chunk_order 升序",
          [row["chunk_order"] for row in body.get("rows", [])] == [1, 2, 3],
          f"orders={[row.get('chunk_order') for row in body.get('rows', [])]}")
    check("GET 分页返回的 document_id 与导入一致",
          body.get("document_id") == document_id, f"document_id={body.get('document_id')}")

    chunk_id = str(body["rows"][0]["chunk_id"]) if body.get("rows") else "0"
    resp = test_client.put(f"/knowledge-units/{document_id}/chunks/{chunk_id}",
                           json={"content": "修改后的正文：断电后清洁散热口。"})
    check("PUT 修改导入切片 返回200", resp.status_code == 200,
          f"status={resp.status_code}, body={resp.text[:120]}")

    resp = test_client.delete(f"/knowledge-units/{document_id}/chunks")
    body = resp.json() if resp.status_code == 200 else {}
    check("DELETE 按文档清空切片 返回deleted_count=3",
          resp.status_code == 200 and body.get("deleted_count") == 3,
          f"status={resp.status_code}, body={resp.text[:120]}")


def main():
    config = get_config()
    origin_collection = config.chunks_collection
    client = StorageClients.get_milvus_client()

    try:
        upload_context = part1_upload_stage()
        part2_task_stage(upload_context)

        config.chunks_collection = TEMP_COLLECTION
        document_id = part3_import_stage(config)
        part4_query_closed_loop(config, document_id)

        config.chunks_collection = origin_collection
        part3b_legacy_collection(config)
    finally:
        config.chunks_collection = origin_collection
        for name in (TEMP_COLLECTION, LEGACY_COLLECTION):
            if client.has_collection(name):
                client.drop_collection(name)

    failed = [item for item in _results if not item[1]]
    print("\n" + "=" * 72)
    print(f"合计 {len(_results)} 项, 通过 {len(_results) - len(failed)} 项, 失败 {len(failed)} 项")
    for name, passed, detail in failed:
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))
    print("=" * 72)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
