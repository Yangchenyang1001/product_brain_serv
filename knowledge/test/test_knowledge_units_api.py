"""
知识切片管理接口测试  /knowledge-units/**

分两部分执行:
    Part 1  对已启动的导入服务(http://127.0.0.1:8000)做契约冒烟, 只读或幂等操作, 不改动库内数据
    Part 2  进程内启动同一套 app, 指向临时集合跑完整增删改查成功路径, 结束后自动 drop 临时集合

运行方式:
    knowledge\\knowledge\\.venv\\Scripts\\python.exe knowledge\\test\\test_knowledge_units_api.py
"""
import logging
import os
import sys
import uuid
from pathlib import Path

# Windows 控制台默认 GBK, 统一改用 UTF-8 避免中文与符号输出报错
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.getLogger("httpx").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

# 显式指定 knowledge/.env, 避免依赖 cwd 导致环境变量加载失败
load_dotenv(KNOWLEDGE_DIR / ".env")

import random
import requests
from fastapi.testclient import TestClient
from pymilvus import DataType

BASE_URL = os.getenv("IMPORT_SERVICE_BASE_URL", "http://127.0.0.1:8000")
TEMP_COLLECTION = f"kb_chunks_api_test_{uuid.uuid4().hex[:8]}"
EMBEDDING_DIM = 1024

_results = []


def check(name: str, passed: bool, detail: str = ""):
    """记录单条断言结果。"""
    _results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ──────────────────────────── Part 1: 真实服务契约冒烟 ────────────────────────────

def part1_contract_smoke():
    section(f"Part 1  已启动服务契约冒烟  base_url={BASE_URL}")

    try:
        openapi_resp = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
    except requests.RequestException as exc:
        check("服务可访问", False, f"{type(exc).__name__}: {exc}")
        return
    check("服务可访问", openapi_resp.status_code == 200, f"status={openapi_resp.status_code}")

    paths = openapi_resp.json().get("paths", {})
    chunks_path = "/knowledge-units/{document_id}/chunks"
    chunk_path = "/knowledge-units/{document_id}/chunks/{chunk_id}"
    check("分页/新增/删文档路由已注册",
          {"get", "post", "delete"} <= set(paths.get(chunks_path, {}).keys()),
          f"methods={sorted(paths.get(chunks_path, {}).keys())}")
    check("修改/删单条路由已注册",
          {"put", "delete"} <= set(paths.get(chunk_path, {}).keys()),
          f"methods={sorted(paths.get(chunk_path, {}).keys())}")

    random_doc_id = str(uuid.uuid4())
    chunks_url = f"{BASE_URL}/knowledge-units/{random_doc_id}/chunks"

    # 1. document_id 非UUID -> 400
    resp = requests.get(f"{BASE_URL}/knowledge-units/not-a-uuid/chunks", timeout=30)
    check("GET 非法document_id 返回400", resp.status_code == 400,
          f"status={resp.status_code}, body={resp.text[:120]}")

    # 2. 分页参数非法 -> 400
    resp = requests.get(chunks_url, params={"page_num": 0, "page_size": 10}, timeout=30)
    check("GET page_num=0 返回400", resp.status_code == 400,
          f"status={resp.status_code}, body={resp.text[:120]}")

    resp = requests.get(chunks_url, params={"page_num": 1, "page_size": -1}, timeout=30)
    check("GET page_size=-1 返回400", resp.status_code == 400,
          f"status={resp.status_code}, body={resp.text[:120]}")

    # 3. 不存在的文档 -> 200 且空列表
    resp = requests.get(chunks_url, timeout=30)
    body = resp.json() if resp.status_code == 200 else {}
    check("GET 不存在文档 返回200空页",
          resp.status_code == 200 and body.get("total") == 0 and body.get("rows") == [],
          f"status={resp.status_code}, body={resp.text[:160]}")

    # 4. 请求体校验 -> 422
    resp = requests.post(chunks_url, json={"content": "   "}, timeout=30)
    check("POST 空白正文 返回422", resp.status_code == 422,
          f"status={resp.status_code}, body={resp.text[:120]}")

    resp = requests.post(chunks_url, json={"content": "正常内容", "chunk_order": 0}, timeout=30)
    check("POST chunk_order=0 返回422", resp.status_code == 422,
          f"status={resp.status_code}, body={resp.text[:120]}")

    # 5. 文档无切片时新增 -> 404 (不会写入任何数据)
    resp = requests.post(chunks_url, json={"content": "契约测试内容"}, timeout=60)
    check("POST 无切片文档 返回404", resp.status_code == 404,
          f"status={resp.status_code}, body={resp.text[:160]}")

    # 6. chunk_id 非法 -> 400
    resp = requests.put(f"{chunks_url}/abc", json={"content": "内容"}, timeout=30)
    check("PUT 非法chunk_id 返回400", resp.status_code == 400,
          f"status={resp.status_code}, body={resp.text[:120]}")

    resp = requests.delete(f"{chunks_url}/0", timeout=30)
    check("DELETE 单条 chunk_id=0 返回400", resp.status_code == 400,
          f"status={resp.status_code}, body={resp.text[:120]}")

    # 7. 按文档删除: 随机UUID无切片 -> 幂等 200
    resp = requests.delete(chunks_url, timeout=30)
    body = resp.json() if resp.status_code == 200 else {}
    check("DELETE 文档 无切片时幂等200",
          resp.status_code == 200 and body.get("deleted_count") == 0,
          f"status={resp.status_code}, body={resp.text[:160]}")


# ──────────────────────────── Part 2: 临时集合全链路 ────────────────────────────

def _create_temp_collection():
    """创建带 document_id/chunk_order 的临时切片集合, 并写入两条种子切片。"""
    from knowledge.utils.client.storage_clients import StorageClients
    from knowledge.utils.id_util import next_chunk_id

    client = StorageClients.get_milvus_client()
    if client.has_collection(TEMP_COLLECTION):
        client.drop_collection(TEMP_COLLECTION)

    # 主键由应用生成: AutoID 集合在 upsert 时会更换 chunk_id, 无法作为稳定资源标识
    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="chunk_order", datatype=DataType.INT64)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    for field_name in ("content", "title", "parent_title", "file_title", "item_name"):
        schema.add_field(field_name=field_name, datatype=DataType.VARCHAR, max_length=65535)

    index = client.prepare_index_params()
    index.add_index(field_name="dense_vector", index_name="dense_idx",
                    index_type="AUTOINDEX", metric_type="COSINE")
    index.add_index(field_name="sparse_vector", index_name="sparse_idx",
                    index_type="SPARSE_INVERTED_INDEX", metric_type="IP")

    client.create_collection(collection_name=TEMP_COLLECTION, schema=schema, index_params=index)

    document_id = str(uuid.uuid4())
    seed_contents = ["种子切片一：设备上电前检查电源指示灯。", "种子切片二：按住复位键五秒恢复出厂设置。"]
    seed_rows = []
    for order, content in enumerate(seed_contents, start=1):
        seed_rows.append({
            "chunk_id": next_chunk_id(),
            "document_id": document_id,
            "chunk_order": order,
            "content": content,
            "title": f"标题{order}",
            "parent_title": "整本手册",
            "file_title": "接口测试手册",
            "item_name": "测试商品",
            "dense_vector": [random.uniform(-1, 1) for _ in range(EMBEDDING_DIM)],
            "sparse_vector": {1: 0.5, 7: 0.25},
        })
    client.insert(collection_name=TEMP_COLLECTION, data=seed_rows)
    client.flush(TEMP_COLLECTION)
    return client, document_id, sum(len(content) for content in seed_contents)


def part2_full_crud():
    section(f"Part 2  临时集合全链路验证  collection={TEMP_COLLECTION}")

    from knowledge.api.import_router import create_app
    from knowledge.processor.import_process.config import get_config

    config = get_config()
    origin_collection = config.chunks_collection
    client = None
    try:
        client, document_id, seed_total_length = _create_temp_collection()
        # 让服务层读写走临时集合
        config.chunks_collection = TEMP_COLLECTION

        test_client = TestClient(create_app())
        chunks_url = f"/knowledge-units/{document_id}/chunks"

        # 1. 分页查询
        resp = test_client.get(chunks_url, params={"page_num": 1, "page_size": 10})
        body = resp.json()
        check("GET 分页 返回2条种子切片",
              resp.status_code == 200 and body["total"] == 2 and len(body["rows"]) == 2,
              f"status={resp.status_code}, total={body.get('total')}")
        check("GET 分页 统计正文总长度",
              body.get("total_text_length") == seed_total_length,
              f"expect={seed_total_length}, actual={body.get('total_text_length')}")
        first_row = body["rows"][0]
        check("GET 分页 按chunk_order升序",
              [row["chunk_order"] for row in body["rows"]] == [1, 2],
              f"orders={[row['chunk_order'] for row in body['rows']]}")
        check("GET 分页 content_length与正文一致",
              first_row["content_length"] == len(first_row["content"]),
              f"content_length={first_row['content_length']}")
        check("GET 分页 不返回向量字段",
              "dense_vector" not in first_row and "sparse_vector" not in first_row,
              f"keys={sorted(first_row.keys())}")

        # 2. 分页边界
        resp = test_client.get(chunks_url, params={"page_num": 2, "page_size": 1})
        body = resp.json()
        check("GET 第2页 每页1条 命中chunk_order=2",
              resp.status_code == 200 and body["total"] == 2
              and len(body["rows"]) == 1 and body["rows"][0]["chunk_order"] == 2,
              f"status={resp.status_code}, rows={[r['chunk_order'] for r in body.get('rows', [])]}")

        # 3. 新增切片(BGE-M3 首次调用会加载模型)
        resp = test_client.post(chunks_url, json={"content": "新增切片：长按开关键十秒重启设备。"}, timeout=300)
        body = resp.json()
        check("POST 新增切片 返回201",
              resp.status_code == 201, f"status={resp.status_code}, body={resp.text[:200]}")
        new_chunk_id = body.get("chunk_id")
        check("POST 新增切片 追加到末尾chunk_order=3",
              body.get("chunk_order") == 3, f"chunk_order={body.get('chunk_order')}")
        check("POST 新增切片 chunk_count=3",
              body.get("chunk_count") == 3, f"chunk_count={body.get('chunk_count')}")
        check("POST 新增切片 content_length正确",
              body.get("content_length") == len("新增切片：长按开关键十秒重启设备。"),
              f"content_length={body.get('content_length')}")

        # 4. 新增时显式指定次序
        resp = test_client.post(chunks_url, timeout=300,
                               json={"content": "指定次序切片：指示灯红色常亮表示充电中。", "chunk_order": 10})
        body = resp.json()
        check("POST 显式chunk_order=10 生效",
              resp.status_code == 201 and body.get("chunk_order") == 10,
              f"status={resp.status_code}, chunk_order={body.get('chunk_order')}")
        ordered_chunk_id = body.get("chunk_id")

        # 5. 修改切片
        new_content = "修改后切片：指示灯绿色常亮表示充电完成。"
        resp = test_client.put(f"{chunks_url}/{new_chunk_id}", json={"content": new_content}, timeout=300)
        body = resp.json()
        check("PUT 修改切片 返回200且正文已更新",
              resp.status_code == 200 and body.get("content") == new_content,
              f"status={resp.status_code}, content={body.get('content')}")
        check("PUT 修改切片 content_length同步",
              body.get("content_length") == len(new_content),
              f"content_length={body.get('content_length')}")
        check("PUT 修改切片 主键保持不变",
              str(body.get("chunk_id")) == str(new_chunk_id),
              f"expect={new_chunk_id}, actual={body.get('chunk_id')}")

        # 6. 修改不存在的切片
        resp = test_client.put(f"{chunks_url}/999999999999999999", json={"content": "任意内容"}, timeout=60)
        check("PUT 不存在切片 返回404", resp.status_code == 404,
              f"status={resp.status_code}, body={resp.text[:120]}")

        # 7. 删除单条
        resp = test_client.delete(f"{chunks_url}/{ordered_chunk_id}", timeout=60)
        body = resp.json()
        check("DELETE 单条 返回200且剩余3条",
              resp.status_code == 200 and body.get("chunk_count") == 3,
              f"status={resp.status_code}, body={body}")

        resp = test_client.delete(f"{chunks_url}/{ordered_chunk_id}", timeout=60)
        check("DELETE 已删除切片 返回404", resp.status_code == 404,
              f"status={resp.status_code}, body={resp.text[:120]}")

        # 8. 删除整个文档
        resp = test_client.delete(chunks_url, timeout=60)
        body = resp.json()
        check("DELETE 文档 物理删除3条",
              resp.status_code == 200 and body.get("deleted_count") == 3,
              f"status={resp.status_code}, body={body}")

        resp = test_client.get(chunks_url, timeout=60)
        body = resp.json()
        check("DELETE 文档后 分页为空",
              resp.status_code == 200 and body.get("total") == 0,
              f"total={body.get('total')}")

        resp = test_client.post(chunks_url, json={"content": "文档已清空后新增"}, timeout=60)
        check("DELETE 文档后 新增返回404", resp.status_code == 404,
              f"status={resp.status_code}, body={resp.text[:120]}")

        resp = test_client.delete(chunks_url, timeout=60)
        body = resp.json()
        check("DELETE 文档 重复调用幂等200",
              resp.status_code == 200 and body.get("deleted_count") == 0,
              f"status={resp.status_code}, body={body}")
    except Exception as exc:
        check("Part 2 执行异常", False, f"{type(exc).__name__}: {exc}")
    finally:
        # 恢复配置并清理临时集合
        config.chunks_collection = origin_collection
        try:
            if client is not None and client.has_collection(TEMP_COLLECTION):
                client.drop_collection(TEMP_COLLECTION)
                print(f"\n  临时集合已清理: {TEMP_COLLECTION}")
        except Exception as exc:
            print(f"\n  临时集合清理失败: {type(exc).__name__}: {exc}")


def part3_autoid_guard():
    """AutoID 集合无法稳定主键, 写入必须被显式拒绝而不是悄悄换ID。"""
    section("Part 3  AutoID 集合写入保护")

    from knowledge.processor.import_process.config import get_config
    from knowledge.utils import milvus_utils
    from knowledge.utils.client.storage_clients import StorageClients

    client = StorageClients.get_milvus_client()
    autoid_collection = f"kb_chunks_autoid_{uuid.uuid4().hex[:8]}"
    config = get_config()
    origin_collection = config.chunks_collection

    schema = client.create_schema(enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field(field_name="document_id", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="chunk_order", datatype=DataType.INT64)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    client.create_collection(collection_name=autoid_collection, schema=schema)

    config.chunks_collection = autoid_collection
    try:
        milvus_utils._CHUNK_ID_WRITABLE_CACHE.pop(autoid_collection, None)
        try:
            milvus_utils.insert_chunk({
                "chunk_id": 1,
                "document_id": str(uuid.uuid4()),
                "chunk_order": 1,
                "content": "AutoID 集合守卫用例",
            })
            check("AutoID 集合拒绝写入", False, "未抛出 ChunkIdNotWritableError")
        except milvus_utils.ChunkIdNotWritableError as exc:
            check("AutoID 集合拒绝写入", "AutoID" in str(exc), f"detail={exc}")
    except Exception as exc:
        check("Part 3 执行异常", False, f"{type(exc).__name__}: {exc}")
    finally:
        config.chunks_collection = origin_collection
        try:
            if client.has_collection(autoid_collection):
                client.drop_collection(autoid_collection)
                print(f"\n  临时集合已清理: {autoid_collection}")
        except Exception as exc:
            print(f"\n  临时集合清理失败: {type(exc).__name__}: {exc}")


def main():
    print("=" * 70)
    print("掌柜智库 - 知识切片管理接口测试")
    print("=" * 70)

    part1_contract_smoke()
    part2_full_crud()
    part3_autoid_guard()

    section("测试结果汇总")
    failed = [item for item in _results if not item[1]]
    for name, passed, detail in _results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  ({detail})" if not passed and detail else ""))
    print(f"\n  合计 {len(_results)} 项, 通过 {len(_results) - len(failed)} 项, 失败 {len(failed)} 项")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
