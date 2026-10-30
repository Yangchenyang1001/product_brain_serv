"""
  @Author:桌角是小黑
  @Time:2026/10/13
  @Desc:
"""
from pymilvus import DataType, client, MilvusClient, CollectionSchema
# 创建客户端
def get_milvus_client():
    client = MilvusClient(
        uri="http://localhost:19530",
        token="",
    )
    res = client.list_collections()
    print(res)
    return client

def build_schema(client: MilvusClient) -> CollectionSchema:
    # 创建 Schema
    schema = client.create_schema(auto_id=True)

    # 添加字段
    schema.add_field(
        field_name="pk",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,        # 自动生成主键
        max_length=100
    )
    schema.add_field(
        field_name="file_title",
        datatype=DataType.VARCHAR,
        max_length=65535
    )
    schema.add_field(
        field_name="item_name",
        datatype=DataType.VARCHAR,
        max_length=65535
    )
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=1024             # BGE-M3 输出维度
    )
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR
    )

    return schema

def build_index(client: MilvusClient):
    index_params = client.prepare_index_params()

    # 稠密向量索引（自动选择最优算法）
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="AUTOINDEX",       # 自动选择 HNSW 等算法
        metric_type="COSINE"          # 余弦相似度
    )

    # 稀疏向量索引（倒排索引）
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP"              # 内积
    )
    return index_params


def create_collection(client: MilvusClient, collection_name: str = "kb_item_names_v1"):
    from pprint import pprint
    client.drop_collection(collection_name=collection_name)
    if not client.has_collection(collection_name=collection_name):
        print("collection demo_collection not exists, create it")
        client.create_collection(
            collection_name=collection_name,  # collection 名称
            schema=build_schema(client),  # collection 的 schema
            index_params=build_index(client),  # collection 的 index
        )
        # 查看 collection
        print(client.list_collections())
        # 查看 collection 描述
        print(client.describe_collection(collection_name=collection_name))
if __name__ == '__main__':
    create_collection(get_milvus_client())