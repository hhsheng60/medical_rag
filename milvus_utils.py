import streamlit as st
# Use MilvusClient for Cloud version
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import time
import os

# ========== 替换为云服务配置（删除原config导入，直接定义参数） ==========
# Milvus Cloud配置
MILVUS_URI = "https://in03-6c505f6fb47cb4f.serverless.aws-eu-central-1.cloud.zilliz.com"
MILVUS_TOKEN = "6bc550fd0dc172355af188dccc6be4264cd7122fa9e9be9f8915c312989b7c0a86830513bc117c2a3a6804482f02a8823e5991d1"

# 原有配置参数（保持和原项目一致）
COLLECTION_NAME = "medical_rag_collection"
EMBEDDING_DIM = 1024  # 和BGE-m3模型维度一致
MAX_ARTICLES_TO_INDEX = 1000
INDEX_METRIC_TYPE = "COSINE"
INDEX_TYPE = "IVF_FLAT"
INDEX_PARAMS = {"nlist": 128}
SEARCH_PARAMS = {"nprobe": 10}
TOP_K = 3
id_to_doc_map = {}  # 全局文档映射

@st.cache_resource
def get_milvus_client():
    """Initializes and returns a MilvusClient instance for Zilliz Cloud."""
    try:
        st.write(f"Initializing Milvus Cloud client with URI: {MILVUS_URI}")
        # 连接云服务Milvus（核心修改）
        client = MilvusClient(
            uri=MILVUS_URI,
            token=MILVUS_TOKEN
        )
        st.success("Milvus Cloud client initialized!")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Milvus Cloud client: {e}")
        return None

@st.cache_resource
def setup_milvus_collection(_client):
    """Ensures the specified collection exists and is set up correctly in Milvus Cloud."""
    if not _client:
        st.error("Milvus client not available.")
        return False
    try:
        collection_name = COLLECTION_NAME
        dim = EMBEDDING_DIM

        has_collection = collection_name in _client.list_collections()

        if not has_collection:
            st.write(f"Collection '{collection_name}' not found. Creating...")
            # 定义集合schema（保持原有字段结构）
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=500),
            ]
            schema = CollectionSchema(fields, f"PubMed Lite RAG (dim={dim})")

            # 创建集合（云服务兼容的方式）
            _client.create_collection(
                collection_name=collection_name,
                schema=schema
            )
            st.write(f"Collection '{collection_name}' created.")

            # 创建索引（适配云服务，简化参数）
            st.write(f"Creating index ({INDEX_TYPE})...")
            _client.create_index(
                collection_name=collection_name,
                field_name="embedding",
                index_params={
                    "index_type": INDEX_TYPE,
                    "metric_type": INDEX_METRIC_TYPE,
                    "params": INDEX_PARAMS
                }
            )
            st.success(f"Index created for collection '{collection_name}'.")
        else:
            st.write(f"Found existing collection: '{collection_name}'.")

        # 获取集合数据量（适配云服务返回格式）
        try:
            stats = _client.get_collection_stats(collection_name)
            current_count = int(stats.get("row_count", 0))
            st.write(f"Collection '{collection_name}' ready. Current entity count: {current_count}")
        except Exception:
            st.write(f"Collection '{collection_name}' ready.")

        return True

    except Exception as e:
        st.error(f"Error setting up Milvus collection '{COLLECTION_NAME}': {e}")
        return False


def index_data_if_needed(client, data, embedding_model):
    """Checks if data needs indexing and performs it using Milvus Cloud."""
    global id_to_doc_map

    if not client:
        st.error("Milvus client not available for indexing.")
        return False

    collection_name = COLLECTION_NAME
    # 获取当前数据量（适配云服务）
    try:
        stats = client.get_collection_stats(collection_name)
        current_count = int(stats.get("row_count", 0))
    except Exception:
        st.write(f"Could not retrieve entity count, attempting to (re)setup collection.")
        if not setup_milvus_collection(client):
            return False
        current_count = 0

    st.write(f"Entities currently in Milvus collection '{collection_name}': {current_count}")

    data_to_index = data[:MAX_ARTICLES_TO_INDEX]
    needed_count = 0
    docs_for_embedding = []
    data_to_insert = []
    temp_id_map = {}

    # 数据预处理（保持原有逻辑）
    with st.spinner("Preparing data for indexing..."):
        for i, doc in enumerate(data_to_index):
             title = doc.get('title', '') or ""
             abstract = doc.get('abstract', '') or ""
             content = f"Title: {title}\nAbstract: {abstract}".strip()
             if not content:
                 continue

             doc_id = i
             needed_count += 1
             temp_id_map[doc_id] = {
                 'title': title, 'abstract': abstract, 'content': content
             }
             docs_for_embedding.append(content)
             data_to_insert.append({
                 "id": doc_id,
                 "embedding": None,
                 "content_preview": content[:500]
             })

    # 数据插入（保持原有逻辑，仅适配云服务insert接口）
    if current_count < needed_count and docs_for_embedding:
        st.warning(f"Indexing required ({current_count}/{needed_count} documents found). This may take a while...")

        st.write(f"Embedding {len(docs_for_embedding)} documents...")
        with st.spinner("Generating embeddings..."):
            start_embed = time.time()
            embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
            end_embed = time.time()
            st.write(f"Embedding took {end_embed - start_embed:.2f} seconds.")

        for i, emb in enumerate(embeddings):
            data_to_insert[i]["embedding"] = emb

        st.write("Inserting data into Milvus Cloud...")
        with st.spinner("Inserting..."):
            try:
                start_insert = time.time()
                # 云服务insert接口和本地一致，无需修改
                res = client.insert(collection_name=collection_name, data=data_to_insert)
                end_insert = time.time()
                inserted_count = len(data_to_insert)
                st.success(f"Successfully attempted to index {inserted_count} documents. Insert took {end_insert - start_insert:.2f} seconds.")
                id_to_doc_map.update(temp_id_map)
                return True
            except Exception as e:
                st.error(f"Error inserting data into Milvus Cloud: {e}")
                return False
    elif current_count >= needed_count:
        st.write("Data count suggests indexing is complete.")
        if not id_to_doc_map:
            id_to_doc_map.update(temp_id_map)
        return True
    else:
         st.error("No valid text content found in the data to index.")
         return False


def search_similar_documents(client, query, embedding_model):
    """Searches Milvus Cloud for documents similar to the query using MilvusClient."""
    if not client or not embedding_model:
        st.error("Milvus client or embedding model not available for search.")
        return [], []

    collection_name = COLLECTION_NAME
    try:
        query_embedding = embedding_model.encode([query])[0]

        # 云服务兼容的搜索参数（简化，避免参数冲突）
        search_params = {
            "collection_name": collection_name,
            "data": [query_embedding],
            "anns_field": "embedding",
            "limit": TOP_K,
            "output_fields": ["id"],
            "search_params": SEARCH_PARAMS  # 直接传递搜索参数
        }

        # 执行搜索（云服务标准接口）
        res = client.search(**search_params)

        # 结果处理（保持原有逻辑）
        if not res or not res[0]:
            return [], []

        hit_ids = [hit['id'] for hit in res[0]]
        distances = [hit['distance'] for hit in res[0]]
        return hit_ids, distances
    except Exception as e:
        st.error(f"Error during Milvus Cloud search: {e}")
        return [], []