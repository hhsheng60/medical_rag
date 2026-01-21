# app.py
import os
import json
import hashlib
import streamlit as st
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

# ===================== 配置项 =====================
# 配置 HuggingFace 国内镜像（解决模型下载/加载问题）
os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# 模型配置（和 preprocess.py 保持一致）
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_KWARGS = {
    "normalize_embeddings": True,
    "prompt": "为检索任务生成表示向量："
}

# ========== Zilliz Cloud / Milvus 云服务配置 ==========
MILVUS_URI = os.getenv(
    "MILVUS_URI",
    "https://in03-6c505f6fb47cb4f.serverless.aws-eu-central-1.cloud.zilliz.com"
)

# 强烈建议把 token 放到环境变量里：MILVUS_TOKEN="xxxx"
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "YOUR_MILVUS_TOKEN_HERE")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_rag_collection")

# 数据路径
PROCESSED_DATA_PATH = "data/processed/processed_data.json"

# 关键：Milvus/Zilliz 的 VARCHAR 上限按 UTF-8 字节计（最大 65535 bytes）
# 建议留余量，避免边界/多字节字符导致超限
MAX_CONTENT_BYTES = 64000

# 你的 schema 里 id max_length=64（同样按字节计），避免 chunk_id 超长
MAX_ID_BYTES = 64


# ===================== 工具函数（按字节） =====================
def utf8_bytes_len(s: str) -> int:
    return len((s or "").encode("utf-8"))


def safe_id(raw_id: str, max_bytes: int = MAX_ID_BYTES) -> str:
    """
    确保主键 id 的 UTF-8 字节长度 <= max_bytes。
    若超长，使用“前缀 + hash”保证稳定且唯一性较高。
    """
    raw_id = str(raw_id)
    if utf8_bytes_len(raw_id) <= max_bytes:
        return raw_id

    h = hashlib.md5(raw_id.encode("utf-8")).hexdigest()[:10]
    prefix = raw_id[:30]
    candidate = f"{prefix}_{h}"

    # 极端情况下再截断
    b = candidate.encode("utf-8")
    if len(b) <= max_bytes:
        return candidate
    return b[:max_bytes].decode("utf-8", errors="ignore")


def split_text_by_utf8_bytes(text: str, max_bytes: int = MAX_CONTENT_BYTES):
    """
    按 UTF-8 字节上限切分，保证每个 chunk 的 utf8 字节数 <= max_bytes
    使用二分查找定位每段最大切片位置，避免中文多字节溢出。
    """
    text = text or ""
    chunks = []
    start = 0
    n = len(text)

    while start < n:
        lo, hi = start + 1, n
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if utf8_bytes_len(text[start:mid]) <= max_bytes:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        chunk = text[start:best]
        if chunk.strip():
            chunks.append(chunk)
        start = best

    return chunks


# ===================== 初始化函数 =====================
@st.cache_resource
def init_model():
    """初始化 BGE-m3 模型（缓存，避免重复加载）"""
    st.info("正在加载 BGE-m3 模型...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    st.success("BGE-m3 模型加载完成！")
    return model


@st.cache_resource
def init_milvus():
    """初始化 Milvus/Zilliz 客户端，并创建集合 + 索引（如果不存在）"""
    if MILVUS_TOKEN == "YOUR_MILVUS_TOKEN_HERE":
        st.warning("你还没有设置 MILVUS_TOKEN 环境变量，将无法正常连接云服务。")

    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    st.success("Milvus/Zilliz 云服务客户端连接成功！")

    # 定义集合 schema（VARCHAR 最大 65535 bytes）
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    ]
    schema = CollectionSchema(fields=fields, description="医疗 RAG 数据集（长文档分块）")

    if not client.has_collection(collection_name=COLLECTION_NAME):
        client.create_collection(collection_name=COLLECTION_NAME, schema=schema)

        client.create_index(
            collection_name=COLLECTION_NAME,
            field_name="embedding",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128},
            },
        )
        st.info(f"集合 {COLLECTION_NAME} 创建完成，并已建立索引。")
    else:
        st.info(f"集合 {COLLECTION_NAME} 已存在，直接使用。")

    return client


def load_data_to_milvus(client: MilvusClient, model: SentenceTransformer):
    """
    加载预处理后的数据到 Milvus：
    - 以 UTF-8 字节为准判断是否超长
    - 超长则按 UTF-8 字节安全分块，并对每个 chunk 重新生成向量
    """
    # 检查是否已有数据
    try:
        stats = client.get_collection_stats(collection_name=COLLECTION_NAME)
        row_count = stats.get("row_count", 0)
        if isinstance(row_count, str) and row_count.isdigit():
            row_count = int(row_count)
        if row_count and row_count > 0:
            st.info(f"Milvus 中已有 {row_count} 条记录，跳过加载。")
            return
    except Exception as e:
        st.warning(f"获取数据量失败：{e}，继续尝试加载。")

    if not os.path.exists(PROCESSED_DATA_PATH):
        st.error(f"预处理数据文件不存在：{PROCESSED_DATA_PATH}")
        st.stop()

    with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as f:
        processed_data = json.load(f)

    if not processed_data:
        st.warning("预处理数据为空，跳过加载。")
        return

    insert_data = []
    total_chunks = 0

    with st.spinner("正在处理文档（按 UTF-8 字节分块）..."):
        for doc in processed_data:
            original_id = safe_id(doc.get("id", ""))
            original_content = doc.get("content", "")
            original_embedding = doc.get("embedding", None)

            # 1) 短文档：按字节判断，直接插入（保持原 embedding）
            if utf8_bytes_len(original_content) <= MAX_CONTENT_BYTES:
                if original_embedding is None:
                    # 兜底：如果没有 embedding，就现场生成
                    original_embedding = model.encode(original_content, **EMBEDDING_KWARGS).tolist()

                insert_data.append(
                    {
                        "id": original_id,
                        "content": original_content,
                        "embedding": original_embedding,
                    }
                )
                total_chunks += 1

            # 2) 长文档：按字节安全分块，chunk 重新编码向量
            else:
                st.info(
                    f"文档 {original_id} UTF-8 字节长度 {utf8_bytes_len(original_content)}，自动分块..."
                )
                chunks = split_text_by_utf8_bytes(original_content, MAX_CONTENT_BYTES)

                for idx, chunk in enumerate(chunks):
                    chunk_id = safe_id(f"{original_id}_c{idx}")
                    chunk_embedding = model.encode(chunk, **EMBEDDING_KWARGS).tolist()

                    # 最终安全校验（防止任何意外超限）
                    if utf8_bytes_len(chunk) > 65535:
                        # 理论上不会发生，因为 MAX_CONTENT_BYTES < 65535
                        chunk = chunk.encode("utf-8")[:65535].decode("utf-8", errors="ignore")

                    insert_data.append(
                        {
                            "id": chunk_id,
                            "content": chunk,
                            "embedding": chunk_embedding,
                        }
                    )
                    total_chunks += 1

    with st.spinner("正在插入数据到 Milvus/Zilliz..."):
        # 如果数据量很大，建议分批插入（这里给一个安全的 batch）
        batch_size = 256
        for i in range(0, len(insert_data), batch_size):
            client.insert(
                collection_name=COLLECTION_NAME,
                data=insert_data[i : i + batch_size],
            )

    st.success(
        f"成功加载 {total_chunks} 个文本块到 Milvus（原文档数：{len(processed_data)}）！"
    )


def search_similar_docs(client: MilvusClient, model: SentenceTransformer, query: str, top_k: int = 3):
    """检索相似文本块"""
    query_embedding = model.encode(query, **EMBEDDING_KWARGS).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["id", "content"],
        search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
    )

    similar_docs = []
    for res in results[0]:
        doc_id = res["entity"]["id"]
        is_chunk = "_c" in doc_id or "_chunk" in doc_id
        similarity = round(1 - res["distance"], 4)  # COSINE 距离转相似度（近似用法）

        similar_docs.append(
            {
                "id": doc_id,
                "content": res["entity"]["content"],
                "similarity": similarity,
                "is_chunk": is_chunk,
            }
        )

    return similar_docs


# ===================== Streamlit 前端 =====================
def main():
    st.set_page_config(
        page_title="医疗RAG问答系统（BGE-m3 + 长文档分块）",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 医疗RAG问答系统（基于 BGE-m3 向量模型 + 长文档分块）")

    # 初始化
    model = init_model()
    client = init_milvus()

    # 加载数据（仅首次）
    load_data_to_milvus(client, model)

    # 问答交互
    st.divider()
    query = st.text_input("请输入你的问题：", placeholder="例如：高血压的日常注意事项？")
    top_k = st.slider("检索相似文本块数量：", min_value=1, max_value=5, value=3)

    if st.button("检索答案", type="primary"):
        if not query.strip():
            st.warning("请输入有效问题！")
            return

        with st.spinner("正在检索相似文本块..."):
            similar_docs = search_similar_docs(client, model, query, top_k)

        st.subheader("📝 相似文本块检索结果")
        if not similar_docs:
            st.info("未检索到相似文本块。")
        else:
            for idx, doc in enumerate(similar_docs, 1):
                chunk_note = "（长文档分块）" if doc["is_chunk"] else ""
                with st.expander(
                    f"文本块 {idx}（相似度：{doc['similarity']:.4f}）{chunk_note}"
                ):
                    st.write(f"文本块ID：{doc['id']}")
                    st.write(doc["content"])

        st.subheader("💡 问答结果")
        if similar_docs:
            st.write(similar_docs[0]["content"])
        else:
            st.write("暂无相关信息。")


if __name__ == "__main__":
    main()
