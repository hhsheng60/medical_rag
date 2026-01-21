# -*- coding: utf-8 -*-
import streamlit as st
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import pymilvus
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import pandas as pd
import time

# ====================== 第一步：设置页面配置（必须是第一个Streamlit命令） ======================
st.set_page_config(
    page_title="医疗RAG问答系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== 第二步：初始化各类模型和Milvus连接 ======================
# 1. 初始化BGE-m3向量模型（优先加载本地缓存）
@st.cache_resource
def init_bge_model():
    st.info("正在加载BGE-m3向量模型...")
    try:
        # 优先使用本地缓存路径（替换为你实际的BGE缓存路径，若没有则自动下载）
        # 示例本地路径：r"C:\Users\你的用户名\.cache\torch\sentence_transformers\BAAI_bge-m3"
        model = SentenceTransformer('BAAI/bge-m3')
        st.success("BGE-m3向量模型加载成功！")
        return model
    except Exception as e:
        st.error(f"BGE模型加载失败：{str(e)}")
        return None

# 2. 初始化Qwen-1.8B-Chat大模型（已修正为你的正确路径）
@st.cache_resource
def init_qwen_model():
    st.info("正在加载Qwen-1.8B-Chat大模型...")
    try:
        # 你的正确Qwen模型路径（使用r前缀避免转义）
        model_path = model_path = r"D:\data-mining-knowledge-processing\medical_rag\models_cache\qwen-1_8b-chat"
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            use_safetensors=True
        ).eval()
        st.success("Qwen-1.8B-Chat大模型加载成功！")
        return tokenizer, model
    except Exception as e:
        st.error(f"Qwen模型加载失败：{str(e)}")
        return None, None

# 3. 连接Milvus数据库
@st.cache_resource
def init_milvus():
    st.info("正在连接Milvus数据库...")
    try:
        # 本地Milvus连接（默认端口19530，若有修改请对应调整）
        connections.connect(
            alias="default",
            host="127.0.0.1",
            port="19530"
        )
        # 检查并创建医疗文档集合（如果不存在）
        collection_name = "medical_docs"
        if not utility.has_collection(collection_name):
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)  # BGE-m3的向量维度
            ]
            schema = CollectionSchema(fields, description="医疗文档向量库")
            collection = Collection(name=collection_name, schema=schema)
            # 创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            st.success("Milvus集合创建成功！")
        else:
            collection = Collection(collection_name)
        st.success("Milvus数据库连接成功！")
        return collection
    except Exception as e:
        st.error(f"Milvus连接失败：{str(e)}")
        return None

# ====================== 第三步：核心功能函数 ======================
# 向量检索函数
def search_docs(query, bge_model, milvus_collection, top_k=3):
    # 生成查询向量
    query_embedding = bge_model.encode([query])[0].tolist()
    # 加载集合并检索
    milvus_collection.load()
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    results = milvus_collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["content"]
    )
    # 整理检索结果
    docs = []
    for hit in results[0]:
        docs.append(hit.entity.get("content"))
    return docs

# 大模型回答生成函数
def generate_answer(query, docs, tokenizer, qwen_model):
    # 构建提示词
    prompt = f"""
    你是一名专业的医疗顾问，请根据以下参考文档回答用户问题。
    参考文档：
    {chr(10).join(docs)}
    
    用户问题：{query}
    
    回答要求：
    1. 基于参考文档回答，不要编造信息；
    2. 语言通俗易懂，结构清晰；
    3. 如果参考文档没有相关信息，说明“暂无相关医疗信息”。
    """
    # 生成回答
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = qwen_model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
    return answer

# ====================== 第四步：页面UI和交互逻辑 ======================
def main():
    # 初始化组件
    bge_model = init_bge_model()
    qwen_tokenizer, qwen_model = init_qwen_model()
    milvus_collection = init_milvus()

    # 页面标题
    st.title("🏥 医疗RAG智能问答系统")
    st.divider()

    # 侧边栏
    with st.sidebar:
        st.subheader("系统配置")
        top_k = st.slider("检索文档数量", min_value=1, max_value=5, value=3)
        st.info("系统已加载：\n1. BGE-m3向量模型\n2. Qwen-1.8B-Chat大模型\n3. Milvus向量数据库")

    # 主界面
    query = st.text_input("请输入你的医疗问题：", placeholder="例如：高血压的日常注意事项有哪些？")
    if st.button("获取回答", type="primary", disabled=(None in [bge_model, qwen_tokenizer, qwen_model, milvus_collection])):
        if not query.strip():
            st.warning("请输入有效的问题！")
        else:
            with st.spinner("正在检索相关医疗文档..."):
                docs = search_docs(query, bge_model, milvus_collection, top_k)
            st.subheader("📚 检索到的参考文档")
            for i, doc in enumerate(docs, 1):
                st.write(f"{i}. {doc}")
            st.divider()
            with st.spinner("正在生成专业回答..."):
                answer = generate_answer(query, docs, qwen_tokenizer, qwen_model)
            st.subheader("💡 智能回答")
            st.write(answer)

if __name__ == "__main__":
    main()
    #streamlit run app.py启动