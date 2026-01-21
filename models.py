import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_embedding_model(model_name):
    """加载嵌入模型（适配Windows+Python3.11）"""
    try:
        st.write(f"📥 正在加载嵌入模型: {model_name}")
        model = SentenceTransformer(model_name)
        st.success(f"✅ 嵌入模型 {model_name} 加载完成！")
        return model
    except Exception as e:
        st.error(f"❌ 加载嵌入模型失败: {str(e)}")
        return None

@st.cache_resource
def load_generation_model(model_name):
    """加载生成模型和tokenizer（适配Windows+Python3.11）"""
    try:
        st.write(f"📥 正在加载生成模型: {model_name}")
        
        # 加载tokenizer（适配Qwen模型）
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 加载模型（Windows CPU/GPU自动适配）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        st.success(f"✅ 生成模型 {model_name} 加载完成！")
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ 加载生成模型失败: {str(e)}")
        return None, None