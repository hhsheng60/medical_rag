import os
# 配置HuggingFace国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import re
from sentence_transformers import SentenceTransformer
import numpy as np

# 自动创建所需文件夹
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# 配置参数（核心：适配你的JSON字段）
RAW_DATA_PATH = "data/raw/medical_data.json"
PROCESSED_DATA_PATH = "data/processed/processed_data.json"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_KWARGS = {
    "normalize_embeddings": True,
    "prompt": "为检索任务生成表示向量："
}

def clean_text(text):
    """清理文本：去除无效字符、补全截断内容"""
    if not isinstance(text, str):
        return ""
    # 去除不可见字符、多余空格，补全可能的截断（比如你的示例里最后少了引号）
    text = re.sub(r'[\s\u3000\x00-\x1f\x7f]', ' ', text).strip()
    # 去除末尾的不完整字符（比如你的示例里最后是 "c"，可能是截断）
    text = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5.,?!;:]+$', '', text)
    return text

def load_raw_data(file_path):
    """加载原始JSON数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"原始数据文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # 处理可能的JSON截断问题
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"JSON解析警告：{e}，尝试修复截断...")
            # 读取全部内容并尝试补全
            f.seek(0)
            raw_content = f.read().strip()
            # 补全末尾缺失的引号/大括号
            if not raw_content.endswith('}'):
                raw_content += '"}'
            data = json.loads(raw_content)
    
    # 适配单条字典/列表格式
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError(f"原始数据必须是列表/字典类型，当前是: {type(data)}")
    
    print(f"成功加载原始JSON文件，共 {len(data)} 条数据")
    return data

def preprocess_data(raw_data):
    """预处理数据：精准提取context字段"""
    print("正在加载BGE-m3模型（首次运行会下载，耐心等待）...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    processed_docs = []
    
    for idx, item in enumerate(raw_data):
        print(f"\n处理第 {idx} 条数据：")
        # 精准提取你的JSON字段
        if isinstance(item, dict):
            # 优先提取context字段（你的核心文本）
            content = clean_text(item.get('context', ''))
            # 用corpus_name+索引作为ID（也可以用item里的其他唯一字段）
            corpus_name = item.get('corpus_name', 'Medical')
            doc_id = f"{corpus_name}_{idx}"
            
            print(f"  原始context内容：[{item.get('context', '空')}]")
            print(f"  清理后文本：[{content}]")
        else:
            print(f"  跳过无效数据：类型为{type(item)}")
            continue
        
        # 检查文本是否有效（长度≥5才保留，避免无意义内容）
        if len(content) < 5:
            print(f"  跳过：文本过短（长度{len(content)}）")
            continue
        
        # 生成BGE-m3向量
        embedding = model.encode(content, **EMBEDDING_KWARGS).tolist()
        
        processed_docs.append({
            "id": doc_id,
            "content": content,
            "embedding": embedding
        })
        print(f"  处理成功：生成1024维向量，文档ID={doc_id}")
    
    print(f"\n预处理完成，有效数据 {len(processed_docs)} 条")
    return processed_docs

def save_processed_data(processed_data, file_path):
    """保存预处理后的数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    print(f"预处理后的数据已保存到: {os.path.abspath(file_path)}")

if __name__ == "__main__":
    try:
        print("开始处理原始JSON数据集（使用BGE-m3模型）...")
        raw_data = load_raw_data(RAW_DATA_PATH)
        processed_data = preprocess_data(raw_data)
        save_processed_data(processed_data, PROCESSED_DATA_PATH)
        print("数据预处理全部完成！")
    except Exception as e:
        print(f"预处理失败：{type(e).__name__} - {e}")
        raise