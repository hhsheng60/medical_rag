# Medical RAG 医疗问答项目

这是一个基于RAG（检索增强生成）技术的医疗问答实验项目，使用Milvus向量数据库和BGE-m3模型进行语义检索，结合Qwen-1.8B-Chat模型生成回答。

## 环境搭建
1.  **创建并激活虚拟环境**
    ```bash
    python -m venv rag_env
    # Windows
    rag_env\Scripts\activate
    # Mac/Linux
    source rag_env/bin/activate
    ```

2.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

## 模型准备
本项目未将模型文件上传至GitHub（体积过大），请按以下步骤准备：

1.  **下载Qwen-1.8B-Chat模型文件**
    你需要从Hugging Face下载以下核心文件，并放置到项目根目录的 `models_cache/qwen-1.8b-chat` 文件夹中：
    - `config.json`
    - `configuration_qwen.py`
    - `generation_config.json`
    - `model-00001-of-00002.safetensors`
    - `model-00002-of-00002.safetensors`
    - `model.safetensors.index.json`
    - `modeling_qwen.py`
    - `qwen_token.py`
    - `tokenization_qwen.py`
    - `tokenizer_config.json`

2.  **确认目录结构**
    最终目录结构应如下：
    ```
    medical_rag/
    └── models_cache/
        └── qwen-1.8b-chat/
            ├── config.json
            ├── ...
            └── tokenizer_config.json
    ```

## 运行项目
在激活虚拟环境并完成模型准备后，执行以下命令启动项目：
```bash
python app.py
```

---
