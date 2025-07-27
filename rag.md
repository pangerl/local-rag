# Local-RAG 项目需求文档

## 1. 项目概述

### 1.1. 项目简介
本项目旨在开发一个名为 **Local-RAG** 的轻量级、纯本地运行的检索增强生成（RAG）知识库。该系统专为向外部系统提供服务而设计，同时支持通过命令行进行大规模数据初始化的双重模式。它的核心是通过 API 接口或本地脚本，实现对指定路径下文档的上传、处理、向量化存储，并根据用户查询，高效地从知识库中检索、重排序并返回最相关的信息片段。

### 1.2. 项目目标
*   **完全离线**：通过手动下载模型并从本地文件系统加载，实现完全离线的部署和运行，最大化数据隐私与安全。
*   **中文优化**：采用 `jieba` 分词结合滑动窗口的策略进行文本分段，确保中文语料被切分成语义连贯且大小可控的知识片段。
*   **双重入口**：提供稳定的 FastAPI 接口用于实时、小批量的文档写入；同时提供一个本地 Python 脚本用于高效、无超时的批量数据导入。
*   **高性能模型**：使用 `BAAI/bge-small-zh-v1.5` 作为嵌入模型，`BAAI/bge-reranker-base` 作为重排序模型。
*   **现代化技术栈**：基于 Python 3.13、FastAPI 和 `uv` 环境管理。

## 2. 核心技术栈

| 组件 | 技术/库 | 用途 |
| :--- | :--- | :--- |
| **Web 框架** | FastAPI | 提供异步 API 服务 |
| **Python 版本** | Python | 3.13+ | 应用开发语言 |
| **环境与依赖管理** | `uv` | 创建虚拟环境并管理项目依赖 |
| **向量数据库** | ChromaDB | 本地存储文档向量 |
| **模型加载框架** | sentence-transformers | 从本地路径加载 BGE 系列模型 |
| **中文分词** | `jieba` | **（新增）** 用于对中文文本进行精确分词 |
| **代码规范** | Black, Ruff | 保证代码风格和质量 |

## 3. 模块设计与实现

### 3.1. 项目结构
```
local-rag/
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── models.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── retriever.py
│   └── main.py
├── data/
│   └── chroma_db/
├── models/
│   ├── bge-small-zh-v1.5/
│   └── bge-reranker-base/
├── scripts/
│   └── bulk_ingest.py
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

### 3.2. 模块详解

#### 3.2.1. `app/core/config.py` - 配置模块
*   配置项不变，明确指定模型库的根目录。
    *   `MODEL_BASE_PATH`: `str = "models"`
    *   `EMBEDDING_MODEL_DIR`: `str = "bge-small-zh-v1.5"`
    *   `RERANKER_MODEL_DIR`: `str = "bge-reranker-base"`

#### 3.2.2. `app/services/loader.py` - 模型加载器
*   **功能**：严格从配置文件指定的本地路径加载模型。它**不会**触发任何网络下载行为。
*   **实现**: 在初始化 `SentenceTransformer` 时，直接传入本地模型文件夹的绝对路径，例如: `SentenceTransformer(os.path.join(settings.MODEL_BASE_PATH, settings.EMBEDDING_MODEL_DIR))`。

#### 3.2.3. `app/services/retriever.py` - 核心服务模块
*   **核心变更点：中文优化的文本分片逻辑**
    *   **取代**：不再使用 `RecursiveCharacterTextSplitter`。
    *   **新流程**: `initialize_collection` 方法中的分片步骤将遵循以下流程：
        1.  **加载文档内容**：将整个文档读取为一个长字符串。
        2.  **Jieba 分词**：使用 `jieba.cut()` 对长字符串进行分词，得到一个词元（token）列表。
        3.  **滑动窗口分块**：
            *   基于词元列表，应用滑动窗口逻辑。
            *   `chunk_size` 和 `chunk_overlap` 参数现在指的是**词元的数量**，而不是字符数。
            *   例如，对于 `chunk_size=300`, `chunk_overlap=50`，第一个分块是第 0 到 299 个词，第二个分块是第 250 到 549 个词，依此类推。
        4.  **重组为文本**：将每个词元块（列表）使用空格或直接连接的方式 `"".join(chunk_words)` 重新组合成最终的文本分片字符串。
        5.  **后续处理**：将这些语义更完整的文本分片送入嵌入模型进行向量化。

## 4. API 接口详细设计

### 4.1. Endpoint: `/api/v1/ingest`
*   **路径**: `/api/v1/ingest`
*   **方法**: `POST`
*   **关键变更**: 请求体中 `chunk_size` 和 `chunk_overlap` 的含义已改变。
*   **请求体 (JSON)**:
    ```json
    {
      "document_path": "/path/to/external/doc.txt",
      "chunk_size": 300,  // 指的是 Jieba 分词后的词元数量
      "chunk_overlap": 50 // 指的是重叠的词元数量
    }
    ```
*   **API 文档**: 必须在 Pydantic 模型或 FastAPI 路由的文档字符串中明确注释 `chunk_size` 和 `chunk_overlap` 是**基于词元 (word count) 的**。

### 4.2. Endpoint: `/api/v1/retrieve`
接口定义不变，`retrieval_k` 默认为 10，`top_k` 默认为 3。

## 5. 本地批量导入脚本 (`scripts/bulk_ingest.py`)

*   **功能**: 与 API 接口同步，使用 `jieba` 和滑动窗口进行分片。
*   **命令行参数**:
    *   `--path` (或 `-p`): 必需，文档或目录的绝对路径。
    *   `--chunk-size` (或 `-s`): 可选，**每个分块的词元数量**，默认 300。
    *   `--chunk-overlap` (或 `-o`): 可选，**重叠的词元数量**，默认 50。
*   **使用示例**:
    ```bash
    # 使用默认词元数导入整个目录
    python -m scripts.bulk_ingest --path "/path/to/document_library/"
    ```

## 6. 环境设置与运行

### 6.1. **第一步：手动准备离线模型**
此步骤是实现完全离线部署的关键，必须在运行应用前完成。

1.  **创建目录**: 在项目根目录下创建 `models` 文件夹。
2.  **下载模型**:
    *   **嵌入模型**: 访问 `https://huggingface.co/BAAI/bge-small-zh-v1.5`，点击 "Files and versions" 标签页，下载所有文件。将它们放入 `local-rag/models/bge-small-zh-v1.5/` 目录中。
    *   **重排序模型**: 访问 `https://huggingface.co/BAAI/bge-reranker-base`，同样下载所有文件，并将它们放入 `local-rag/models/bge-reranker-base/` 目录中。
    *   *推荐方式*：如果安装了 `git-lfs`，可以使用 `git clone` 命令直接克隆模型仓库到对应目录。

### 6.2. **第二步：设置 Python 环境**
1.  **安装 uv**: 根据官方文档安装 `uv`。
2.  **创建并激活环境**:
    ```bash
    uv venv
    source .venv/bin/activate
    ```
3.  **准备`requirements.txt`**:
    ```txt
    # Web Framework
    fastapi
    uvicorn[standard]

    # Data & Config
    pydantic
    python-dotenv

    # AI & Vector DB
    sentence-transformers
    chromadb
    langchain
    langchain-community
    jieba  # 新增中文分词库

    # Code Quality (Optional)
    black
    ruff
    ```
4.  **安装依赖**:
    ```bash
    uv pip install -r requirements.txt
    ```

### 6.3. **第三步：运行**
*   **运行 API 服务**:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
*   **执行批量导入**:
    ```bash
    python -m scripts.bulk_ingest --path "/path/to/your/docs"
    ```