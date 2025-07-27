# Design Document

## Overview

Local-RAG 系统采用模块化架构设计，核心组件包括配置管理、模型加载、文档处理、向量存储和检索服务。系统基于 FastAPI 提供 RESTful API 服务，使用 ChromaDB 作为本地向量数据库，通过 jieba 分词实现中文优化的文本分片策略。整个系统设计为完全离线运行，确保数据隐私和安全性。

## Architecture

```mermaid
graph TB
    subgraph "API Layer"
        A[FastAPI Application]
        B[/api/v1/ingest]
        C[/api/v1/retrieve]
    end
    
    subgraph "Service Layer"
        D[Document Processor]
        E[Vector Retriever]
        F[Model Loader]
    end
    
    subgraph "Core Layer"
        G[Configuration Manager]
        H[Text Chunker with Jieba]
        I[Error Handler]
        J[Logger]
    end
    
    subgraph "Storage Layer"
        K[ChromaDB]
        L[Local Models]
    end
    
    subgraph "External"
        M[Bulk Ingest Script]
        N[Document Files]
    end
    
    A --> B
    A --> C
    B --> D
    C --> E
    D --> H
    D --> F
    E --> F
    F --> L
    D --> K
    E --> K
    G --> F
    G --> K
    H --> J
    I --> J
    M --> D
    N --> D
```

## Components and Interfaces

### 1. Configuration Management (`app/core/config.py`)

**职责**: 集中管理系统配置参数

```python
class Settings:
    # 模型配置
    MODEL_BASE_PATH: str = "models"
    EMBEDDING_MODEL_DIR: str = "bge-small-zh-v1.5"
    RERANKER_MODEL_DIR: str = "bge-reranker-base"
    
    # 数据库配置
    CHROMA_DB_PATH: str = "data/chroma_db"
    COLLECTION_NAME: str = "documents"
    
    # 分片配置
    DEFAULT_CHUNK_SIZE: int = 300
    DEFAULT_CHUNK_OVERLAP: int = 50
    
    # 检索配置
    DEFAULT_RETRIEVAL_K: int = 10
    DEFAULT_TOP_K: int = 3
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
```

### 2. Model Loader (`app/services/loader.py`)

**职责**: 从本地路径加载预训练模型

```python
class ModelLoader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_model = None
        self.reranker_model = None
    
    def load_embedding_model(self) -> SentenceTransformer:
        """加载嵌入模型，严格从本地路径加载"""
        
    def load_reranker_model(self) -> SentenceTransformer:
        """加载重排序模型，严格从本地路径加载"""
        
    def validate_model_files(self) -> bool:
        """验证模型文件是否存在且完整"""
```

### 3. Text Chunker (`app/core/chunker.py`)

**职责**: 使用 jieba 分词实现中文优化的文本分片

```python
class JiebaChunker:
    def __init__(self):
        # 初始化 jieba 分词器
        
    def chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        使用 jieba 分词和滑动窗口进行文本分片
        
        Args:
            text: 输入文本
            chunk_size: 每个分片的词元数量
            chunk_overlap: 相邻分片的重叠词元数量
            
        Returns:
            分片后的文本列表
        """
        
    def _tokenize(self, text: str) -> List[str]:
        """使用 jieba 进行分词"""
        
    def _sliding_window(self, tokens: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
        """实现滑动窗口分片逻辑"""
        
    def _reconstruct_text(self, token_chunks: List[List[str]]) -> List[str]:
        """将词元块重组为文本"""
```

### 4. Document Processor (`app/services/processor.py`)

**职责**: 处理文档上传、格式验证、文本提取和向量化

```python
class DocumentProcessor:
    def __init__(self, model_loader: ModelLoader, chunker: JiebaChunker, db_client):
        self.model_loader = model_loader
        self.chunker = chunker
        self.db_client = db_client
        
    def process_document(self, document_path: str, chunk_size: int, chunk_overlap: int) -> Dict:
        """处理单个文档的完整流程"""
        
    def validate_file_format(self, file_path: str) -> bool:
        """验证文件格式是否支持 (.txt, .md)"""
        
    def extract_text(self, file_path: str) -> str:
        """从文件中提取纯文本内容"""
        
    def vectorize_chunks(self, chunks: List[str]) -> List[List[float]]:
        """将文本分片转换为向量"""
        
    def store_vectors(self, chunks: List[str], vectors: List[List[float]], metadata: Dict):
        """将向量和元数据存储到 ChromaDB"""
```

### 5. Vector Retriever (`app/services/retriever.py`)

**职责**: 执行向量检索和重排序

```python
class VectorRetriever:
    def __init__(self, model_loader: ModelLoader, db_client):
        self.model_loader = model_loader
        self.db_client = db_client
        
    def retrieve(self, query: str, retrieval_k: int = 10, top_k: int = 3) -> List[Dict]:
        """执行检索和重排序的完整流程"""
        
    def vector_search(self, query_vector: List[float], k: int) -> List[Dict]:
        """在向量数据库中执行相似性搜索"""
        
    def rerank_results(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """使用重排序模型对候选结果进行重排序"""
```

### 6. API Endpoints (`app/api.py`)

**职责**: 提供 RESTful API 接口

```python
# 数据模型
class IngestRequest(BaseModel):
    document_path: str = Field(..., description="文档路径")
    chunk_size: int = Field(300, description="每个分片的词元数量")
    chunk_overlap: int = Field(50, description="相邻分片的重叠词元数量")

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="查询文本")
    retrieval_k: int = Field(10, description="候选文档数量")
    top_k: int = Field(3, description="返回结果数量")

# API 路由
@app.post("/api/v1/ingest")
async def ingest_document(request: IngestRequest) -> Dict:
    """文档摄取接口"""

@app.post("/api/v1/retrieve")
async def retrieve_documents(request: RetrieveRequest) -> Dict:
    """文档检索接口"""
```

## Data Models

### 1. Document Metadata

```python
@dataclass
class DocumentMetadata:
    file_path: str          # 原始文件路径
    file_name: str          # 文件名
    file_size: int          # 文件大小
    file_format: str        # 文件格式 (.txt, .md)
    chunk_index: int        # 分片索引
    chunk_size: int         # 分片大小（词元数）
    chunk_overlap: int      # 重叠大小（词元数）
    created_at: datetime    # 创建时间
    content_hash: str       # 内容哈希值
```

### 2. Retrieval Result

```python
@dataclass
class RetrievalResult:
    content: str            # 文档分片内容
    score: float           # 相关性分数
    metadata: DocumentMetadata  # 文档元数据
    rank: int              # 排序位置
```

## Error Handling

### 1. 错误分类和处理策略

```python
class LocalRAGException(Exception):
    """系统基础异常类"""
    pass

class ModelLoadError(LocalRAGException):
    """模型加载错误"""
    pass

class DocumentProcessError(LocalRAGException):
    """文档处理错误"""
    pass

class DatabaseError(LocalRAGException):
    """数据库操作错误"""
    pass

class ValidationError(LocalRAGException):
    """参数验证错误"""
    pass
```

### 2. HTTP 错误映射

- `ModelLoadError` → HTTP 500 (Internal Server Error)
- `DocumentProcessError` → HTTP 400 (Bad Request)
- `DatabaseError` → HTTP 500 (Internal Server Error)
- `ValidationError` → HTTP 422 (Unprocessable Entity)
- `FileNotFoundError` → HTTP 404 (Not Found)
- `UnsupportedFormatError` → HTTP 400 (Bad Request)

### 3. 错误响应格式

```python
class ErrorResponse(BaseModel):
    error_code: str
    error_message: str
    details: Optional[Dict] = None
    timestamp: datetime
```

## Testing Strategy

### 1. 单元测试

- **配置管理测试**: 验证配置加载和验证逻辑
- **模型加载测试**: 测试本地模型加载功能（使用模拟模型）
- **文本分片测试**: 验证 jieba 分词和滑动窗口逻辑
- **文档处理测试**: 测试文件格式验证和文本提取
- **向量检索测试**: 验证检索和重排序逻辑

### 2. 集成测试

- **API 端点测试**: 测试完整的 API 请求-响应流程
- **数据库集成测试**: 验证 ChromaDB 的存储和检索功能
- **端到端测试**: 从文档上传到检索的完整流程测试

### 3. 性能测试

- **文档处理性能**: 测试不同大小文档的处理时间
- **检索性能**: 测试不同数据量下的检索响应时间
- **并发测试**: 验证系统在并发请求下的表现

### 4. 测试数据准备

```python
# 测试用例结构
tests/
├── unit/
│   ├── test_config.py
│   ├── test_chunker.py
│   ├── test_processor.py
│   └── test_retriever.py
├── integration/
│   ├── test_api.py
│   └── test_database.py
├── fixtures/
│   ├── sample_documents/
│   │   ├── test.txt
│   │   └── test.md
│   └── mock_models/
└── conftest.py
```

## Deployment Considerations

### 1. 目录结构

```
local-rag/
├── app/                    # 应用代码
├── data/                   # 数据存储
│   └── chroma_db/         # ChromaDB 数据文件
├── models/                 # 本地模型文件
│   ├── bge-small-zh-v1.5/
│   └── bge-reranker-base/
├── logs/                   # 日志文件
├── scripts/               # 批量处理脚本
└── tests/                 # 测试代码
```

### 2. 环境要求

- Python 3.13+
- 足够的磁盘空间存储模型文件（约 2-3GB）
- 足够的内存运行模型（建议 8GB+）

### 3. 配置管理

- 使用环境变量覆盖默认配置
- 支持不同环境的配置文件
- 配置验证和错误提示

### 4. 日志和监控

```python
# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "default"
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["file", "console"]
    }
}
```