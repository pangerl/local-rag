"""
API 数据模型
定义请求和响应的 Pydantic 模型，包含参数验证和文档注释
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from app.core.config import settings


class ChunkingOptions(BaseModel):
    """分块选项"""
    chunk_size: Optional[int] = Field(
        default=None,
        description="文本分片大小（词元数量），如果不指定则使用系统默认值 300",
        example=500,
        ge=50,
        le=2000
    )
    chunk_overlap: Optional[int] = Field(
        default=None,
        description="相邻分片间的重叠词元数量，如果不指定则使用系统默认值 50",
        example=50,
        ge=0,
        le=500
    )

    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        """验证 chunk_overlap 不能大于等于 chunk_size"""
        if v is not None and info.data.get('chunk_size') is not None:
            if v >= info.data['chunk_size']:
                raise ValueError('chunk_overlap 必须小于 chunk_size')
        return v


class IngestRequest(ChunkingOptions):
    """
    文档摄取请求模型

    用于 /api/v1/ingest 接口的请求参数验证
    """

    document_path: str = Field(
        ...,
        description="文档文件路径",
        example="documents/example.txt",
        min_length=1,
        max_length=500
    )

    @field_validator('document_path')
    @classmethod
    def validate_document_path(cls, v):
        """验证文档路径格式"""
        if not v or not v.strip():
            raise ValueError('document_path 不能为空')

        v = v.strip()

        # 检查文件扩展名
        if not any(v.lower().endswith(ext) for ext in settings.SUPPORTED_FORMATS):
            raise ValueError(f'不支持的文件格式，仅支持: {", ".join(settings.SUPPORTED_FORMATS)}')

        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_path": "documents/python_tutorial.txt",
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }
    )


class IngestLoadRequest(ChunkingOptions):
    """
    批量摄取请求模型
    """
    path: str = Field(..., description="要加载的本地目录的路径")


class IngestLoadResponse(BaseModel):
    """
    批量摄取响应模型
    """
    success: bool
    total_files: int
    processed_files: int
    failed_files: int
    failed_details: List[Dict[str, str]]
    total_chunks_created: int
    total_chunks_stored: int
    total_processing_time: float


class IngestResponse(BaseModel):
    """
    文档摄取响应模型

    返回文档处理结果和统计信息
    """

    success: bool = Field(
        ...,
        description="处理是否成功"
    )

    message: str = Field(
        ...,
        description="处理结果消息"
    )

    document_path: str = Field(
        ...,
        description="处理的文档路径"
    )

    chunks_created: int = Field(
        ...,
        description="创建的文档分片数量",
        ge=0
    )

    chunks_stored: int = Field(
        ...,
        description="成功存储的分片数量",
        ge=0
    )

    text_length: int = Field(
        ...,
        description="文档文本总长度（字符数）",
        ge=0
    )

    processing_time: float = Field(
        ...,
        description="处理耗时（秒）",
        ge=0
    )

    chunk_size: int = Field(
        ...,
        description="实际使用的分片大小（词元数量）",
        ge=1
    )

    chunk_overlap: int = Field(
        ...,
        description="实际使用的分片重叠（词元数量）",
        ge=0
    )

    embedding_dimension: Optional[int] = Field(
        default=None,
        description="嵌入向量维度",
        ge=1
    )

    collection_name: Optional[str] = Field(
        default=None,
        description="存储的数据库集合名称"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="处理完成时间"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "文档处理成功",
                "document_path": "documents/python_tutorial.txt",
                "chunks_created": 15,
                "chunks_stored": 15,
                "text_length": 4500,
                "processing_time": 2.35,
                "chunk_size": 300,
                "chunk_overlap": 50,
                "embedding_dimension": 768,
                "collection_name": "documents",
                "timestamp": "2024-01-01T12:00:00"
            }
        }
    )


class RetrieveRequest(BaseModel):
    """
    文档检索请求模型

    用于 /api/v1/retrieve 接口的请求参数验证
    """

    query: str = Field(
        ...,
        description="查询文本，用于搜索相关的文档片段",
        example="Python编程语言的特性",
        min_length=1,
        max_length=1000
    )

    retrieval_k: Optional[int] = Field(
        default=None,
        description="检索的候选文档片段数量，如果不指定则使用系统默认值 10",
        example=10,
        ge=1,
        le=100
    )

    top_k: Optional[int] = Field(
        default=None,
        description="返回的最终结果数量，如果不指定则使用系统默认值 3",
        example=3,
        ge=1,
        le=50
    )

    use_reranker: Optional[bool] = Field(
        default=True,
        description="是否使用重排序模型提高检索精度，默认为 True"
    )

    @field_validator('top_k')
    @classmethod
    def validate_top_k(cls, v, info):
        """验证 top_k 不能大于 retrieval_k"""
        if v is not None and info.data.get('retrieval_k') is not None:
            if v > info.data['retrieval_k']:
                raise ValueError('top_k 不能大于 retrieval_k')
        return v

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """验证查询文本"""
        if not v or not v.strip():
            raise ValueError('query 不能为空')
        return v.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Python编程语言的特性和优势",
                "retrieval_k": 10,
                "top_k": 3,
                "use_reranker": True
            }
        }
    )


class DocumentChunk(BaseModel):
    """
    文档分片模型

    表示检索结果中的单个文档片段
    """

    id: str = Field(
        ...,
        description="分片唯一标识符"
    )

    text: str = Field(
        ...,
        description="分片文本内容"
    )

    similarity_score: float = Field(
        ...,
        description="与查询的相似度分数（0-1之间，越高越相似）",
        ge=0,
        le=1
    )

    rerank_score: Optional[float] = Field(
        default=None,
        description="重排序分数（如果使用了重排序模型）",
        ge=0,
        le=1
    )

    metadata: Dict[str, Any] = Field(
        ...,
        description="分片元数据，包含文档路径、分片索引等信息"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "python_tutorial_0_abc123",
                "text": "Python是一种高级编程语言，具有简洁的语法和强大的功能。",
                "similarity_score": 0.85,
                "rerank_score": 0.92,
                "metadata": {
                    "document_path": "documents/python_tutorial.txt",
                    "chunk_index": 0,
                    "total_chunks": 15,
                    "chunk_length": 45,
                    "created_at": "2024-01-01T12:00:00"
                }
            }
        }
    )


class RetrieveResponse(BaseModel):
    """
    文档检索响应模型

    返回检索结果和相关统计信息
    """

    success: bool = Field(
        ...,
        description="检索是否成功"
    )

    message: str = Field(
        ...,
        description="检索结果消息"
    )

    query: str = Field(
        ...,
        description="原始查询文本"
    )

    results: List[DocumentChunk] = Field(
        ...,
        description="检索到的文档片段列表，按相关性降序排列"
    )

    total_candidates: int = Field(
        ...,
        description="检索到的候选片段总数",
        ge=0
    )

    returned_count: int = Field(
        ...,
        description="实际返回的结果数量",
        ge=0
    )

    retrieval_k: int = Field(
        ...,
        description="使用的候选片段检索数量",
        ge=1
    )

    top_k: int = Field(
        ...,
        description="使用的最终结果数量",
        ge=1
    )

    use_reranker: bool = Field(
        ...,
        description="是否使用了重排序模型"
    )

    timing: Dict[str, float] = Field(
        ...,
        description="检索各阶段的耗时统计（秒）"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="检索完成时间"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "检索成功",
                "query": "Python编程语言的特性",
                "results": [
                    {
                        "id": "python_tutorial_0_abc123",
                        "text": "Python是一种高级编程语言，具有简洁的语法和强大的功能。",
                        "similarity_score": 0.85,
                        "rerank_score": 0.92,
                        "metadata": {
                            "document_path": "documents/python_tutorial.txt",
                            "chunk_index": 0,
                            "total_chunks": 15
                        }
                    }
                ],
                "total_candidates": 8,
                "returned_count": 1,
                "retrieval_k": 10,
                "top_k": 3,
                "use_reranker": True,
                "timing": {
                    "total_time": 0.45,
                    "search_time": 0.25,
                    "rerank_time": 0.20
                },
                "timestamp": "2024-01-01T12:00:00"
            }
        }
    )


class ErrorResponse(BaseModel):
    """
    错误响应模型

    统一的错误响应格式
    """

    success: bool = Field(
        default=False,
        description="操作是否成功，错误时固定为 False"
    )

    error: str = Field(
        ...,
        description="错误类型"
    )

    message: str = Field(
        ...,
        description="错误详细信息"
    )

    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="错误详细信息，如验证错误的字段信息"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="错误发生时间"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": False,
                "error": "ValidationError",
                "message": "请求参数验证失败",
                "details": {
                    "field": "chunk_size",
                    "value": 2500,
                    "constraint": "必须小于等于 2000"
                },
                "timestamp": "2024-01-01T12:00:00"
            }
        }
    )


class HealthResponse(BaseModel):
    """
    健康检查响应模型

    返回系统各组件的健康状态
    """

    status: str = Field(
        ...,
        description="整体健康状态：healthy, unhealthy, error"
    )

    components: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="各组件的健康状态详情"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="健康检查时间"
    )

    error: Optional[str] = Field(
        default=None,
        description="健康检查过程中的错误信息"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "components": {
                    "database": {
                        "status": "healthy",
                        "database_path_exists": True,
                        "connection_status": True,
                        "default_collection_exists": True
                    },
                    "models": {
                        "status": "healthy",
                        "embedding_model_loaded": True,
                        "reranker_model_loaded": True
                    }
                },
                "timestamp": "2024-01-01T12:00:00",
                "error": None
            }
        }
    )


class DeleteDocumentRequest(BaseModel):
    """
    文档删除请求模型

    用于删除文档接口的请求参数验证
    """

    document_path: str = Field(
        ...,
        description="要删除的文档文件路径",
        example="documents/example.txt",
        min_length=1,
        max_length=500
    )

    @field_validator('document_path')
    @classmethod
    def validate_document_path(cls, v):
        """验证文档路径格式"""
        if not v or not v.strip():
            raise ValueError('document_path 不能为空')
        return v.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_path": "documents/python_tutorial.txt"
            }
        }
    )


class DeleteDocumentResponse(BaseModel):
    """
    文档删除响应模型

    返回文档删除结果和统计信息
    """

    success: bool = Field(
        ...,
        description="删除是否成功"
    )

    message: str = Field(
        ...,
        description="删除结果消息"
    )

    document_path: str = Field(
        ...,
        description="删除的文档路径"
    )

    chunks_deleted: int = Field(
        ...,
        description="删除的文档分片数量",
        ge=0
    )

    processing_time: float = Field(
        ...,
        description="删除耗时（秒）",
        ge=0
    )

    status: str = Field(
        ...,
        description="删除状态：success, not_found, error"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="删除完成时间"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "success": True,
                "message": "文档删除成功",
                "document_path": "documents/python_tutorial.txt",
                "chunks_deleted": 15,
                "processing_time": 0.25,
                "status": "success",
                "timestamp": "2024-01-01T12:00:00"
            }
        }
    )


class DocumentInfo(BaseModel):
    """文档基本信息，用于统计响应"""
    document_path: str = Field(..., description="文档路径")
    chunk_count: int = Field(..., description="该文档的分片数量")
    created_at: Optional[str] = Field(default=None, description="文档首次入库时间")
    text_length: Optional[int] = Field(default=None, description="文档文本总长度（字符数）")
    chunk_size: Optional[int] = Field(default=None, description="分片大小")
    file_size: Optional[int] = Field(default=None, description="文件大小（字节）")


class StatsResponse(BaseModel):
    """
    统计信息响应模型

    返回系统运行统计信息和文档列表
    """

    system_stats: Dict[str, Any] = Field(
        ...,
        description="系统运行统计信息"
    )

    documents: Optional[List[DocumentInfo]] = Field(
        default=[],
        description="已处理的文档列表"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="统计信息生成时间"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "system_stats": {
                    "total_documents": 5,
                    "total_chunks": 120
                },
                "documents": [
                    {
                        "document_path": "docs/doc1.txt",
                        "chunk_count": 20,
                        "created_at": "2024-07-29T10:00:00",
                        "text_length": 15000,
                        "chunk_size": 300
                    },
                    {
                        "document_path": "docs/doc2.pdf",
                        "chunk_count": 100,
                        "created_at": "2024-07-29T11:00:00",
                        "text_length": 80000,
                        "chunk_size": 300
                    }
                ],
                "timestamp": "2024-07-30T12:00:00"
            }
        }
    )
