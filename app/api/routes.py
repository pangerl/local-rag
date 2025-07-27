"""
API 路由模块
定义 FastAPI 路由和端点处理逻辑
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from datetime import datetime

from app.api.models import (
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    ErrorResponse,
    HealthResponse,
    StatsResponse,
    DocumentChunk
)
from app.core.config import Settings
from app.services.document_service import DocumentService
from app.services.retriever import VectorRetriever
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.core.exceptions import (
    DocumentProcessError,
    UnsupportedFormatError,
    FileNotFoundError,
    DatabaseError,
    ModelLoadError
)

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["Local RAG API"])

# 全局服务实例（将在应用启动时初始化）
_document_service: DocumentService = None
_retriever: VectorRetriever = None


def get_document_service() -> DocumentService:
    """获取文档服务实例"""
    if _document_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="文档服务未初始化"
        )
    return _document_service


def get_retriever() -> VectorRetriever:
    """获取检索器实例"""
    if _retriever is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="检索服务未初始化"
        )
    return _retriever


def init_services(settings: Settings):
    """初始化服务实例"""
    global _document_service, _retriever
    
    try:
        # 初始化服务组件
        db_service = ChromaDBService(settings)
        model_loader = ModelLoader(settings)
        
        # 创建服务实例
        _document_service = DocumentService(settings, db_service, model_loader)
        _retriever = VectorRetriever(settings, db_service, model_loader)
        
        logger.info("API 服务初始化完成")
        
    except Exception as e:
        logger.error(f"API 服务初始化失败: {str(e)}")
        raise


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="文档摄取",
    description="上传和处理文档，将文档分片并存储到向量数据库中",
    responses={
        201: {"description": "文档处理成功"},
        400: {"model": ErrorResponse, "description": "请求参数错误"},
        404: {"model": ErrorResponse, "description": "文件不存在"},
        422: {"model": ErrorResponse, "description": "参数验证失败"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    }
)
async def ingest_document(
    request: IngestRequest,
    document_service: DocumentService = Depends(get_document_service)
) -> IngestResponse:
    """
    文档摄取接口
    
    处理文档上传和向量化，支持以下功能：
    - 文档格式验证（.txt, .md）
    - 基于词元数量的文本分片
    - 文档向量化和存储
    - 完整的错误处理
    
    Args:
        request: 文档摄取请求参数
        document_service: 文档处理服务
        
    Returns:
        IngestResponse: 处理结果和统计信息
        
    Raises:
        HTTPException: 各种错误情况的 HTTP 异常
    """
    try:
        logger.info(f"开始处理文档摄取请求: {request.document_path}")
        
        # 处理文档
        result = document_service.process_document(
            document_path=request.document_path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        # 构建响应
        response = IngestResponse(
            success=True,
            message="文档处理成功",
            document_path=result["document_path"],
            chunks_created=result["chunks_created"],
            chunks_stored=result["chunks_stored"],
            text_length=result["text_length"],
            processing_time=result["processing_time"],
            chunk_size=result["chunk_size"],
            chunk_overlap=result["chunk_overlap"],
            embedding_dimension=result.get("embedding_dimension"),
            collection_name=result.get("collection_name")
        )
        
        logger.info(f"文档摄取完成: {request.document_path}, 分片数: {result['chunks_created']}")
        return response
        
    except FileNotFoundError as e:
        logger.error(f"文件不存在: {request.document_path}, 错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文件不存在: {request.document_path}"
        )
        
    except UnsupportedFormatError as e:
        logger.error(f"不支持的文件格式: {request.document_path}, 错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式，仅支持 .txt 和 .md 格式"
        )
        
    except DocumentProcessError as e:
        logger.error(f"文档处理错误: {request.document_path}, 错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"文档处理失败: {str(e)}"
        )
        
    except DatabaseError as e:
        logger.error(f"数据库错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据库操作失败，请稍后重试"
        )
        
    except ModelLoadError as e:
        logger.error(f"模型加载错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型加载失败，请检查模型配置"
        )
        
    except Exception as e:
        logger.error(f"未知错误: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器内部错误，请稍后重试"
        )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    status_code=status.HTTP_200_OK,
    summary="文档检索",
    description="根据查询文本检索相关的文档片段",
    responses={
        200: {"description": "检索成功"},
        400: {"model": ErrorResponse, "description": "请求参数错误"},
        422: {"model": ErrorResponse, "description": "参数验证失败"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    }
)
async def retrieve_documents(
    request: RetrieveRequest,
    retriever: VectorRetriever = Depends(get_retriever)
) -> RetrieveResponse:
    """
    文档检索接口
    
    根据查询文本检索相关的文档片段，支持以下功能：
    - 查询向量化
    - 相似性搜索
    - 结果重排序（可选）
    - 可配置的检索参数
    
    Args:
        request: 文档检索请求参数
        retriever: 向量检索器
        
    Returns:
        RetrieveResponse: 检索结果和统计信息
        
    Raises:
        HTTPException: 各种错误情况的 HTTP 异常
    """
    try:
        logger.info(f"开始处理检索请求: {request.query[:50]}...")
        
        # 执行检索
        result = retriever.retrieve(
            query=request.query,
            retrieval_k=request.retrieval_k,
            top_k=request.top_k,
            use_reranker=request.use_reranker
        )
        
        # 转换结果格式
        document_chunks = []
        for chunk_data in result["results"]:
            chunk = DocumentChunk(
                id=chunk_data["id"],
                text=chunk_data["text"],
                similarity_score=chunk_data["similarity_score"],
                rerank_score=chunk_data.get("rerank_score"),
                metadata=chunk_data["metadata"]
            )
            document_chunks.append(chunk)
        
        # 构建响应
        response = RetrieveResponse(
            success=True,
            message="检索成功",
            query=result["query"],
            results=document_chunks,
            total_candidates=result["total_candidates"],
            returned_count=result["returned_count"],
            retrieval_k=result["retrieval_k"],
            top_k=result["top_k"],
            use_reranker=result["use_reranker"],
            timing=result["timing"]
        )
        
        logger.info(f"检索完成: 返回 {result['returned_count']} 个结果，耗时 {result['timing']['total_time']:.3f}s")
        return response
        
    except DatabaseError as e:
        logger.error(f"数据库错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据库操作失败，请稍后重试"
        )
        
    except ModelLoadError as e:
        logger.error(f"模型加载错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="模型加载失败，请检查模型配置"
        )
        
    except Exception as e:
        logger.error(f"检索过程中发生未知错误: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器内部错误，请稍后重试"
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="健康检查",
    description="检查系统各组件的健康状态"
)
async def health_check(
    document_service: DocumentService = Depends(get_document_service),
    retriever: VectorRetriever = Depends(get_retriever)
) -> HealthResponse:
    """
    健康检查接口
    
    检查系统各组件的健康状态，包括：
    - 数据库连接状态
    - 模型加载状态
    - 服务组件状态
    
    Returns:
        HealthResponse: 健康检查结果
    """
    try:
        logger.info("执行系统健康检查")
        
        # 获取各服务的健康状态
        doc_health = document_service.health_check()
        retriever_health = retriever.health_check()
        
        # 合并健康状态
        overall_status = "healthy"
        if doc_health["status"] != "healthy" or retriever_health["status"] != "healthy":
            overall_status = "unhealthy"
        
        components = {
            "document_service": doc_health,
            "retriever_service": retriever_health
        }
        
        response = HealthResponse(
            status=overall_status,
            components=components
        )
        
        logger.info(f"健康检查完成，状态: {overall_status}")
        return response
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}", exc_info=True)
        return HealthResponse(
            status="error",
            components={},
            error=str(e)
        )


@router.get(
    "/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="系统统计",
    description="获取系统运行统计信息"
)
async def get_stats(
    document_service: DocumentService = Depends(get_document_service),
    retriever: VectorRetriever = Depends(get_retriever)
) -> StatsResponse:
    """
    系统统计接口
    
    获取系统运行统计信息，包括：
    - 文档处理统计
    - 检索统计
    - 存储统计
    
    Returns:
        StatsResponse: 统计信息
    """
    try:
        logger.info("获取系统统计信息")
        
        # 获取各服务的统计信息
        processing_stats = document_service.get_processing_stats()
        retrieval_stats = retriever.get_retrieval_stats()
        
        # 存储统计信息已包含在处理统计中
        storage_stats = {
            "total_documents": processing_stats.get("storage_total_documents", 0),
            "total_chunks": processing_stats.get("storage_total_chunks", 0),
            "average_chunks_per_document": processing_stats.get("average_chunks_per_document", 0)
        }
        
        response = StatsResponse(
            processing_stats=processing_stats,
            retrieval_stats=retrieval_stats,
            storage_stats=storage_stats
        )
        
        logger.info("系统统计信息获取完成")
        return response
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取统计信息失败，请稍后重试"
        )