"""
API 路由模块
定义 FastAPI 路由和端点处理逻辑
"""

import logging
import tempfile
import os
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Request, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from pathlib import Path

from app.api.models import (
    IngestRequest,
    IngestResponse,
    IngestLoadRequest,
    IngestLoadResponse,
    RetrieveRequest,
    RetrieveResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    ErrorResponse,
    HealthResponse,
    StatsResponse,
    DocumentChunk,
    DocumentInfo
)
from app.core.config import settings, Settings
from app.services.document_service import DocumentService
from app.services.retriever import VectorRetriever
from app.services.database import ChromaDBService
from app.services.models import EmbeddingModel, RerankerModel
from app.core.exceptions import (
    DocumentProcessError,
    UnsupportedFormatError,
    FileNotFoundError,
    DatabaseError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)

# 支持的文档格式
SUPPORTED_EXTENSIONS = settings.SUPPORTED_FORMATS

# 数据目录
DATA_DIR = Path(settings.DATA_PATH)

# 配置模板目录
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["Local RAG API"])

# 创建管理页面路由器（不带前缀）
admin_router = APIRouter(tags=["Admin Interface"])

class ModelManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedding_model: Optional[EmbeddingModel] = None
        self.reranker_model: Optional[RerankerModel] = None
        self._document_service: Optional[DocumentService] = None
        self._retriever: Optional[VectorRetriever] = None
        self._db_service: Optional[ChromaDBService] = None

    def load_models(self):
        try:
            logger.info("开始加载嵌入模型...")
            self.embedding_model = EmbeddingModel(
                model_path=str(self.settings.embedding_model_path),
                device=self.settings.EMBEDDING_DEVICE,
                max_length=self.settings.EMBEDDING_MAX_LENGTH,
            )
            logger.info("嵌入模型加载成功")

            logger.info("开始加载重排序模型...")
            self.reranker_model = RerankerModel(
                model_path=str(self.settings.reranker_model_path),
                device=self.settings.RERANKER_DEVICE,
                max_length=self.settings.RERANKER_MAX_LENGTH,
            )
            logger.info("重排序模型加载成功")

            self._db_service = ChromaDBService(self.settings)
            self._document_service = DocumentService(self.settings, self._db_service, self.embedding_model)
            self._retriever = VectorRetriever(self.settings, self._db_service, self.embedding_model, self.reranker_model)
            logger.info("所有服务初始化完成")

        except Exception as e:
            logger.error(f"模型或服务加载失败: {e}", exc_info=True)
            raise ModelLoadError(f"模型或服务加载失败: {e}") from e

    def unload_models(self):
        logger.info("开始卸载模型和服务...")
        del self.embedding_model
        del self.reranker_model
        del self._document_service
        del self._retriever
        del self._db_service
        self.embedding_model = None
        self.reranker_model = None
        self._document_service = None
        self._retriever = None
        self._db_service = None
        logger.info("模型和服务已卸载")

    def get_document_service(self) -> DocumentService:
        if not self._document_service:
            raise HTTPException(status_code=503, detail="文档服务未初始化")
        return self._document_service

    def get_retriever(self) -> VectorRetriever:
        if not self._retriever:
            raise HTTPException(status_code=503, detail="检索服务未初始化")
        return self._retriever

model_manager = ModelManager(settings)

def get_document_service() -> DocumentService:
    return model_manager.get_document_service()

def get_retriever() -> VectorRetriever:
    return model_manager.get_retriever()


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
    logger.info(f"开始处理文档摄取请求: {request.document_path}")

    # 安全性检查：确保文件路径在 DATA_DIR 内
    doc_path = Path(request.document_path).resolve()
    if not doc_path.is_file():
        raise FileNotFoundError(f"文件不存在: {request.document_path}")
    if DATA_DIR.resolve() not in doc_path.parents:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="禁止访问此文件路径"
        )

    # 验证文件格式
    file_extension = doc_path.suffix.lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"不支持的文件格式 '{file_extension}'。支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}")

    # 获取文件大小
    file_size = doc_path.stat().st_size

    # 异步处理文档
    result = await run_in_threadpool(
        document_service.process_document,
        document_path=str(doc_path),
        original_filename=request.document_path,
        file_size=file_size,
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


@router.post(
    "/ingest/load",
    response_model=IngestLoadResponse,
    status_code=status.HTTP_200_OK,
    summary="批量摄取目录",
    description="从本地目录批量加载和处理文档",
)
async def ingest_load(
    request: IngestLoadRequest,
    document_service: DocumentService = Depends(get_document_service)
) -> IngestLoadResponse:
    """
    批量摄取目录接口
    """
    logger.info(f"开始批量处理目录: {request.path}")
    load_path = (DATA_DIR / request.path).resolve()
    if not load_path.is_dir():
        raise FileNotFoundError(f"目录不存在: {load_path}")
    if DATA_DIR.resolve() not in load_path.parents and load_path != DATA_DIR.resolve():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="禁止访问此目录路径"
        )

    result = await run_in_threadpool(
        document_service.process_directory,
        directory_path=load_path,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap
    )
    return IngestLoadResponse(**result)


@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="文档上传摄取",
    description="上传文件并处理文档，将文档分片并存储到向量数据库中",
    responses={
        201: {"description": "文档处理成功"},
        400: {"model": ErrorResponse, "description": "请求参数错误"},
        413: {"model": ErrorResponse, "description": "文件过大"},
        422: {"model": ErrorResponse, "description": "参数验证失败"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    }
)
async def ingest_uploaded_document(
    file: UploadFile = File(..., description="要上传的文档文件"),
    chunk_size: Optional[int] = Form(default=None, description="文本分片大小（词元数量）", ge=50, le=2000),
    chunk_overlap: Optional[int] = Form(default=None, description="相邻分片间的重叠词元数量", ge=0, le=500),
    document_service: DocumentService = Depends(get_document_service)
) -> IngestResponse:
    """
    文档上传摄取接口

    处理文件上传和文档向量化，支持以下功能：
    - 多种文档格式支持（.txt, .md, .pdf, .docx, .doc, .html, .xml, .eml, .msg）
    - 文件大小验证
    - 基于词元数量的文本分片
    - 文档向量化和存储
    - 完整的错误处理

    Args:
        file: 上传的文档文件
        chunk_size: 文本分片大小（词元数量）
        chunk_overlap: 相邻分片间的重叠词元数量
        document_service: 文档处理服务

    Returns:
        IngestResponse: 处理结果和统计信息

    Raises:
        HTTPException: 各种错误情况的 HTTP 异常
    """
    logger.info(f"开始处理文档上传摄取请求: {file.filename}")

    # 验证文件名和格式
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件名不能为空")

    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(f"不支持的文件格式 '{file_extension}'。支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}")

    # 验证分片参数
    if chunk_overlap is not None and chunk_size is not None and chunk_overlap >= chunk_size:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="chunk_overlap 必须小于 chunk_size")

    temp_file_path = None
    try:
        # 创建临时文件并流式写入
        max_file_size = 50 * 1024 * 1024  # 50MB
        file_size = 0

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            while chunk := await file.read(8192):
                file_size += len(chunk)
                if file_size > max_file_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"文件过大，最大支持 {max_file_size // (1024*1024)}MB"
                    )
                temp_file.write(chunk)

        if file_size == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件内容为空")

        logger.info(f"文件已保存到临时路径: {temp_file_path}, 大小: {file_size} 字节")

        # 异步处理文档
        result = await run_in_threadpool(
            document_service.process_document,
            document_path=temp_file_path,
            original_filename=file.filename,
            file_size=file_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # 构建响应
        response = IngestResponse(
            success=True,
            message="文档上传和处理成功",
            document_path=file.filename,
            chunks_created=result["chunks_created"],
            chunks_stored=result["chunks_stored"],
            text_length=result["text_length"],
            processing_time=result["processing_time"],
            chunk_size=result["chunk_size"],
            chunk_overlap=result["chunk_overlap"],
            embedding_dimension=result.get("embedding_dimension"),
            collection_name=result.get("collection_name")
        )

        logger.info(f"文档上传摄取完成: {file.filename}, 分片数: {result['chunks_created']}")
        return response

    finally:
        # 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                logger.info(f"临时文件已删除: {temp_file_path}")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {temp_file_path}, 错误: {str(e)}")


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
    logger.info(f"开始处理检索请求: {request.query[:50]}...")

    # 执行检索
    result = await run_in_threadpool(
        retriever.retrieve,
        query=request.query,
        retrieval_k=request.retrieval_k,
        top_k=request.top_k,
        use_reranker=request.use_reranker
    )

    # 转换结果格式
    document_chunks = [DocumentChunk(**chunk_data) for chunk_data in result["results"]]

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
    logger.info("执行系统健康检查")

    # 获取各服务的健康状态
    doc_health = await run_in_threadpool(document_service.health_check)
    retriever_health = await run_in_threadpool(retriever.health_check)

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


@router.get(
    "/documents",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="获取所有文档信息",
    description="返回已处理的所有文档的列表及其详细信息"
)
async def get_documents(
    document_service: DocumentService = Depends(get_document_service)
) -> StatsResponse:
    """
    获取所有文档信息接口

    返回一个包含所有已处理文档详细信息的列表。

    Returns:
        list[DocumentInfo]: 文档信息列表。
    """
    logger.info("获取所有文档信息")

    # 并行获取统计信息和文档列表
    stats_future = run_in_threadpool(document_service.get_system_stats)
    documents_future = run_in_threadpool(document_service.list_documents_with_stats)

    stats = await stats_future
    documents = await documents_future

    document_info_list = [DocumentInfo(**doc) for doc in documents]
    logger.info(f"成功获取 {len(document_info_list)} 篇文档的信息")

    response = StatsResponse(
        system_stats=stats,
        documents=document_info_list
    )

    return response


@router.delete(
    "/documents/{document_path:path}",
    response_model=DeleteDocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="删除文档",
    description="删除指定的文档及其所有分片数据",
    responses={
        200: {"description": "文档删除成功"},
        404: {"model": ErrorResponse, "description": "文档不存在"},
        500: {"model": ErrorResponse, "description": "服务器内部错误"}
    }
)
async def delete_document(
    document_path: str,
    document_service: DocumentService = Depends(get_document_service)
) -> DeleteDocumentResponse:
    """
    删除文档接口

    删除指定的文档及其在向量数据库中的所有分片数据，支持以下功能：
    - 验证文档路径
    - 删除向量数据库中的分片
    - 返回删除统计信息
    - 完整的错误处理

    Args:
        document_path: 要删除的文档路径
        document_service: 文档处理服务

    Returns:
        DeleteDocumentResponse: 删除结果和统计信息

    Raises:
        HTTPException: 各种错误情况的 HTTP 异常
    """
    logger.info(f"开始处理文档删除请求: {document_path}")

    # 删除文档
    result = await run_in_threadpool(document_service.delete_document, document_path)

    # 构建响应
    if result["status"] == "not_found":
        raise FileNotFoundError(f"文档不存在: {document_path}")

    response = DeleteDocumentResponse(
        success=True,
        message="文档删除成功",
        document_path=result["document_path"],
        chunks_deleted=result["chunks_deleted"],
        processing_time=result["processing_time"],
        status=result["status"]
    )

    logger.info(f"文档删除完成: {document_path}, 删除分片数: {result['chunks_deleted']}")
    return response


@admin_router.get(
    "/admin",
    response_class=HTMLResponse,
    summary="管理页面",
    description="显示文档管理界面，支持文档上传、查看和删除操作"
)
async def admin_page(request: Request):
    """
    管理页面接口

    返回文档管理的HTML界面，提供以下功能：
    - 文档列表查看
    - 文档上传功能
    - 文档删除功能
    - 系统状态监控

    Args:
        request: FastAPI请求对象

    Returns:
        HTMLResponse: 管理页面HTML内容
    """
    logger.info("访问管理页面")
    # 渲染管理页面模板
    return templates.TemplateResponse(
        "admin.html",
        {"request": request}
    )


@admin_router.get(
    "/admin/search",
    response_class=HTMLResponse,
    summary="检索查询页面",
    description="显示检索查询界面，支持在知识库中搜索相关信息"
)
async def admin_search_page(request: Request):
    """
    检索查询页面接口

    返回检索查询的HTML界面，提供以下功能：
    - 输入查询内容
    - 配置检索参数
    - 显示检索结果
    - 结果详细信息展示

    Args:
        request: FastAPI请求对象

    Returns:
        HTMLResponse: 检索查询页面HTML内容
    """
    logger.info("访问检索查询页面")
    # 渲染检索查询页面模板
    return templates.TemplateResponse(
        "search.html",
        {"request": request}
    )
