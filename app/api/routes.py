"""
API 路由模块
定义 FastAPI 路由和端点处理逻辑
"""

import logging
import tempfile
import os
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime
from pathlib import Path

from app.api.models import (
    IngestRequest,
    IngestResponse,
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

# 支持的文档格式
SUPPORTED_EXTENSIONS = ['.txt', '.md', '.pdf', '.docx', '.doc', '.html', '.xml', '.eml', '.msg']

# 配置模板目录
templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# 创建路由器
router = APIRouter(prefix="/api/v1", tags=["Local RAG API"])

# 创建管理页面路由器（不带前缀）
admin_router = APIRouter(tags=["Admin Interface"])

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

        # 验证文件路径和格式
        doc_path = Path(request.document_path)
        if not doc_path.is_file():
            raise FileNotFoundError(f"文件不存在: {request.document_path}")

        file_extension = doc_path.suffix.lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(f"不支持的文件格式 '{file_extension}'。支持的格式: {', '.join(SUPPORTED_EXTENSIONS)}")

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
            detail=str(e)
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
    temp_file_path = None

    try:
        logger.info(f"开始处理文档上传摄取请求: {file.filename}")

        # 验证文件名
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件名不能为空"
            )

        # 验证文件格式
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件格式，仅支持: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        # 验证文件大小（限制为50MB）
        max_file_size = 50 * 1024 * 1024  # 50MB
        file_content = await file.read()
        file_size = len(file_content)

        if file_size > max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"文件过大，最大支持 {max_file_size // (1024*1024)}MB"
            )

        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="文件内容为空"
            )

        # 验证分片参数
        if chunk_overlap is not None and chunk_size is not None:
            if chunk_overlap >= chunk_size:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="chunk_overlap 必须小于 chunk_size"
                )

        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension,
            prefix=f"upload_{Path(file.filename).stem}_"
        ) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        logger.info(f"文件已保存到临时路径: {temp_file_path}")

        # 处理文档
        result = document_service.process_document(
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
            document_path=file.filename,  # 使用原始文件名而不是临时路径
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

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise

    except UnsupportedFormatError as e:
        logger.error(f"不支持的文件格式: {file.filename}, 错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式: {str(e)}"
        )

    except DocumentProcessError as e:
        logger.error(f"文档处理错误: {file.filename}, 错误: {str(e)}")
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
        logger.error(f"文档上传摄取过程中发生未知错误: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器内部错误，请稍后重试"
        )

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
    try:
        logger.info("获取所有文档信息")
                # 获取统计信息
        stats = document_service.get_system_stats()
        documents = document_service.list_documents_with_stats()
        document_info_list = [DocumentInfo(**doc) for doc in documents]
        logger.info(f"成功获取 {len(document_info_list)} 篇文档的信息")

        response = StatsResponse(
            system_stats=stats,
            documents=document_info_list
        )

        return response
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取文档列表失败，请稍后重试"
        )


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
    try:
        logger.info(f"开始处理文档删除请求: {document_path}")

        # 删除文档
        result = document_service.delete_document(document_path)

        # 构建响应
        if result["status"] == "not_found":
            response = DeleteDocumentResponse(
                success=False,
                message=f"文档不存在: {document_path}",
                document_path=document_path,
                chunks_deleted=0,
                processing_time=result["processing_time"],
                status="not_found"
            )
            logger.warning(f"文档删除失败，文档不存在: {document_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"文档不存在: {document_path}"
            )
        else:
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

    except HTTPException:
        # 重新抛出 HTTP 异常
        raise

    except FileNotFoundError as e:
        logger.error(f"文档不存在: {document_path}, 错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"文档不存在: {document_path}"
        )

    except DocumentProcessError as e:
        logger.error(f"文档删除错误: {document_path}, 错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"文档删除失败: {str(e)}"
        )

    except DatabaseError as e:
        logger.error(f"数据库错误: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="数据库操作失败，请稍后重试"
        )

    except Exception as e:
        logger.error(f"删除文档时发生未知错误: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="服务器内部错误，请稍后重试"
        )


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
    try:
        logger.info("访问管理页面")

        # 渲染管理页面模板
        return templates.TemplateResponse(
            "admin.html",
            {"request": request}
        )

    except Exception as e:
        logger.error(f"渲染管理页面失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="管理页面加载失败，请稍后重试"
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
    try:
        logger.info("访问检索查询页面")

        # 渲染检索查询页面模板
        return templates.TemplateResponse(
            "search.html",
            {"request": request}
        )

    except Exception as e:
        logger.error(f"渲染检索查询页面失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="检索查询页面加载失败，请稍后重试"
        )
