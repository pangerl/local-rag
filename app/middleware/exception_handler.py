"""
统一异常处理中间件
提供全局异常处理和错误响应格式化
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from app.core.exceptions import (
    LocalRAGException,
    ModelLoadError,
    DocumentProcessError,
    DatabaseError,
    ValidationError as CustomValidationError,
    UnsupportedFormatError,
    FileNotFoundError
)
from app.api.models import ErrorResponse

logger = logging.getLogger(__name__)


class ExceptionHandlerMiddleware:
    """统一异常处理中间件"""
    
    def __init__(self):
        """初始化异常处理中间件"""
        self.error_mappings = self._setup_error_mappings()
        logger.info("异常处理中间件初始化完成")
    
    def _setup_error_mappings(self) -> Dict[type, Dict[str, Any]]:
        """设置异常类型到 HTTP 状态码的映射"""
        return {
            # 自定义异常映射
            ModelLoadError: {
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "error_type": "ModelLoadError",
                "default_message": "模型加载失败，请检查模型文件是否存在且完整"
            },
            DocumentProcessError: {
                "status_code": status.HTTP_400_BAD_REQUEST,
                "error_type": "DocumentProcessError",
                "default_message": "文档处理失败"
            },
            DatabaseError: {
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "error_type": "DatabaseError",
                "default_message": "数据库操作失败，请稍后重试"
            },
            UnsupportedFormatError: {
                "status_code": status.HTTP_400_BAD_REQUEST,
                "error_type": "UnsupportedFormatError",
                "default_message": "不支持的文件格式，仅支持 .txt 和 .md 格式"
            },
            FileNotFoundError: {
                "status_code": status.HTTP_404_NOT_FOUND,
                "error_type": "FileNotFoundError",
                "default_message": "文件不存在"
            },
            CustomValidationError: {
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "error_type": "ValidationError",
                "default_message": "参数验证失败"
            },
            # 标准异常映射
            ValidationError: {
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "error_type": "ValidationError",
                "default_message": "请求参数验证失败"
            },
            ValueError: {
                "status_code": status.HTTP_400_BAD_REQUEST,
                "error_type": "ValueError",
                "default_message": "参数值错误"
            },
            KeyError: {
                "status_code": status.HTTP_400_BAD_REQUEST,
                "error_type": "KeyError",
                "default_message": "缺少必需的参数"
            },
            PermissionError: {
                "status_code": status.HTTP_403_FORBIDDEN,
                "error_type": "PermissionError",
                "default_message": "权限不足"
            },
            TimeoutError: {
                "status_code": status.HTTP_408_REQUEST_TIMEOUT,
                "error_type": "TimeoutError",
                "default_message": "请求超时"
            }
        }
    
    def _get_client_info(self, request: Request) -> Dict[str, Any]:
        """获取客户端信息"""
        return {
            "ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path
        }
    
    def _create_error_response(
        self,
        exception: Exception,
        request: Request,
        status_code: int = None,
        error_type: str = None,
        message: str = None
    ) -> ErrorResponse:
        """创建标准化的错误响应"""
        
        # 获取异常映射信息
        exc_type = type(exception)
        mapping = self.error_mappings.get(exc_type, {})
        
        # 确定状态码
        if status_code is None:
            status_code = mapping.get("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # 确定错误类型
        if error_type is None:
            error_type = mapping.get("error_type", exc_type.__name__)
        
        # 确定错误消息
        if message is None:
            if hasattr(exception, 'message'):
                message = exception.message
            elif hasattr(exception, 'detail'):
                message = exception.detail
            else:
                message = mapping.get("default_message", str(exception))
        
        # 构建详细信息
        details = {
            "exception_type": exc_type.__name__,
            "timestamp": time.time(),
            "path": request.url.path,
            "method": request.method
        }
        
        # 添加自定义异常的详细信息
        if isinstance(exception, LocalRAGException) and hasattr(exception, 'details'):
            details.update(exception.details)
        
        # 添加 Pydantic 验证错误的详细信息
        if isinstance(exception, ValidationError):
            validation_errors = []
            for error in exception.errors():
                validation_errors.append({
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                })
            details["validation_errors"] = validation_errors
        
        # 添加 HTTP 异常的详细信息
        if isinstance(exception, HTTPException):
            details["status_code"] = exception.status_code
        
        return ErrorResponse(
            error=error_type,
            message=message,
            details=details
        )
    
    def _log_exception(
        self,
        exception: Exception,
        request: Request,
        status_code: int,
        response_time: float = None
    ):
        """记录异常日志"""
        client_info = self._get_client_info(request)
        
        # 构建日志消息
        log_msg = (
            f"[{client_info['ip']}] {client_info['method']} {client_info['path']} - "
            f"异常: {type(exception).__name__}: {str(exception)} - "
            f"状态码: {status_code}"
        )
        
        if response_time is not None:
            log_msg += f" - 耗时: {response_time:.3f}s"
        
        # 根据状态码选择日志级别
        if status_code >= 500:
            logger.error(log_msg, exc_info=True)
        elif status_code >= 400:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        # 记录详细的异常信息（仅用于调试）
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"异常详情: {traceback.format_exc()}")
            logger.debug(f"请求头: {dict(request.headers)}")
    
    async def handle_validation_error(
        self,
        request: Request,
        exc: ValidationError
    ) -> JSONResponse:
        """处理 Pydantic 验证错误"""
        error_response = self._create_error_response(exc, request)
        self._log_exception(exc, request, error_response.details.get("status_code", 422))
        
        return JSONResponse(
            status_code=422,
            content=jsonable_encoder(error_response.model_dump())
        )
    
    async def handle_local_rag_exception(
        self,
        request: Request,
        exc: LocalRAGException
    ) -> JSONResponse:
        """处理 Local RAG 自定义异常"""
        error_response = self._create_error_response(exc, request)
        mapping = self.error_mappings.get(type(exc), {})
        status_code = mapping.get("status_code", 500)
        
        self._log_exception(exc, request, status_code)
        
        return JSONResponse(
            status_code=status_code,
            content=jsonable_encoder(error_response.model_dump())
        )
    
    async def handle_http_exception(
        self,
        request: Request,
        exc: HTTPException
    ) -> JSONResponse:
        """处理 HTTP 异常"""
        error_response = self._create_error_response(
            exc, request, exc.status_code, f"HTTP{exc.status_code}", exc.detail
        )
        
        self._log_exception(exc, request, exc.status_code)
        
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder(error_response.model_dump())
        )
    
    async def handle_general_exception(
        self,
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """处理通用异常"""
        error_response = self._create_error_response(exc, request)
        mapping = self.error_mappings.get(type(exc), {})
        status_code = mapping.get("status_code", 500)
        
        self._log_exception(exc, request, status_code)
        
        return JSONResponse(
            status_code=status_code,
            content=jsonable_encoder(error_response.model_dump())
        )


# 全局异常处理器实例
exception_handler = ExceptionHandlerMiddleware()