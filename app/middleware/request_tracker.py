"""
请求追踪中间件
提供请求 ID 生成和追踪功能，集成性能监控
"""

import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.monitoring import metrics_collector
from app.core.logging_config import log_performance

logger = logging.getLogger(__name__)


class RequestTrackerMiddleware(BaseHTTPMiddleware):
    """请求追踪中间件"""
    
    def __init__(self, app, include_request_body: bool = False):
        """
        初始化请求追踪中间件
        
        Args:
            app: FastAPI 应用实例
            include_request_body: 是否在日志中包含请求体（默认 False）
        """
        super().__init__(app)
        self.include_request_body = include_request_body
        logger.info("请求追踪中间件初始化完成")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """处理请求追踪"""
        # 生成请求 ID
        request_id = str(uuid.uuid4())[:8]
        
        # 将请求 ID 添加到请求状态中
        request.state.request_id = request_id
        
        # 获取客户端信息
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # 记录请求开始
        start_time = time.time()
        
        # 构建请求日志信息
        request_info = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length")
        }
        
        # 记录请求开始日志
        logger.info(
            f"[{request_id}] 请求开始: {request.method} {request.url.path} "
            f"来自 {client_ip}"
        )
        
        # 如果需要，记录请求体（仅用于调试）
        if self.include_request_body and logger.isEnabledFor(logging.DEBUG):
            try:
                # 注意：这会消耗请求体，需要小心处理
                body = await request.body()
                if body:
                    logger.debug(f"[{request_id}] 请求体: {body.decode('utf-8')[:500]}...")
            except Exception as e:
                logger.debug(f"[{request_id}] 无法读取请求体: {e}")
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 获取请求和响应大小
            request_size = None
            if request.headers.get("content-length"):
                try:
                    request_size = int(request.headers["content-length"])
                except ValueError:
                    pass
            
            response_size = None
            if response.headers.get("content-length"):
                try:
                    response_size = int(response.headers["content-length"])
                except ValueError:
                    pass
            
            # 记录请求指标到监控系统
            metrics_collector.record_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=response.status_code,
                response_time=process_time,
                request_size=request_size,
                response_size=response_size
            )
            
            # 记录性能日志
            log_performance(
                func_name=f"{request.method} {request.url.path}",
                duration=process_time,
                status_code=response.status_code,
                request_id=request_id,
                client_ip=client_ip,
                request_size=request_size,
                response_size=response_size
            )
            
            # 添加处理时间到响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            # 记录请求完成日志
            status_emoji = "✅" if response.status_code < 400 else "❌"
            logger.info(
                f"[{request_id}] 请求完成: {request.method} {request.url.path} - "
                f"状态码: {response.status_code} {status_emoji} - "
                f"耗时: {process_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration": process_time,
                    "client_ip": client_ip,
                    "request_size": request_size,
                    "response_size": response_size
                }
            )
            
            # 如果响应时间过长，记录警告
            if process_time > 5.0:
                logger.warning(
                    f"[{request_id}] 慢请求警告: {request.method} {request.url.path} "
                    f"耗时 {process_time:.3f}s",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "duration": process_time,
                        "slow_request": True
                    }
                )
            
            return response
            
        except Exception as e:
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录异常请求指标
            metrics_collector.record_request(
                endpoint=request.url.path,
                method=request.method,
                status_code=500,  # 异常默认为 500
                response_time=process_time
            )
            
            # 记录异常日志
            logger.error(
                f"[{request_id}] 请求异常: {request.method} {request.url.path} - "
                f"异常: {type(e).__name__}: {str(e)} - "
                f"耗时: {process_time:.3f}s",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "duration": process_time,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "error": True
                }
            )
            
            # 重新抛出异常，让异常处理器处理
            raise


def get_request_id(request: Request) -> str:
    """
    从请求中获取请求 ID
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        str: 请求 ID，如果不存在则返回 'unknown'
    """
    return getattr(request.state, 'request_id', 'unknown')