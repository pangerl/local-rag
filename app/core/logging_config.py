"""
日志配置模块
配置系统日志记录，支持结构化日志和性能监控
"""

import logging
import logging.handlers
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .config import settings


class StructuredFormatter(logging.Formatter):
    """结构化日志格式化器"""
    
    def __init__(self, include_extra: bool = True):
        """
        初始化结构化格式化器
        
        Args:
            include_extra: 是否包含额外字段
        """
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为 JSON"""
        # 基础字段
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in {
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'lineno', 'funcName', 'created',
                    'msecs', 'relativeCreated', 'thread', 'threadName',
                    'processName', 'process', 'getMessage', 'exc_info',
                    'exc_text', 'stack_info'
                }:
                    extra_fields[key] = value
            
            if extra_fields:
                log_data["extra"] = extra_fields
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class PerformanceLogFilter(logging.Filter):
    """性能日志过滤器"""
    
    def __init__(self, min_duration: float = 0.1):
        """
        初始化性能日志过滤器
        
        Args:
            min_duration: 最小记录时长（秒）
        """
        super().__init__()
        self.min_duration = min_duration
    
    def filter(self, record: logging.LogRecord) -> bool:
        """过滤性能日志"""
        # 检查是否有 duration 字段
        if hasattr(record, 'duration'):
            return record.duration >= self.min_duration
        return True


class RequestContextFilter(logging.Filter):
    """请求上下文过滤器"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """添加请求上下文信息"""
        # 尝试从当前上下文获取请求信息
        try:
            from contextvars import copy_context
            context = copy_context()
            
            # 这里可以添加从上下文变量获取请求 ID 等信息的逻辑
            # 目前先设置默认值
            if not hasattr(record, 'request_id'):
                record.request_id = getattr(record, 'request_id', 'unknown')
            
        except Exception:
            record.request_id = 'unknown'
        
        return True


def get_logging_config() -> Dict[str, Any]:
    """获取日志配置字典"""
    
    # 确保日志目录存在
    log_dir = settings.log_file_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "structured": {
                "()": StructuredFormatter,
                "include_extra": True
            },
            "performance": {
                "format": "%(asctime)s - PERF - %(name)s - %(message)s - Duration: %(duration).3fs",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "filters": {
            "performance_filter": {
                "()": PerformanceLogFilter,
                "min_duration": 0.1
            },
            "request_context": {
                "()": RequestContextFilter
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "default",
                "stream": "ext://sys.stdout",
                "filters": ["request_context"]
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "detailed",
                "filename": str(settings.log_file_path),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8",
                "filters": ["request_context"]
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(settings.log_file_path.parent / "error.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3,
                "encoding": "utf-8",
                "filters": ["request_context"]
            },
            "structured_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "structured",
                "filename": str(settings.log_file_path.parent / "structured.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf-8",
                "filters": ["request_context"]
            },
            "performance_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "performance",
                "filename": str(settings.log_file_path.parent / "performance.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3,
                "encoding": "utf-8",
                "filters": ["performance_filter"]
            }
        },
        "loggers": {
            "local_rag": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "file", "error_file", "structured_file"],
                "propagate": False
            },
            "local_rag.performance": {
                "level": "INFO",
                "handlers": ["performance_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file"],
                "propagate": False
            }
        },
        "root": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console", "file"]
        }
    }


def setup_logging():
    """设置日志配置"""
    import logging.config
    
    config = get_logging_config()
    logging.config.dictConfig(config)
    
    # 获取主日志记录器
    logger = logging.getLogger("local_rag")
    logger.info("增强日志系统初始化完成")
    
    # 获取性能日志记录器
    perf_logger = logging.getLogger("local_rag.performance")
    
    return logger


def get_performance_logger():
    """获取性能日志记录器"""
    return logging.getLogger("local_rag.performance")


def log_performance(func_name: str, duration: float, **kwargs):
    """
    记录性能日志
    
    Args:
        func_name: 函数名称
        duration: 执行时长
        **kwargs: 额外参数
    """
    perf_logger = get_performance_logger()
    
    # 构建日志消息
    message_parts = [f"Function: {func_name}"]
    for key, value in kwargs.items():
        message_parts.append(f"{key}: {value}")
    
    message = " | ".join(message_parts)
    
    # 记录性能日志
    perf_logger.info(message, extra={"duration": duration, "function": func_name, **kwargs})