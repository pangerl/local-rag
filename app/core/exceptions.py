"""
异常处理模块
定义系统中使用的自定义异常类
"""

from typing import Optional, Dict, Any


class LocalRAGException(Exception):
    """系统基础异常类"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


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


class UnsupportedFormatError(LocalRAGException):
    """不支持的文件格式错误"""
    pass


class FileNotFoundError(LocalRAGException):
    """文件未找到错误"""
    pass