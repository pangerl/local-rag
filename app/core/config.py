"""
配置管理模块
集中管理系统配置参数，支持环境变量覆盖
"""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """系统配置类"""

    # 数据文件基础路径
    DATA_PATH: str = Field(default="data", description="数据文件基础路径")

    # 模型配置
    EMBEDDING_MODEL_DIR: str = Field(default="Qwen3-Embedding-0.6B", description="嵌入模型目录名")
    RERANKER_MODEL_DIR: str = Field(default="Qwen3-Reranker-0.6B", description="重排序模型目录名")
    EMBEDDING_INSTRUCTION: str = Field(
        default="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        description="嵌入模型的查询指令，如果模型不需要则设为空字符串"
    )
    RERANKER_INSTRUCTION: str = Field(
        default="Given a web search query, retrieve relevant passages that answer the query",
        description="重排序模型的指令，如果模型不需要则设为空字符串"
    )
    EMBEDDING_DEVICE: str = Field(default="cpu", description="嵌入模型运行设备 (例如 'cpu', 'cuda')")
    RERANKER_DEVICE: str = Field(default="cpu", description="重排序模型运行设备 (例如 'cpu', 'cuda')")
    EMBEDDING_MAX_LENGTH: int = Field(default=8192, description="嵌入模型最大序列长度")
    EMBEDDING_BATCH_SIZE: int = Field(default=32, description="嵌入模型批处理大小")

    # 数据库配置
    COLLECTION_NAME: str = Field(default="documents", description="文档集合名称")

    # 分片配置
    DEFAULT_CHUNK_SIZE: int = Field(default=500, description="默认分片大小（词元数）")
    DEFAULT_CHUNK_OVERLAP: int = Field(default=50, description="默认分片重叠（词元数）")

    # 检索配置
    DEFAULT_RETRIEVAL_K: int = Field(default=10, description="默认候选文档数量")
    DEFAULT_TOP_K: int = Field(default=3, description="默认返回结果数量")
    RERANKER_MAX_LENGTH: int = Field(default=512, description="重排序模型最大序列长度")

    # 日志配置
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FILE: str = Field(default="logs/app.log", description="日志文件路径")

    # API 配置
    API_HOST: str = Field(default="0.0.0.0", description="API 服务主机")
    API_PORT: int = Field(default=8000, description="API 服务端口")

    # 支持的文件格式
    SUPPORTED_FORMATS: list = Field(default=['.txt', '.md', '.pdf', '.docx', '.doc'], description="支持的文档格式")

    @computed_field
    @property
    def MODEL_BASE_PATH(self) -> str:
        """模型文件基础路径"""
        return str(Path(self.DATA_PATH) / "models")

    @computed_field
    @property
    def CHROMA_DB_PATH(self) -> str:
        """ChromaDB 数据库路径"""
        return str(Path(self.DATA_PATH) / "chroma_db")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

    @property
    def embedding_model_path(self) -> Path:
        """获取嵌入模型完整路径"""
        return Path(self.MODEL_BASE_PATH) / self.EMBEDDING_MODEL_DIR

    @property
    def reranker_model_path(self) -> Path:
        """获取重排序模型完整路径"""
        return Path(self.MODEL_BASE_PATH) / self.RERANKER_MODEL_DIR

    @property
    def chroma_db_full_path(self) -> Path:
        """获取 ChromaDB 完整路径"""
        return Path(self.CHROMA_DB_PATH)

    @property
    def log_file_path(self) -> Path:
        """获取日志文件完整路径"""
        return Path(self.LOG_FILE)

    def validate_paths(self) -> dict:
        """验证关键路径是否存在"""
        validation_results = {
            "embedding_model": self.embedding_model_path.exists(),
            "reranker_model": self.reranker_model_path.exists(),
            "chroma_db_dir": self.chroma_db_full_path.parent.exists(),
            "log_dir": self.log_file_path.parent.exists()
        }
        return validation_results


# 全局配置实例
settings = Settings()
