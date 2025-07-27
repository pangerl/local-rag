"""
配置模块测试
"""

import pytest
from pathlib import Path
from app.core.config import Settings


def test_settings_default_values():
    """测试配置默认值"""
    settings = Settings()
    
    # 测试模型配置
    assert settings.MODEL_BASE_PATH == "models"
    assert settings.EMBEDDING_MODEL_DIR == "bge-small-zh-v1.5"
    assert settings.RERANKER_MODEL_DIR == "bge-reranker-base"
    
    # 测试数据库配置
    assert settings.CHROMA_DB_PATH == "data/chroma_db"
    assert settings.COLLECTION_NAME == "documents"
    
    # 测试分片配置
    assert settings.DEFAULT_CHUNK_SIZE == 300
    assert settings.DEFAULT_CHUNK_OVERLAP == 50
    
    # 测试检索配置
    assert settings.DEFAULT_RETRIEVAL_K == 10
    assert settings.DEFAULT_TOP_K == 3


def test_settings_path_properties():
    """测试路径属性"""
    settings = Settings()
    
    # 测试嵌入模型路径
    expected_embedding_path = Path("models") / "bge-small-zh-v1.5"
    assert settings.embedding_model_path == expected_embedding_path
    
    # 测试重排序模型路径
    expected_reranker_path = Path("models") / "bge-reranker-base"
    assert settings.reranker_model_path == expected_reranker_path
    
    # 测试 ChromaDB 路径
    expected_chroma_path = Path("data/chroma_db")
    assert settings.chroma_db_full_path == expected_chroma_path
    
    # 测试日志文件路径
    expected_log_path = Path("logs/app.log")
    assert settings.log_file_path == expected_log_path


def test_validate_paths():
    """测试路径验证"""
    settings = Settings()
    
    validation_results = settings.validate_paths()
    
    # 验证返回的字典包含所有必要的键
    expected_keys = ["embedding_model", "reranker_model", "chroma_db_dir", "log_dir"]
    for key in expected_keys:
        assert key in validation_results
        assert isinstance(validation_results[key], bool)


def test_supported_formats():
    """测试支持的文件格式"""
    settings = Settings()
    
    assert ".txt" in settings.SUPPORTED_FORMATS
    assert ".md" in settings.SUPPORTED_FORMATS
    assert len(settings.SUPPORTED_FORMATS) == 2