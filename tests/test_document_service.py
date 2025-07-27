"""
文档处理服务单元测试
测试 DocumentService 的完整文档处理流程
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.core.config import Settings
from app.services.document_service import DocumentService
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.core.exceptions import (
    DocumentProcessError, 
    UnsupportedFormatError, 
    FileNotFoundError,
    DatabaseError,
    ModelLoadError
)


class TestDocumentService:
    """DocumentService 测试类"""
    
    @pytest.fixture
    def mock_settings(self):
        """创建模拟配置对象"""
        settings = Mock(spec=Settings)
        settings.DEFAULT_CHUNK_SIZE = 300
        settings.DEFAULT_CHUNK_OVERLAP = 50
        return settings
    
    @pytest.fixture
    def mock_db_service(self):
        """创建模拟数据库服务"""
        return Mock(spec=ChromaDBService)
    
    @pytest.fixture
    def mock_model_loader(self):
        """创建模拟模型加载器"""
        return Mock(spec=ModelLoader)
    
    @pytest.fixture
    def temp_file(self):
        """创建临时测试文件"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write("这是一个测试文档的内容。\n包含多行文本用于测试分片功能。")
        temp_file.close()
        
        yield temp_file.name
        
        # 清理临时文件
        Path(temp_file.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def document_service(self, mock_settings, mock_db_service, mock_model_loader):
        """创建 DocumentService 实例"""
        return DocumentService(mock_settings, mock_db_service, mock_model_loader)
    
    def test_init(self, document_service, mock_settings, mock_db_service, mock_model_loader):
        """测试 DocumentService 初始化"""
        assert document_service.settings == mock_settings
        assert document_service.db_service == mock_db_service
        assert document_service.model_loader == mock_model_loader
        assert document_service.document_processor is not None
        assert document_service.chunker is not None
        assert document_service.vector_store is not None
        assert document_service.processing_stats["total_documents_processed"] == 0
    
    @patch('app.services.document_service.DocumentService._validate_document')
    @patch('app.services.document_service.DocumentService._extract_document_text')
    @patch('app.services.document_service.DocumentService._chunk_document_text')
    @patch('app.services.document_service.DocumentService._store_document_chunks')
    def test_process_document_success(self, mock_store, mock_chunk, mock_extract, 
                                    mock_validate, document_service):
        """测试文档处理成功"""
        # 设置模拟返回值
        mock_extract.return_value = "测试文档内容"
        mock_chunk.return_value = ["分片1", "分片2", "分片3"]
        mock_store.return_value = {
            "chunks_stored": 3,
            "embedding_dimension": 768,
            "collection_name": "test_collection"
        }
        
        result = document_service.process_document("test.txt", chunk_size=200, chunk_overlap=30)
        
        assert result["status"] == "success"
        assert result["document_path"] == "test.txt"
        assert result["chunks_created"] == 3
        assert result["chunks_stored"] == 3
        assert result["chunk_size"] == 200
        assert result["chunk_overlap"] == 30
        assert "processing_time" in result
        
        # 验证调用
        mock_validate.assert_called_once_with("test.txt")
        mock_extract.assert_called_once_with("test.txt")
        mock_chunk.assert_called_once_with("测试文档内容", 200, 30)
        mock_store.assert_called_once_with("test.txt", ["分片1", "分片2", "分片3"])
        
        # 验证统计信息更新
        assert document_service.processing_stats["total_documents_processed"] == 1
        assert document_service.processing_stats["total_chunks_created"] == 3
    
    @patch('app.services.document_service.DocumentService._validate_document')
    def test_process_document_validation_failure(self, mock_validate, document_service):
        """测试文档验证失败"""
        mock_validate.side_effect = FileNotFoundError("文件不存在")
        
        with pytest.raises(FileNotFoundError):
            document_service.process_document("nonexistent.txt")
    
    @patch('app.services.document_service.DocumentService._validate_document')
    @patch('app.services.document_service.DocumentService._extract_document_text')
    def test_process_document_extraction_failure(self, mock_extract, mock_validate, document_service):
        """测试文本提取失败"""
        mock_extract.side_effect = DocumentProcessError("文本提取失败")
        
        with pytest.raises(DocumentProcessError):
            document_service.process_document("test.txt")
    
    @patch('app.services.document_service.DocumentService._validate_document')
    @patch('app.services.document_service.DocumentService._extract_document_text')
    @patch('app.services.document_service.DocumentService._chunk_document_text')
    @patch('app.services.document_service.DocumentService._store_document_chunks')
    def test_update_document_success(self, mock_store, mock_chunk, mock_extract, 
                                   mock_validate, document_service):
        """测试文档更新成功"""
        # 设置模拟返回值
        mock_extract.return_value = "更新的文档内容"
        mock_chunk.return_value = ["新分片1", "新分片2"]
        
        # 模拟 vector_store.update_document_chunks
        document_service.vector_store.update_document_chunks = Mock(return_value={
            "old_chunks_deleted": 3,
            "new_chunks_stored": 2
        })
        
        result = document_service.update_document("test.txt")
        
        assert result["status"] == "success"
        assert result["document_path"] == "test.txt"
        assert result["old_chunks_deleted"] == 3
        assert result["new_chunks_stored"] == 2
        assert "processing_time" in result
        
        # 验证调用
        document_service.vector_store.update_document_chunks.assert_called_once_with(
            "test.txt", ["新分片1", "新分片2"]
        )
    
    def test_delete_document_success(self, document_service):
        """测试文档删除成功"""
        # 模拟 vector_store.delete_document_chunks
        document_service.vector_store.delete_document_chunks = Mock(return_value={
            "status": "success",
            "chunks_deleted": 5
        })
        
        result = document_service.delete_document("test.txt")
        
        assert result["status"] == "success"
        assert result["document_path"] == "test.txt"
        assert result["chunks_deleted"] == 5
        assert "processing_time" in result
        
        document_service.vector_store.delete_document_chunks.assert_called_once_with("test.txt")
    
    def test_get_document_info(self, document_service):
        """测试获取文档信息"""
        # 模拟 document_processor.get_file_info
        document_service.document_processor.get_file_info = Mock(return_value={
            "path": "test.txt",
            "exists": True,
            "format": "txt",
            "size": 1024,
            "supported": True
        })
        
        # 模拟 vector_store.get_document_chunks
        document_service.vector_store.get_document_chunks = Mock(return_value={
            "status": "success",
            "total_chunks": 3,
            "chunks": [{"id": "1", "text": "分片1"}]
        })
        
        result = document_service.get_document_info("test.txt")
        
        assert result["document_path"] == "test.txt"
        assert result["file_info"]["exists"] is True
        assert result["storage_info"]["is_stored"] is True
        assert result["storage_info"]["total_chunks"] == 3
    
    @patch('app.services.document_service.DocumentService.process_document')
    def test_batch_process_documents_success(self, mock_process, document_service):
        """测试批量处理文档成功"""
        # 设置模拟返回值
        mock_process.side_effect = [
            {"status": "success", "chunks_created": 2},
            {"status": "success", "chunks_created": 3},
            {"status": "success", "chunks_created": 1}
        ]
        
        document_paths = ["doc1.txt", "doc2.txt", "doc3.txt"]
        result = document_service.batch_process_documents(document_paths)
        
        assert result["total_documents"] == 3
        assert result["successful_documents"] == 3
        assert result["failed_documents"] == 0
        assert len(result["processing_results"]) == 3
        assert len(result["errors"]) == 0
        assert "total_processing_time" in result
    
    @patch('app.services.document_service.DocumentService.process_document')
    def test_batch_process_documents_partial_failure(self, mock_process, document_service):
        """测试批量处理文档部分失败"""
        # 设置模拟返回值（第二个文档失败）
        mock_process.side_effect = [
            {"status": "success", "chunks_created": 2},
            DocumentProcessError("处理失败"),
            {"status": "success", "chunks_created": 1}
        ]
        
        document_paths = ["doc1.txt", "doc2.txt", "doc3.txt"]
        result = document_service.batch_process_documents(document_paths)
        
        assert result["total_documents"] == 3
        assert result["successful_documents"] == 2
        assert result["failed_documents"] == 1
        assert len(result["processing_results"]) == 2
        assert len(result["errors"]) == 1
        assert result["errors"][0]["document_path"] == "doc2.txt"
    
    def test_get_processing_stats(self, document_service):
        """测试获取处理统计信息"""
        # 设置初始统计信息
        document_service.processing_stats["total_documents_processed"] = 5
        document_service.processing_stats["total_chunks_created"] = 25
        
        # 模拟 vector_store.get_storage_stats
        document_service.vector_store.get_storage_stats = Mock(return_value={
            "total_documents": 5,
            "total_chunks": 25,
            "average_chunks_per_document": 5.0
        })
        
        stats = document_service.get_processing_stats()
        
        assert stats["total_documents_processed"] == 5
        assert stats["total_chunks_created"] == 25
        assert stats["storage_total_documents"] == 5
        assert stats["storage_total_chunks"] == 25
        assert stats["average_chunks_per_document"] == 5.0
    
    def test_validate_document_file_not_found(self, document_service):
        """测试文件不存在验证"""
        with pytest.raises(FileNotFoundError):
            document_service._validate_document("nonexistent.txt")
    
    @patch('pathlib.Path.exists')
    def test_validate_document_unsupported_format(self, mock_exists, document_service):
        """测试不支持的文件格式验证"""
        mock_exists.return_value = True
        document_service.document_processor.validate_file_format = Mock(return_value=False)
        
        with pytest.raises(UnsupportedFormatError):
            document_service._validate_document("test.pdf")
    
    def test_extract_document_text_success(self, document_service):
        """测试文本提取成功"""
        document_service.document_processor.extract_text = Mock(return_value=(True, "测试内容"))
        
        result = document_service._extract_document_text("test.txt")
        
        assert result == "测试内容"
    
    def test_extract_document_text_failure(self, document_service):
        """测试文本提取失败"""
        document_service.document_processor.extract_text = Mock(return_value=(False, ""))
        
        with pytest.raises(DocumentProcessError):
            document_service._extract_document_text("test.txt")
    
    def test_extract_document_text_empty(self, document_service):
        """测试提取空文本"""
        document_service.document_processor.extract_text = Mock(return_value=(True, "   "))
        
        with pytest.raises(DocumentProcessError):
            document_service._extract_document_text("test.txt")
    
    def test_chunk_document_text_success(self, document_service):
        """测试文本分片成功"""
        document_service.chunker.chunk_text = Mock(return_value=["分片1", "分片2"])
        
        result = document_service._chunk_document_text("测试内容", 300, 50)
        
        assert result == ["分片1", "分片2"]
        document_service.chunker.chunk_text.assert_called_once_with("测试内容", 300, 50)
    
    def test_chunk_document_text_empty_result(self, document_service):
        """测试文本分片结果为空"""
        document_service.chunker.chunk_text = Mock(return_value=[])
        
        with pytest.raises(DocumentProcessError):
            document_service._chunk_document_text("测试内容", 300, 50)
    
    def test_chunk_document_text_failure(self, document_service):
        """测试文本分片失败"""
        document_service.chunker.chunk_text = Mock(side_effect=Exception("分片失败"))
        
        with pytest.raises(DocumentProcessError):
            document_service._chunk_document_text("测试内容", 300, 50)
    
    def test_store_document_chunks_success(self, document_service):
        """测试文档分片存储成功"""
        document_service.vector_store.store_document_chunks = Mock(return_value={
            "status": "success",
            "chunks_stored": 3
        })
        
        result = document_service._store_document_chunks("test.txt", ["分片1", "分片2", "分片3"])
        
        assert result["status"] == "success"
        assert result["chunks_stored"] == 3
    
    def test_store_document_chunks_failure(self, document_service):
        """测试文档分片存储失败"""
        document_service.vector_store.store_document_chunks = Mock(return_value={
            "status": "failed",
            "error": "存储失败"
        })
        
        with pytest.raises(DatabaseError):
            document_service._store_document_chunks("test.txt", ["分片1"])
    
    def test_update_processing_stats(self, document_service):
        """测试更新处理统计信息"""
        initial_docs = document_service.processing_stats["total_documents_processed"]
        initial_chunks = document_service.processing_stats["total_chunks_created"]
        
        document_service._update_processing_stats(5, 2.5)
        
        assert document_service.processing_stats["total_documents_processed"] == initial_docs + 1
        assert document_service.processing_stats["total_chunks_created"] == initial_chunks + 5
        assert document_service.processing_stats["total_processing_time"] == 2.5
        assert document_service.processing_stats["last_processed_at"] is not None
    
    def test_health_check_healthy(self, document_service):
        """测试健康检查（健康状态）"""
        # 模拟各组件健康状态
        document_service.db_service.health_check = Mock(return_value={"status": "healthy"})
        document_service.model_loader.get_model_info = Mock(return_value={
            "models_loaded": True,
            "embedding_model_loaded": True,
            "reranker_model_loaded": True
        })
        
        result = document_service.health_check()
        
        assert result["status"] == "healthy"
        assert result["components"]["database"]["status"] == "healthy"
        assert result["components"]["models"]["status"] == "healthy"
        assert result["components"]["document_processor"]["status"] == "healthy"
        assert result["components"]["chunker"]["status"] == "healthy"
    
    def test_health_check_unhealthy(self, document_service):
        """测试健康检查（不健康状态）"""
        # 模拟数据库不健康
        document_service.db_service.health_check = Mock(return_value={"status": "unhealthy"})
        document_service.model_loader.get_model_info = Mock(return_value={
            "models_loaded": True,
            "embedding_model_loaded": True,
            "reranker_model_loaded": True
        })
        
        result = document_service.health_check()
        
        assert result["status"] == "unhealthy"
    
    def test_health_check_error(self, document_service):
        """测试健康检查异常"""
        document_service.db_service.health_check = Mock(side_effect=Exception("健康检查失败"))
        
        result = document_service.health_check()
        
        assert result["status"] == "error"
        assert "健康检查失败" in result["error"]