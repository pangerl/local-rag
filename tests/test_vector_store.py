"""
向量存储服务单元测试
测试 VectorStore 的向量化、存储和元数据管理功能
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection

from app.core.config import Settings
from app.services.vector_store import VectorStore
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.core.exceptions import DatabaseError, ModelLoadError


class TestVectorStore:
    """VectorStore 测试类"""
    
    @pytest.fixture
    def mock_settings(self):
        """创建模拟配置对象"""
        settings = Mock(spec=Settings)
        settings.embedding_model_path = "models/bge-small-zh-v1.5"
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
    def vector_store(self, mock_settings, mock_db_service, mock_model_loader):
        """创建 VectorStore 实例"""
        return VectorStore(mock_settings, mock_db_service, mock_model_loader)
    
    def test_init(self, vector_store, mock_settings, mock_db_service, mock_model_loader):
        """测试 VectorStore 初始化"""
        assert vector_store.settings == mock_settings
        assert vector_store.db_service == mock_db_service
        assert vector_store.model_loader == mock_model_loader
        assert vector_store.embedding_model is None
    
    def test_ensure_embedding_model_first_load(self, vector_store, mock_model_loader):
        """测试首次加载嵌入模型"""
        mock_model = Mock(spec=SentenceTransformer)
        mock_model_loader.load_embedding_model.return_value = mock_model
        
        result = vector_store._ensure_embedding_model()
        
        assert result == mock_model
        assert vector_store.embedding_model == mock_model
        mock_model_loader.load_embedding_model.assert_called_once()
    
    def test_ensure_embedding_model_cached(self, vector_store, mock_model_loader):
        """测试嵌入模型缓存"""
        mock_model = Mock(spec=SentenceTransformer)
        vector_store.embedding_model = mock_model
        
        result = vector_store._ensure_embedding_model()
        
        assert result == mock_model
        mock_model_loader.load_embedding_model.assert_not_called()
    
    def test_ensure_embedding_model_failure(self, vector_store, mock_model_loader):
        """测试嵌入模型加载失败"""
        mock_model_loader.load_embedding_model.side_effect = Exception("模型加载失败")
        
        with pytest.raises(ModelLoadError) as exc_info:
            vector_store._ensure_embedding_model()
        
        assert "嵌入模型加载失败" in str(exc_info.value)
    
    @patch.object(VectorStore, '_ensure_embedding_model')
    def test_generate_embeddings_success(self, mock_ensure_model, vector_store):
        """测试文本向量化成功"""
        # 设置模拟模型
        mock_model = Mock(spec=SentenceTransformer)
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_ensure_model.return_value = mock_model
        
        texts = ["文本1", "文本2"]
        result = vector_store._generate_embeddings(texts)
        
        assert np.array_equal(result, mock_embeddings)
        mock_model.encode.assert_called_once_with(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
    
    @patch.object(VectorStore, '_ensure_embedding_model')
    def test_generate_embeddings_failure(self, mock_ensure_model, vector_store):
        """测试文本向量化失败"""
        mock_model = Mock(spec=SentenceTransformer)
        mock_model.encode.side_effect = Exception("向量化失败")
        mock_ensure_model.return_value = mock_model
        
        with pytest.raises(ModelLoadError) as exc_info:
            vector_store._generate_embeddings(["文本"])
        
        assert "文本向量化失败" in str(exc_info.value)
    
    def test_prepare_metadata(self, vector_store):
        """测试元数据准备"""
        document_path = "test/document.txt"
        chunk_index = 1
        chunk_text = "这是测试文本"
        total_chunks = 5
        
        with patch('app.services.vector_store.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T12:00:00"
            
            metadata = vector_store._prepare_metadata(
                document_path, chunk_index, chunk_text, total_chunks
            )
        
        assert metadata["document_path"] == document_path
        assert metadata["chunk_index"] == chunk_index
        assert metadata["total_chunks"] == total_chunks
        assert metadata["chunk_length"] == len(chunk_text)
        assert metadata["created_at"] == "2024-01-01T12:00:00"
        assert metadata["chunk_id"] == f"{document_path}_{chunk_index}"
    
    @patch.object(VectorStore, '_generate_embeddings')
    @patch.object(VectorStore, '_prepare_metadata')
    def test_store_document_chunks_success(self, mock_prepare_metadata, mock_generate_embeddings, 
                                         vector_store, mock_db_service):
        """测试文档分片存储成功"""
        # 设置模拟数据
        document_path = "test/document.txt"
        chunks = ["分片1", "分片2"]
        mock_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_collection = Mock(spec=Collection)
        mock_collection.name = "test_collection"
        
        mock_generate_embeddings.return_value = mock_embeddings
        mock_db_service.create_or_get_collection.return_value = mock_collection
        mock_prepare_metadata.side_effect = [
            {"document_path": document_path, "chunk_index": 0},
            {"document_path": document_path, "chunk_index": 1}
        ]
        
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value.hex = "abcd1234abcd1234"
            
            result = vector_store.store_document_chunks(document_path, chunks)
        
        assert result["document_path"] == document_path
        assert result["chunks_stored"] == 2
        assert result["status"] == "success"
        assert result["collection_name"] == "test_collection"
        assert result["embedding_dimension"] == 2
        
        # 验证调用
        mock_generate_embeddings.assert_called_once_with(chunks)
        mock_db_service.create_or_get_collection.assert_called_once()
        mock_collection.add.assert_called_once()
    
    def test_store_document_chunks_empty(self, vector_store):
        """测试存储空分片列表"""
        result = vector_store.store_document_chunks("test.txt", [])
        
        assert result["chunks_stored"] == 0
        assert result["status"] == "skipped"
        assert "没有分片需要存储" in result["message"]
    
    @patch.object(VectorStore, '_generate_embeddings')
    def test_store_document_chunks_embedding_failure(self, mock_generate_embeddings, vector_store):
        """测试向量化失败的情况"""
        mock_generate_embeddings.side_effect = ModelLoadError("向量化失败")
        
        with pytest.raises(ModelLoadError):
            vector_store.store_document_chunks("test.txt", ["分片"])
    
    @patch.object(VectorStore, 'delete_document_chunks')
    @patch.object(VectorStore, 'store_document_chunks')
    def test_update_document_chunks_success(self, mock_store, mock_delete, vector_store):
        """测试文档分片更新成功"""
        document_path = "test.txt"
        chunks = ["新分片1", "新分片2"]
        
        mock_delete.return_value = {"chunks_deleted": 3}
        mock_store.return_value = {"chunks_stored": 2}
        
        result = vector_store.update_document_chunks(document_path, chunks)
        
        assert result["document_path"] == document_path
        assert result["old_chunks_deleted"] == 3
        assert result["new_chunks_stored"] == 2
        assert result["status"] == "success"
        
        mock_delete.assert_called_once_with(document_path)
        mock_store.assert_called_once_with(document_path, chunks)
    
    def test_delete_document_chunks_success(self, vector_store, mock_db_service):
        """测试删除文档分片成功"""
        document_path = "test.txt"
        mock_collection = Mock(spec=Collection)
        mock_collection.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "metadatas": [{"chunk_index": 0}, {"chunk_index": 1}, {"chunk_index": 2}]
        }
        
        mock_db_service.create_or_get_collection.return_value = mock_collection
        
        result = vector_store.delete_document_chunks(document_path)
        
        assert result["document_path"] == document_path
        assert result["chunks_deleted"] == 3
        assert result["status"] == "success"
        
        mock_collection.get.assert_called_once_with(
            where={"document_path": document_path},
            include=["metadatas"]
        )
        mock_collection.delete.assert_called_once_with(ids=["id1", "id2", "id3"])
    
    def test_delete_document_chunks_not_found(self, vector_store, mock_db_service):
        """测试删除不存在的文档分片"""
        document_path = "nonexistent.txt"
        mock_collection = Mock(spec=Collection)
        mock_collection.get.return_value = {"ids": []}
        
        mock_db_service.create_or_get_collection.return_value = mock_collection
        
        result = vector_store.delete_document_chunks(document_path)
        
        assert result["document_path"] == document_path
        assert result["chunks_deleted"] == 0
        assert result["status"] == "not_found"
    
    def test_get_document_chunks_success(self, vector_store, mock_db_service):
        """测试获取文档分片成功"""
        document_path = "test.txt"
        mock_collection = Mock(spec=Collection)
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["分片1", "分片2"],
            "metadatas": [
                {"chunk_index": 1, "document_path": document_path},
                {"chunk_index": 0, "document_path": document_path}
            ]
        }
        
        mock_db_service.create_or_get_collection.return_value = mock_collection
        
        result = vector_store.get_document_chunks(document_path)
        
        assert result["document_path"] == document_path
        assert result["total_chunks"] == 2
        assert result["status"] == "success"
        
        # 验证分片按索引排序
        chunks = result["chunks"]
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[1]["metadata"]["chunk_index"] == 1
    
    def test_get_document_chunks_not_found(self, vector_store, mock_db_service):
        """测试获取不存在的文档分片"""
        document_path = "nonexistent.txt"
        mock_collection = Mock(spec=Collection)
        mock_collection.get.return_value = {"ids": []}
        
        mock_db_service.create_or_get_collection.return_value = mock_collection
        
        result = vector_store.get_document_chunks(document_path)
        
        assert result["document_path"] == document_path
        assert result["total_chunks"] == 0
        assert result["status"] == "not_found"
    
    def test_list_stored_documents_success(self, vector_store, mock_db_service):
        """测试列出存储文档成功"""
        mock_collection = Mock(spec=Collection)
        mock_collection.get.return_value = {
            "metadatas": [
                {
                    "document_path": "doc1.txt",
                    "total_chunks": 3,
                    "created_at": "2024-01-01T12:00:00"
                },
                {
                    "document_path": "doc1.txt",
                    "total_chunks": 3,
                    "created_at": "2024-01-01T12:00:00"
                },
                {
                    "document_path": "doc2.txt",
                    "total_chunks": 2,
                    "created_at": "2024-01-02T12:00:00"
                }
            ]
        }
        
        mock_db_service.create_or_get_collection.return_value = mock_collection
        
        result = vector_store.list_stored_documents()
        
        assert result["total_documents"] == 2
        assert result["total_chunks"] == 3
        assert result["status"] == "success"
        
        # 验证文档统计
        documents = result["documents"]
        doc1 = next(d for d in documents if d["document_path"] == "doc1.txt")
        doc2 = next(d for d in documents if d["document_path"] == "doc2.txt")
        
        assert doc1["chunk_count"] == 2
        assert doc2["chunk_count"] == 1
    
    def test_list_stored_documents_empty(self, vector_store, mock_db_service):
        """测试列出空存储"""
        mock_collection = Mock(spec=Collection)
        mock_collection.get.return_value = {"metadatas": []}
        
        mock_db_service.create_or_get_collection.return_value = mock_collection
        
        result = vector_store.list_stored_documents()
        
        assert result["total_documents"] == 0
        assert result["total_chunks"] == 0
        assert result["status"] == "empty"
    
    @patch.object(VectorStore, 'list_stored_documents')
    def test_get_storage_stats(self, mock_list_documents, vector_store, mock_db_service, mock_settings):
        """测试获取存储统计信息"""
        mock_db_service.get_collection_info.return_value = {
            "name": "test_collection",
            "count": 10
        }
        mock_list_documents.return_value = {
            "total_documents": 3,
            "total_chunks": 10
        }
        # 设置 db_service.settings 属性
        mock_db_service.settings = Mock()
        mock_db_service.settings.chroma_db_full_path = "/path/to/db"
        
        result = vector_store.get_storage_stats()
        
        assert result["collection_name"] == "test_collection"
        assert result["total_chunks"] == 10
        assert result["total_documents"] == 3
        assert result["average_chunks_per_document"] == 10 / 3
        assert result["status"] == "success"
    
    def test_clear_all_data(self, vector_store, mock_db_service):
        """测试清空所有数据"""
        mock_collection = Mock(spec=Collection)
        mock_collection.name = "test_collection"
        mock_db_service.reset_collection.return_value = mock_collection
        
        result = vector_store.clear_all_data()
        
        assert result["status"] == "success"
        assert result["collection_name"] == "test_collection"
        assert "所有数据已清空" in result["message"]
        
        mock_db_service.reset_collection.assert_called_once()
    
    def test_database_error_handling(self, vector_store, mock_db_service):
        """测试数据库错误处理"""
        mock_db_service.create_or_get_collection.side_effect = Exception("数据库错误")
        
        with pytest.raises(DatabaseError) as exc_info:
            vector_store.delete_document_chunks("test.txt")
        
        assert "删除文档分片失败" in str(exc_info.value)