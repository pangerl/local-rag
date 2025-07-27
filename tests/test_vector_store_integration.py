"""
向量存储服务集成测试
测试 VectorStore 与真实 ChromaDB 和模型的集成功能
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.services.vector_store import VectorStore
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader


class TestVectorStoreIntegration:
    """VectorStore 集成测试类"""
    
    @pytest.fixture
    def temp_db_dir(self):
        """创建临时数据库目录用于测试"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_settings(self, temp_db_dir):
        """创建测试用的配置对象"""
        settings = Settings()
        settings.CHROMA_DB_PATH = temp_db_dir
        settings.COLLECTION_NAME = "test_vector_store"
        return settings
    
    @pytest.fixture
    def db_service(self, test_settings):
        """创建 ChromaDBService 实例"""
        service = ChromaDBService(test_settings)
        yield service
        service.disconnect()
    
    @pytest.fixture
    def mock_model_loader(self):
        """创建模拟模型加载器"""
        loader = Mock(spec=ModelLoader)
        
        # 创建模拟的嵌入模型
        mock_model = Mock(spec=SentenceTransformer)
        
        def mock_encode(texts, **kwargs):
            # 返回模拟的嵌入向量（每个文本对应一个3维向量）
            embeddings = []
            for i, text in enumerate(texts):
                # 基于文本内容生成简单的向量
                vector = [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1]
                embeddings.append(vector)
            return np.array(embeddings)
        
        mock_model.encode = mock_encode
        loader.load_embedding_model.return_value = mock_model
        
        return loader
    
    @pytest.fixture
    def vector_store(self, test_settings, db_service, mock_model_loader):
        """创建 VectorStore 实例"""
        return VectorStore(test_settings, db_service, mock_model_loader)
    
    def test_store_and_retrieve_document_chunks(self, vector_store):
        """测试存储和检索文档分片的完整流程"""
        document_path = "test_document.txt"
        chunks = [
            "这是第一个文档分片，包含一些测试内容。",
            "这是第二个文档分片，包含更多测试内容。",
            "这是第三个文档分片，用于验证存储功能。"
        ]
        
        # 存储文档分片
        store_result = vector_store.store_document_chunks(document_path, chunks)
        
        assert store_result["status"] == "success"
        assert store_result["chunks_stored"] == 3
        assert store_result["document_path"] == document_path
        assert "embedding_dimension" in store_result
        
        # 检索文档分片
        retrieve_result = vector_store.get_document_chunks(document_path)
        
        assert retrieve_result["status"] == "success"
        assert retrieve_result["total_chunks"] == 3
        assert retrieve_result["document_path"] == document_path
        
        # 验证分片内容
        retrieved_chunks = retrieve_result["chunks"]
        assert len(retrieved_chunks) == 3
        
        for i, chunk in enumerate(retrieved_chunks):
            assert chunk["text"] == chunks[i]
            assert chunk["metadata"]["chunk_index"] == i
            assert chunk["metadata"]["document_path"] == document_path
            assert chunk["metadata"]["total_chunks"] == 3
    
    def test_update_document_chunks(self, vector_store):
        """测试更新文档分片"""
        document_path = "update_test.txt"
        
        # 初始存储
        original_chunks = ["原始分片1", "原始分片2"]
        vector_store.store_document_chunks(document_path, original_chunks)
        
        # 验证初始存储
        result = vector_store.get_document_chunks(document_path)
        assert result["total_chunks"] == 2
        
        # 更新分片
        new_chunks = ["新分片1", "新分片2", "新分片3"]
        update_result = vector_store.update_document_chunks(document_path, new_chunks)
        
        assert update_result["status"] == "success"
        assert update_result["old_chunks_deleted"] == 2
        assert update_result["new_chunks_stored"] == 3
        
        # 验证更新后的内容
        result = vector_store.get_document_chunks(document_path)
        assert result["total_chunks"] == 3
        
        retrieved_texts = [chunk["text"] for chunk in result["chunks"]]
        assert retrieved_texts == new_chunks
    
    def test_delete_document_chunks(self, vector_store):
        """测试删除文档分片"""
        document_path = "delete_test.txt"
        chunks = ["删除测试分片1", "删除测试分片2"]
        
        # 存储分片
        vector_store.store_document_chunks(document_path, chunks)
        
        # 验证存储成功
        result = vector_store.get_document_chunks(document_path)
        assert result["total_chunks"] == 2
        
        # 删除分片
        delete_result = vector_store.delete_document_chunks(document_path)
        
        assert delete_result["status"] == "success"
        assert delete_result["chunks_deleted"] == 2
        
        # 验证删除成功
        result = vector_store.get_document_chunks(document_path)
        assert result["status"] == "not_found"
        assert result["total_chunks"] == 0
    
    def test_list_stored_documents(self, vector_store):
        """测试列出存储的文档"""
        # 存储多个文档
        documents = {
            "doc1.txt": ["文档1分片1", "文档1分片2"],
            "doc2.txt": ["文档2分片1", "文档2分片2", "文档2分片3"],
            "doc3.txt": ["文档3分片1"]
        }
        
        for doc_path, chunks in documents.items():
            vector_store.store_document_chunks(doc_path, chunks)
        
        # 列出文档
        result = vector_store.list_stored_documents()
        
        assert result["status"] == "success"
        assert result["total_documents"] == 3
        assert result["total_chunks"] == 6  # 2 + 3 + 1
        
        # 验证文档信息
        doc_paths = [doc["document_path"] for doc in result["documents"]]
        assert "doc1.txt" in doc_paths
        assert "doc2.txt" in doc_paths
        assert "doc3.txt" in doc_paths
        
        # 验证分片计数
        for doc in result["documents"]:
            expected_count = len(documents[doc["document_path"]])
            assert doc["chunk_count"] == expected_count
    
    def test_storage_stats(self, vector_store):
        """测试存储统计信息"""
        # 存储一些测试数据
        vector_store.store_document_chunks("stats_test1.txt", ["分片1", "分片2"])
        vector_store.store_document_chunks("stats_test2.txt", ["分片3", "分片4", "分片5"])
        
        # 获取统计信息
        stats = vector_store.get_storage_stats()
        
        assert stats["status"] == "success"
        assert stats["total_documents"] == 2
        assert stats["total_chunks"] == 5
        assert stats["average_chunks_per_document"] == 2.5
        assert "collection_name" in stats
        assert "database_path" in stats
        assert "embedding_model" in stats
    
    def test_clear_all_data(self, vector_store):
        """测试清空所有数据"""
        # 存储一些测试数据
        vector_store.store_document_chunks("clear_test1.txt", ["分片1"])
        vector_store.store_document_chunks("clear_test2.txt", ["分片2"])
        
        # 验证数据存在
        result = vector_store.list_stored_documents()
        assert result["total_documents"] == 2
        
        # 清空数据
        clear_result = vector_store.clear_all_data()
        assert clear_result["status"] == "success"
        
        # 验证数据已清空
        result = vector_store.list_stored_documents()
        assert result["total_documents"] == 0
        assert result["status"] == "empty"
    
    def test_empty_chunks_handling(self, vector_store):
        """测试空分片列表的处理"""
        result = vector_store.store_document_chunks("empty_test.txt", [])
        
        assert result["status"] == "skipped"
        assert result["chunks_stored"] == 0
        assert "没有分片需要存储" in result["message"]
    
    def test_nonexistent_document_operations(self, vector_store):
        """测试对不存在文档的操作"""
        nonexistent_path = "nonexistent.txt"
        
        # 获取不存在的文档
        result = vector_store.get_document_chunks(nonexistent_path)
        assert result["status"] == "not_found"
        assert result["total_chunks"] == 0
        
        # 删除不存在的文档
        result = vector_store.delete_document_chunks(nonexistent_path)
        assert result["status"] == "not_found"
        assert result["chunks_deleted"] == 0
    
    def test_metadata_integrity(self, vector_store):
        """测试元数据完整性"""
        document_path = "metadata_test.txt"
        chunks = ["元数据测试分片1", "元数据测试分片2", "元数据测试分片3"]
        
        # 存储分片
        vector_store.store_document_chunks(document_path, chunks)
        
        # 检索并验证元数据
        result = vector_store.get_document_chunks(document_path)
        
        for i, chunk in enumerate(result["chunks"]):
            metadata = chunk["metadata"]
            
            assert metadata["document_path"] == document_path
            assert metadata["chunk_index"] == i
            assert metadata["total_chunks"] == 3
            assert metadata["chunk_length"] == len(chunks[i])
            assert "created_at" in metadata
            assert "chunk_id" in metadata
            assert metadata["chunk_id"] == f"{document_path}_{i}"
    
    def test_concurrent_document_operations(self, vector_store):
        """测试并发文档操作"""
        # 同时存储多个文档
        documents = [
            ("concurrent1.txt", ["并发测试1-分片1", "并发测试1-分片2"]),
            ("concurrent2.txt", ["并发测试2-分片1", "并发测试2-分片2"]),
            ("concurrent3.txt", ["并发测试3-分片1", "并发测试3-分片2"])
        ]
        
        # 存储所有文档
        for doc_path, chunks in documents:
            result = vector_store.store_document_chunks(doc_path, chunks)
            assert result["status"] == "success"
        
        # 验证所有文档都存储成功
        all_docs = vector_store.list_stored_documents()
        assert all_docs["total_documents"] == 3
        assert all_docs["total_chunks"] == 6
        
        # 验证每个文档的内容
        for doc_path, expected_chunks in documents:
            result = vector_store.get_document_chunks(doc_path)
            assert result["status"] == "success"
            assert result["total_chunks"] == len(expected_chunks)
            
            retrieved_texts = [chunk["text"] for chunk in result["chunks"]]
            assert retrieved_texts == expected_chunks