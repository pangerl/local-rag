"""
向量检索器单元测试
测试 VectorRetriever 的查询向量化、相似性搜索和结果重排序功能
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection

from app.core.config import Settings
from app.services.retriever import VectorRetriever
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.core.exceptions import ModelLoadError, DatabaseError


class TestVectorRetriever:
    """VectorRetriever 测试类"""
    
    @pytest.fixture
    def mock_settings(self):
        """创建模拟配置对象"""
        settings = Mock(spec=Settings)
        settings.DEFAULT_RETRIEVAL_K = 10
        settings.DEFAULT_TOP_K = 3
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
    def retriever(self, mock_settings, mock_db_service, mock_model_loader):
        """创建 VectorRetriever 实例"""
        return VectorRetriever(mock_settings, mock_db_service, mock_model_loader)
    
    def test_init(self, retriever, mock_settings, mock_db_service, mock_model_loader):
        """测试 VectorRetriever 初始化"""
        assert retriever.settings == mock_settings
        assert retriever.db_service == mock_db_service
        assert retriever.model_loader == mock_model_loader
        assert retriever.embedding_model is None
        assert retriever.reranker_model is None
        assert retriever.retrieval_stats["total_queries"] == 0
    
    def test_ensure_embedding_model_first_load(self, retriever, mock_model_loader):
        """测试首次加载嵌入模型"""
        mock_model = Mock(spec=SentenceTransformer)
        mock_model_loader.load_embedding_model.return_value = mock_model
        
        result = retriever._ensure_embedding_model()
        
        assert result == mock_model
        assert retriever.embedding_model == mock_model
        mock_model_loader.load_embedding_model.assert_called_once()
    
    def test_ensure_embedding_model_cached(self, retriever, mock_model_loader):
        """测试嵌入模型缓存"""
        mock_model = Mock(spec=SentenceTransformer)
        retriever.embedding_model = mock_model
        
        result = retriever._ensure_embedding_model()
        
        assert result == mock_model
        mock_model_loader.load_embedding_model.assert_not_called()
    
    def test_ensure_embedding_model_failure(self, retriever, mock_model_loader):
        """测试嵌入模型加载失败"""
        mock_model_loader.load_embedding_model.side_effect = Exception("模型加载失败")
        
        with pytest.raises(ModelLoadError) as exc_info:
            retriever._ensure_embedding_model()
        
        assert "嵌入模型加载失败" in str(exc_info.value)
    
    def test_ensure_reranker_model_first_load(self, retriever, mock_model_loader):
        """测试首次加载重排序模型"""
        mock_model = Mock(spec=SentenceTransformer)
        mock_model_loader.load_reranker_model.return_value = mock_model
        
        result = retriever._ensure_reranker_model()
        
        assert result == mock_model
        assert retriever.reranker_model == mock_model
        mock_model_loader.load_reranker_model.assert_called_once()
    
    def test_ensure_reranker_model_cached(self, retriever, mock_model_loader):
        """测试重排序模型缓存"""
        mock_model = Mock(spec=SentenceTransformer)
        retriever.reranker_model = mock_model
        
        result = retriever._ensure_reranker_model()
        
        assert result == mock_model
        mock_model_loader.load_reranker_model.assert_not_called()
    
    @patch.object(VectorRetriever, '_ensure_embedding_model')
    def test_vectorize_query_success(self, mock_ensure_model, retriever):
        """测试查询向量化成功"""
        # 设置模拟模型
        mock_model = Mock(spec=SentenceTransformer)
        mock_vector = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_vector
        mock_ensure_model.return_value = mock_model
        
        result = retriever._vectorize_query("测试查询")
        
        assert np.array_equal(result, mock_vector[0])
        mock_model.encode.assert_called_once_with(
            ["测试查询"],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
    
    @patch.object(VectorRetriever, '_ensure_embedding_model')
    def test_vectorize_query_failure(self, mock_ensure_model, retriever):
        """测试查询向量化失败"""
        mock_model = Mock(spec=SentenceTransformer)
        mock_model.encode.side_effect = Exception("向量化失败")
        mock_ensure_model.return_value = mock_model
        
        with pytest.raises(ModelLoadError) as exc_info:
            retriever._vectorize_query("测试查询")
        
        assert "查询向量化失败" in str(exc_info.value)
    
    def test_search_similar_chunks_success(self, retriever, mock_db_service):
        """测试相似性搜索成功"""
        # 设置模拟集合
        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [["id1", "id2", "id3"]],
            "documents": [["文档1", "文档2", "文档3"]],
            "metadatas": [[{"path": "doc1.txt"}, {"path": "doc2.txt"}, {"path": "doc3.txt"}]],
            "distances": [[0.1, 0.2, 0.3]]
        }
        mock_db_service.create_or_get_collection.return_value = mock_collection
        
        query_vector = np.array([0.1, 0.2, 0.3])
        result = retriever._search_similar_chunks(query_vector, 5)
        
        assert result["total_found"] == 3
        assert len(result["chunks"]) == 3
        
        # 验证第一个结果
        first_chunk = result["chunks"][0]
        assert first_chunk["id"] == "id1"
        assert first_chunk["text"] == "文档1"
        assert first_chunk["similarity_score"] == 0.9  # 1.0 - 0.1
        assert first_chunk["distance"] == 0.1
        
        # 验证调用
        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_vector.tolist()],
            n_results=5,
            include=["documents", "metadatas", "distances"]
        )
    
    def test_search_similar_chunks_no_results(self, retriever, mock_db_service):
        """测试相似性搜索无结果"""
        # 设置模拟集合返回空结果
        mock_collection = Mock(spec=Collection)
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        mock_db_service.create_or_get_collection.return_value = mock_collection
        
        query_vector = np.array([0.1, 0.2, 0.3])
        result = retriever._search_similar_chunks(query_vector, 5)
        
        assert result["total_found"] == 0
        assert len(result["chunks"]) == 0
    
    def test_search_similar_chunks_failure(self, retriever, mock_db_service):
        """测试相似性搜索失败"""
        mock_db_service.create_or_get_collection.side_effect = Exception("数据库错误")
        
        query_vector = np.array([0.1, 0.2, 0.3])
        
        with pytest.raises(DatabaseError) as exc_info:
            retriever._search_similar_chunks(query_vector, 5)
        
        assert "相似性搜索失败" in str(exc_info.value)
    
    @patch.object(VectorRetriever, '_ensure_reranker_model')
    def test_rerank_results_success(self, mock_ensure_model, retriever):
        """测试结果重排序成功"""
        # 设置模拟重排序模型
        mock_model = Mock()
        mock_scores = np.array([0.8, 0.9, 0.7])
        mock_model.predict.return_value = mock_scores
        mock_ensure_model.return_value = mock_model
        
        chunks = [
            {"id": "1", "text": "文档1", "similarity_score": 0.5},
            {"id": "2", "text": "文档2", "similarity_score": 0.6},
            {"id": "3", "text": "文档3", "similarity_score": 0.4}
        ]
        
        result = retriever._rerank_results("测试查询", chunks)
        
        # 验证重排序分数被添加
        assert all("rerank_score" in chunk for chunk in result)
        
        # 验证按重排序分数降序排序
        assert result[0]["id"] == "2"  # 最高分 0.9
        assert result[1]["id"] == "1"  # 中等分 0.8
        assert result[2]["id"] == "3"  # 最低分 0.7
        
        # 验证调用
        expected_pairs = [
            ["测试查询", "文档1"],
            ["测试查询", "文档2"],
            ["测试查询", "文档3"]
        ]
        mock_model.predict.assert_called_once_with(expected_pairs)
    
    @patch.object(VectorRetriever, '_ensure_reranker_model')
    def test_rerank_results_empty_chunks(self, mock_ensure_model, retriever):
        """测试重排序空结果列表"""
        result = retriever._rerank_results("测试查询", [])
        
        assert result == []
        mock_ensure_model.assert_not_called()
    
    @patch.object(VectorRetriever, '_ensure_reranker_model')
    def test_rerank_results_failure(self, mock_ensure_model, retriever):
        """测试重排序失败时返回原始结果"""
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("重排序失败")
        mock_ensure_model.return_value = mock_model
        
        chunks = [{"id": "1", "text": "文档1"}]
        result = retriever._rerank_results("测试查询", chunks)
        
        # 应该返回原始结果
        assert result == chunks
    
    @patch.object(VectorRetriever, '_vectorize_query')
    @patch.object(VectorRetriever, '_search_similar_chunks')
    @patch.object(VectorRetriever, '_rerank_results')
    def test_retrieve_success_with_reranker(self, mock_rerank, mock_search, mock_vectorize, retriever):
        """测试完整检索流程（使用重排序）"""
        # 设置模拟返回值
        mock_query_vector = np.array([0.1, 0.2, 0.3])
        mock_vectorize.return_value = mock_query_vector
        
        mock_chunks = [
            {"id": "1", "text": "文档1", "similarity_score": 0.8},
            {"id": "2", "text": "文档2", "similarity_score": 0.7},
            {"id": "3", "text": "文档3", "similarity_score": 0.6}
        ]
        mock_search.return_value = {"chunks": mock_chunks, "total_found": 3}
        
        mock_reranked_chunks = [
            {"id": "2", "text": "文档2", "rerank_score": 0.9},
            {"id": "1", "text": "文档1", "rerank_score": 0.8},
            {"id": "3", "text": "文档3", "rerank_score": 0.7}
        ]
        mock_rerank.return_value = mock_reranked_chunks
        
        result = retriever.retrieve("测试查询", retrieval_k=5, top_k=2, use_reranker=True)
        
        assert result["query"] == "测试查询"
        assert result["returned_count"] == 2  # top_k=2
        assert result["total_candidates"] == 3
        assert result["retrieval_k"] == 5
        assert result["top_k"] == 2
        assert result["use_reranker"] is True
        assert len(result["results"]) == 2
        assert result["results"][0]["id"] == "2"  # 重排序后的第一个
        
        # 验证调用
        mock_vectorize.assert_called_once_with("测试查询")
        mock_search.assert_called_once_with(mock_query_vector, 5)
        mock_rerank.assert_called_once_with("测试查询", mock_chunks)
    
    @patch.object(VectorRetriever, '_vectorize_query')
    @patch.object(VectorRetriever, '_search_similar_chunks')
    def test_retrieve_success_without_reranker(self, mock_search, mock_vectorize, retriever):
        """测试完整检索流程（不使用重排序）"""
        # 设置模拟返回值
        mock_query_vector = np.array([0.1, 0.2, 0.3])
        mock_vectorize.return_value = mock_query_vector
        
        mock_chunks = [
            {"id": "1", "text": "文档1", "similarity_score": 0.8},
            {"id": "2", "text": "文档2", "similarity_score": 0.7}
        ]
        mock_search.return_value = {"chunks": mock_chunks, "total_found": 2}
        
        result = retriever.retrieve("测试查询", use_reranker=False)
        
        assert result["use_reranker"] is False
        assert result["returned_count"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["id"] == "1"  # 原始顺序
    
    @patch.object(VectorRetriever, '_vectorize_query')
    def test_retrieve_vectorization_failure(self, mock_vectorize, retriever):
        """测试检索过程中向量化失败"""
        mock_vectorize.side_effect = ModelLoadError("向量化失败")
        
        with pytest.raises(ModelLoadError):
            retriever.retrieve("测试查询")
    
    @patch.object(VectorRetriever, 'retrieve')
    def test_batch_retrieve_success(self, mock_retrieve, retriever):
        """测试批量检索成功"""
        # 设置模拟返回值
        mock_retrieve.side_effect = [
            {"query": "查询1", "returned_count": 2},
            {"query": "查询2", "returned_count": 1},
            {"query": "查询3", "returned_count": 3}
        ]
        
        queries = ["查询1", "查询2", "查询3"]
        result = retriever.batch_retrieve(queries)
        
        assert result["total_queries"] == 3
        assert result["successful_queries"] == 3
        assert result["failed_queries"] == 0
        assert len(result["query_results"]) == 3
        assert len(result["errors"]) == 0
        assert "total_processing_time" in result
    
    @patch.object(VectorRetriever, 'retrieve')
    def test_batch_retrieve_partial_failure(self, mock_retrieve, retriever):
        """测试批量检索部分失败"""
        # 设置模拟返回值（第二个查询失败）
        mock_retrieve.side_effect = [
            {"query": "查询1", "returned_count": 2},
            DatabaseError("检索失败"),
            {"query": "查询3", "returned_count": 1}
        ]
        
        queries = ["查询1", "查询2", "查询3"]
        result = retriever.batch_retrieve(queries)
        
        assert result["total_queries"] == 3
        assert result["successful_queries"] == 2
        assert result["failed_queries"] == 1
        assert len(result["query_results"]) == 2
        assert len(result["errors"]) == 1
        assert result["errors"][0]["query"] == "查询2"
    
    def test_get_retrieval_stats(self, retriever, mock_db_service):
        """测试获取检索统计信息"""
        # 设置初始统计信息
        retriever.retrieval_stats["total_queries"] = 5
        retriever.retrieval_stats["average_retrieval_time"] = 0.1
        
        # 模拟数据库统计信息
        mock_db_service.get_collection_info.return_value = {
            "count": 100,
            "name": "test_collection"
        }
        
        stats = retriever.get_retrieval_stats()
        
        assert stats["total_queries"] == 5
        assert stats["average_retrieval_time"] == 0.1
        assert stats["total_documents_in_db"] == 100
        assert stats["collection_name"] == "test_collection"
    
    def test_update_retrieval_stats(self, retriever):
        """测试更新检索统计信息"""
        # 初始状态
        assert retriever.retrieval_stats["total_queries"] == 0
        
        # 更新统计信息
        retriever._update_retrieval_stats(1.0, 0.8, 0.2)
        
        assert retriever.retrieval_stats["total_queries"] == 1
        assert retriever.retrieval_stats["total_retrieval_time"] == 0.8
        assert retriever.retrieval_stats["total_rerank_time"] == 0.2
        assert retriever.retrieval_stats["average_retrieval_time"] == 0.8
        assert retriever.retrieval_stats["average_rerank_time"] == 0.2
        assert retriever.retrieval_stats["last_query_at"] is not None
        
        # 再次更新
        retriever._update_retrieval_stats(2.0, 1.0, 0.4)
        
        assert retriever.retrieval_stats["total_queries"] == 2
        assert retriever.retrieval_stats["average_retrieval_time"] == 0.9  # (0.8 + 1.0) / 2
        assert abs(retriever.retrieval_stats["average_rerank_time"] - 0.3) < 1e-10  # (0.2 + 0.4) / 2
    
    def test_health_check_healthy(self, retriever, mock_db_service, mock_model_loader):
        """测试健康检查（健康状态）"""
        # 模拟各组件健康状态
        mock_db_service.health_check.return_value = {"status": "healthy"}
        mock_model_loader.get_model_info.return_value = {
            "models_loaded": True,
            "embedding_model_loaded": True,
            "reranker_model_loaded": True
        }
        mock_db_service.get_collection_info.return_value = {"count": 100}
        
        result = retriever.health_check()
        
        assert result["status"] == "healthy"
        assert result["components"]["database"]["status"] == "healthy"
        assert result["components"]["models"]["status"] == "healthy"
        assert result["components"]["data"]["status"] == "healthy"
    
    def test_health_check_unhealthy(self, retriever, mock_db_service, mock_model_loader):
        """测试健康检查（不健康状态）"""
        # 模拟数据库不健康
        mock_db_service.health_check.return_value = {"status": "unhealthy"}
        mock_model_loader.get_model_info.return_value = {
            "models_loaded": True,
            "embedding_model_loaded": True,
            "reranker_model_loaded": True
        }
        
        result = retriever.health_check()
        
        assert result["status"] == "unhealthy"
    
    @patch.object(VectorRetriever, 'retrieve')
    def test_test_retrieval_success(self, mock_retrieve, retriever):
        """测试检索功能测试成功"""
        mock_retrieve.return_value = {
            "returned_count": 3,
            "total_candidates": 10,
            "timing": {"total_time": 0.5, "search_time": 0.3, "rerank_time": 0.2}
        }
        
        result = retriever.test_retrieval("测试查询")
        
        assert result["status"] == "success"
        assert result["test_query"] == "测试查询"
        assert result["results_count"] == 3
        assert result["total_candidates"] == 10
        assert "timing" in result
        
        mock_retrieve.assert_called_once_with("测试查询", retrieval_k=5, top_k=3, use_reranker=True)
    
    @patch.object(VectorRetriever, 'retrieve')
    def test_test_retrieval_failure(self, mock_retrieve, retriever):
        """测试检索功能测试失败"""
        mock_retrieve.side_effect = DatabaseError("检索失败")
        
        result = retriever.test_retrieval("测试查询")
        
        assert result["status"] == "failed"
        assert result["test_query"] == "测试查询"
        assert "检索功能测试失败" in result["error"]
        assert result["error_type"] == "DatabaseError"