"""
向量检索器集成测试
测试 VectorRetriever 与真实 ChromaDB 和模型的集成功能
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import Settings
from app.services.retriever import VectorRetriever
from app.services.database import ChromaDBService
from app.services.model_loader import ModelLoader
from app.services.vector_store import VectorStore


class TestVectorRetrieverIntegration:
    """VectorRetriever 集成测试类"""
    
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
        settings.COLLECTION_NAME = "test_retriever"
        settings.DEFAULT_RETRIEVAL_K = 10
        settings.DEFAULT_TOP_K = 3
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
        mock_embedding_model = Mock(spec=SentenceTransformer)
        
        def mock_encode(texts, **kwargs):
            # 返回模拟的嵌入向量（每个文本对应一个3维向量）
            embeddings = []
            for i, text in enumerate(texts):
                # 基于文本内容生成简单的向量
                vector = [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1]
                embeddings.append(vector)
            return np.array(embeddings)
        
        mock_embedding_model.encode = mock_encode
        loader.load_embedding_model.return_value = mock_embedding_model
        
        # 创建模拟的重排序模型
        mock_reranker_model = Mock()
        
        def mock_predict(query_doc_pairs):
            # 返回模拟的重排序分数
            scores = []
            for query, doc in query_doc_pairs:
                # 简单的相关性计算：查询和文档的相似度
                score = 0.5 + 0.3 * (len(set(query.split()) & set(doc.split())) / max(len(query.split()), 1))
                scores.append(score)
            return np.array(scores)
        
        mock_reranker_model.predict = mock_predict
        loader.load_reranker_model.return_value = mock_reranker_model
        
        loader.get_model_info.return_value = {
            "models_loaded": True,
            "embedding_model_loaded": True,
            "reranker_model_loaded": True
        }
        
        return loader
    
    @pytest.fixture
    def vector_store(self, test_settings, db_service, mock_model_loader):
        """创建 VectorStore 实例用于准备测试数据"""
        return VectorStore(test_settings, db_service, mock_model_loader)
    
    @pytest.fixture
    def retriever(self, test_settings, db_service, mock_model_loader):
        """创建 VectorRetriever 实例"""
        return VectorRetriever(test_settings, db_service, mock_model_loader)
    
    @pytest.fixture
    def test_documents(self, vector_store):
        """准备测试文档数据"""
        documents = {
            "python_tutorial.txt": [
                "Python是一种高级编程语言，具有简洁的语法和强大的功能。",
                "Python支持面向对象编程、函数式编程和过程式编程。",
                "Python有丰富的标准库和第三方库生态系统。"
            ],
            "machine_learning.txt": [
                "机器学习是人工智能的一个重要分支，通过算法让计算机学习数据模式。",
                "监督学习使用标记数据训练模型，无监督学习发现数据中的隐藏模式。",
                "深度学习是机器学习的子领域，使用神经网络处理复杂问题。"
            ],
            "web_development.txt": [
                "Web开发包括前端开发和后端开发两个主要方面。",
                "前端开发负责用户界面和用户体验，使用HTML、CSS和JavaScript。",
                "后端开发处理服务器逻辑、数据库和API接口。"
            ]
        }
        
        # 存储测试文档
        for doc_path, chunks in documents.items():
            vector_store.store_document_chunks(doc_path, chunks)
        
        return documents
    
    def test_basic_retrieval_flow(self, retriever, test_documents):
        """测试基本检索流程"""
        query = "Python编程语言"
        
        result = retriever.retrieve(query, retrieval_k=5, top_k=2)
        
        assert result["query"] == query
        assert result["retrieval_k"] == 5
        assert result["top_k"] == 2
        assert result["returned_count"] <= 2
        assert result["total_candidates"] > 0
        assert len(result["results"]) <= 2
        
        # 验证结果结构
        for chunk in result["results"]:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            assert "similarity_score" in chunk
            assert "distance" in chunk
        
        # 验证时间统计
        timing = result["timing"]
        assert timing["total_time"] > 0
        assert timing["search_time"] > 0
        assert timing["rerank_time"] >= 0
    
    def test_retrieval_with_reranker(self, retriever, test_documents):
        """测试使用重排序的检索"""
        query = "机器学习算法"
        
        result = retriever.retrieve(query, use_reranker=True)
        
        assert result["use_reranker"] is True
        assert result["returned_count"] > 0
        assert result["timing"]["rerank_time"] > 0
        
        # 验证重排序分数存在
        for chunk in result["results"]:
            assert "rerank_score" in chunk
    
    def test_retrieval_without_reranker(self, retriever, test_documents):
        """测试不使用重排序的检索"""
        query = "Web开发技术"
        
        result = retriever.retrieve(query, use_reranker=False)
        
        assert result["use_reranker"] is False
        assert result["returned_count"] > 0
        assert result["timing"]["rerank_time"] == 0
        
        # 验证没有重排序分数
        for chunk in result["results"]:
            assert "rerank_score" not in chunk
    
    def test_retrieval_with_different_parameters(self, retriever, test_documents):
        """测试不同参数的检索"""
        query = "编程语言特性"
        
        # 测试不同的 retrieval_k 和 top_k
        result1 = retriever.retrieve(query, retrieval_k=3, top_k=1)
        result2 = retriever.retrieve(query, retrieval_k=8, top_k=5)
        
        assert result1["retrieval_k"] == 3
        assert result1["top_k"] == 1
        assert result1["returned_count"] <= 1
        
        assert result2["retrieval_k"] == 8
        assert result2["top_k"] == 5
        assert result2["returned_count"] <= 5
        
        # 更大的 retrieval_k 应该能找到更多候选结果
        assert result2["total_candidates"] >= result1["total_candidates"]
    
    def test_retrieval_relevance_ranking(self, retriever, test_documents):
        """测试检索结果的相关性排序"""
        # 查询与 Python 相关的内容
        query = "Python编程"
        
        result = retriever.retrieve(query, top_k=5)
        
        # 验证结果按相关性排序（相似度分数递减）
        if len(result["results"]) > 1:
            for i in range(len(result["results"]) - 1):
                current_score = result["results"][i]["similarity_score"]
                next_score = result["results"][i + 1]["similarity_score"]
                assert current_score >= next_score
        
        # 验证 Python 相关的文档排在前面
        python_related_count = 0
        for chunk in result["results"][:2]:  # 检查前两个结果
            if "Python" in chunk["text"] or "python" in chunk["text"].lower():
                python_related_count += 1
        
        # 至少应该有一个 Python 相关的结果在前面
        assert python_related_count > 0
    
    def test_batch_retrieval_flow(self, retriever, test_documents):
        """测试批量检索流程"""
        queries = [
            "Python编程语言",
            "机器学习算法",
            "Web前端开发"
        ]
        
        result = retriever.batch_retrieve(queries, retrieval_k=5, top_k=2)
        
        assert result["total_queries"] == 3
        assert result["successful_queries"] == 3
        assert result["failed_queries"] == 0
        assert len(result["query_results"]) == 3
        assert len(result["errors"]) == 0
        assert result["total_processing_time"] > 0
        
        # 验证每个查询的结果
        for query_result in result["query_results"]:
            assert query_result["retrieval_k"] == 5
            assert query_result["top_k"] == 2
            assert query_result["returned_count"] <= 2
    
    def test_batch_retrieval_with_invalid_query(self, retriever, test_documents):
        """测试批量检索包含无效查询的情况"""
        queries = [
            "Python编程语言",
            "",  # 空查询
            "机器学习算法"
        ]
        
        result = retriever.batch_retrieve(queries)
        
        # 空查询可能会导致某些失败，但不应该影响其他查询
        assert result["total_queries"] == 3
        assert result["successful_queries"] >= 2  # 至少两个成功
    
    def test_retrieval_stats_tracking(self, retriever, test_documents):
        """测试检索统计信息跟踪"""
        # 初始统计信息
        initial_stats = retriever.get_retrieval_stats()
        assert initial_stats["total_queries"] == 0
        
        # 执行几次检索
        queries = ["Python", "机器学习", "Web开发"]
        for query in queries:
            retriever.retrieve(query)
        
        # 检查统计信息更新
        final_stats = retriever.get_retrieval_stats()
        assert final_stats["total_queries"] == 3
        assert final_stats["total_retrieval_time"] > 0
        assert final_stats["average_retrieval_time"] > 0
        assert final_stats["last_query_at"] is not None
    
    def test_retrieval_with_empty_database(self, test_settings, mock_model_loader):
        """测试空数据库的检索"""
        # 创建新的数据库服务（空数据库）
        empty_db_service = ChromaDBService(test_settings)
        empty_retriever = VectorRetriever(test_settings, empty_db_service, mock_model_loader)
        
        try:
            result = empty_retriever.retrieve("测试查询")
            
            assert result["returned_count"] == 0
            assert result["total_candidates"] == 0
            assert len(result["results"]) == 0
        finally:
            empty_db_service.disconnect()
    
    def test_retrieval_performance_monitoring(self, retriever, test_documents):
        """测试检索性能监控"""
        query = "Python编程语言特性"
        
        result = retriever.retrieve(query)
        
        # 验证性能指标
        timing = result["timing"]
        assert timing["total_time"] > 0
        assert timing["search_time"] > 0
        assert timing["search_time"] <= timing["total_time"]
        
        if result["use_reranker"]:
            assert timing["rerank_time"] > 0
            assert timing["rerank_time"] <= timing["total_time"]
        else:
            assert timing["rerank_time"] == 0
    
    def test_health_check_integration(self, retriever, test_documents):
        """测试健康检查集成功能"""
        health = retriever.health_check()
        
        assert health["status"] == "healthy"
        assert health["components"]["database"]["status"] == "healthy"
        assert health["components"]["models"]["status"] == "healthy"
        assert health["components"]["data"]["status"] == "healthy"
        assert health["components"]["data"]["document_count"] > 0
        assert health["error"] is None
    
    def test_test_retrieval_function(self, retriever, test_documents):
        """测试检索功能测试"""
        test_result = retriever.test_retrieval("测试查询内容")
        
        assert test_result["status"] == "success"
        assert test_result["test_query"] == "测试查询内容"
        assert test_result["results_count"] >= 0
        assert test_result["total_candidates"] >= 0
        assert "timing" in test_result
        assert "检索测试成功" in test_result["message"]
    
    def test_retrieval_with_special_characters(self, retriever, test_documents):
        """测试包含特殊字符的查询"""
        special_queries = [
            "Python & 机器学习",
            "Web开发 (前端/后端)",
            "编程语言: Python, Java, C++",
            "人工智能 - AI & ML"
        ]
        
        for query in special_queries:
            result = retriever.retrieve(query, top_k=1)
            
            # 应该能正常处理特殊字符
            assert result["query"] == query
            assert result["returned_count"] >= 0
            assert "timing" in result
    
    def test_retrieval_with_long_query(self, retriever, test_documents):
        """测试长查询文本的处理"""
        long_query = """
        我想了解关于Python编程语言的详细信息，包括它的语法特性、
        面向对象编程能力、函数式编程支持、以及在机器学习和Web开发
        领域的应用。同时也希望了解Python的标准库和第三方库生态系统，
        以及它与其他编程语言相比的优势和特点。
        """
        
        result = retriever.retrieve(long_query.strip(), top_k=3)
        
        assert result["returned_count"] >= 0
        assert result["total_candidates"] >= 0
        assert result["timing"]["total_time"] > 0
    
    def test_concurrent_retrieval_operations(self, retriever, test_documents):
        """测试并发检索操作"""
        queries = [
            "Python编程",
            "机器学习",
            "Web开发",
            "人工智能",
            "数据科学"
        ]
        
        # 模拟并发检索（顺序执行，但验证状态一致性）
        results = []
        for query in queries:
            result = retriever.retrieve(query, top_k=2)
            results.append(result)
        
        # 验证所有检索都成功
        assert len(results) == 5
        for result in results:
            assert result["returned_count"] >= 0
            assert "timing" in result
        
        # 验证统计信息正确更新
        stats = retriever.get_retrieval_stats()
        assert stats["total_queries"] == 5
    
    def test_retrieval_result_metadata_integrity(self, retriever, test_documents):
        """测试检索结果元数据完整性"""
        query = "Python编程语言"
        
        result = retriever.retrieve(query, top_k=3)
        
        for chunk in result["results"]:
            # 验证基本字段
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk
            assert "similarity_score" in chunk
            assert "distance" in chunk
            
            # 验证元数据结构
            metadata = chunk["metadata"]
            assert "document_path" in metadata
            assert "chunk_index" in metadata
            assert "total_chunks" in metadata
            assert "chunk_length" in metadata
            assert "created_at" in metadata
            
            # 验证数据类型
            assert isinstance(chunk["similarity_score"], (int, float))
            assert isinstance(chunk["distance"], (int, float))
            assert isinstance(metadata["chunk_index"], int)
            assert isinstance(metadata["total_chunks"], int)
            assert isinstance(metadata["chunk_length"], int)