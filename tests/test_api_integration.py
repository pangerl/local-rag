"""
API 端点集成测试
测试 FastAPI 应用的完整功能
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from fastapi.testclient import TestClient
from sentence_transformers import SentenceTransformer

from app.main import app
from app.core.config import Settings
from app.api.routes import init_services


class TestAPIIntegration:
    """API 集成测试类"""
    
    @pytest.fixture(scope="class")
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture(scope="class")
    def test_settings(self, temp_dir):
        """创建测试用的配置对象"""
        settings = Settings()
        settings.CHROMA_DB_PATH = str(Path(temp_dir) / "chroma_db")
        settings.COLLECTION_NAME = "test_api"
        settings.DEFAULT_CHUNK_SIZE = 100
        settings.DEFAULT_CHUNK_OVERLAP = 20
        settings.DEFAULT_RETRIEVAL_K = 5
        settings.DEFAULT_TOP_K = 2
        return settings
    
    @pytest.fixture(scope="class")
    def mock_model_loader(self):
        """创建模拟模型加载器"""
        with patch('app.api.routes.ModelLoader') as mock_loader_class:
            loader = Mock()
            
            # 创建模拟的嵌入模型
            mock_embedding_model = Mock(spec=SentenceTransformer)
            
            def mock_encode(texts, **kwargs):
                embeddings = []
                for i, text in enumerate(texts):
                    vector = [0.1 + i * 0.1, 0.2 + i * 0.1, 0.3 + i * 0.1]
                    embeddings.append(vector)
                return np.array(embeddings)
            
            mock_embedding_model.encode = mock_encode
            loader.load_embedding_model.return_value = mock_embedding_model
            
            # 创建模拟的重排序模型
            mock_reranker_model = Mock()
            
            def mock_predict(query_doc_pairs):
                scores = []
                for query, doc in query_doc_pairs:
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
            
            mock_loader_class.return_value = loader
            yield loader
    
    @pytest.fixture(scope="class")
    def client(self, test_settings, mock_model_loader):
        """创建测试客户端"""
        # 初始化服务
        init_services(test_settings)
        
        # 创建测试客户端
        with TestClient(app) as client:
            yield client
    
    @pytest.fixture
    def test_document(self, temp_dir):
        """创建测试文档"""
        content = """这是一个测试文档的内容。

文档包含多个段落，用于测试 API 接口的完整功能。
每个段落都有足够的内容来生成多个文本分片。

第二段落包含更多的文本内容，确保能够测试分片功能。
这里有更多的句子来增加文本长度。
分词器会将这些内容分解为词元，然后进行滑动窗口分片。

第三段落继续添加内容，测试 API 处理功能。
文档处理服务会将整个文档分解为多个分片。
每个分片都会被向量化并存储到数据库中。"""
        
        file_path = Path(temp_dir) / "test_api_document.txt"
        file_path.write_text(content, encoding='utf-8')
        return str(file_path)
    
    def test_root_endpoint(self, client):
        """测试根路径端点"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Local RAG API"
        assert data["version"] == "1.0.0"
        assert "docs_url" in data
        assert "health_check" in data
    
    def test_health_check_endpoint(self, client):
        """测试健康检查端点"""
        response = client.get("/api/v1/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
    
    def test_ingest_document_success(self, client, test_document):
        """测试文档摄取成功"""
        request_data = {
            "document_path": test_document,
            "chunk_size": 80,
            "chunk_overlap": 15
        }
        
        response = client.post("/api/v1/ingest", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert data["success"] is True
        assert data["document_path"] == test_document
        assert data["chunks_created"] > 0
        assert data["chunks_stored"] == data["chunks_created"]
        assert data["text_length"] > 0
        assert data["processing_time"] > 0
        assert data["chunk_size"] == 80
        assert data["chunk_overlap"] == 15
        assert "timestamp" in data
    
    def test_ingest_document_file_not_found(self, client):
        """测试文档摄取文件不存在"""
        request_data = {
            "document_path": "nonexistent_file.txt"
        }
        
        response = client.post("/api/v1/ingest", json=request_data)
        
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "文件不存在" in data["message"]
    
    def test_ingest_document_unsupported_format(self, client, temp_dir):
        """测试文档摄取不支持的格式"""
        # 创建不支持的格式文件
        unsupported_file = Path(temp_dir) / "test.pdf"
        unsupported_file.write_text("PDF content", encoding='utf-8')
        
        request_data = {
            "document_path": str(unsupported_file)
        }
        
        response = client.post("/api/v1/ingest", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert "不支持的文件格式" in data["message"]
    
    def test_ingest_document_validation_error(self, client):
        """测试文档摄取参数验证错误"""
        request_data = {
            "document_path": "test.txt",
            "chunk_size": 2500,  # 超出范围
            "chunk_overlap": 50
        }
        
        response = client.post("/api/v1/ingest", json=request_data)
        
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "ValidationError"
    
    def test_retrieve_documents_success(self, client, test_document):
        """测试文档检索成功"""
        # 首先摄取文档
        ingest_data = {"document_path": test_document}
        ingest_response = client.post("/api/v1/ingest", json=ingest_data)
        assert ingest_response.status_code == 201
        
        # 然后检索
        retrieve_data = {
            "query": "测试文档内容",
            "retrieval_k": 5,
            "top_k": 2,
            "use_reranker": True
        }
        
        response = client.post("/api/v1/retrieve", json=retrieve_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["query"] == "测试文档内容"
        assert data["retrieval_k"] == 5
        assert data["top_k"] == 2
        assert data["use_reranker"] is True
        assert data["returned_count"] <= 2
        assert "timing" in data
        assert "timestamp" in data
        
        # 验证结果结构
        for result in data["results"]:
            assert "id" in result
            assert "text" in result
            assert "similarity_score" in result
            assert "metadata" in result
    
    def test_retrieve_documents_with_defaults(self, client, test_document):
        """测试使用默认参数的文档检索"""
        # 首先摄取文档
        ingest_data = {"document_path": test_document}
        client.post("/api/v1/ingest", json=ingest_data)
        
        # 使用默认参数检索
        retrieve_data = {
            "query": "文档处理服务"
        }
        
        response = client.post("/api/v1/retrieve", json=retrieve_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["retrieval_k"] == 5  # 默认值
        assert data["top_k"] == 2  # 默认值
        assert data["use_reranker"] is True  # 默认值
    
    def test_retrieve_documents_validation_error(self, client):
        """测试检索参数验证错误"""
        retrieve_data = {
            "query": "",  # 空查询
            "retrieval_k": 5,
            "top_k": 2
        }
        
        response = client.post("/api/v1/retrieve", json=retrieve_data)
        
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"] == "ValidationError"
    
    def test_retrieve_documents_empty_database(self, client):
        """测试空数据库的检索"""
        retrieve_data = {
            "query": "不存在的内容",
            "top_k": 3
        }
        
        response = client.post("/api/v1/retrieve", json=retrieve_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["returned_count"] == 0
        assert len(data["results"]) == 0
        assert data["total_candidates"] == 0
    
    def test_stats_endpoint(self, client, test_document):
        """测试统计信息端点"""
        # 先处理一些文档以生成统计数据
        ingest_data = {"document_path": test_document}
        client.post("/api/v1/ingest", json=ingest_data)
        
        retrieve_data = {"query": "测试查询"}
        client.post("/api/v1/retrieve", json=retrieve_data)
        
        # 获取统计信息
        response = client.get("/api/v1/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "processing_stats" in data
        assert "retrieval_stats" in data
        assert "storage_stats" in data
        assert "timestamp" in data
        
        # 验证统计数据
        assert data["processing_stats"]["total_documents_processed"] >= 1
        assert data["retrieval_stats"]["total_queries"] >= 1
        assert data["storage_stats"]["total_documents"] >= 1
    
    def test_complete_workflow(self, client, test_document):
        """测试完整的工作流程"""
        # 1. 检查健康状态
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200
        
        # 2. 摄取文档
        ingest_data = {
            "document_path": test_document,
            "chunk_size": 60,
            "chunk_overlap": 10
        }
        ingest_response = client.post("/api/v1/ingest", json=ingest_data)
        assert ingest_response.status_code == 201
        
        ingest_result = ingest_response.json()
        assert ingest_result["success"] is True
        chunks_created = ingest_result["chunks_created"]
        
        # 3. 检索文档
        retrieve_data = {
            "query": "测试文档段落",
            "retrieval_k": 3,
            "top_k": 2,
            "use_reranker": True
        }
        retrieve_response = client.post("/api/v1/retrieve", json=retrieve_data)
        assert retrieve_response.status_code == 200
        
        retrieve_result = retrieve_response.json()
        assert retrieve_result["success"] is True
        assert retrieve_result["returned_count"] <= 2
        
        # 4. 获取统计信息
        stats_response = client.get("/api/v1/stats")
        assert stats_response.status_code == 200
        
        stats_result = stats_response.json()
        assert stats_result["processing_stats"]["total_documents_processed"] >= 1
        assert stats_result["processing_stats"]["total_chunks_created"] >= chunks_created
        assert stats_result["retrieval_stats"]["total_queries"] >= 1
    
    def test_error_handling_consistency(self, client):
        """测试错误处理的一致性"""
        # 测试各种错误情况的响应格式
        
        # 1. 参数验证错误
        response1 = client.post("/api/v1/ingest", json={"document_path": ""})
        assert response1.status_code == 422
        data1 = response1.json()
        assert data1["success"] is False
        assert data1["error"] == "ValidationError"
        assert "timestamp" in data1
        
        # 2. 文件不存在错误
        response2 = client.post("/api/v1/ingest", json={"document_path": "nonexistent.txt"})
        assert response2.status_code == 404
        data2 = response2.json()
        assert data2["success"] is False
        assert "timestamp" in data2
        
        # 3. 检索参数错误
        response3 = client.post("/api/v1/retrieve", json={"query": "test", "retrieval_k": 0})
        assert response3.status_code == 422
        data3 = response3.json()
        assert data3["success"] is False
        assert data3["error"] == "ValidationError"
    
    def test_concurrent_requests(self, client, test_document):
        """测试并发请求处理"""
        import concurrent.futures
        
        # 先摄取文档
        ingest_data = {"document_path": test_document}
        client.post("/api/v1/ingest", json=ingest_data)
        
        # 并发检索请求
        def make_retrieve_request(query_suffix):
            retrieve_data = {"query": f"测试查询 {query_suffix}"}
            return client.post("/api/v1/retrieve", json=retrieve_data)
        
        # 执行并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_retrieve_request, i) for i in range(5)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证所有请求都成功
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_api_documentation_endpoints(self, client):
        """测试 API 文档端点"""
        # 测试 OpenAPI 规范
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_spec = response.json()
        assert openapi_spec["info"]["title"] == "Local RAG API"
        assert openapi_spec["info"]["version"] == "1.0.0"
        
        # 验证端点定义
        paths = openapi_spec["paths"]
        assert "/api/v1/ingest" in paths
        assert "/api/v1/retrieve" in paths
        assert "/api/v1/health" in paths
        assert "/api/v1/stats" in paths