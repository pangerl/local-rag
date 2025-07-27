"""
API 数据模型单元测试
测试 Pydantic 模型的验证逻辑和数据结构
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.api.models import (
    IngestRequest,
    IngestResponse,
    RetrieveRequest,
    RetrieveResponse,
    DocumentChunk,
    ErrorResponse,
    HealthResponse,
    StatsResponse
)


class TestIngestRequest:
    """IngestRequest 模型测试类"""
    
    def test_valid_request(self):
        """测试有效的摄取请求"""
        data = {
            "document_path": "documents/test.txt",
            "chunk_size": 300,
            "chunk_overlap": 50
        }
        
        request = IngestRequest(**data)
        
        assert request.document_path == "documents/test.txt"
        assert request.chunk_size == 300
        assert request.chunk_overlap == 50
    
    def test_valid_request_with_defaults(self):
        """测试使用默认值的有效请求"""
        data = {
            "document_path": "documents/test.md"
        }
        
        request = IngestRequest(**data)
        
        assert request.document_path == "documents/test.md"
        assert request.chunk_size is None
        assert request.chunk_overlap is None
    
    def test_invalid_document_path_empty(self):
        """测试空文档路径"""
        data = {
            "document_path": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest(**data)
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_invalid_document_path_unsupported_format(self):
        """测试不支持的文件格式"""
        data = {
            "document_path": "documents/test.pdf"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest(**data)
        
        assert "不支持的文件格式" in str(exc_info.value)
    
    def test_invalid_chunk_size_too_small(self):
        """测试 chunk_size 过小"""
        data = {
            "document_path": "documents/test.txt",
            "chunk_size": 30
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest(**data)
        
        assert "greater than or equal to 50" in str(exc_info.value)
    
    def test_invalid_chunk_size_too_large(self):
        """测试 chunk_size 过大"""
        data = {
            "document_path": "documents/test.txt",
            "chunk_size": 2500
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest(**data)
        
        assert "less than or equal to 2000" in str(exc_info.value)
    
    def test_invalid_chunk_overlap_too_large(self):
        """测试 chunk_overlap 过大"""
        data = {
            "document_path": "documents/test.txt",
            "chunk_size": 300,
            "chunk_overlap": 300
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IngestRequest(**data)
        
        assert "chunk_overlap 必须小于 chunk_size" in str(exc_info.value)
    
    def test_document_path_whitespace_trimming(self):
        """测试文档路径空白字符处理"""
        data = {
            "document_path": "  documents/test.txt  "
        }
        
        request = IngestRequest(**data)
        
        assert request.document_path == "documents/test.txt"
    
    def test_supported_file_extensions(self):
        """测试支持的文件扩展名"""
        supported_files = [
            "test.txt",
            "test.TXT",
            "test.md",
            "test.MD",
            "documents/file.txt",
            "path/to/file.md"
        ]
        
        for file_path in supported_files:
            data = {"document_path": file_path}
            request = IngestRequest(**data)
            assert request.document_path == file_path


class TestIngestResponse:
    """IngestResponse 模型测试类"""
    
    def test_valid_response(self):
        """测试有效的摄取响应"""
        data = {
            "success": True,
            "message": "文档处理成功",
            "document_path": "documents/test.txt",
            "chunks_created": 15,
            "chunks_stored": 15,
            "text_length": 4500,
            "processing_time": 2.35,
            "chunk_size": 300,
            "chunk_overlap": 50,
            "embedding_dimension": 768,
            "collection_name": "documents"
        }
        
        response = IngestResponse(**data)
        
        assert response.success is True
        assert response.message == "文档处理成功"
        assert response.chunks_created == 15
        assert response.chunks_stored == 15
        assert isinstance(response.timestamp, datetime)
    
    def test_response_with_optional_fields(self):
        """测试包含可选字段的响应"""
        data = {
            "success": False,
            "message": "处理失败",
            "document_path": "documents/test.txt",
            "chunks_created": 0,
            "chunks_stored": 0,
            "text_length": 0,
            "processing_time": 0.1,
            "chunk_size": 300,
            "chunk_overlap": 50
        }
        
        response = IngestResponse(**data)
        
        assert response.success is False
        assert response.embedding_dimension is None
        assert response.collection_name is None


class TestRetrieveRequest:
    """RetrieveRequest 模型测试类"""
    
    def test_valid_request(self):
        """测试有效的检索请求"""
        data = {
            "query": "Python编程语言",
            "retrieval_k": 10,
            "top_k": 3,
            "use_reranker": True
        }
        
        request = RetrieveRequest(**data)
        
        assert request.query == "Python编程语言"
        assert request.retrieval_k == 10
        assert request.top_k == 3
        assert request.use_reranker is True
    
    def test_valid_request_with_defaults(self):
        """测试使用默认值的有效请求"""
        data = {
            "query": "机器学习算法"
        }
        
        request = RetrieveRequest(**data)
        
        assert request.query == "机器学习算法"
        assert request.retrieval_k is None
        assert request.top_k is None
        assert request.use_reranker is True  # 默认值
    
    def test_invalid_query_empty(self):
        """测试空查询"""
        data = {
            "query": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RetrieveRequest(**data)
        
        assert "String should have at least 1 character" in str(exc_info.value)
    
    def test_invalid_query_too_long(self):
        """测试查询过长"""
        data = {
            "query": "a" * 1001
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RetrieveRequest(**data)
        
        assert "at most 1000 characters" in str(exc_info.value)
    
    def test_invalid_retrieval_k_too_small(self):
        """测试 retrieval_k 过小"""
        data = {
            "query": "测试查询",
            "retrieval_k": 0
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RetrieveRequest(**data)
        
        assert "greater than or equal to 1" in str(exc_info.value)
    
    def test_invalid_retrieval_k_too_large(self):
        """测试 retrieval_k 过大"""
        data = {
            "query": "测试查询",
            "retrieval_k": 101
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RetrieveRequest(**data)
        
        assert "less than or equal to 100" in str(exc_info.value)
    
    def test_invalid_top_k_greater_than_retrieval_k(self):
        """测试 top_k 大于 retrieval_k"""
        data = {
            "query": "测试查询",
            "retrieval_k": 5,
            "top_k": 10
        }
        
        with pytest.raises(ValidationError) as exc_info:
            RetrieveRequest(**data)
        
        assert "top_k 不能大于 retrieval_k" in str(exc_info.value)
    
    def test_query_whitespace_trimming(self):
        """测试查询文本空白字符处理"""
        data = {
            "query": "  Python编程语言  "
        }
        
        request = RetrieveRequest(**data)
        
        assert request.query == "Python编程语言"


class TestDocumentChunk:
    """DocumentChunk 模型测试类"""
    
    def test_valid_chunk(self):
        """测试有效的文档分片"""
        data = {
            "id": "test_chunk_1",
            "text": "这是一个测试文档分片",
            "similarity_score": 0.85,
            "rerank_score": 0.92,
            "metadata": {
                "document_path": "test.txt",
                "chunk_index": 0,
                "total_chunks": 5
            }
        }
        
        chunk = DocumentChunk(**data)
        
        assert chunk.id == "test_chunk_1"
        assert chunk.text == "这是一个测试文档分片"
        assert chunk.similarity_score == 0.85
        assert chunk.rerank_score == 0.92
        assert chunk.metadata["document_path"] == "test.txt"
    
    def test_chunk_without_rerank_score(self):
        """测试没有重排序分数的分片"""
        data = {
            "id": "test_chunk_1",
            "text": "这是一个测试文档分片",
            "similarity_score": 0.85,
            "metadata": {
                "document_path": "test.txt",
                "chunk_index": 0
            }
        }
        
        chunk = DocumentChunk(**data)
        
        assert chunk.rerank_score is None
    
    def test_invalid_similarity_score_out_of_range(self):
        """测试相似度分数超出范围"""
        data = {
            "id": "test_chunk_1",
            "text": "测试文本",
            "similarity_score": 1.5,  # 超出范围
            "metadata": {}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            DocumentChunk(**data)
        
        assert "less than or equal to 1" in str(exc_info.value)


class TestRetrieveResponse:
    """RetrieveResponse 模型测试类"""
    
    def test_valid_response(self):
        """测试有效的检索响应"""
        chunk_data = {
            "id": "test_chunk_1",
            "text": "测试文档分片",
            "similarity_score": 0.85,
            "metadata": {"document_path": "test.txt"}
        }
        
        data = {
            "success": True,
            "message": "检索成功",
            "query": "测试查询",
            "results": [chunk_data],
            "total_candidates": 5,
            "returned_count": 1,
            "retrieval_k": 10,
            "top_k": 3,
            "use_reranker": True,
            "timing": {
                "total_time": 0.45,
                "search_time": 0.25,
                "rerank_time": 0.20
            }
        }
        
        response = RetrieveResponse(**data)
        
        assert response.success is True
        assert response.query == "测试查询"
        assert len(response.results) == 1
        assert response.total_candidates == 5
        assert response.returned_count == 1
        assert isinstance(response.timestamp, datetime)
    
    def test_response_with_empty_results(self):
        """测试空结果的响应"""
        data = {
            "success": True,
            "message": "未找到相关结果",
            "query": "测试查询",
            "results": [],
            "total_candidates": 0,
            "returned_count": 0,
            "retrieval_k": 10,
            "top_k": 3,
            "use_reranker": False,
            "timing": {
                "total_time": 0.1,
                "search_time": 0.1,
                "rerank_time": 0.0
            }
        }
        
        response = RetrieveResponse(**data)
        
        assert len(response.results) == 0
        assert response.total_candidates == 0
        assert response.returned_count == 0


class TestErrorResponse:
    """ErrorResponse 模型测试类"""
    
    def test_valid_error_response(self):
        """测试有效的错误响应"""
        data = {
            "error": "ValidationError",
            "message": "请求参数验证失败",
            "details": {
                "field": "chunk_size",
                "value": 2500,
                "constraint": "必须小于等于 2000"
            }
        }
        
        response = ErrorResponse(**data)
        
        assert response.success is False
        assert response.error == "ValidationError"
        assert response.message == "请求参数验证失败"
        assert response.details["field"] == "chunk_size"
        assert isinstance(response.timestamp, datetime)
    
    def test_error_response_without_details(self):
        """测试没有详细信息的错误响应"""
        data = {
            "error": "FileNotFoundError",
            "message": "文件不存在"
        }
        
        response = ErrorResponse(**data)
        
        assert response.details is None


class TestHealthResponse:
    """HealthResponse 模型测试类"""
    
    def test_valid_health_response(self):
        """测试有效的健康检查响应"""
        data = {
            "status": "healthy",
            "components": {
                "database": {
                    "status": "healthy",
                    "connection_status": True
                },
                "models": {
                    "status": "healthy",
                    "embedding_model_loaded": True
                }
            }
        }
        
        response = HealthResponse(**data)
        
        assert response.status == "healthy"
        assert response.components["database"]["status"] == "healthy"
        assert response.components["models"]["status"] == "healthy"
        assert isinstance(response.timestamp, datetime)
        assert response.error is None
    
    def test_health_response_with_error(self):
        """测试包含错误的健康检查响应"""
        data = {
            "status": "error",
            "components": {},
            "error": "数据库连接失败"
        }
        
        response = HealthResponse(**data)
        
        assert response.status == "error"
        assert response.error == "数据库连接失败"


class TestStatsResponse:
    """StatsResponse 模型测试类"""
    
    def test_valid_stats_response(self):
        """测试有效的统计信息响应"""
        data = {
            "processing_stats": {
                "total_documents_processed": 25,
                "total_chunks_created": 450,
                "total_processing_time": 125.5
            },
            "retrieval_stats": {
                "total_queries": 150,
                "average_retrieval_time": 0.35,
                "average_rerank_time": 0.15
            },
            "storage_stats": {
                "total_documents": 25,
                "total_chunks": 450,
                "collection_name": "documents"
            }
        }
        
        response = StatsResponse(**data)
        
        assert response.processing_stats["total_documents_processed"] == 25
        assert response.retrieval_stats["total_queries"] == 150
        assert response.storage_stats["total_documents"] == 25
        assert isinstance(response.timestamp, datetime)


class TestModelSerialization:
    """模型序列化测试类"""
    
    def test_ingest_request_json_serialization(self):
        """测试摄取请求的 JSON 序列化"""
        request = IngestRequest(
            document_path="test.txt",
            chunk_size=300,
            chunk_overlap=50
        )
        
        json_data = request.model_dump()
        
        assert json_data["document_path"] == "test.txt"
        assert json_data["chunk_size"] == 300
        assert json_data["chunk_overlap"] == 50
    
    def test_retrieve_response_json_serialization(self):
        """测试检索响应的 JSON 序列化"""
        chunk = DocumentChunk(
            id="test_1",
            text="测试文本",
            similarity_score=0.8,
            metadata={"path": "test.txt"}
        )
        
        response = RetrieveResponse(
            success=True,
            message="成功",
            query="测试",
            results=[chunk],
            total_candidates=1,
            returned_count=1,
            retrieval_k=10,
            top_k=3,
            use_reranker=True,
            timing={"total_time": 0.1}
        )
        
        json_data = response.model_dump()
        
        assert json_data["success"] is True
        assert len(json_data["results"]) == 1
        assert json_data["results"][0]["id"] == "test_1"
    
    def test_model_schema_generation(self):
        """测试模型 Schema 生成"""
        schema = IngestRequest.model_json_schema()
        
        assert "properties" in schema
        assert "document_path" in schema["properties"]
        assert "chunk_size" in schema["properties"]
        assert "chunk_overlap" in schema["properties"]
        
        # 验证字段描述
        assert "description" in schema["properties"]["document_path"]
        assert "example" in schema["properties"]["document_path"]