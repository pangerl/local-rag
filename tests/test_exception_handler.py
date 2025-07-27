"""
异常处理中间件测试
"""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi import Request, HTTPException, status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.middleware.exception_handler import ExceptionHandlerMiddleware
from app.core.exceptions import (
    ModelLoadError,
    DocumentProcessError,
    DatabaseError,
    UnsupportedFormatError,
    FileNotFoundError
)


class TestExceptionHandlerMiddleware:
    """异常处理中间件测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.handler = ExceptionHandlerMiddleware()
        self.mock_request = Mock(spec=Request)
        self.mock_request.client = Mock()
        self.mock_request.client.host = "127.0.0.1"
        self.mock_request.headers = {"user-agent": "test-client"}
        self.mock_request.method = "POST"
        self.mock_request.url = Mock()
        self.mock_request.url.path = "/api/v1/test"
        self.mock_request.url.__str__ = lambda: "http://localhost:8000/api/v1/test"
    
    def test_error_mappings_setup(self):
        """测试错误映射设置"""
        mappings = self.handler.error_mappings
        
        # 测试自定义异常映射
        assert ModelLoadError in mappings
        assert mappings[ModelLoadError]["status_code"] == 500
        assert mappings[ModelLoadError]["error_type"] == "ModelLoadError"
        
        assert DocumentProcessError in mappings
        assert mappings[DocumentProcessError]["status_code"] == 400
        
        assert DatabaseError in mappings
        assert mappings[DatabaseError]["status_code"] == 500
        
        assert UnsupportedFormatError in mappings
        assert mappings[UnsupportedFormatError]["status_code"] == 400
        
        assert FileNotFoundError in mappings
        assert mappings[FileNotFoundError]["status_code"] == 404
        
        # 测试标准异常映射
        assert ValueError in mappings
        assert mappings[ValueError]["status_code"] == 400
        
        assert KeyError in mappings
        assert mappings[KeyError]["status_code"] == 400
    
    def test_get_client_info(self):
        """测试获取客户端信息"""
        client_info = self.handler._get_client_info(self.mock_request)
        
        assert client_info["ip"] == "127.0.0.1"
        assert client_info["user_agent"] == "test-client"
        assert client_info["method"] == "POST"
        assert client_info["path"] == "/api/v1/test"
    
    def test_create_error_response_with_custom_exception(self):
        """测试创建自定义异常的错误响应"""
        exc = ModelLoadError("模型文件不存在", {"model_path": "/path/to/model"})
        
        error_response = self.handler._create_error_response(exc, self.mock_request)
        
        assert error_response.error == "ModelLoadError"
        assert error_response.message == "模型文件不存在"
        assert "model_path" in error_response.details
        assert error_response.details["exception_type"] == "ModelLoadError"
        assert error_response.details["path"] == "/api/v1/test"
        assert error_response.details["method"] == "POST"
    
    def test_create_error_response_with_http_exception(self):
        """测试创建 HTTP 异常的错误响应"""
        exc = HTTPException(status_code=404, detail="资源不存在")
        
        error_response = self.handler._create_error_response(exc, self.mock_request)
        
        assert error_response.error == "HTTPException"
        assert error_response.message == "资源不存在"
        assert error_response.details["status_code"] == 404
    
    def test_create_error_response_with_validation_error(self):
        """测试创建验证错误的错误响应"""
        # 模拟 Pydantic ValidationError
        mock_error = Mock()
        mock_error.errors.return_value = [
            {
                "loc": ("field1",),
                "msg": "field required",
                "type": "value_error.missing"
            },
            {
                "loc": ("field2", "nested"),
                "msg": "invalid value",
                "type": "value_error.invalid"
            }
        ]
        
        error_response = self.handler._create_error_response(mock_error, self.mock_request)
        
        assert "validation_errors" in error_response.details
        validation_errors = error_response.details["validation_errors"]
        assert len(validation_errors) == 2
        assert validation_errors[0]["field"] == "field1"
        assert validation_errors[1]["field"] == "field2.nested"
    
    @patch('app.middleware.exception_handler.logger')
    def test_log_exception_error_level(self, mock_logger):
        """测试错误级别的异常日志记录"""
        exc = DatabaseError("数据库连接失败")
        
        self.handler._log_exception(exc, self.mock_request, 500, 1.234)
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "[127.0.0.1]" in call_args
        assert "POST /api/v1/test" in call_args
        assert "DatabaseError" in call_args
        assert "数据库连接失败" in call_args
        assert "状态码: 500" in call_args
        assert "耗时: 1.234s" in call_args
    
    @patch('app.middleware.exception_handler.logger')
    def test_log_exception_warning_level(self, mock_logger):
        """测试警告级别的异常日志记录"""
        exc = UnsupportedFormatError("不支持的文件格式")
        
        self.handler._log_exception(exc, self.mock_request, 400)
        
        mock_logger.warning.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_local_rag_exception(self):
        """测试处理 Local RAG 自定义异常"""
        exc = DocumentProcessError("文档处理失败", {"file_path": "/test.txt"})
        
        response = await self.handler.handle_local_rag_exception(self.mock_request, exc)
        
        assert response.status_code == 400
        
        content = json.loads(response.body)
        assert content["error"] == "DocumentProcessError"
        assert content["message"] == "文档处理失败"
        assert "file_path" in content["details"]
    
    @pytest.mark.asyncio
    async def test_handle_http_exception(self):
        """测试处理 HTTP 异常"""
        exc = HTTPException(status_code=404, detail="资源不存在")
        
        response = await self.handler.handle_http_exception(self.mock_request, exc)
        
        assert response.status_code == 404
        
        content = json.loads(response.body)
        assert content["error"] == "HTTP404"
        assert content["message"] == "资源不存在"
    
    @pytest.mark.asyncio
    async def test_handle_general_exception(self):
        """测试处理通用异常"""
        exc = ValueError("参数值错误")
        
        response = await self.handler.handle_general_exception(self.mock_request, exc)
        
        assert response.status_code == 400
        
        content = json.loads(response.body)
        assert content["error"] == "ValueError"
        assert content["message"] == "参数值错误"
    
    @pytest.mark.asyncio
    async def test_handle_unmapped_exception(self):
        """测试处理未映射的异常"""
        class CustomException(Exception):
            pass
        
        exc = CustomException("自定义异常")
        
        response = await self.handler.handle_general_exception(self.mock_request, exc)
        
        assert response.status_code == 500
        
        content = json.loads(response.body)
        assert content["error"] == "CustomException"
        assert content["message"] == "自定义异常"


@pytest.fixture
def test_app():
    """创建测试应用"""
    from fastapi import FastAPI
    from app.middleware.exception_handler import exception_handler
    
    app = FastAPI()
    
    # 注册异常处理器
    app.add_exception_handler(Exception, exception_handler.handle_general_exception)
    
    @app.get("/test-error/{error_type}")
    async def test_error_endpoint(error_type: str):
        """测试错误端点"""
        if error_type == "model_load":
            raise ModelLoadError("模型加载失败")
        elif error_type == "document_process":
            raise DocumentProcessError("文档处理失败")
        elif error_type == "database":
            raise DatabaseError("数据库错误")
        elif error_type == "unsupported_format":
            raise UnsupportedFormatError("不支持的格式")
        elif error_type == "file_not_found":
            raise FileNotFoundError("文件不存在")
        elif error_type == "value_error":
            raise ValueError("参数值错误")
        else:
            raise Exception("通用异常")
    
    return app


class TestExceptionHandlerIntegration:
    """异常处理器集成测试"""
    
    def test_model_load_error_integration(self, test_app):
        """测试模型加载错误的集成处理"""
        client = TestClient(test_app)
        
        response = client.get("/test-error/model_load")
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "ModelLoadError"
        assert "模型加载失败" in data["message"]
    
    def test_document_process_error_integration(self, test_app):
        """测试文档处理错误的集成处理"""
        client = TestClient(test_app)
        
        response = client.get("/test-error/document_process")
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "DocumentProcessError"
        assert "文档处理失败" in data["message"]
    
    def test_database_error_integration(self, test_app):
        """测试数据库错误的集成处理"""
        client = TestClient(test_app)
        
        response = client.get("/test-error/database")
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "DatabaseError"
    
    def test_unsupported_format_error_integration(self, test_app):
        """测试不支持格式错误的集成处理"""
        client = TestClient(test_app)
        
        response = client.get("/test-error/unsupported_format")
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "UnsupportedFormatError"
    
    def test_file_not_found_error_integration(self, test_app):
        """测试文件不存在错误的集成处理"""
        client = TestClient(test_app)
        
        response = client.get("/test-error/file_not_found")
        
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "FileNotFoundError"
    
    def test_value_error_integration(self, test_app):
        """测试参数值错误的集成处理"""
        client = TestClient(test_app)
        
        response = client.get("/test-error/value_error")
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "ValueError"
    
    def test_general_exception_integration(self, test_app):
        """测试通用异常的集成处理"""
        client = TestClient(test_app)
        
        response = client.get("/test-error/unknown")
        
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Exception"
        assert "details" in data
        assert "timestamp" in data["details"]
        assert "path" in data["details"]