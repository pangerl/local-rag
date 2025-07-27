"""
请求追踪中间件测试
"""

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from app.middleware.request_tracker import RequestTrackerMiddleware, get_request_id


class TestRequestTrackerMiddleware:
    """请求追踪中间件测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.app = FastAPI()
        self.middleware = RequestTrackerMiddleware(self.app)
    
    @pytest.mark.asyncio
    async def test_dispatch_successful_request(self):
        """测试成功请求的处理"""
        # 模拟请求
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"user-agent": "test-client"}
        mock_request.method = "GET"
        mock_request.url = Mock()
        mock_request.url.path = "/test"
        mock_request.url.__str__ = lambda: "http://localhost:8000/test"
        mock_request.state = Mock()
        
        # 模拟响应
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        
        # 模拟 call_next
        async def mock_call_next(request):
            await asyncio.sleep(0.1)  # 模拟处理时间
            return mock_response
        
        with patch('app.middleware.request_tracker.logger') as mock_logger:
            response = await self.middleware.dispatch(mock_request, mock_call_next)
        
        # 验证请求 ID 被设置
        assert hasattr(mock_request.state, 'request_id')
        assert len(mock_request.state.request_id) == 8
        
        # 验证响应头被设置
        assert "X-Request-ID" in mock_response.headers
        assert "X-Process-Time" in mock_response.headers
        
        # 验证日志记录
        assert mock_logger.info.call_count >= 2  # 开始和完成日志
        
        # 验证返回响应
        assert response == mock_response
    
    @pytest.mark.asyncio
    async def test_dispatch_exception_handling(self):
        """测试异常处理"""
        import asyncio
        
        # 模拟请求
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"user-agent": "test-client"}
        mock_request.method = "POST"
        mock_request.url = Mock()
        mock_request.url.path = "/error"
        mock_request.url.__str__ = lambda: "http://localhost:8000/error"
        mock_request.state = Mock()
        
        # 模拟抛出异常的 call_next
        async def mock_call_next(request):
            raise ValueError("测试异常")
        
        with patch('app.middleware.request_tracker.logger') as mock_logger:
            with pytest.raises(ValueError, match="测试异常"):
                await self.middleware.dispatch(mock_request, mock_call_next)
        
        # 验证异常日志被记录
        mock_logger.error.assert_called_once()
        error_call = mock_logger.error.call_args[0][0]
        assert "请求异常" in error_call
        assert "ValueError" in error_call
        assert "测试异常" in error_call
    
    @pytest.mark.asyncio
    async def test_dispatch_slow_request_warning(self):
        """测试慢请求警告"""
        import asyncio
        
        # 模拟请求
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"user-agent": "test-client"}
        mock_request.method = "GET"
        mock_request.url = Mock()
        mock_request.url.path = "/slow"
        mock_request.url.__str__ = lambda: "http://localhost:8000/slow"
        mock_request.state = Mock()
        
        # 模拟响应
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_response.headers = {}
        
        # 模拟慢请求
        async def mock_call_next(request):
            # 模拟超过 5 秒的处理时间
            with patch('time.time', side_effect=[0, 6.0]):
                return mock_response
        
        with patch('app.middleware.request_tracker.logger') as mock_logger:
            with patch('time.time', side_effect=[0, 6.0]):
                await self.middleware.dispatch(mock_request, mock_call_next)
        
        # 验证慢请求警告被记录
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "慢请求警告" in warning_call
        assert "6.000s" in warning_call
    
    @pytest.mark.asyncio
    async def test_dispatch_with_request_body_logging(self):
        """测试请求体日志记录"""
        # 创建包含请求体日志的中间件
        middleware = RequestTrackerMiddleware(self.app, include_request_body=True)
        
        # 模拟请求
        mock_request = Mock(spec=Request)
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"user-agent": "test-client", "content-type": "application/json"}
        mock_request.method = "POST"
        mock_request.url = Mock()
        mock_request.url.path = "/api/test"
        mock_request.url.__str__ = lambda: "http://localhost:8000/api/test"
        mock_request.state = Mock()
        mock_request.body = AsyncMock(return_value=b'{"test": "data"}')
        
        # 模拟响应
        mock_response = Mock(spec=Response)
        mock_response.status_code = 201
        mock_response.headers = {}
        
        async def mock_call_next(request):
            return mock_response
        
        with patch('app.middleware.request_tracker.logger') as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            await middleware.dispatch(mock_request, mock_call_next)
        
        # 验证请求体被记录（在调试模式下）
        debug_calls = [call for call in mock_logger.debug.call_args_list 
                      if "请求体" in str(call)]
        assert len(debug_calls) > 0
    
    def test_get_request_id_with_existing_id(self):
        """测试获取已存在的请求 ID"""
        mock_request = Mock(spec=Request)
        mock_request.state = Mock()
        mock_request.state.request_id = "test123"
        
        request_id = get_request_id(mock_request)
        assert request_id == "test123"
    
    def test_get_request_id_without_id(self):
        """测试获取不存在的请求 ID"""
        mock_request = Mock(spec=Request)
        mock_request.state = Mock()
        # 没有 request_id 属性
        
        request_id = get_request_id(mock_request)
        assert request_id == "unknown"


@pytest.fixture
def test_app_with_middleware():
    """创建带有请求追踪中间件的测试应用"""
    app = FastAPI()
    
    # 添加请求追踪中间件
    app.add_middleware(RequestTrackerMiddleware)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}
    
    @app.get("/error")
    async def error_endpoint():
        raise ValueError("测试错误")
    
    @app.get("/slow")
    async def slow_endpoint():
        import asyncio
        await asyncio.sleep(0.1)
        return {"message": "slow response"}
    
    return app


class TestRequestTrackerIntegration:
    """请求追踪中间件集成测试"""
    
    def test_successful_request_integration(self, test_app_with_middleware):
        """测试成功请求的集成处理"""
        client = TestClient(test_app_with_middleware)
        
        with patch('app.middleware.request_tracker.logger') as mock_logger:
            response = client.get("/test")
        
        assert response.status_code == 200
        assert response.json() == {"message": "success"}
        
        # 验证响应头
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8
        
        # 验证日志记录
        info_calls = mock_logger.info.call_args_list
        assert len(info_calls) >= 2
        
        # 检查开始日志
        start_log = info_calls[0][0][0]
        assert "请求开始" in start_log
        assert "GET /test" in start_log
        
        # 检查完成日志
        complete_log = info_calls[1][0][0]
        assert "请求完成" in complete_log
        assert "状态码: 200" in complete_log
        assert "✅" in complete_log
    
    def test_error_request_integration(self, test_app_with_middleware):
        """测试错误请求的集成处理"""
        client = TestClient(test_app_with_middleware)
        
        with patch('app.middleware.request_tracker.logger') as mock_logger:
            response = client.get("/error")
        
        assert response.status_code == 500
        
        # 验证错误日志记录
        mock_logger.error.assert_called()
        error_call = mock_logger.error.call_args[0][0]
        assert "请求异常" in error_call
        assert "GET /error" in error_call
    
    def test_request_id_consistency(self, test_app_with_middleware):
        """测试请求 ID 的一致性"""
        app = test_app_with_middleware
        
        @app.get("/check-id")
        async def check_id_endpoint(request: Request):
            return {"request_id": get_request_id(request)}
        
        client = TestClient(app)
        response = client.get("/check-id")
        
        assert response.status_code == 200
        data = response.json()
        
        # 验证请求 ID 在响应头和响应体中一致
        assert data["request_id"] == response.headers["X-Request-ID"]
        assert len(data["request_id"]) == 8
    
    def test_multiple_requests_different_ids(self, test_app_with_middleware):
        """测试多个请求具有不同的 ID"""
        client = TestClient(test_app_with_middleware)
        
        response1 = client.get("/test")
        response2 = client.get("/test")
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        id1 = response1.headers["X-Request-ID"]
        id2 = response2.headers["X-Request-ID"]
        
        assert id1 != id2
        assert len(id1) == 8
        assert len(id2) == 8