"""
监控功能测试
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch

from app.core.monitoring import MetricsCollector, SystemMonitor, PerformanceMetric, RequestMetrics


class TestMetricsCollector:
    """指标收集器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.collector = MetricsCollector(max_metrics=100)
    
    def test_record_metric(self):
        """测试记录性能指标"""
        self.collector.record_metric("test_metric", 1.23, {"tag": "test"})
        
        assert len(self.collector.metrics) == 1
        metric = self.collector.metrics[0]
        assert metric.name == "test_metric"
        assert metric.value == 1.23
        assert metric.tags == {"tag": "test"}
        assert isinstance(metric.timestamp, float)
    
    def test_record_request(self):
        """测试记录请求指标"""
        self.collector.record_request(
            endpoint="/test",
            method="GET",
            status_code=200,
            response_time=0.123,
            request_size=100,
            response_size=200
        )
        
        assert len(self.collector.request_metrics) == 1
        request_metric = self.collector.request_metrics[0]
        assert request_metric.endpoint == "/test"
        assert request_metric.method == "GET"
        assert request_metric.status_code == 200
        assert request_metric.response_time == 0.123
        assert request_metric.request_size == 100
        assert request_metric.response_size == 200
        
        # 检查计数器是否更新
        assert self.collector.counters["requests_total_GET_/test"] == 1
        assert self.collector.counters["requests_status_200"] == 1
        
        # 检查直方图是否更新
        assert len(self.collector.histograms["response_time_GET_/test"]) == 1
        assert self.collector.histograms["response_time_GET_/test"][0] == 0.123
    
    def test_increment_counter(self):
        """测试增加计数器"""
        self.collector.increment_counter("test_counter", 5)
        assert self.collector.counters["test_counter"] == 5
        
        self.collector.increment_counter("test_counter", 3)
        assert self.collector.counters["test_counter"] == 8
        
        # 测试带标签的计数器
        self.collector.increment_counter("tagged_counter", 1, {"env": "test"})
        assert self.collector.counters["tagged_counter_env_test"] == 1
    
    def test_set_gauge(self):
        """测试设置仪表盘值"""
        self.collector.set_gauge("test_gauge", 42.0)
        assert self.collector.gauges["test_gauge"] == 42.0
        
        # 测试带标签的仪表盘
        self.collector.set_gauge("tagged_gauge", 24.0, {"type": "memory"})
        assert self.collector.gauges["tagged_gauge_type_memory"] == 24.0
    
    def test_get_metrics_summary_empty(self):
        """测试获取空指标摘要"""
        summary = self.collector.get_metrics_summary(300)
        
        assert summary["time_window"] == 300
        assert summary["request_count"] == 0
        assert summary["avg_response_time"] == 0
        assert summary["error_rate"] == 0
        assert summary["requests_per_second"] == 0
        assert summary["status_codes"] == {}
        assert summary["endpoints"] == {}
        assert isinstance(summary["counters"], dict)
        assert isinstance(summary["gauges"], dict)
    
    def test_get_metrics_summary_with_data(self):
        """测试获取有数据的指标摘要"""
        # 添加一些测试数据
        current_time = time.time()
        
        # 模拟请求指标
        for i in range(5):
            request_metric = RequestMetrics(
                endpoint="/test",
                method="GET",
                status_code=200 if i < 4 else 500,
                response_time=0.1 + i * 0.05,
                timestamp=current_time - i * 10
            )
            self.collector.request_metrics.append(request_metric)
        
        # 更新计数器
        self.collector.counters["test_counter"] = 10
        self.collector.gauges["test_gauge"] = 42.0
        
        summary = self.collector.get_metrics_summary(300)
        
        assert summary["request_count"] == 5
        assert summary["error_rate"] == 0.2  # 1 error out of 5 requests
        assert summary["status_codes"][200] == 4
        assert summary["status_codes"][500] == 1
        assert "GET /test" in summary["endpoints"]
        assert summary["counters"]["test_counter"] == 10
        assert summary["gauges"]["test_gauge"] == 42.0
    
    def test_max_metrics_limit(self):
        """测试最大指标数量限制"""
        collector = MetricsCollector(max_metrics=3)
        
        # 添加超过限制的指标
        for i in range(5):
            collector.record_metric(f"metric_{i}", i)
        
        # 应该只保留最后 3 个指标
        assert len(collector.metrics) == 3
        assert collector.metrics[0].name == "metric_2"
        assert collector.metrics[2].name == "metric_4"


class TestSystemMonitor:
    """系统监控器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.collector = MetricsCollector()
        self.monitor = SystemMonitor(self.collector)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_collect_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """测试收集系统指标"""
        # 模拟系统指标
        mock_cpu.return_value = 25.5
        
        mock_memory_info = Mock()
        mock_memory_info.percent = 60.0
        mock_memory_info.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory_info.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_memory.return_value = mock_memory_info
        
        mock_disk_info = Mock()
        mock_disk_info.percent = 45.0
        mock_disk_info.used = 100 * 1024 * 1024 * 1024  # 100GB
        mock_disk_info.free = 150 * 1024 * 1024 * 1024  # 150GB
        mock_disk.return_value = mock_disk_info
        
        # 收集指标
        self.monitor._collect_system_metrics()
        
        # 验证指标是否被记录
        gauges = self.collector.gauges
        assert gauges["system_cpu_percent"] == 25.5
        assert gauges["system_memory_percent"] == 60.0
        assert gauges["system_memory_used_mb"] == 8192.0
        assert gauges["system_memory_available_mb"] == 4096.0
        assert gauges["system_disk_percent"] == 45.0
        assert gauges["system_disk_used_gb"] == 100.0
        assert gauges["system_disk_free_gb"] == 150.0
    
    def test_get_system_status(self):
        """测试获取系统状态"""
        status = self.monitor.get_system_status()
        
        assert "timestamp" in status
        assert "system" in status
        assert "process" in status
        
        # 检查系统信息
        system_info = status["system"]
        assert "cpu_count" in system_info
        assert "cpu_percent" in system_info
        assert "memory" in system_info
        assert "disk" in system_info
        
        # 检查进程信息
        process_info = status["process"]
        assert "pid" in process_info
        assert "create_time" in process_info
        assert "cpu_percent" in process_info
        assert "memory_info" in process_info
        assert "num_threads" in process_info
    
    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        assert not self.monitor._monitoring
        
        # 启动监控
        self.monitor.start_monitoring(interval=1)
        assert self.monitor._monitoring
        assert self.monitor._monitor_thread is not None
        
        # 等待一下让监控线程运行
        time.sleep(0.1)
        
        # 停止监控
        self.monitor.stop_monitoring()
        assert not self.monitor._monitoring


class TestPerformanceMetric:
    """性能指标数据类测试"""
    
    def test_performance_metric_creation(self):
        """测试性能指标创建"""
        metric = PerformanceMetric(
            name="test_metric",
            value=1.23,
            timestamp=time.time(),
            tags={"env": "test"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 1.23
        assert isinstance(metric.timestamp, float)
        assert metric.tags == {"env": "test"}


class TestRequestMetrics:
    """请求指标数据类测试"""
    
    def test_request_metrics_creation(self):
        """测试请求指标创建"""
        metric = RequestMetrics(
            endpoint="/test",
            method="POST",
            status_code=201,
            response_time=0.456,
            timestamp=time.time(),
            request_size=100,
            response_size=200
        )
        
        assert metric.endpoint == "/test"
        assert metric.method == "POST"
        assert metric.status_code == 201
        assert metric.response_time == 0.456
        assert isinstance(metric.timestamp, float)
        assert metric.request_size == 100
        assert metric.response_size == 200


@pytest.mark.asyncio
class TestMonitoringIntegration:
    """监控功能集成测试"""
    
    async def test_monitoring_workflow(self):
        """测试完整的监控工作流程"""
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)
        
        # 记录一些指标
        collector.record_metric("cpu_usage", 25.5)
        collector.record_request("/api/test", "GET", 200, 0.123)
        collector.increment_counter("api_calls", 1)
        collector.set_gauge("active_connections", 10)
        
        # 获取摘要
        summary = collector.get_metrics_summary(300)
        assert summary["request_count"] == 1
        assert summary["counters"]["api_calls"] == 1
        assert summary["gauges"]["active_connections"] == 10
        
        # 获取系统状态
        status = monitor.get_system_status()
        assert "system" in status
        assert "process" in status