"""
系统监控模块
提供性能指标收集和系统状态监控
"""

import time
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class RequestMetrics:
    """请求指标数据类"""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    timestamp: float
    request_size: Optional[int] = None
    response_size: Optional[int] = None


class MetricsCollector:
    """指标收集器"""

    def __init__(self, max_metrics: int = 10000):
        """
        初始化指标收集器

        Args:
            max_metrics: 最大保存的指标数量
        """
        self.max_metrics = max_metrics
        self.metrics: deque = deque(maxlen=max_metrics)
        self.request_metrics: deque = deque(maxlen=max_metrics)
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

        logger.info(f"指标收集器初始化完成，最大指标数量: {max_metrics}")

    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """
        记录性能指标

        Args:
            name: 指标名称
            value: 指标值
            tags: 标签字典
        """
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=value,
                timestamp=time.time(),
                tags=tags or {}
            )
            self.metrics.append(metric)

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None
    ):
        """
        记录请求指标

        Args:
            endpoint: 端点路径
            method: HTTP 方法
            status_code: 状态码
            response_time: 响应时间
            request_size: 请求大小
            response_size: 响应大小
        """
        with self._lock:
            request_metric = RequestMetrics(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time=response_time,
                timestamp=time.time(),
                request_size=request_size,
                response_size=response_size
            )
            self.request_metrics.append(request_metric)

            # 更新计数器
            self.counters[f"requests_total_{method}_{endpoint}"] += 1
            self.counters[f"requests_status_{status_code}"] += 1

            # 更新响应时间直方图
            self.histograms[f"response_time_{method}_{endpoint}"].append(response_time)

    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """
        增加计数器

        Args:
            name: 计数器名称
            value: 增加的值
            tags: 标签字典
        """
        with self._lock:
            key = name
            if tags:
                tag_str = "_".join(f"{k}_{v}" for k, v in sorted(tags.items()))
                key = f"{name}_{tag_str}"
            self.counters[key] += value

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """
        设置仪表盘值

        Args:
            name: 仪表盘名称
            value: 值
            tags: 标签字典
        """
        with self._lock:
            key = name
            if tags:
                tag_str = "_".join(f"{k}_{v}" for k, v in sorted(tags.items()))
                key = f"{name}_{tag_str}"
            self.gauges[key] = value

    def get_metrics_summary(self, time_window: int = 300) -> Dict[str, Any]:
        """
        获取指标摘要

        Args:
            time_window: 时间窗口（秒）

        Returns:
            Dict: 指标摘要
        """
        current_time = time.time()
        cutoff_time = current_time - time_window

        with self._lock:
            # 过滤时间窗口内的请求
            recent_requests = [
                req for req in self.request_metrics
                if req.timestamp >= cutoff_time
            ]

            if not recent_requests:
                return {
                    "time_window": time_window,
                    "request_count": 0,
                    "avg_response_time": 0,
                    "error_rate": 0,
                    "requests_per_second": 0,
                    "status_codes": {},
                    "endpoints": {},
                    "counters": dict(self.counters),
                    "gauges": dict(self.gauges)
                }

            # 计算统计信息
            total_requests = len(recent_requests)
            total_response_time = sum(req.response_time for req in recent_requests)
            error_requests = sum(1 for req in recent_requests if req.status_code >= 400)

            avg_response_time = total_response_time / total_requests
            error_rate = error_requests / total_requests
            requests_per_second = total_requests / time_window

            # 按状态码分组
            status_codes = defaultdict(int)
            for req in recent_requests:
                status_codes[req.status_code] += 1

            # 按端点分组
            endpoints = defaultdict(list)
            for req in recent_requests:
                endpoints[f"{req.method} {req.endpoint}"].append(req.response_time)

            endpoint_stats = {}
            for endpoint, times in endpoints.items():
                endpoint_stats[endpoint] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times)
                }

            return {
                "time_window": time_window,
                "request_count": total_requests,
                "avg_response_time": avg_response_time,
                "error_rate": error_rate,
                "requests_per_second": requests_per_second,
                "status_codes": dict(status_codes),
                "endpoints": endpoint_stats,
                "counters": dict(self.counters),
                "gauges": dict(self.gauges)
            }


class SystemMonitor:
    """系统监控器"""

    def __init__(self, metrics_collector: MetricsCollector):
        """
        初始化系统监控器

        Args:
            metrics_collector: 指标收集器
        """
        self.metrics_collector = metrics_collector
        self.process = psutil.Process(os.getpid())
        self._monitoring = False
        self._monitor_thread = None

        logger.info("系统监控器初始化完成")

    def start_monitoring(self, interval: int = 30):
        """
        开始系统监控

        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring:
            logger.warning("系统监控已在运行中")
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()

        logger.info(f"系统监控已启动，监控间隔: {interval}秒")

    def stop_monitoring(self):
        """停止系统监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

        logger.info("系统监控已停止")

    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self._monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"系统监控错误: {e}", exc_info=True)
                time.sleep(interval)

    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)

            # 内存使用情况
            memory = psutil.virtual_memory()
            self.metrics_collector.set_gauge("system_memory_percent", memory.percent)
            self.metrics_collector.set_gauge("system_memory_used_mb", memory.used / 1024 / 1024)
            self.metrics_collector.set_gauge("system_memory_available_mb", memory.available / 1024 / 1024)

            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            self.metrics_collector.set_gauge("system_disk_percent", disk.percent)
            self.metrics_collector.set_gauge("system_disk_used_gb", disk.used / 1024 / 1024 / 1024)
            self.metrics_collector.set_gauge("system_disk_free_gb", disk.free / 1024 / 1024 / 1024)

            # 进程指标
            process_memory = self.process.memory_info()
            self.metrics_collector.set_gauge("process_memory_rss_mb", process_memory.rss / 1024 / 1024)
            self.metrics_collector.set_gauge("process_memory_vms_mb", process_memory.vms / 1024 / 1024)
            self.metrics_collector.set_gauge("process_cpu_percent", self.process.cpu_percent())

            # 文件描述符数量
            try:
                num_fds = self.process.num_fds()
                self.metrics_collector.set_gauge("process_open_fds", num_fds)
            except AttributeError:
                # Windows 系统不支持 num_fds
                pass

            # 线程数量
            num_threads = self.process.num_threads()
            self.metrics_collector.set_gauge("process_threads", num_threads)

            logger.debug(f"系统指标收集完成 - CPU: {cpu_percent}%, 内存: {memory.percent}%, 磁盘: {disk.percent}%")

        except Exception as e:
            logger.error(f"收集系统指标失败: {e}", exc_info=True)

    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态

        Returns:
            Dict: 系统状态信息
        """
        try:
            # 基本系统信息
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # 进程信息
            process_info = {
                "pid": self.process.pid,
                "create_time": self.process.create_time(),
                "cpu_percent": self.process.cpu_percent(),
                "memory_info": self.process.memory_info()._asdict(),
                "num_threads": self.process.num_threads()
            }

            # 尝试获取文件描述符数量（仅 Unix 系统）
            try:
                process_info["num_fds"] = self.process.num_fds()
            except AttributeError:
                pass

            return {
                "timestamp": time.time(),
                "system": {
                    "cpu_count": cpu_count,
                    "cpu_percent": cpu_percent,
                    "memory": {
                        "total": memory.total,
                        "available": memory.available,
                        "percent": memory.percent,
                        "used": memory.used
                    },
                    "disk": {
                        "total": disk.total,
                        "used": disk.used,
                        "free": disk.free,
                        "percent": disk.percent
                    }
                },
                "process": process_info
            }

        except Exception as e:
            logger.error(f"获取系统状态失败: {e}", exc_info=True)
            return {
                "timestamp": time.time(),
                "error": str(e)
            }


# 全局指标收集器和系统监控器实例
metrics_collector = MetricsCollector()
system_monitor = SystemMonitor(metrics_collector)