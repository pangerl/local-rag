"""
监控 API 端点
提供系统监控和指标查询接口
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from app.core.monitoring import metrics_collector, system_monitor

logger = logging.getLogger(__name__)

# 创建监控路由器
monitoring_router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring"])


class MetricsSummaryResponse(BaseModel):
    """指标摘要响应模型"""
    time_window: int
    request_count: int
    avg_response_time: float
    error_rate: float
    requests_per_second: float
    status_codes: Dict[int, int]
    endpoints: Dict[str, Dict[str, Any]]
    counters: Dict[str, int]
    gauges: Dict[str, float]


class SystemStatusResponse(BaseModel):
    """系统状态响应模型"""
    timestamp: float
    system: Dict[str, Any]
    process: Dict[str, Any]
    error: Optional[str] = None


@monitoring_router.get(
    "/metrics",
    response_model=MetricsSummaryResponse,
    summary="获取系统指标摘要",
    description="获取指定时间窗口内的系统性能指标摘要"
)
async def get_metrics_summary(
    time_window: int = Query(
        default=300,
        ge=60,
        le=3600,
        description="时间窗口（秒），范围 60-3600"
    )
) -> MetricsSummaryResponse:
    """
    获取系统指标摘要
    
    Args:
        time_window: 时间窗口（秒）
        
    Returns:
        MetricsSummaryResponse: 指标摘要
    """
    try:
        logger.info(f"获取指标摘要，时间窗口: {time_window}秒")
        
        summary = metrics_collector.get_metrics_summary(time_window)
        
        return MetricsSummaryResponse(**summary)
        
    except Exception as e:
        logger.error(f"获取指标摘要失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"获取指标摘要失败: {str(e)}"
        )


@monitoring_router.get(
    "/system",
    response_model=SystemStatusResponse,
    summary="获取系统状态",
    description="获取当前系统资源使用情况和进程状态"
)
async def get_system_status() -> SystemStatusResponse:
    """
    获取系统状态
    
    Returns:
        SystemStatusResponse: 系统状态信息
    """
    try:
        logger.info("获取系统状态")
        
        status = system_monitor.get_system_status()
        
        return SystemStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"获取系统状态失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"获取系统状态失败: {str(e)}"
        )


@monitoring_router.get(
    "/health/detailed",
    summary="详细健康检查",
    description="获取详细的系统健康状态，包括各组件状态和性能指标"
)
async def detailed_health_check() -> Dict[str, Any]:
    """
    详细健康检查
    
    Returns:
        Dict: 详细健康状态
    """
    try:
        logger.info("执行详细健康检查")
        
        # 获取系统状态
        system_status = system_monitor.get_system_status()
        
        # 获取最近 5 分钟的指标摘要
        metrics_summary = metrics_collector.get_metrics_summary(300)
        
        # 判断系统健康状态
        health_status = "healthy"
        issues = []
        
        # 检查系统资源
        if "system" in system_status:
            system_info = system_status["system"]
            
            # CPU 使用率检查
            if system_info.get("cpu_percent", 0) > 80:
                health_status = "warning"
                issues.append(f"CPU 使用率过高: {system_info['cpu_percent']:.1f}%")
            
            # 内存使用率检查
            memory_percent = system_info.get("memory", {}).get("percent", 0)
            if memory_percent > 85:
                health_status = "warning"
                issues.append(f"内存使用率过高: {memory_percent:.1f}%")
            
            # 磁盘使用率检查
            disk_percent = system_info.get("disk", {}).get("percent", 0)
            if disk_percent > 90:
                health_status = "warning"
                issues.append(f"磁盘使用率过高: {disk_percent:.1f}%")
        
        # 检查错误率
        error_rate = metrics_summary.get("error_rate", 0)
        if error_rate > 0.1:  # 错误率超过 10%
            health_status = "warning"
            issues.append(f"错误率过高: {error_rate:.1%}")
        
        # 检查平均响应时间
        avg_response_time = metrics_summary.get("avg_response_time", 0)
        if avg_response_time > 2.0:  # 平均响应时间超过 2 秒
            health_status = "warning"
            issues.append(f"平均响应时间过长: {avg_response_time:.3f}s")
        
        return {
            "status": health_status,
            "timestamp": system_status.get("timestamp"),
            "issues": issues,
            "system": system_status,
            "metrics": metrics_summary,
            "recommendations": _get_health_recommendations(issues)
        }
        
    except Exception as e:
        logger.error(f"详细健康检查失败: {e}", exc_info=True)
        return {
            "status": "error",
            "timestamp": None,
            "issues": [f"健康检查失败: {str(e)}"],
            "error": str(e)
        }


def _get_health_recommendations(issues: list) -> list:
    """
    根据问题生成健康建议
    
    Args:
        issues: 问题列表
        
    Returns:
        list: 建议列表
    """
    recommendations = []
    
    for issue in issues:
        if "CPU 使用率过高" in issue:
            recommendations.append("建议检查是否有异常进程占用 CPU，考虑优化代码或增加服务器资源")
        elif "内存使用率过高" in issue:
            recommendations.append("建议检查内存泄漏，优化内存使用或增加内存容量")
        elif "磁盘使用率过高" in issue:
            recommendations.append("建议清理日志文件，删除不必要的文件或扩展磁盘容量")
        elif "错误率过高" in issue:
            recommendations.append("建议检查错误日志，修复导致错误的问题")
        elif "平均响应时间过长" in issue:
            recommendations.append("建议优化代码性能，检查数据库查询或增加服务器资源")
    
    if not recommendations:
        recommendations.append("系统运行正常，继续保持良好的监控")
    
    return recommendations


@monitoring_router.post(
    "/metrics/reset",
    summary="重置指标",
    description="重置所有收集的指标数据"
)
async def reset_metrics() -> Dict[str, str]:
    """
    重置指标数据
    
    Returns:
        Dict: 操作结果
    """
    try:
        logger.info("重置指标数据")
        
        # 清空指标数据
        metrics_collector.metrics.clear()
        metrics_collector.request_metrics.clear()
        metrics_collector.counters.clear()
        metrics_collector.gauges.clear()
        metrics_collector.histograms.clear()
        
        logger.info("指标数据重置完成")
        
        return {"message": "指标数据重置成功"}
        
    except Exception as e:
        logger.error(f"重置指标数据失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"重置指标数据失败: {str(e)}"
        )