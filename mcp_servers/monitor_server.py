"""智能运维监控 MCP Server

支持真实 Prometheus API 查询（需配置 PROMETHEUS_BASE_URL），
未配置时降级为 mock 数据。
"""

import logging
import functools
import json
import random
import os
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastmcp import FastMCP

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Monitor_MCP_Server")

# 从环境变量读取 Prometheus 配置
PROMETHEUS_BASE_URL = os.environ.get("PROMETHEUS_BASE_URL", "")
PROMETHEUS_TIMEOUT = float(os.environ.get("PROMETHEUS_REQUEST_TIMEOUT", "10.0"))

mcp = FastMCP("Monitor")


def log_tool_call(func):
    """装饰器：记录工具调用的日志，包括方法名、参数和返回状态"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        method_name = func.__name__

        # 记录调用信息
        logger.info(f"=" * 80)
        logger.info(f"调用方法: {method_name}")

        # 记录参数（排除self等）
        if kwargs:
            # 使用 json.dumps 格式化参数，处理可能的序列化错误
            try:
                params_str = json.dumps(kwargs, ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                params_str = str(kwargs)
            logger.info(f"参数信息:\n{params_str}")
        else:
            logger.info("参数信息: 无")

        # 执行方法
        try:
            result = func(*args, **kwargs)

            # 记录返回状态
            logger.info(f"返回状态: SUCCESS")

            # 记录返回结果摘要（避免日志过长）
            if isinstance(result, dict):
                summary = {k: v if not isinstance(v, (list, dict)) else f"<{type(v).__name__} with {len(v)} items>"
                          for k, v in list(result.items())[:5]}
                logger.info(f"返回结果摘要: {json.dumps(summary, ensure_ascii=False)}")
            else:
                logger.info(f"返回结果: {result}")

            logger.info(f"=" * 80)
            return result

        except Exception as e:
            # 记录错误状态
            logger.error(f"返回状态: ERROR")
            logger.error(f"错误信息: {str(e)}")
            logger.error(f"=" * 80)
            raise

    return wrapper


# ============================================================
# 辅助函数
# ============================================================


def _query_prometheus(query: str) -> Optional[list]:
    """查询真实 Prometheus API

    Args:
        query: PromQL 查询语句

    Returns:
        Optional[list]: 查询结果列表，失败返回 None
    """
    if not PROMETHEUS_BASE_URL:
        return None

    try:
        import requests

        url = f"{PROMETHEUS_BASE_URL}/api/v1/query"
        resp = requests.get(url, params={"query": query}, timeout=PROMETHEUS_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                return data["data"]["result"]
        logger.warning(f"Prometheus 查询失败: {resp.status_code} - {resp.text[:200]}")
        return None
    except ImportError:
        logger.warning("requests 未安装，无法查询真实 Prometheus")
        return None
    except Exception as e:
        logger.error(f"Prometheus 查询异常: {e}")
        return None

def parse_time_or_default(time_str: Optional[str], default_offset_hours: int = 0) -> datetime:
    """解析时间字符串或返回默认时间。

    Args:
        time_str: 时间字符串（格式：YYYY-MM-DD HH:MM:SS）
        default_offset_hours: 默认时间偏移（小时）

    Returns:
        datetime: 解析后的时间对象
    """
    if time_str:
        try:
            return datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    # 返回默认时间（当前时间 + 偏移）
    return datetime.now() + timedelta(hours=default_offset_hours)


def generate_time_series(base_time: datetime, minutes_offset: int, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """生成时间序列字符串。

    Args:
        base_time: 基准时间
        minutes_offset: 分钟偏移量
        format_str: 时间格式字符串

    Returns:
        str: 格式化的时间字符串
    """
    result_time = base_time + timedelta(minutes=minutes_offset)
    return result_time.strftime(format_str)





# ============================================================
# 监控数据查询工具
# ============================================================

def _query_cpu_from_prometheus(service_name: str) -> Optional[Dict[str, Any]]:
    """从 Prometheus 查询真实 CPU 数据"""
    results = _query_prometheus(f'node_cpu_seconds_total{{job="{service_name}"}}')
    if not results:
        return None

    data_points = []
    for r in results:
        metric = r.get("metric", {})
        value = r.get("value", [None, 0])
        ts = datetime.fromtimestamp(value[0]) if value[0] else datetime.now()
        data_points.append({
            "timestamp": ts.strftime("%H:%M"),
            "value": float(value[1]) if len(value) > 1 else 0,
            "instance": metric.get("instance", ""),
        })

    values = [d["value"] for d in data_points]
    return {
        "service_name": service_name,
        "metric_name": "cpu_usage_percent",
        "interval": "1m",
        "data_points": data_points,
        "statistics": {
            "avg": round(sum(values) / len(values), 2) if values else 0,
            "max": max(values) if values else 0,
            "min": min(values) if values else 0,
        },
        "source": "prometheus",
    }


def _generate_mock_cpu(service_name: str, start_dt, end_dt, interval_minutes: int) -> Dict[str, Any]:
    """生成 mock CPU 数据"""
    data_points = []
    current_time = start_dt
    time_index = 0
    base_cpu = 10.0

    while current_time <= end_dt:
        if time_index < 3:
            cpu_value = base_cpu + (time_index * 0.5)
        else:
            growth_factor = (time_index - 2) * 8.5
            cpu_value = min(base_cpu + growth_factor, 96.0)
        cpu_value = round(cpu_value + random.uniform(-2, 2), 1)
        cpu_value = max(0, min(100, cpu_value))

        data_points.append({
            "timestamp": current_time.strftime("%H:%M"),
            "value": cpu_value,
        })
        current_time += timedelta(minutes=interval_minutes)
        time_index += 1

    values = [d["value"] for d in data_points]
    spike_detected = max(values) > 80.0 if values else False

    return {
        "service_name": service_name,
        "metric_name": "cpu_usage_percent",
        "interval": f"{interval_minutes}m",
        "data_points": data_points,
        "statistics": {
            "avg": round(sum(values) / len(values), 2) if values else 0,
            "max": max(values) if values else 0,
            "min": min(values) if values else 0,
            "spike_detected": spike_detected,
        },
        "alert_info": {
            "triggered": spike_detected,
            "threshold": 80.0,
            "message": "CPU 使用率持续超过 80% 阈值" if spike_detected else "CPU 使用率正常",
        },
        "source": "mock",
    }


@mcp.tool()
@log_tool_call
def query_cpu_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m"
) -> Dict[str, Any]:
    """查询服务的 CPU 使用率监控数据。

    支持 Prometheus API（需配置 PROMETHEUS_BASE_URL），未配置时使用模拟数据。

    Args:
        service_name: 服务名称（必填）
        start_time: 开始时间（可选，字符串格式 "YYYY-MM-DD HH:MM:SS"）
        end_time: 结束时间（可选）
        interval: 数据聚合间隔（可选，1m/5m/1h）

    Returns:
        Dict: CPU 监控数据
    """
    # 先尝试真实 Prometheus
    prom_result = _query_cpu_from_prometheus(service_name)
    if prom_result:
        return prom_result

    # 降级到 mock
    start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time, default_offset_hours=0)

    interval_minutes = 1
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60

    return _generate_mock_cpu(service_name, start_dt, end_dt, interval_minutes)


def _generate_mock_memory(service_name: str, start_dt, end_dt, interval_minutes: int) -> Dict[str, Any]:
    """生成 mock 内存数据"""
    data_points = []
    current_time = start_dt
    time_index = 0
    base_memory = 30.0
    total_gb = 8.0

    while current_time <= end_dt:
        if time_index < 3:
            memory_value = base_memory + (time_index * 1.0)
        else:
            growth_factor = (time_index - 2) * 5.5
            memory_value = min(base_memory + growth_factor, 85.0)
        memory_value = round(memory_value + random.uniform(-1, 1), 1)
        memory_value = max(0, min(100, memory_value))

        data_points.append({
            "timestamp": current_time.strftime("%H:%M"),
            "value": memory_value,
            "used_gb": round((memory_value / 100.0) * total_gb, 2),
            "total_gb": total_gb,
        })
        current_time += timedelta(minutes=interval_minutes)
        time_index += 1

    values = [d["value"] for d in data_points]
    memory_pressure = max(values) > 70.0 if values else False

    return {
        "service_name": service_name,
        "metric_name": "memory_usage_percent",
        "interval": f"{interval_minutes}m",
        "data_points": data_points,
        "statistics": {
            "avg": round(sum(values) / len(values), 2) if values else 0,
            "max": max(values) if values else 0,
            "min": min(values) if values else 0,
            "memory_pressure": memory_pressure,
        },
        "alert_info": {
            "triggered": memory_pressure,
            "threshold": 70.0,
            "message": "内存使用率超过 70% 阈值" if memory_pressure else "内存使用率正常",
        },
        "source": "mock",
    }


def _query_memory_from_prometheus(service_name: str) -> Optional[Dict[str, Any]]:
    """从 Prometheus 查询真实内存数据"""
    results = _query_prometheus(f'node_memory_MemTotal_bytes{{job="{service_name}"}}')
    if not results:
        return None

    data_points = []
    for r in results:
        metric = r.get("metric", {})
        value = r.get("value", [None, 0])
        ts = datetime.fromtimestamp(value[0]) if value[0] else datetime.now()
        data_points.append({
            "timestamp": ts.strftime("%H:%M"),
            "value": float(value[1]) / 1024 / 1024 / 1024 if len(value) > 1 else 0,
            "instance": metric.get("instance", ""),
        })

    values = [d["value"] for d in data_points]
    return {
        "service_name": service_name,
        "metric_name": "memory_usage_gb",
        "interval": "1m",
        "data_points": data_points,
        "statistics": {
            "avg": round(sum(values) / len(values), 2) if values else 0,
            "max": max(values) if values else 0,
            "min": min(values) if values else 0,
        },
        "source": "prometheus",
    }


@mcp.tool()
@log_tool_call
def query_memory_metrics(
    service_name: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    interval: str = "1m"
) -> Dict[str, Any]:
    """查询服务的内存使用监控数据。

    支持 Prometheus API（需配置 PROMETHEUS_BASE_URL），未配置时使用模拟数据。

    Args:
        service_name: 服务名称（必填）
        start_time: 开始时间（可选）
        end_time: 结束时间（可选）
        interval: 数据聚合间隔（可选，1m/5m/1h）

    Returns:
        Dict: 内存监控数据
    """
    # 先尝试真实 Prometheus
    prom_result = _query_memory_from_prometheus(service_name)
    if prom_result:
        return prom_result

    # 降级到 mock
    start_dt = parse_time_or_default(start_time, default_offset_hours=-1)
    end_dt = parse_time_or_default(end_time, default_offset_hours=0)

    interval_minutes = 1
    if interval.endswith('m'):
        interval_minutes = int(interval[:-1])
    elif interval.endswith('h'):
        interval_minutes = int(interval[:-1]) * 60

    return _generate_mock_memory(service_name, start_dt, end_dt, interval_minutes)




if __name__ == "__main__":
    # 使用 streamable-http 模式，运行在 8004 端口
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8004, path="/mcp")
