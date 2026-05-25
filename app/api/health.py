"""健康检查接口"""

from typing import Any
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.config import config
from app.core.milvus_client import milvus_manager
from app.db.database import is_connected
from loguru import logger

router = APIRouter()


@router.get("/health")
async def health_check():

    """健康检查接口
    检查服务状态、Milvus 和 PostgreSQL 连接状态

    Returns:
        JSONResponse: 健康检查结果
    """
    health_data: dict[str, Any] = {
        "service": config.app_name,
        "version": config.app_version,
        "status": "healthy"
    }

    # 检查 Milvus 连接状态
    try:
        milvus_healthy = milvus_manager.health_check()
        milvus_status = "connected" if milvus_healthy else "disconnected"
        milvus_message = "Milvus 连接正常" if milvus_healthy else "Milvus 连接异常"
        health_data["milvus"] = {
            "status": milvus_status,
            "message": milvus_message
        }
    except Exception as e:
        logger.warning(f"Milvus 健康检查失败: {e}")
        health_data["milvus"] = {
            "status": "error",
            "message": f"Milvus 检查失败: {str(e)}"
        }

    # 检查 PostgreSQL 连接状态
    try:
        pg_connected = is_connected()
        pg_status = "connected" if pg_connected else "disconnected"
        pg_message = "PostgreSQL 连接正常" if pg_connected else "PostgreSQL 未连接（降级模式）"
        health_data["postgresql"] = {
            "status": pg_status,
            "message": pg_message
        }
    except Exception as e:
        logger.warning(f"PostgreSQL 健康检查失败: {e}")
        health_data["postgresql"] = {
            "status": "error",
            "message": f"PostgreSQL 检查失败: {str(e)}"
        }

    # 判断整体健康状态
    overall_status = "healthy"
    status_code = 200
    errors = []

    if health_data["milvus"]["status"] != "connected":
        overall_status = "degraded"
        errors.append("Milvus 不可用")

    if health_data["postgresql"]["status"] != "connected":
        overall_status = "degraded" if overall_status == "healthy" else overall_status
        errors.append("PostgreSQL 不可用（降级模式）")

    if errors:
        health_data["warnings"] = errors

    health_data["status"] = overall_status

    return JSONResponse(
        status_code=status_code,
        content={
            "code": 200 if overall_status != "unhealthy" else 503,
            "message": "服务运行正常" if overall_status == "healthy" else f"服务降级: {', '.join(errors)}",
            "data": health_data
        }
    )
