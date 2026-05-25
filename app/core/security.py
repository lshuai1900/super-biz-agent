"""API Key 鉴权中间件

- API_KEY 为空时不启用鉴权
- API_KEY 不为空时，请求必须携带 X-API-Key Header
- /api/health 永远不加鉴权
- 缺失 API Key 返回 401，错误 API Key 返回 403
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import config

# 需要鉴权的路径前缀
PROTECTED_PREFIXES = (
    "/api/chat",
    "/api/agent",
    "/api/aiops",
    "/api/upload",
    "/api/evaluation",
)

# 永远放行的路径
PUBLIC_PATHS = (
    "/health",
    "/api/health",
    "/api/mcp/status",
    "/api/mcp/tools",
    "/api/mcp/test_tool",
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API Key 鉴权中间件"""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 公开路径放行
        if path in PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/openapi") or path == "/":
            return await call_next(request)

        # 检查是否需要鉴权
        needs_auth = path.startswith(PROTECTED_PREFIXES)

        if not needs_auth:
            return await call_next(request)

        # API_KEY 为空时跳过鉴权（本地开发模式）
        if not config.api_key:
            return await call_next(request)

        # 校验 API Key
        api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")

        if not api_key:
            logger.warning(f"API Key 缺失: {request.method} {path}")
            return JSONResponse(status_code=401, content={"detail": "API Key 缺失，请在 Header 中提供 X-API-Key"})

        if api_key != config.api_key:
            logger.warning(f"API Key 错误: {request.method} {path}")
            return JSONResponse(status_code=403, content={"detail": "API Key 无效"})

        return await call_next(request)
