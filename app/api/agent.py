"""统一 Agent API 接口

提供统一入口：POST /api/agent 和 POST /api/agent_stream
"""

import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.config import config
from app.models.agent import AgentRequest, MCPStatusResponse, MCPToolTestRequest
from app.services.router_agent_service import router_agent_service
from app.agent.mcp_client import get_mcp_client, _mcp_client

router = APIRouter()


@router.post("/agent")
async def unified_agent(request: AgentRequest):
    """统一 Agent 接口（非流式）

    支持 mode=auto | chat | rag | aiops 四种模式
    """
    try:
        logger.info(f"[统一Agent] 请求: session={request.session_id}, mode={request.mode}, question='{request.question[:60]}...'")

        result = await router_agent_service.query(
            question=request.question,
            session_id=request.session_id,
            mode=request.mode,
        )

        return {
            "code": 200,
            "message": "success",
            "data": result,
        }

    except Exception as e:
        logger.error(f"[统一Agent] 错误: {e}")
        return {
            "code": 500,
            "message": str(e),
            "data": None,
        }


@router.post("/agent_stream")
async def unified_agent_stream(request: AgentRequest):
    """统一 Agent 流式接口（SSE）

    事件类型：
    - route: 路由选择
    - content: 文本内容
    - event: AIOps 事件
    - complete: 完成
    - error: 错误
    """
    logger.info(f"[统一Agent流式] 请求: session={request.session_id}, mode={request.mode}")

    async def event_generator():
        try:
            async for chunk in router_agent_service.query_stream(
                question=request.question,
                session_id=request.session_id,
                mode=request.mode,
            ):
                yield {
                    "event": "message",
                    "data": json.dumps(chunk, ensure_ascii=False),
                }

            logger.info(f"[统一Agent流式] 完成")

        except Exception as e:
            logger.error(f"[统一Agent流式] 错误: {e}")
            yield {
                "event": "message",
                "data": json.dumps({"type": "error", "data": str(e)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.get("/mcp/status")
async def mcp_status():
    """MCP 服务状态（含 real/mock 标识）"""
    try:
        servers = {}
        tools = []
        connected = False

        if _mcp_client is not None:
            connected = True
            try:
                tools_raw = await _mcp_client.get_tools()
                for t in tools_raw:
                    tools.append({
                        "name": getattr(t, "name", str(t)),
                        "description": getattr(t, "description", ""),
                    })
            except Exception:
                pass

        # 判断每个 server 是 real 还是 mock
        server_status = {}
        for name, server_cfg in config.mcp_servers.items():
            if name == "cls":
                has_creds = bool(config.tencentcloud_secret_id and config.tencentcloud_secret_key)
                server_status[name] = {
                    "type": "real" if has_creds else "mock",
                    "mode": "CLS SDK" if has_creds else "模拟数据",
                    "endpoint": server_cfg.get("url", ""),
                }
            elif name == "monitor":
                has_prom = bool(config.prometheus_base_url and config.prometheus_base_url != "http://127.0.0.1:9090")
                server_status[name] = {
                    "type": "real" if has_prom else "mock",
                    "mode": "Prometheus HTTP API" if has_prom else "模拟数据",
                    "endpoint": server_cfg.get("url", ""),
                }
            else:
                server_status[name] = {
                    "type": "unknown",
                    "mode": "未知",
                    "endpoint": server_cfg.get("url", ""),
                }

        return {
            "code": 200,
            "message": "success",
            "data": {
                "connected": connected,
                "servers": config.mcp_servers,
                "server_status": server_status,
                "tools": tools,
            },
        }
    except Exception as e:
        logger.error(f"MCP 状态查询失败: {e}")
        return {"code": 500, "message": str(e), "data": None}


@router.get("/mcp/tools")
async def mcp_tools():
    """列出所有 MCP 工具"""
    tools = []
    try:
        if _mcp_client is not None:
            tools_raw = await _mcp_client.get_tools()
            for t in tools_raw:
                tools.append({
                    "name": getattr(t, "name", str(t)),
                    "description": getattr(t, "description", ""),
                    "args": getattr(t, "args", {}),
                })
    except Exception as e:
        logger.error(f"获取 MCP 工具列表失败: {e}")

    return {"code": 200, "message": "success", "data": {"tools": tools}}


@router.post("/mcp/test_tool")
async def mcp_test_tool(request: MCPToolTestRequest):
    """测试 MCP 工具"""
    try:
        if _mcp_client is None:
            return {"code": 500, "message": "MCP 客户端未初始化", "data": None}

        tools_raw = await _mcp_client.get_tools()
        target_tool = None
        for t in tools_raw:
            if getattr(t, "name", "") == request.tool:
                target_tool = t
                break

        if target_tool is None:
            return {"code": 404, "message": f"工具 {request.tool} 未找到", "data": None}

        result = await target_tool.ainvoke(request.params)
        return {"code": 200, "message": "success", "data": {"result": str(result)}}

    except Exception as e:
        logger.error(f"测试 MCP 工具失败: {e}")
        return {"code": 500, "message": str(e), "data": None}
