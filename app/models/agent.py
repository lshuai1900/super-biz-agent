"""统一 Agent 请求/响应模型"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """统一 Agent 请求"""
    session_id: str = Field(default="default", description="会话ID")
    question: str = Field(..., description="用户问题")
    mode: str = Field(default="auto", description="模式: auto | chat | rag | aiops")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "test-001",
                "question": "data-sync-service 出现 CPU 告警，帮我排查",
                "mode": "auto"
            }
        }


class AgentResponse(BaseModel):
    """统一 Agent 响应"""
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]


class MCPStatusResponse(BaseModel):
    """MCP 状态响应"""
    servers: Dict[str, Any]
    tools: List[Dict[str, Any]]
    connected: bool


class MCPToolTestRequest(BaseModel):
    """MCP 工具测试请求"""
    server: str = Field(default="cls", description="MCP 服务器名称")
    tool: str = Field(..., description="工具名称")
    params: Dict[str, Any] = Field(default_factory=dict, description="工具参数")
