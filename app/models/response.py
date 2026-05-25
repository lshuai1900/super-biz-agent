"""响应数据模型"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SourceInfo(BaseModel):
    """知识库引用来源"""

    file_name: str = Field(default="", description="文件名")
    section: str = Field(default="", description="文档段落")
    page: int = Field(default=1, description="页码")
    chunk_id: str = Field(default="", description="chunk ID")
    score: float = Field(default=0.0, description="检索相关度分数")
    content_preview: str = Field(default="", description="内容预览")
    l2_distance: Optional[float] = Field(default=None, description="L2 距离")


class ChatResponse(BaseModel):
    """对话响应"""

    answer: str = Field(..., description="AI 回答")
    session_id: str = Field(..., description="会话 ID")


class ChatDataResponse(BaseModel):
    """带来源的对话数据"""

    success: bool = Field(default=True)
    answer: str = Field(default="")
    sources: List[SourceInfo] = Field(default_factory=list)
    errorMessage: Optional[str] = Field(default=None)


class SessionInfoResponse(BaseModel):
    """会话信息响应"""

    session_id: str = Field(..., description="会话 ID")
    message_count: int = Field(..., description="消息数量")
    history: List[Dict[str, Any]] = Field(..., description="历史消息列表")


class ApiResponse(BaseModel):
    """通用 API 响应"""

    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    data: Optional[Any] = Field(None, description="数据")


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="状态")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="版本号")
