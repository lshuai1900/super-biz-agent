"""RAG 评估模型"""

from pydantic import BaseModel, Field


class GenerateDatasetRequest(BaseModel):
    """生成测试集请求"""
    source_dir: str = Field(default="aiops-docs", description="文档目录")
    count: int = Field(default=3, ge=1, le=20, description="生成数量 (最大 20)")
    max_docs: int = Field(default=3, ge=1, le=50, description="最多读取文档数")
    max_chunks: int = Field(default=20, ge=1, le=100, description="最多使用 chunk 数")
    timeout_seconds: int = Field(default=120, ge=10, le=600, description="超时时间 (秒)")
    use_cache: bool = Field(default=True, description="是否使用缓存")
    force: bool = Field(default=False, description="强制重新生成（跳过缓存）")
    quick: bool = Field(default=False, description="快速演示模式（自动设置小规模参数）")
    question_types: list[str] = Field(default_factory=list, description="问题类型（可选）")


class RunEvalRequest(BaseModel):
    """运行评估请求"""
    dataset_id: str | None = Field(default=None, description="数据集ID")
    use_dataset: bool = Field(default=True, description="是否使用已有数据集")
