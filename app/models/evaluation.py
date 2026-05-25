"""RAG 评估模型"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvalItem(BaseModel):
    """评估数据项"""
    question: str
    ground_truth: str
    contexts: List[str]
    source: str = ""


class EvalResult(BaseModel):
    """评估结果"""
    eval_run_id: str
    timestamp: str
    metrics: Dict[str, float]
    total_items: int
    failed_items: int
    details: List[Dict[str, Any]]


class GenerateDatasetRequest(BaseModel):
    """生成测试集请求"""
    source_dir: str = Field(default="aiops-docs", description="文档目录")
    count: int = Field(default=10, ge=1, le=100, description="生成数量")


class RunEvalRequest(BaseModel):
    """运行评估请求"""
    dataset_id: Optional[str] = Field(default=None, description="数据集ID")
    use_dataset: bool = Field(default=True, description="是否使用已有数据集")
