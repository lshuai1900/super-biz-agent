"""评估数据模型"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvalQAItem(BaseModel):
    """QA 测试集条目"""
    question: str
    ground_truth: str
    contexts: List[str] = Field(default_factory=list)
    source: str = ""


class EvalRunRecord(BaseModel):
    """评估运行记录"""
    eval_run_id: str
    timestamp: str
    metrics: Dict[str, float]
    total_items: int
    failed_items: int
    details: List[Dict[str, Any]]
    dataset_path: str = ""


class EvalMetrics(BaseModel):
    """评估指标"""
    context_precision: float = 0.0
    context_recall: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: Optional[float] = None
