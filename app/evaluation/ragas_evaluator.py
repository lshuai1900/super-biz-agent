"""Ragas 自动质量评估

对 RAG 检索做 Context Precision / Context Recall，
对生成做 Faithfulness，并能基于真实文档自动生成 QA 测试集。
评估结果写入 PostgreSQL + JSON 文件双存储。
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger

from app.config import config
from app.services.memory_service import memory_service


class RagasEvaluator:
    """Ragas 评估器"""

    def __init__(self):
        self.data_dir = "./reports"
        os.makedirs(self.data_dir, exist_ok=True)

    async def evaluate(
        self,
        dataset: List[Dict[str, Any]],
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """运行 RAG 评估

        Args:
            dataset: QA 数据集，每项含 question, ground_truth, contexts
            run_id: 运行 ID（自动生成）

        Returns:
            Dict: 包含 metrics, details 等
        """
        run_id = run_id or str(uuid4())[:8]
        logger.info(f"开始 Ragas 评估: run_id={run_id}, dataset_size={len(dataset)}")

        # 尝试使用真实 Ragas 库
        try:
            result = await self._evaluate_with_ragas(dataset, run_id)
        except ImportError as e:
            logger.warning(f"ragas 库不可用: {e}")
            return {
                "eval_run_id": run_id,
                "error": True,
                "message": "Ragas 库未安装。请执行: pip install ragas datasets",
                "metrics": {},
                "total_items": len(dataset),
                "details": [],
            }
        except Exception as e:
            logger.error(f"Ragas 评估失败: {e}")
            return {
                "eval_run_id": run_id,
                "error": True,
                "message": f"评估执行出错: {str(e)}",
                "metrics": {},
                "total_items": len(dataset),
                "details": [],
            }

        # 写入数据库
        await self._persist_result(result, run_id, dataset)

        # 保存到文件
        self._save_result(result)

        return result

    async def _evaluate_with_ragas(
        self, dataset: List[Dict[str, Any]], run_id: str
    ) -> Dict[str, Any]:
        """使用真实 Ragas 库进行评估"""
        from datasets import Dataset as HFDataset
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas import evaluate as ragas_evaluate

        # 准备数据
        data = {
            "question": [item["question"] for item in dataset],
            "ground_truth": [item.get("ground_truth", "") for item in dataset],
            "contexts": [item.get("contexts", []) for item in dataset],
        }

        # 用当前 RAG 系统生成回答
        data["answer"] = []
        for item in dataset:
            answer = await self._get_rag_answer(item["question"])
            data["answer"].append(answer)

        hf_dataset = HFDataset.from_dict(data)

        # 运行评估
        result = ragas_evaluate(
            hf_dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        )

        metrics = {
            "context_precision": float(result.get("context_precision", 0)),
            "context_recall": float(result.get("context_recall", 0)),
            "faithfulness": float(result.get("faithfulness", 0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0)),
        }

        details = []
        for i, item in enumerate(dataset):
            details.append({
                "index": i,
                "question": item["question"],
                "answer": data["answer"][i] if i < len(data["answer"]) else "",
                "ground_truth": item.get("ground_truth", ""),
                "contexts": item.get("contexts", []),
                "source": item.get("source", ""),
            })

        return {
            "eval_run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "total_items": len(dataset),
            "failed_items": 0,
            "details": details,
        }

    async def _persist_result(
        self, result: Dict[str, Any], run_id: str, dataset: List[Dict[str, Any]]
    ):
        """将评估结果写入 PostgreSQL"""
        if result.get("error"):
            logger.warning(f"评估存在错误，跳过数据库写入")
            return

        try:
            # 写入 eval_runs
            await memory_service.save_eval_run(
                run_id=run_id,
                dataset_size=len(dataset),
                metrics=result.get("metrics", {}),
                report_path=f"reports/ragas_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            # 写入 eval_items
            items = []
            for i, item in enumerate(dataset):
                detail = result.get("details", [])[i] if i < len(result.get("details", [])) else {}
                items.append({
                    "question": item.get("question", ""),
                    "answer": detail.get("answer", ""),
                    "ground_truth": item.get("ground_truth", ""),
                    "contexts": item.get("contexts", []),
                    "metrics": {},
                    "source": item.get("source", "aiops-docs"),
                })

            await memory_service.save_eval_items(run_id, items)
            logger.info(f"评估结果已写入数据库: run_id={run_id}")
        except Exception as e:
            logger.warning(f"评估结果写入数据库失败（不影响评估报告）: {e}")

    async def _get_rag_answer(self, question: str) -> str:
        """通过 RAG 系统获取回答"""
        try:
            from app.services.rag_agent_service import rag_agent_service
            return await rag_agent_service.query(question, session_id="_eval_")
        except Exception as e:
            logger.warning(f"RAG 回答获取失败: {e}")
            return ""

    def _save_result(self, result: Dict[str, Any]):
        """保存评估结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = Path(self.data_dir) / f"ragas_eval_{timestamp}.json"

        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"评估结果已保存: {json_path}")

    def load_results(self) -> List[Dict[str, Any]]:
        """从文件加载所有评估结果（兼容旧接口）"""
        results = []
        for f in sorted(Path(self.data_dir).glob("ragas_eval_*.json")):
            try:
                results.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning(f"加载结果失败: {f.name}: {e}")
        return results


# 全局单例
ragas_evaluator = RagasEvaluator()
