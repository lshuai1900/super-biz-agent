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

    def _setup_ragas_llm(self):
        """配置 Ragas 使用的 LLM 和 embedding"""
        try:
            from langchain_openai import ChatOpenAI
            from langchain_openai import OpenAIEmbeddings
            from ragas.llms import LangchainLLM
            from ragas.embeddings import LangchainEmbeddings

            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            api_key = config.dashscope_api_key

            if not api_key:
                logger.warning("DASHSCOPE_API_KEY 未配置，Ragas 使用默认 LLM")
                return None, None

            # 为 Ragas 设置评判 LLM
            judge_llm = ChatOpenAI(
                model="qwen-plus",
                temperature=0.0,
                base_url=base_url,
                api_key=api_key,
            )

            # 为 Ragas 设置 embedding
            judge_embeddings = OpenAIEmbeddings(
                model="text-embedding-v4",
                base_url=base_url,
                api_key=api_key,
            )

            # 保存到实例变量供 evaluate 方法使用
            self._ragas_llm = LangchainLLM(judge_llm)
            self._ragas_embeddings = LangchainEmbeddings(judge_embeddings)

            return self._ragas_llm, self._ragas_embeddings
        except Exception as e:
            logger.warning(f"Ragas LLM/embedding 配置失败，使用默认: {e}")
            self._ragas_llm = None
            self._ragas_embeddings = None
            return None, None

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
        import pandas as pd

        # 配置 Ragas 使用的 LLM 和 embedding
        self._setup_ragas_llm()

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

        # 将 LLM/embeddings 注入 metrics
        if getattr(self, '_ragas_llm', None):
            context_precision.llm = self._ragas_llm
            context_recall.llm = self._ragas_llm
            faithfulness.llm = self._ragas_llm
            answer_relevancy.llm = self._ragas_llm
        if getattr(self, '_ragas_embeddings', None):
            answer_relevancy.embeddings = self._ragas_embeddings

        # 运行评估
        result = ragas_evaluate(
            hf_dataset,
            metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        )

        # 转换为 DataFrame 获取 per-item 和 overall metrics
        result_df = result.to_pandas()

        # 总体指标（取均值）
        metrics = {}
        for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            if col in result_df.columns:
                metrics[col] = float(result_df[col].mean())

        # Per-item 指标
        details = []
        for i, item in enumerate(dataset):
            item_metrics = {}
            for col in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
                if col in result_df.columns and i < len(result_df):
                    val = result_df[col].iloc[i]
                    item_metrics[col] = float(val) if pd.notna(val) else 0.0

            details.append({
                "index": i,
                "question": item["question"],
                "answer": data["answer"][i] if i < len(data["answer"]) else "",
                "ground_truth": item.get("ground_truth", ""),
                "contexts": item.get("contexts", []),
                "source": item.get("source", ""),
                "metrics": item_metrics,
            })

        # 找出低分样例（faithfulness < 0.7 或 context_recall < 0.5）
        low_score_samples = [
            d for d in details
            if d["metrics"].get("faithfulness", 1.0) < 0.7
            or d["metrics"].get("context_recall", 1.0) < 0.5
        ]

        return {
            "eval_run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "total_items": len(dataset),
            "failed_items": len([d for d in details if not d.get("answer")]),
            "details": details,
            "low_score_samples": low_score_samples[:5],
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

            # 写入 eval_items（含 item-level metrics）
            items = []
            for i, item in enumerate(dataset):
                detail = result.get("details", [])[i] if i < len(result.get("details", [])) else {}
                items.append({
                    "question": item.get("question", ""),
                    "answer": detail.get("answer", ""),
                    "ground_truth": item.get("ground_truth", ""),
                    "contexts": item.get("contexts", []),
                    "metrics": detail.get("metrics", {}),
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
