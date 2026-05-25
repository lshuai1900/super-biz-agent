"""RAG 评估 API 接口"""

import json
from fastapi import APIRouter
from loguru import logger

from app.models.evaluation import GenerateDatasetRequest, RunEvalRequest
from app.evaluation.dataset_generator import dataset_generator
from app.evaluation.ragas_evaluator import ragas_evaluator
from app.services.memory_service import memory_service

router = APIRouter()


@router.post("/evaluation/generate_dataset")
async def generate_dataset(request: GenerateDatasetRequest):
    """生成 QA 测试集"""
    try:
        dataset = await dataset_generator.generate_dataset(
            source_dir=request.source_dir,
            count=request.count,
        )
        return {
            "code": 200,
            "message": "success",
            "data": {
                "total": len(dataset),
                "items": dataset,
            },
        }
    except Exception as e:
        logger.error(f"生成数据集失败: {e}")
        return {"code": 500, "message": str(e), "data": None}


@router.post("/evaluation/run")
async def run_evaluation(request: RunEvalRequest):
    """运行 RAG 评估"""
    try:
        dataset = None
        if request.use_dataset:
            # 从已有数据集中加载
            results = ragas_evaluator.load_results()
            if results and request.dataset_id:
                for r in results:
                    if r.get("eval_run_id") == request.dataset_id and r.get("details"):
                        dataset = r["details"]
                        break

            if not dataset and results:
                latest = results[-1]
                if latest.get("details"):
                    dataset = latest["details"]

        if not dataset:
            # 自动生成数据集
            dataset = await dataset_generator.generate_dataset(
                source_dir="aiops-docs",
                count=5,
            )

        if not dataset:
            return {"code": 400, "message": "没有可用的评估数据集", "data": None}

        result = await ragas_evaluator.evaluate(dataset)
        return {"code": 200, "message": "success", "data": result}

    except Exception as e:
        logger.error(f"运行评估失败: {e}")
        return {"code": 500, "message": str(e), "data": None}


@router.get("/evaluation/results")
async def get_evaluation_results():
    """获取所有评估结果（从数据库读取）"""
    try:
        # 优先从数据库读取
        db_runs = await memory_service.get_eval_runs(limit=20)
        db_latest = await memory_service.get_latest_eval_run()

        # 同时加载文件结果作为补充
        file_results = ragas_evaluator.load_results()

        # 合并：以数据库为主，文件结果为辅
        if db_runs:
            return {
                "code": 200,
                "message": "success",
                "data": {
                    "total": len(db_runs),
                    "latest": db_latest,
                    "results": db_runs,
                    "file_results_count": len(file_results),
                },
            }

        # 完全降级到文件结果
        latest = file_results[-1] if file_results else None
        return {
            "code": 200,
            "message": "success",
            "data": {
                "total": len(file_results),
                "latest": latest,
                "results": file_results,
                "source": "file",
            },
        }
    except Exception as e:
        logger.error(f"获取评估结果失败: {e}")
        return {"code": 500, "message": str(e), "data": None}


@router.get("/evaluation/results/{run_id}")
async def get_evaluation_run(run_id: str):
    """获取指定评估运行的详细信息"""
    try:
        run = await memory_service.get_eval_run(run_id)
        if run:
            return {"code": 200, "message": "success", "data": run}

        # 从文件查找
        file_results = ragas_evaluator.load_results()
        for r in file_results:
            if r.get("eval_run_id") == run_id:
                return {"code": 200, "message": "success", "data": r}

        return {"code": 404, "message": f"评估运行 {run_id} 未找到", "data": None}
    except Exception as e:
        logger.error(f"获取评估运行详情失败: {e}")
        return {"code": 500, "message": str(e), "data": None}
