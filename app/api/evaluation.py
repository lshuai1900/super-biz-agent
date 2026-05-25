"""RAG 评估 API 接口"""

from fastapi import APIRouter
from loguru import logger

from app.evaluation.dataset_generator import dataset_generator
from app.evaluation.ragas_evaluator import ragas_evaluator
from app.models.evaluation import GenerateDatasetRequest, RunEvalRequest
from app.services.memory_service import memory_service

router = APIRouter()


@router.post("/evaluation/generate_dataset")
async def generate_dataset(request: GenerateDatasetRequest):
    """生成 QA 测试集

    参数：
    - count: 生成数量（默认 3，最大 20）
    - max_docs: 最多读取文档数（默认 3）
    - max_chunks: 最多使用 chunk 数（默认 20）
    - timeout_seconds: 超时时间（默认 120s）
    - use_cache: 是否使用缓存（默认 true）
    - force: 强制重新生成（默认 false）
    - quick: 快速演示模式（默认 false，自动限制参数规模）
    """
    try:
        # 快速模式：覆盖计数限制
        if request.quick:
            request.count = min(request.count, 3)
            request.max_docs = min(request.max_docs, 2)
            request.max_chunks = min(request.max_chunks, 10)
            request.timeout_seconds = min(request.timeout_seconds, 60)

        result = await dataset_generator.generate_dataset(
            source_dir=request.source_dir,
            count=request.count,
            max_docs=request.max_docs,
            max_chunks=request.max_chunks,
            timeout_seconds=request.timeout_seconds,
            use_cache=request.use_cache,
            force=request.force,
            quick=request.quick,
        )

        return {
            "code": 200 if result.get("success") else (206 if result.get("partial") else 400),
            "message": result.get("message", "success"),
            "data": {
                "total": result.get("total", 0),
                "items": result.get("items", []),
                "partial": result.get("partial", False),
                "from_cache": result.get("from_cache", False),
            },
        }
    except Exception as e:
        logger.error(f"生成数据集失败: {e}")
        return {
            "code": 500,
            "message": f"生成失败: {e}",
            "data": {"total": 0, "items": [], "partial": True},
        }


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
            # 自动生成数据集（快速模式）
            gen_result = await dataset_generator.generate_dataset(
                source_dir="aiops-docs",
                count=5,
                max_docs=3,
                max_chunks=20,
                quick=True,
            )
            dataset = gen_result.get("items", [])

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
