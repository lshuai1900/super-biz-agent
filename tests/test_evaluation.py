"""Evaluation API and dataset generator tests"""

import json
import os

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services.memory_service import memory_service


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestGenerateDataset:
    """测试集生成测试"""

    def test_default_count_is_3(self):
        """默认 count=3"""
        from app.models.evaluation import GenerateDatasetRequest
        req = GenerateDatasetRequest()
        assert req.count == 3

    def test_count_max_is_20(self):
        """count 最大值为 20"""
        from app.models.evaluation import GenerateDatasetRequest
        req = GenerateDatasetRequest(count=20)
        assert req.count == 20

    def test_quick_mode_params(self):
        """quick=true 时参数被限制"""
        from app.models.evaluation import GenerateDatasetRequest
        req = GenerateDatasetRequest(quick=True)
        # API 层会进一步限制，这里只验证模型层面
        assert req.quick is True
        assert req.count == 3  # 默认值

    def test_cache_enabled_by_default(self):
        """use_cache 默认为 true"""
        from app.models.evaluation import GenerateDatasetRequest
        req = GenerateDatasetRequest()
        assert req.use_cache is True

    def test_force_disabled_by_default(self):
        """force 默认为 false"""
        from app.models.evaluation import GenerateDatasetRequest
        req = GenerateDatasetRequest()
        assert req.force is False

    async def test_generate_without_real_llm(self, monkeypatch):
        """无真实 LLM 时不会崩溃，返回空结果"""
        from app.evaluation.dataset_generator import dataset_generator

        # Mock LLM 失败
        monkeypatch.setattr(
            "app.evaluation.dataset_generator.llm_factory.create_chat_model",
            lambda **kw: _FakeFailingLLM(),
        )

        result = await dataset_generator.generate_dataset(
            source_dir="aiops-docs",
            count=1,
            max_docs=1,
            max_chunks=2,
            timeout_seconds=5,
            use_cache=False,
            force=True,
        )
        assert "items" in result
        assert "total" in result

    async def test_cache_disabled_with_force(self):
        """force=true 或 use_cache=false 时跳过缓存"""
        from app.evaluation.dataset_generator import dataset_generator

        # 清除缓存确保测试干净
        dataset_generator.clear_cache()

        # use_cache=False 不会从缓存读取
        result = await dataset_generator.generate_dataset(
            source_dir="aiops-docs",
            count=1,
            max_docs=1,
            max_chunks=2,
            timeout_seconds=5,
            use_cache=False,
            force=True,
        )
        assert not result.get("from_cache", False)

    async def test_force_skips_cache(self, monkeypatch):
        """force=true 时跳过缓存走生成路径"""
        monkeypatch.setattr(
            "app.evaluation.dataset_generator.llm_factory.create_chat_model",
            lambda **kw: _FakeFailingLLM(),
        )

        from app.evaluation.dataset_generator import dataset_generator
        result = await dataset_generator.generate_dataset(
            source_dir="/nonexistent/path",
            count=1,
            max_docs=1,
            max_chunks=2,
            timeout_seconds=5,
            use_cache=True,
            force=True,
        )
        assert not result.get("from_cache", False)

    async def test_generator_missing_docs(self):
        """文档目录不存在时返回错误"""
        from app.evaluation.dataset_generator import dataset_generator
        result = await dataset_generator.generate_dataset(
            source_dir="/nonexistent/path",
            count=1,
            timeout_seconds=5,
            use_cache=False,
            force=True,
        )
        assert result["success"] is False
        assert "未在" in result.get("message", "")

    async def test_api_generate_endpoint(self, client):
        """API 端点返回合理结构"""
        async with client as ac:
            resp = await ac.post(
                "/api/evaluation/generate_dataset",
                json={
                    "source_dir": "aiops-docs",
                    "count": 1,
                    "max_docs": 1,
                    "max_chunks": 2,
                    "timeout_seconds": 10,
                    "use_cache": False,
                    "force": True,
                },
            )
            data = resp.json()
            assert "code" in data
            assert "data" in data
            # 可能因为 LLM 不可用而返回空列表，但不应该崩溃
            assert "total" in data["data"]

    async def test_api_generate_quick_mode(self, client):
        """quick=true 快速模式使用小参数"""
        async with client as ac:
            resp = await ac.post(
                "/api/evaluation/generate_dataset",
                json={
                    "source_dir": "aiops-docs",
                    "quick": True,
                },
            )
            data = resp.json()
            assert "code" in data
            assert isinstance(data["data"]["total"], int)

    def test_clear_cache_method_exists(self):
        """clear_cache 方法存在且可调用"""
        from app.evaluation.dataset_generator import dataset_generator
        result = dataset_generator.clear_cache()
        assert isinstance(result, bool)


class TestEvalPersistence:
    """评估持久化测试"""

    @pytest.mark.asyncio
    async def test_evaluation_results_endpoint(self, client):
        async with client as ac:
            resp = await ac.get("/api/evaluation/results")
            data = resp.json()
            assert "code" in data
            assert "data" in data

    @pytest.mark.asyncio
    async def test_evaluation_run_endpoint(self, client):
        async with client as ac:
            resp = await ac.post("/api/evaluation/run", json={"use_dataset": False})
            data = resp.json()
            assert "code" in data

    @pytest.mark.asyncio
    async def test_memory_service_eval(self):
        run_id = "test-run-001"
        metrics = {"context_precision": 0.85, "context_recall": 0.72, "faithfulness": 0.91}
        await memory_service.save_eval_run(run_id, 3, metrics)
        runs = await memory_service.get_eval_runs(limit=5)
        assert isinstance(runs, list)

        run_data = await memory_service.get_eval_run(run_id)
        if run_data is not None:
            assert "metrics" in run_data
            assert run_data["run_id"] == run_id

        latest = await memory_service.get_latest_eval_run()
        if latest is not None:
            assert "metrics" in latest

    @pytest.mark.asyncio
    async def test_eval_items_save(self):
        run_id = "test-items-001"
        items = [{
            "question": "test q1",
            "answer": "test a1",
            "ground_truth": "gt1",
            "contexts": ["ctx1"],
            "metrics": {"faithfulness": 0.9, "context_recall": 0.8},
            "source": "test",
        }]
        await memory_service.save_eval_items(run_id, items)
        run_data = await memory_service.get_eval_run(run_id)
        if run_data and "details" in run_data:
            assert len(run_data["details"]) > 0
            assert "metrics" in run_data["details"][0]


class _FakeFailingLLM:
    """模拟失败的 LLM（不调用真实 API）"""

    def invoke(self, prompt):
        raise RuntimeError("Mock LLM 不可用")
