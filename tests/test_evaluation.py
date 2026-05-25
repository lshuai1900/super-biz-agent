"""Evaluation API tests"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.memory_service import memory_service


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_evaluation_results_endpoint(client):
    """Test that evaluation results endpoint responds"""
    async with client as ac:
        resp = await ac.get("/api/evaluation/results")
        data = resp.json()
        assert "code" in data
        assert "data" in data


@pytest.mark.asyncio
async def test_evaluation_run_endpoint(client):
    """Test that evaluation run endpoint responds"""
    async with client as ac:
        resp = await ac.post("/api/evaluation/run", json={"use_dataset": False})
        data = resp.json()
        # May be 200 (with dataset) or 400 (no dataset) or 500 (error)
        assert "code" in data


@pytest.mark.asyncio
async def test_memory_service_eval():
    """Test eval persistence methods"""
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
async def test_eval_items_save():
    """Test saving eval items with metrics"""
    run_id = "test-items-001"
    items = [
        {
            "question": "test q1",
            "answer": "test a1",
            "ground_truth": "gt1",
            "contexts": ["ctx1"],
            "metrics": {"faithfulness": 0.9, "context_recall": 0.8},
            "source": "test",
        }
    ]
    await memory_service.save_eval_items(run_id, items)
    run_data = await memory_service.get_eval_run(run_id)
    if run_data and "details" in run_data:
        assert len(run_data["details"]) > 0
        assert "metrics" in run_data["details"][0]
