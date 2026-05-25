"""Health check API tests"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_endpoint(client):
    async with client as ac:
        resp = await ac.get("/health")
        data = resp.json()
        assert "code" in data
        assert "data" in data
        assert "service" in data["data"]
        assert "milvus" in data["data"]


@pytest.mark.asyncio
async def test_root_endpoint(client):
    async with client as ac:
        resp = await ac.get("/")
        # Root returns HTML frontend (FileResponse) when index.html exists
        content_type = resp.headers.get("content-type", "")
        if "text/html" in content_type:
            assert resp.status_code == 200
            body = resp.text
            assert "DOCTYPE html" in body or "html" in body.lower()
        else:
            # Fallback: JSON response when index.html doesn't exist
            data = resp.json()
            assert "message" in data
