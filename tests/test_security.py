"""API Key 鉴权测试"""

import pytest
from httpx import AsyncClient, ASGITransport


class TestAPIKeyAuth:
    """API Key 鉴权测试"""

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self):
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_public_paths_no_auth(self):
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/mcp/status")
            assert resp.status_code == 200

    def test_api_key_middleware_module(self):
        from app.core.security import PROTECTED_PREFIXES, PUBLIC_PATHS
        assert "/health" in PUBLIC_PATHS
        assert any(p.startswith("/api/chat") for p in PROTECTED_PREFIXES)
        assert any(p.startswith("/api/agent") for p in PROTECTED_PREFIXES)

    def test_api_key_disabled_by_default(self):
        from app.config import config
        assert config.api_key == ""

    @pytest.mark.asyncio
    async def test_auth_when_apikey_set(self, monkeypatch):
        monkeypatch.setattr("app.core.security.config.api_key", "test-secret-123")
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/chat", json={"Id": "test", "Question": "hello"})
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_when_apikey_set(self, monkeypatch):
        monkeypatch.setattr("app.core.security.config.api_key", "test-secret-123")
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/chat",
                json={"Id": "test", "Question": "hello"},
                headers={"X-API-Key": "wrong-key"},
            )
            assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_correct_key_passes_auth(self, monkeypatch):
        monkeypatch.setattr("app.core.security.config.api_key", "test-secret-123")
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/api/chat",
                json={"Id": "test", "Question": "hello"},
                headers={"X-API-Key": "test-secret-123"},
            )
            assert resp.status_code not in (401, 403)

    @pytest.mark.asyncio
    async def test_health_always_passes(self, monkeypatch):
        monkeypatch.setattr("app.core.security.config.api_key", "test-secret-123")
        from app.main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/health")
            assert resp.status_code == 200
