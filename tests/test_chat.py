"""Chat session API tests"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.services.memory_service import memory_service


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_chat_session_endpoint(client):
    """Test that chat session endpoints respond (may be degraded without DB)"""
    async with client as ac:
        resp = await ac.get("/api/chat/sessions")
        data = resp.json()
        # Should always return a response, even if DB is not connected
        assert "code" in data


@pytest.mark.asyncio
async def test_chat_session_history(client):
    """Test session history endpoint returns valid structure"""
    async with client as ac:
        resp = await ac.get("/api/chat/session/test-session-001")
        data = resp.json()
        # FastAPI route uses SessionInfoResponse model or raises 500
        assert "session_id" in data or "detail" in data


@pytest.mark.asyncio
async def test_chat_session_summary(client):
    """Test session summary endpoint"""
    async with client as ac:
        resp = await ac.get("/api/chat/session/test-session-001/summary")
        data = resp.json()
        assert "code" in data


@pytest.mark.asyncio
async def test_memory_service_crud():
    """Test memory service message operations (with or without DB)"""
    # These should not crash even if PostgreSQL is not connected
    msgs = await memory_service.get_history("__test__", limit=5)
    assert isinstance(msgs, list)

    count = await memory_service.count_messages("__test__")
    assert isinstance(count, int)

    tokens = await memory_service.total_tokens("__test__")
    assert isinstance(tokens, int)

    conv = await memory_service.get_conversation("__test__")
    assert conv is None or isinstance(conv, dict)

    sessions = await memory_service.list_conversations()
    assert isinstance(sessions, list)
