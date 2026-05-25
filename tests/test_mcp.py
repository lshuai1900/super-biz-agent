"""MCP status API tests"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_mcp_status_endpoint(client):
    """Test MCP status endpoint"""
    async with client as ac:
        resp = await ac.get("/api/mcp/status")
        data = resp.json()
        assert "code" in data
        assert "data" in data
        d = data["data"]
        # Should have server_status indicating real/mock for each server
        assert "server_status" in d or "servers" in d or "connected" in d


@pytest.mark.asyncio
async def test_mcp_tools_endpoint(client):
    """Test MCP tools listing endpoint"""
    async with client as ac:
        resp = await ac.get("/api/mcp/tools")
        data = resp.json()
        assert "code" in data
        assert "data" in data


@pytest.mark.asyncio
async def test_mcp_config():
    """Test MCP server config has expected structure"""
    from app.config import config
    servers = config.mcp_servers
    assert isinstance(servers, dict)
    assert "cls" in servers
    assert "monitor" in servers
    for name, cfg in servers.items():
        assert "transport" in cfg
        assert "url" in cfg


@pytest.mark.asyncio
async def test_mcp_status_has_real_mock(client):
    """Test MCP status returns real/mock info when server_status present"""
    async with client as ac:
        resp = await ac.get("/api/mcp/status")
        data = resp.json()
        if data.get("data") and data["data"].get("server_status"):
            for name, status in data["data"]["server_status"].items():
                assert "type" in status
                assert status["type"] in ("real", "mock", "unknown")
