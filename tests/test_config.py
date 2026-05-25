"""配置模块测试"""

import pytest


class TestConfig:
    """配置模块测试"""

    def test_config_importable(self):
        from app.config import config
        assert config.app_name == "SuperBizAgent"

    def test_allowed_origins_not_wildcard(self):
        from app.config import config
        origins = config.allowed_origins
        # 不应该等于 *
        assert origins != "*"
        # 应该包含至少一个 localhost
        assert "localhost" in origins.lower()

    def test_allowed_origins_parseable(self):
        from app.config import config
        origins = [o.strip() for o in config.allowed_origins.split(",") if o.strip()]
        assert len(origins) > 0
        for origin in origins:
            assert origin.startswith("http://") or origin.startswith("https://")

    def test_api_key_default_empty(self):
        from app.config import config
        assert config.api_key == ""

    def test_upload_config_valid(self):
        from app.config import config
        assert config.max_upload_size_mb > 0
        extensions = config.allowed_upload_extensions.split(",")
        assert len(extensions) > 0

    def test_rag_config_valid(self):
        from app.config import config
        assert config.rag_final_top_k > 0
        assert config.rag_candidate_top_k > 0

    def test_checkpointer_config(self):
        from app.config import config
        assert config.agent_checkpointer in ("memory", "sqlite", "postgres")

    def test_hybrid_search_config(self):
        from app.config import config
        assert 0 <= config.hybrid_vector_weight <= 1
        assert 0 <= config.hybrid_bm25_weight <= 1
        assert config.hybrid_rrf_k > 0

    def test_sources_model(self):
        from app.models.response import SourceInfo
        s = SourceInfo(
            file_name="test.md",
            chunk_id="chunk-1",
            score=0.85,
            content_preview="preview text...",
        )
        d = s.model_dump()
        assert d["file_name"] == "test.md"
        assert d["score"] == 0.85
