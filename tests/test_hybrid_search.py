"""Hybrid Search 和 BM25 测试"""

import pytest


class TestBM25:
    """BM25 算法测试"""

    def test_bm25_init(self):
        from app.services.bm25_search_service import BM25

        corpus = [
            "故障排查 CPU 使用率过高 服务响应慢",
            "内存泄漏 磁盘空间不足 服务器宕机",
            "网络延迟 丢包率 数据库连接池耗尽",
            "CPU 使用率过高 可能是死循环导致 需要查看进程",
        ]
        bm25 = BM25(corpus)
        assert bm25.n_docs == 4
        assert bm25.avgdl > 0

    def test_bm25_score(self):
        from app.services.bm25_search_service import BM25

        corpus = [
            "故障排查 CPU 使用率过高 服务响应慢",
            "内存泄漏 磁盘空间不足 服务器宕机",
            "网络延迟 丢包率 数据库连接池耗尽",
        ]
        bm25 = BM25(corpus)
        scores = bm25.score("CPU 使用率过高")
        assert isinstance(scores, dict)
        # doc 0 和 doc 3 (if existed) 应该得分最高

    def test_bm25_empty_corpus(self):
        from app.services.bm25_search_service import BM25

        bm25 = BM25([])
        assert bm25.n_docs == 0
        scores = bm25.score("test")
        assert scores == {}

    def test_bm25_service_importable(self):
        from app.services.bm25_search_service import bm25_service
        assert bm25_service is not None
        assert not bm25_service.is_ready  # 未 build_index 时为 False

    def test_bm25_build_index_no_milvus(self):
        """无 Milvus 连接时 build_index 不崩溃"""
        from app.services.bm25_search_service import bm25_service
        result = bm25_service.build_index()
        # 应该返回 False（因为 Milvus 不可用）但不应抛异常
        assert isinstance(result, bool)

    def test_bm25_search_always_returns_list(self):
        """search 始终返回列表，不崩溃"""
        from app.services.bm25_search_service import bm25_service
        results = bm25_service.search("test query")
        assert isinstance(results, list)


class TestHybridSearch:
    """混合检索测试"""

    def test_rrf_fusion(self):
        from app.services.hybrid_search_service import _rrf_fusion
        from app.services.vector_search_service import SearchResult

        # 构造测试数据
        vec_results = [
            SearchResult(id="c1", content="doc1", score=0.9, l2_distance=0.1,
                         metadata={"file_name": "a.md"}),
            SearchResult(id="c2", content="doc2", score=0.8, l2_distance=0.2,
                         metadata={"file_name": "b.md"}),
        ]
        bm25_results = [
            {"chunk_id": "c1", "content": "doc1", "score": 0.7, "metadata": {}},
            {"chunk_id": "c3", "content": "doc3", "score": 0.6, "metadata": {}},
        ]

        fused = _rrf_fusion(vec_results, bm25_results, k=60)
        assert len(fused) >= 2
        # 去重：c1 不应该出现两次
        ids = [r.id for r in fused]
        assert len(ids) == len(set(ids))  # 无重复

    def test_hybrid_service_disabled_by_default(self):
        from app.services.hybrid_search_service import hybrid_search_service
        # 默认未启用
        assert not hybrid_search_service.enabled

    def test_hybrid_search_l2_to_similarity(self):
        from app.services.vector_search_service import l2_to_similarity

        # 距离越小，相似度越高
        sim_near = l2_to_similarity(0.0)
        sim_far = l2_to_similarity(10.0)
        assert sim_near > sim_far
        assert 0 < sim_near <= 1.0

    def test_search_result_to_dict(self):
        from app.services.vector_search_service import SearchResult

        r = SearchResult(
            id="test-1",
            content="test content",
            score=0.85,
            l2_distance=0.3,
            metadata={"file_name": "test.md", "source": "aiops-docs"},
        )
        d = r.to_dict()
        assert d["id"] == "test-1"
        assert d["score"] == 0.85
        assert d["file_name"] == "test.md"
        assert "source" in d
