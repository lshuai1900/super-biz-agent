"""Hybrid Search 混合检索服务

结合 BM25 关键词召回和 Milvus 向量召回，使用 RRF 融合策略，
可选 Rerank 提升精确度。

配置：
- ENABLE_HYBRID_SEARCH: 是否启用混合检索
- HYBRID_VECTOR_WEIGHT / HYBRID_BM25_WEIGHT: 加权融合权重
- HYBRID_RRF_K: RRF 参数 k
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from app.config import config
from app.services.vector_search_service import SearchResult, vector_search_service
from app.services.bm25_search_service import bm25_service


def _rrf_fusion(
    vector_results: List[SearchResult],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> List[SearchResult]:
    """RRF + 加权融合

    Args:
        vector_results: 向量检索结果
        bm25_results: BM25 检索结果
        k: RRF 参数
        vector_weight: 向量权重
        bm25_weight: BM25 权重

    Returns:
        融合后的 SearchResult 列表
    """
    scores: Dict[str, float] = {}
    results_map: Dict[str, SearchResult] = {}
    content_map: Dict[str, str] = {}

    # 向量检索 RRF 分数
    for rank, r in enumerate(vector_results):
        rrf_score = 1.0 / (k + rank + 1)
        scores[r.id] = scores.get(r.id, 0) + rrf_score * vector_weight
        results_map[r.id] = r

    # BM25 检索 RRF 分数
    for rank, r in enumerate(bm25_results):
        chunk_id = r.get("chunk_id", "")
        if not chunk_id:
            continue
        rrf_score = 1.0 / (k + rank + 1)
        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score * bm25_weight
        content_map[chunk_id] = r.get("content", "")
        if chunk_id not in results_map:
            results_map[chunk_id] = SearchResult(
                id=chunk_id,
                content=r.get("content", ""),
                score=0.0,
                l2_distance=float("inf"),
                metadata=r.get("metadata", {}),
            )

    # 按融合分数排序
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused = []
    for chunk_id in sorted_ids:
        r = results_map[chunk_id]
        r.score = round(scores[chunk_id], 4)
        fused.append(r)

    return fused


class HybridSearchService:
    """混合检索服务"""

    def __init__(self):
        self.enabled = getattr(config, "enable_hybrid_search", False)
        self.vector_weight = getattr(config, "hybrid_vector_weight", 0.7)
        self.bm25_weight = getattr(config, "hybrid_bm25_weight", 0.3)
        self.rrf_k = getattr(config, "hybrid_rrf_k", 60)
        logger.info(
            f"HybridSearch 初始化: enabled={self.enabled}, "
            f"vector_w={self.vector_weight}, bm25_w={self.bm25_weight}, rrf_k={self.rrf_k}"
        )

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """混合检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            融合后的搜索结果
        """
        final_k = top_k or config.rag_final_top_k

        # 1. 向量检索
        vector_results = vector_search_service.search_similar_documents(
            query, top_k=max(final_k * 2, 10)
        )

        # 2. 如果混合检索未启用，直接返回向量结果
        if not self.enabled:
            logger.debug("Hybrid Search 未启用，仅使用向量检索")
            return vector_results[:final_k]

        # 3. BM25 检索
        bm25_results = bm25_service.search(query, top_k=max(final_k * 2, 10))
        if not bm25_results:
            logger.info("BM25 无结果或未就绪，仅使用向量检索")
            return vector_results[:final_k]

        # 4. RRF 融合
        fused = _rrf_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            k=self.rrf_k,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
        )

        # 5. 截断
        final = fused[:final_k]

        logger.info(
            f"Hybrid Search 完成: vector={len(vector_results)}, "
            f"bm25={len(bm25_results)}, fused={len(fused)}, final={len(final)}"
        )

        # 6. Rerank（如果启用）
        if config.rag_enable_rerank and final:
            try:
                from app.services.rerank_service import rerank_service
                reranked = rerank_service.rerank(query, final)
                if reranked:
                    final = reranked
            except Exception as e:
                logger.warning(f"Rerank 失败，使用融合排序: {e}")

        return final

    def build_bm25_index(self) -> bool:
        """重建 BM25 索引（可在文件上传后调用）"""
        return bm25_service.build_index()


# 全局单例
hybrid_search_service = HybridSearchService()
