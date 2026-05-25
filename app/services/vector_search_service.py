"""向量检索服务模块

支持：
- 候选 top_k (例如 20) 粗召回
- L2 距离转 similarity_score
- 双阈值过滤 (max_l2_distance / min_similarity_score)
- 最终 top_k 默认 3
- 每条结果含 score, metadata, source, chunk_index
"""

import math
from typing import Any, Dict, List, Optional

from loguru import logger
from pymilvus import Collection

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


class SearchResult:
    """搜索结果"""

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        l2_distance: float,
        metadata: Dict[str, Any],
    ):
        self.id = id
        self.content = content
        self.score = score  # 相似度分数 (越高越相似)
        self.l2_distance = l2_distance  # L2 距离 (越小越相似)
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "score": round(self.score, 4),
            "l2_distance": round(self.l2_distance, 4),
            "source": self.metadata.get("source", ""),
            "file_name": self.metadata.get("file_name", ""),
            "chunk_index": self.metadata.get("chunk_index", -1),
            "metadata": self.metadata,
        }


def l2_to_similarity(l2_distance: float, epsilon: float = 1e-8) -> float:
    """将 L2 距离转换为 similarity score (0~1)"""
    return 1.0 / (1.0 + l2_distance + epsilon)


class VectorSearchService:
    """向量检索服务"""

    def __init__(self):
        self.candidate_top_k = getattr(config, "rag_candidate_top_k", 20)
        self.final_top_k = getattr(config, "rag_final_top_k", 3)
        self.min_similarity = getattr(config, "rag_min_similarity_score", 0.0)
        self.max_l2_distance = getattr(config, "rag_max_l2_distance", 2.0)
        self.enable_rerank = getattr(config, "rag_enable_rerank", False)
        logger.info(
            f"向量检索服务初始化: candidate={self.candidate_top_k}, "
            f"final={self.final_top_k}, min_sim={self.min_similarity}, "
            f"max_l2={self.max_l2_distance}, rerank={self.enable_rerank}"
        )

    def search_similar_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """搜索相似文档（双阈值过滤）"""
        try:
            candidate_k = self.candidate_top_k
            if top_k is not None and top_k > candidate_k:
                candidate_k = top_k

            logger.info(f"搜索相似文档: query='{query[:50]}...', candidate_k={candidate_k}")

            # 1. 向量化
            query_vector = vector_embedding_service.embed_query(query)

            # 2. 获取 collection
            collection: Collection = milvus_manager.get_collection()

            # 3. 搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }

            # 4. 粗召回
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=candidate_k,
                output_fields=["id", "content", "metadata"],
            )

            # 5. 解析并过滤
            all_results = []
            for hits in results:
                for hit in hits:
                    l2_dist = hit.distance
                    sim_score = l2_to_similarity(l2_dist)

                    # 双阈值过滤
                    if l2_dist > self.max_l2_distance:
                        continue
                    if sim_score < self.min_similarity:
                        continue

                    meta = dict(hit.entity.get("metadata", {})) if hit.entity.get("metadata") else {}
                    result = SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=sim_score,
                        l2_distance=l2_dist,
                        metadata=meta,
                    )
                    all_results.append(result)

            # 6. 按相似度降序排列
            all_results.sort(key=lambda r: r.score, reverse=True)

            # 7. top_k 截断
            final_top = top_k or self.final_top_k
            final_results = all_results[:final_top]

            logger.info(
                f"检索完成: 粗召回={len(all_results)}, "
                f"最终={len(final_results)}, "
                f"过滤={len(hits) if 'hits' in dir() else 0 - len(all_results)}"
            )

            # 8. 重排（如果启用）
            if self.enable_rerank and final_results:
                try:
                    from app.services.rerank_service import rerank_service
                    reranked = rerank_service.rerank(query, final_results)
                    if reranked:
                        final_results = reranked
                except Exception as e:
                    logger.warning(f"Rerank 失败，使用原始排序: {e}")

            return final_results

        except Exception as e:
            logger.error(f"搜索相似文档失败: {e}")
            raise RuntimeError(f"搜索失败: {e}") from e


# 全局单例
vector_search_service = VectorSearchService()
