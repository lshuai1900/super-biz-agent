"""重排服务 - 支持 DashScope Rerank 或本地降级

配置项：
- RAG_ENABLE_RERANK: 是否启用重排
- RAG_RERANK_MODEL: 重排模型名称（DashScope: gte-rerank）
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from app.config import config


class RerankService:
    """重排服务"""

    def __init__(self):
        self.enabled = getattr(config, "rag_enable_rerank", False)
        self.model = getattr(config, "rag_rerank_model", "gte-rerank")
        self.api_key = getattr(config, "dashscope_api_key", "")
        logger.info(f"Rerank 服务初始化: enabled={self.enabled}, model={self.model}")

    def rerank(
        self,
        query: str,
        results: List[Any],
    ) -> List[Any]:
        """对检索结果进行重排

        Args:
            query: 原始查询
            results: SearchResult 列表

        Returns:
            重排后的 SearchResult 列表
        """
        if not self.enabled or not results:
            return results

        # 尝试 DashScope Rerank API
        reranked = self._dashscope_rerank(query, results)
        if reranked:
            return reranked

        # 降级: 基于已有关联度排序（不做改变）
        logger.info("Rerank 不可用，使用原始排序（按相似度）")
        return results

    def _dashscope_rerank(self, query: str, results: List[Any]) -> Optional[List[Any]]:
        """使用 DashScope Rerank API"""
        try:
            import dashscope
            from dashscope.api_entities.dashscope_response import RerankResponse

            if not self.api_key:
                logger.warning("DASHSCOPE_API_KEY 未配置，无法使用 DashScope Rerank")
                return None

            documents = [r.content for r in results]

            response = dashscope.TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                api_key=self.api_key,
                return_documents=True,
                top_n=len(documents),
            )

            if not response or response.status_code != 200:
                logger.warning(f"DashScope Rerank 返回异常: {response}")
                return None

            # 按重排结果排序
            reranked = []
            if hasattr(response, "output") and response.output:
                results_dict = {r.content: r for r in results}
                for item in response.output.results:
                    doc_text = item.get("document", {}).get("text", "")
                    if doc_text in results_dict:
                        original = results_dict[doc_text]
                        original.score = item.get("relevance_score", original.score)
                        reranked.append(original)

            if reranked:
                # 按新分数降序
                reranked.sort(key=lambda r: r.score, reverse=True)
                logger.info(f"DashScope Rerank 完成: {len(reranked)} 条")
                return reranked

            return None

        except ImportError:
            logger.warning("dashscope 库未安装，无法使用 Rerank")
            return None
        except Exception as e:
            logger.warning(f"DashScope Rerank 失败: {e}")
            return None


# 全局单例
rerank_service = RerankService()
