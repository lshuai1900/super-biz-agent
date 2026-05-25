"""BM25 关键词检索服务

构建基于已入库 chunk 的 BM25 索引，支持关键词检索。
当 BM25 索引不可用时自动降级，不影响向量检索。
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

from loguru import logger
from pymilvus import Collection

from app.core.milvus_client import milvus_manager


class BM25SearchService:
    """BM25 关键词检索"""

    def __init__(self):
        self._index: Optional["BM25"] = None
        self._chunks: Dict[str, Dict[str, Any]] = {}  # chunk_id -> metadata
        self._chunk_ids: List[str] = []
        self._ready = False

    def build_index(self) -> bool:
        """从 Milvus 中加载所有 chunk 构建 BM25 索引"""
        try:
            collection = milvus_manager.get_collection()
            if collection is None:
                logger.warning("BM25: Milvus collection 不可用")
                self._ready = False
                return False

            # 查询所有 chunk
            all_chunks = self._fetch_all_chunks(collection)
            if not all_chunks:
                logger.warning("BM25: 未找到任何 chunk")
                self._ready = False
                return False

            corpus_texts = []
            corpus_ids = []
            for chunk_id, content, meta in all_chunks:
                corpus_texts.append(content or "")
                corpus_ids.append(chunk_id)
                self._chunks[chunk_id] = meta

            self._index = BM25(corpus_texts)
            self._chunk_ids = corpus_ids
            self._ready = True
            logger.info(f"BM25 索引构建完成: {len(corpus_texts)} 条")
            return True

        except Exception as e:
            logger.warning(f"BM25 索引构建失败（降级，不影响向量检索）: {e}")
            self._ready = False
            return False

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 关键词检索

        Returns:
            [{chunk_id, content, metadata, score}, ...]
        """
        if not self._ready or self._index is None:
            logger.debug("BM25: 索引未就绪，跳过关键词检索")
            return []

        try:
            scores = self._index.score(query)
            tokenized_scores = {self._chunk_ids[i]: score for i, score in scores.items()}

            # 按分数降序取 top_k
            sorted_ids = sorted(tokenized_scores.keys(), key=lambda x: tokenized_scores[x], reverse=True)[:top_k]
            results = []
            for chunk_id in sorted_ids:
                meta = self._chunks.get(chunk_id, {})
                results.append({
                    "chunk_id": chunk_id,
                    "content": meta.get("content", ""),
                    "metadata": meta,
                    "score": round(tokenized_scores[chunk_id], 4),
                })
            return results

        except Exception as e:
            logger.warning(f"BM25 检索失败（降级）: {e}")
            return []

    def _fetch_all_chunks(self, collection: Collection) -> List[tuple]:
        """从 Milvus 获取所有 chunk"""
        all_data = []
        # 先获取总数
        try:
            # 使用 query 获取所有数据
            results = collection.query(
                expr="id != ''",
                output_fields=["id", "content", "metadata"],
                limit=10000,
            )
            for r in results:
                chunk_id = r.get("id", "")
                content = r.get("content", "")
                meta = dict(r.get("metadata", {})) if r.get("metadata") else {}
                meta["content"] = content
                all_data.append((chunk_id, content, meta))
        except Exception as e:
            logger.warning(f"BM25: 获取 chunk 数据失败: {e}")

        return all_data

    @property
    def is_ready(self) -> bool:
        return self._ready


class BM25:
    """BM25 算法实现 (Okapi BM25)"""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.n_docs = len(corpus)

        # 计算文档频率
        self.doc_freqs = defaultdict(int)  # term -> doc count
        self.doc_len = []
        self.avgdl = 0.0

        tokenized_corpus = []
        for doc in corpus:
            tokens = self._tokenize(doc)
            tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))
            for token in set(tokens):
                self.doc_freqs[token] += 1

        if self.n_docs > 0:
            self.avgdl = sum(self.doc_len) / self.n_docs
        else:
            self.avgdl = 1.0

        # 构建倒排索引
        self.inverted_index = defaultdict(list)  # term -> [(doc_idx, term_freq)]
        for idx, tokens in enumerate(tokenized_corpus):
            term_counts = defaultdict(int)
            for t in tokens:
                term_counts[t] += 1
            for t, cnt in term_counts.items():
                self.inverted_index[t].append((idx, cnt))

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（按非字母数字字符分割）"""
        if not text:
            return []
        tokens = []
        current = ""
        for ch in text.lower():
            if ch.isalnum():
                current += ch
            else:
                if current:
                    tokens.append(current)
                    current = ""
        if current:
            tokens.append(current)
        return tokens

    def score(self, query: str) -> Dict[int, float]:
        """计算 query 与每个文档的 BM25 分数"""
        query_tokens = self._tokenize(query)
        scores = defaultdict(float)

        for token in query_tokens:
            doc_freq = self.doc_freqs.get(token, 0)
            if doc_freq == 0:
                continue

            idf = math.log(1 + (self.n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

            for doc_idx, term_freq in self.inverted_index.get(token, []):
                if doc_idx >= self.n_docs:
                    continue
                doc_len = self.doc_len[doc_idx] if doc_idx < len(self.doc_len) else self.avgdl
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_idx] += idf * numerator / denominator

        return dict(scores)


# 全局单例
bm25_service = BM25SearchService()
