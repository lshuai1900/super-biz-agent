"""文档分割服务模块 - 基于 LangChain 的智能文档分割

支持：
- Markdown：按 #、##、### 层级切分，长段落 1600 字符滑窗，overlap=100
- Text：统一 1600 字符，overlap=100
- 每个 chunk 携带完整 metadata
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from loguru import logger

from app.config import config


class DocumentSplitterService:
    """文档分割服务 - 智能 Markdown / Text 分割"""

    def __init__(self):
        self.chunk_size = getattr(config, "rag_chunk_size", 1600)
        self.chunk_overlap = getattr(config, "rag_chunk_overlap", 100)

        # Markdown 标题分割器 (三级标题)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

        # 滑窗分割器 (用于长段落二次分割)
        self.sliding_window_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

        # 纯文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

        logger.info(
            f"文档分割服务初始化, chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    def _compute_content_hash(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _build_metadata(
        self,
        file_path: str,
        chunk_index: int,
        doc_id: str,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        path = Path(file_path)
        return {
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_chunk_{chunk_index:04d}",
            "chunk_index": chunk_index,
            "source": file_path,
            "file_name": path.name,
            "file_type": path.suffix.lower(),
            "h1": (headers or {}).get("h1", ""),
            "h2": (headers or {}).get("h2", ""),
            "h3": (headers or {}).get("h3", ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """分割 Markdown 文档

        1. 按 #、##、### 层级切分
        2. 长段落用 1600 字符滑窗，overlap=100
        """
        if not content or not content.strip():
            return []

        doc_id = self._compute_content_hash(content)[:12]

        try:
            # 1. 按标题分割
            md_docs = self.markdown_splitter.split_text(content)

            result_docs = []
            for md_doc in md_docs:
                h1 = md_doc.metadata.get("h1", "")
                h2 = md_doc.metadata.get("h2", "")
                h3 = md_doc.metadata.get("h3", "")

                text = md_doc.page_content

                # 2. 如果段落仍然太长，用滑窗二次分割
                if len(text) > self.chunk_size:
                    sub_chunks = self.sliding_window_splitter.split_text(text)
                    for i, sub in enumerate(sub_chunks):
                        meta = self._build_metadata(
                            file_path, len(result_docs), doc_id,
                            {"h1": h1, "h2": h2, "h3": h3}
                        )
                        meta["content_hash"] = self._compute_content_hash(sub)
                        result_docs.append(Document(page_content=sub, metadata=meta))
                else:
                    meta = self._build_metadata(
                        file_path, len(result_docs), doc_id,
                        {"h1": h1, "h2": h2, "h3": h3}
                    )
                    meta["content_hash"] = self._compute_content_hash(text)
                    result_docs.append(Document(page_content=text, metadata=meta))

            logger.info(f"Markdown 分割完成: {file_path} -> {len(result_docs)} 分片")
            return result_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """分割纯文本：1600 字符，overlap=100"""
        if not content or not content.strip():
            return []

        doc_id = self._compute_content_hash(content)[:12]

        try:
            raw_docs = self.text_splitter.create_documents([content])
            result_docs = []
            for i, doc in enumerate(raw_docs):
                meta = self._build_metadata(file_path, i, doc_id)
                meta["content_hash"] = self._compute_content_hash(doc.page_content)
                result_docs.append(Document(page_content=doc.page_content, metadata=meta))

            logger.info(f"Text 分割完成: {file_path} -> {len(result_docs)} 分片")
            return result_docs

        except Exception as e:
            logger.error(f"Text 分割失败: {file_path}, 错误: {e}")
            raise

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """智能选择分割器"""
        if file_path.endswith(".md"):
            return self.split_markdown(content, file_path)
        return self.split_text(content, file_path)


# 全局单例
document_splitter_service = DocumentSplitterService()
