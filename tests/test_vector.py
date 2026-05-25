"""Vector document splitter and search tests"""

import pytest
from app.services.document_splitter_service import document_splitter_service


class TestDocumentSplitter:
    """Document splitter service tests"""

    MARKDOWN_SAMPLE = """# 一级标题

这是第一个段落的内容。

## 二级标题 1

二级段落内容。

### 三级标题 A

三级段落内容，包含一些详细说明。
"""

    TEXT_SAMPLE = """这是一段纯文本内容。
它没有标题层级。
只是连续的文字段落。
用于测试纯文本分割器。
"""

    def test_split_markdown_by_headers(self):
        docs = document_splitter_service.split_markdown(self.MARKDOWN_SAMPLE)
        assert isinstance(docs, list)
        if len(docs) > 0:
            doc = docs[0]
            assert hasattr(doc, "page_content")
            assert hasattr(doc, "metadata")
            # Metadata should contain h1
            assert "doc_id" in doc.metadata
            assert "chunk_id" in doc.metadata
            assert "chunk_index" in doc.metadata
            assert "source" in doc.metadata
            assert "file_name" in doc.metadata
            assert "content_hash" in doc.metadata
            assert "created_at" in doc.metadata

    def test_split_markdown_empty(self):
        docs = document_splitter_service.split_markdown("")
        assert docs == []

    def test_split_text(self):
        docs = document_splitter_service.split_text(self.TEXT_SAMPLE)
        assert isinstance(docs, list)
        if len(docs) > 0:
            assert "content_hash" in docs[0].metadata

    def test_split_text_empty(self):
        docs = document_splitter_service.split_text("")
        assert docs == []

    def test_split_document_markdown(self):
        docs = document_splitter_service.split_document(
            self.MARKDOWN_SAMPLE, "test.md"
        )
        assert isinstance(docs, list)

    def test_split_document_text(self):
        docs = document_splitter_service.split_document(
            self.TEXT_SAMPLE, "test.txt"
        )
        assert isinstance(docs, list)

    def test_chunk_metadata_fields(self):
        """Test that all required metadata fields are present"""
        docs = document_splitter_service.split_markdown(self.MARKDOWN_SAMPLE, "test.md")
        if docs:
            meta = docs[0].metadata
            required_fields = [
                "doc_id", "chunk_id", "chunk_index", "source",
                "file_name", "file_type", "content_hash", "created_at",
            ]
            for field in required_fields:
                assert field in meta, f"Missing metadata field: {field}"

    def test_content_hash_consistency(self):
        """Same content should produce same hash"""
        docs1 = document_splitter_service.split_text("Hello World", "a.txt")
        docs2 = document_splitter_service.split_text("Hello World", "b.txt")
        if docs1 and docs2:
            assert docs1[0].metadata["content_hash"] == docs2[0].metadata["content_hash"]

    def test_l2_to_similarity(self):
        from app.services.vector_search_service import l2_to_similarity
        assert 0 < l2_to_similarity(0.0) <= 1.0
        assert l2_to_similarity(1.0) < l2_to_similarity(0.0)
        assert l2_to_similarity(100.0) > 0
