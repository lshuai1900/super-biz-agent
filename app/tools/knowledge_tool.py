"""知识检索工具 - 从向量数据库中检索相关信息"""

from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_store_manager import vector_store_manager
from app.services.vector_search_service import vector_search_service


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题

    当用户的问题涉及专业知识、文档内容或需要参考资料时，使用此工具。

    Args:
        query: 用户的问题或查询

    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")

        # 优先使用增强检索
        try:
            search_results = vector_search_service.search_similar_documents(query)
            if search_results:
                context, docs = _format_results_with_refs(search_results)
                logger.info(f"增强检索: {len(search_results)} 条结果")
                return context, docs
        except Exception as e:
            logger.warning(f"增强检索失败，降级到基础检索: {e}")

        # 降级：使用基础检索
        vector_store = vector_store_manager.get_vector_store()
        retriever = vector_store.as_retriever(
            search_kwargs={"k": getattr(config, "rag_final_top_k", 3)}
        )
        docs = retriever.invoke(query)

        if not docs:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []

        context = format_docs(docs)
        logger.info(f"基础检索到 {len(docs)} 个相关文档")
        return context, docs

    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def _format_results_with_refs(search_results) -> Tuple[str, List[Document]]:
    """格式化增强检索结果"""
    docs = []
    formatted_parts = []

    for i, sr in enumerate(search_results, 1):
        meta = sr.metadata
        source = meta.get("source", meta.get("file_name", "未知来源"))
        chunk_idx = meta.get("chunk_index", -1)

        doc = Document(
            page_content=sr.content,
            metadata=meta,
        )
        docs.append(doc)

        # 构建标题链
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in meta and meta[key]:
                headers.append(meta[key])
        header_str = " > ".join(headers) if headers else ""
        sim_score = sr.score

        formatted = f"【参考资料 {i}】(相关度: {sim_score:.3f})"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        if chunk_idx >= 0:
            formatted += f" (分片 {chunk_idx})"
        formatted += f"\n内容:\n{sr.content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts), docs


def format_docs(docs: List[Document]) -> str:
    """格式化文档列表"""
    formatted_parts = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        source = metadata.get("_file_name", metadata.get("source", "未知来源"))

        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        header_str = " > ".join(headers) if headers else ""

        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"

        formatted_parts.append(formatted)

    return "\n".join(formatted_parts)
