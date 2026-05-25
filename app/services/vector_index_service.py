"""向量索引服务模块

支持：
- 重复上传同一文件时，根据 content_hash 去重或覆盖
- 批量索引 uploads 和 aiops-docs
- 返回索引统计：文件数、chunk 数、耗时、失败文件
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from app.services.document_splitter_service import document_splitter_service
from app.services.vector_store_manager import vector_store_manager


class IndexingResult:
    """索引结果类"""

    def __init__(self):
        self.success = False
        self.directory_path = ""
        self.total_files = 0
        self.success_count = 0
        self.fail_count = 0
        self.total_chunks = 0
        self.skipped_chunks = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.error_message = ""
        self.failed_files: Dict[str, str] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "total_chunks": self.total_chunks,
            "skipped_chunks": self.skipped_chunks,
            "duration_ms": int((self.end_time - self.start_time).total_seconds() * 1000)
            if self.start_time and self.end_time else 0,
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }

    def increment_success_count(self):
        self.success_count += 1

    def increment_fail_count(self):
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str):
        self.failed_files[file_path] = error


class VectorIndexService:
    """向量索引服务"""

    def __init__(self):
        self.upload_path = "./uploads"
        self.aiops_docs_path = "./aiops-docs"
        logger.info("向量索引服务初始化完成")

    def _load_existing_hashes(self) -> Set[str]:
        """从向量库中读取已有的 content_hash 集合，用于去重"""
        try:
            from pymilvus import Collection, utility
            from app.core.milvus_client import milvus_manager

            collection = milvus_manager.get_collection()
            collection.load()

            # 查询所有 metadata 中的 content_hash
            results = collection.query(
                expr="",
                output_fields=["metadata"],
                limit=10000,
            )
            hashes = set()
            for r in results:
                meta = r.get("metadata", {})
                if isinstance(meta, dict) and "content_hash" in meta:
                    hashes.add(meta["content_hash"])
            logger.info(f"Loaded {len(hashes)} existing content hashes from Milvus")
            return hashes
        except Exception as e:
            logger.warning(f"Failed to load existing hashes: {e}")
            return set()

    def index_single_file(self, file_path: str, existing_hashes: Optional[Set[str]] = None):
        """索引单个文件，带去重"""
        path = Path(file_path).resolve()
        if not path.exists() or not path.is_file():
            raise ValueError(f"文件不存在: {file_path}")

        logger.info(f"开始索引文件: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            normalized_path = path.as_posix()

            # 分割文档
            documents = document_splitter_service.split_document(content, normalized_path)

            if not documents:
                logger.warning(f"文件为空或无法分割: {file_path}")
                return

            # 去重
            if existing_hashes is not None:
                new_docs = []
                skipped = 0
                for doc in documents:
                    ch = doc.metadata.get("content_hash", "")
                    if ch and ch in existing_hashes:
                        skipped += 1
                    else:
                        new_docs.append(doc)
                        if ch:
                            existing_hashes.add(ch)

                if skipped > 0:
                    logger.info(f"跳过 {skipped} 个已存在的分片")
                if not new_docs:
                    logger.info(f"文件所有分片均已存在，跳过: {file_path}")
                    return
                documents = new_docs

            # 先删除旧数据（覆盖模式）
            try:
                vector_store_manager.delete_by_source(normalized_path)
            except Exception:
                pass

            # 添加文档到向量存储
            ids = vector_store_manager.add_documents(documents)
            logger.info(f"文件索引成功: {file_path}, {len(documents)} 分片, {len(ids)} IDs")

        except Exception as e:
            logger.error(f"索引文件失败: {file_path}, 错误: {e}")
            raise RuntimeError(f"索引文件失败: {e}") from e

    def index_directory(self, directory_path: Optional[str] = None) -> IndexingResult:
        """批量索引目录"""
        result = IndexingResult()
        result.start_time = datetime.now()

        try:
            target_path = directory_path or self.upload_path
            dir_path = Path(target_path).resolve()

            if not dir_path.exists() or not dir_path.is_dir():
                raise ValueError(f"目录不存在: {target_path}")

            result.directory_path = str(dir_path)

            files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))
            if not files:
                result.total_files = 0
                result.success = True
                result.end_time = datetime.now()
                return result

            result.total_files = len(files)
            logger.info(f"开始索引目录: {target_path}, {len(files)} 个文件")

            # 加载已有 hash 用于去重
            existing_hashes = self._load_existing_hashes()

            for file_path in files:
                try:
                    self.index_single_file(str(file_path), existing_hashes)
                    result.increment_success_count()
                except Exception as e:
                    result.increment_fail_count()
                    result.add_failed_file(str(file_path), str(e))
                    logger.error(f"文件索引失败: {file_path.name}, {e}")

            result.success = result.fail_count == 0
            result.end_time = datetime.now()

            logger.info(
                f"目录索引完成: 总数={result.total_files}, "
                f"成功={result.success_count}, 失败={result.fail_count}"
            )

            return result

        except Exception as e:
            logger.error(f"索引目录失败: {e}")
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            return result


# 全局单例
vector_index_service = VectorIndexService()
