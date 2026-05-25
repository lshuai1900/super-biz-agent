"""QA 测试集自动生成器

从知识库文档中抽取片段，调用 LLM 自动生成 QA 对。
支持超时控制、缓存、快速演示模式和部分结果。
"""

import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from app.core.llm_factory import llm_factory

CACHE_DIR = Path("data/evaluation")
CACHE_FILE = CACHE_DIR / "test_dataset.json"


class EvalDatasetGenerator:
    """评估数据集生成器"""

    def __init__(self):
        self.data_dir = Path("./reports")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async def generate_dataset(
        self,
        source_dir: str = "aiops-docs",
        count: int = 3,
        max_docs: int = 3,
        max_chunks: int = 20,
        timeout_seconds: int = 120,
        use_cache: bool = True,
        force: bool = False,
        quick: bool = False,
        question_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """从文档目录自动生成 QA 测试集

        Args:
            source_dir: 文档目录路径
            count: 生成的 QA 对数量（默认 3，最大 20）
            max_docs: 最多读取文档数
            max_chunks: 最多使用 chunk 数
            timeout_seconds: 超时时间
            use_cache: 是否使用缓存
            force: 强制重新生成
            quick: 快速演示模式

        Returns:
            dict: { success, partial, total, items, message, from_cache }
        """
        # 快速模式：限制参数
        if quick:
            count = min(count, 3)
            max_docs = min(max_docs, 2)
            max_chunks = min(max_chunks, 10)
            timeout_seconds = min(timeout_seconds, 60)
            logger.info("快速演示模式：参数已限制为小规模")

        logger.info(
            f"开始生成测试集：count={count}, max_docs={max_docs}, "
            f"max_chunks={max_chunks}, timeout={timeout_seconds}s, "
            f"cache={use_cache}, force={force}, quick={quick}"
        )

        # 检查缓存
        if use_cache and not force and CACHE_FILE.exists():
            try:
                cached = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                cached_count = cached.get("count", 0) if isinstance(cached, dict) else len(cached)
                logger.info(f"测试集缓存命中：{cached_count} 条")
                return {
                    "success": True,
                    "partial": False,
                    "total": len(cached.get("items", cached)) if isinstance(cached, dict) else len(cached),
                    "items": cached.get("items", cached) if isinstance(cached, dict) else cached,
                    "message": f"从缓存加载 {cached_count} 条测试数据，使用 force=true 强制重新生成",
                    "from_cache": True,
                }
            except Exception as e:
                logger.warning(f"缓存读取失败，将重新生成: {e}")

        # 1. 读取文档
        docs_content = self._load_documents(source_dir, max_docs)
        if not docs_content:
            return {
                "success": False,
                "partial": False,
                "total": 0,
                "items": [],
                "message": f"未在 {source_dir} 找到文档。请先上传文档或检查 aiops-docs/ 目录",
                "from_cache": False,
            }

        logger.info(f"已加载文档：{len(docs_content)} 个")

        # 2. 分割文档片段
        chunks = self._split_into_chunks(docs_content, max_chunks)
        logger.info(f"已切分 chunk：{len(chunks)} 个")

        if len(chunks) > count * 2:
            chunks = random.sample(chunks, count * 2)

        # 3. 对每个片段生成 QA 对（带超时）
        qa_pairs = []
        error_msg = None
        partial = False

        try:
            qa_pairs = await asyncio.wait_for(
                self._generate_qa_batch(chunks, count),
                timeout=timeout_seconds,
            )
        except TimeoutError:
            error_msg = (
                f"测试集生成超时 (>{timeout_seconds}s)，"
                f"已生成 {len(qa_pairs)}/{count} 条。"
                f"建议减少 count、max_docs 或 max_chunks 后重试，或使用 quick=true 快速模式"
            )
            partial = len(qa_pairs) > 0
            logger.warning(error_msg)
        except Exception as e:
            error_msg = f"测试集生成失败: {e}"
            partial = len(qa_pairs) > 0
            logger.error(error_msg)

        # 4. 缓存到文件
        if qa_pairs:
            self._save_cache(qa_pairs)
            self._save_dataset(qa_pairs)

        total = len(qa_pairs)
        logger.info(f"测试集生成完成：{total} 条" + (" (部分)" if partial else ""))

        return {
            "success": total > 0,
            "partial": partial,
            "total": total,
            "items": qa_pairs,
            "message": error_msg or f"生成完成：{total} 条",
            "from_cache": False,
        }

    async def _generate_qa_batch(self, chunks: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        """批量生成 QA 对（顺序执行，但可被 asyncio.wait_for 取消）"""
        qa_pairs = []
        actual_count = min(count, len(chunks))

        for i in range(actual_count):
            logger.info(f"正在生成第 {i + 1}/{actual_count} 个测试问题")
            try:
                qa = await self._generate_qa_pair(chunks[i])
                if qa:
                    qa_pairs.append(qa)
            except Exception as e:
                logger.error(f"生成 QA 第 {i + 1} 个失败: {e}")
                continue

        return qa_pairs

    def _load_documents(self, source_dir: str, max_docs: int = 3) -> list[dict[str, str]]:
        """加载目录中的文档（限制数量）"""
        docs = []
        dir_path = Path(source_dir)
        if not dir_path.exists():
            return docs

        files = list(dir_path.glob("*.md")) + list(dir_path.glob("*.txt"))
        # 随机采样避免每次都读同样的文档
        if len(files) > max_docs:
            files = random.sample(files, max_docs)

        for f in files:
            try:
                docs.append({"file": f.name, "content": f.read_text(encoding="utf-8")})
            except Exception as e:
                logger.warning(f"读取文档失败: {f.name}: {e}")

        return docs

    def _split_into_chunks(
        self, docs: list[dict[str, str]], max_chunks: int = 20, chunk_size: int = 1000
    ) -> list[dict[str, Any]]:
        """分割文档为片段（限制总 chunk 数）"""
        chunks = []
        for doc in docs:
            content = doc["content"]
            for i in range(0, len(content), chunk_size):
                chunk_text = content[i:i + chunk_size]
                if len(chunk_text.strip()) > 50:
                    chunks.append({
                        "source": doc["file"],
                        "content": chunk_text,
                    })
                if len(chunks) >= max_chunks:
                    return chunks
        return chunks

    async def _generate_qa_pair(self, chunk: dict[str, Any]) -> dict[str, Any] | None:
        """调用 LLM 从文档片段生成 QA 对"""
        prompt = f"""你是一个运维知识库 QA 生成器。请根据以下文档片段，生成一个高质量的问答对。

文档片段（来自 {chunk['source']}）：
{chunk['content'][:1500]}

请生成：
1. question: 一个基于此文档片段的真实问题
2. ground_truth: 基于文档片段的正确答案（详细、准确）

以 JSON 格式返回，不要包含其他内容：
{{"question": "...", "ground_truth": "..."}}
"""
        try:
            llm = llm_factory.create_chat_model(streaming=False, temperature=0.3)
            result = llm.invoke(prompt)
            text = result.content.strip()

            # 提取 JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]

            text = text.strip()
            if text.startswith("{"):
                data = json.loads(text)
                return {
                    "question": data.get("question", ""),
                    "ground_truth": data.get("ground_truth", ""),
                    "contexts": [chunk["content"]],
                    "source": chunk["source"],
                }
        except Exception as e:
            logger.error(f"QA 生成解析失败: {e}")

        return None

    def _save_cache(self, qa_pairs: list[dict[str, Any]]):
        """保存到缓存文件"""
        try:
            cache_data = {
                "count": len(qa_pairs),
                "generated_at": datetime.now().isoformat(),
                "items": qa_pairs,
            }
            CACHE_FILE.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"缓存已保存: {CACHE_FILE}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def _save_dataset(self, qa_pairs: list[dict[str, Any]]):
        """保存数据集到文件（带时间戳）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.data_dir / f"eval_dataset_{timestamp}.json"
        path.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"数据集已保存: {path}")

    def clear_cache(self) -> bool:
        """清除缓存"""
        try:
            if CACHE_FILE.exists():
                CACHE_FILE.unlink()
                logger.info("测试集缓存已清除")
                return True
        except Exception as e:
            logger.warning(f"清除缓存失败: {e}")
        return False


# 全局单例
dataset_generator = EvalDatasetGenerator()
