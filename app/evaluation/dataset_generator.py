"""QA 测试集自动生成器

从知识库文档中抽取片段，调用 LLM 自动生成 QA 对。
"""

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from loguru import logger

from app.core.llm_factory import llm_factory


class EvalDatasetGenerator:
    """评估数据集生成器"""

    def __init__(self):
        self.data_dir = "./reports"
        os.makedirs(self.data_dir, exist_ok=True)

    async def generate_dataset(
        self,
        source_dir: str = "aiops-docs",
        count: int = 10,
    ) -> List[Dict[str, Any]]:
        """从文档目录自动生成 QA 测试集

        Args:
            source_dir: 文档目录路径
            count: 要生成的 QA 对数量

        Returns:
            List[Dict]: QA 列表，每项包含 question, ground_truth, contexts, source
        """
        logger.info(f"开始生成 QA 测试集: source={source_dir}, count={count}")

        # 1. 读取文档
        docs_content = self._load_documents(source_dir)
        if not docs_content:
            logger.warning(f"未在 {source_dir} 找到文档")
            return []

        # 2. 分割文档片段
        chunks = self._split_into_chunks(docs_content)
        if len(chunks) > count * 2:
            chunks = random.sample(chunks, count * 2)

        # 3. 对每个片段生成 QA 对
        qa_pairs = []
        for i, chunk in enumerate(chunks[:count]):
            try:
                qa = await self._generate_qa_pair(chunk)
                if qa:
                    qa_pairs.append(qa)
                logger.info(f"生成 QA [{i+1}/{min(count, len(chunks))}]: {qa.get('question', '')[:50]}...")
            except Exception as e:
                logger.error(f"生成 QA 失败: {e}")

        # 4. 保存数据集
        if qa_pairs:
            self._save_dataset(qa_pairs)

        logger.info(f"QA 测试集生成完成: {len(qa_pairs)} 条")
        return qa_pairs

    def _load_documents(self, source_dir: str) -> List[Dict[str, str]]:
        """加载目录中的所有文档"""
        docs = []
        dir_path = Path(source_dir)
        if not dir_path.exists():
            return docs

        for f in dir_path.glob("*.md"):
            try:
                docs.append({"file": f.name, "content": f.read_text(encoding="utf-8")})
            except Exception as e:
                logger.warning(f"读取文档失败: {f.name}: {e}")

        for f in dir_path.glob("*.txt"):
            try:
                docs.append({"file": f.name, "content": f.read_text(encoding="utf-8")})
            except Exception as e:
                logger.warning(f"读取文档失败: {f.name}: {e}")

        return docs

    def _split_into_chunks(self, docs: List[Dict[str, str]], chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """分割文档为片段"""
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
        return chunks

    async def _generate_qa_pair(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """调用 LLM 从文档片段生成 QA 对"""
        prompt = f"""你是一个运维知识库 QA 生成器。请根据以下文档片段，生成一个高质量的问答对。

文档片段（来自 {chunk['source']}）：
{chunk['content'][:1500]}

请生成：
1. question: 一个基于此文档片段的真实问题
2. ground_truth: 基于文档片段的正确答案（详细、准确）
3. contexts: 包含此文档片段

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

    def _save_dataset(self, qa_pairs: List[Dict[str, Any]]):
        """保存数据集到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(self.data_dir) / f"eval_dataset_{timestamp}.json"
        path.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"数据集已保存: {path}")


# 全局单例
dataset_generator = EvalDatasetGenerator()
