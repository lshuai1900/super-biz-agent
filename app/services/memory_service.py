"""PostgreSQL 长程记忆与上下文管理服务

支持：
- conversations / messages / session_summaries / eval_runs / eval_items 表
- 自动写入用户和助手消息
- 按 session_id 读取历史
- 超过阈值自动 LLM 摘要早期消息
- 评估结果持久化
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from loguru import logger
from sqlalchemy import select, delete, func as sa_func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import config
from app.db.database import is_connected, get_session_sync, init_db
from app.db.models import (
    ConversationModel, MessageModel, SessionSummaryModel,
    EvalRunModel, EvalItemModel,
)

# ── Token estimation ──────────────────────────────────────────
try:
    import tiktoken
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tokenizer = None
    logger.debug("tiktoken not available, using approximate char-based token count")


def count_tokens(text: str) -> int:
    """估算 token 数量"""
    if _tokenizer:
        return len(_tokenizer.encode(text))
    return len(text) // 2  # rough estimate


class MemoryService:
    """长程记忆服务"""

    def __init__(self):
        self._connected = False

    async def initialize(self):
        """初始化数据库连接并创建表"""
        if self._connected:
            return
        await init_db()
        self._connected = is_connected()
        if self._connected:
            logger.info("MemoryService 初始化完成")
        else:
            logger.warning("MemoryService 初始化失败（降级模式：无持久化）")

    def _check(self) -> bool:
        return self._connected

    async def _get_session(self) -> AsyncSession:
        sess = get_session_sync()
        if sess is None:
            raise ConnectionError("PostgreSQL not connected")
        return sess

    # ── Conversation ──────────────────────────────────────────

    async def create_or_get_conversation(
        self, session_id: str, mode: str = "auto", title: str = ""
    ) -> Optional[Dict[str, Any]]:
        """获取或创建会话"""
        if not self._check():
            return None
        try:
            sess = await self._get_session()
            async with sess:
                existing = await sess.get(ConversationModel, session_id)
                if existing:
                    existing.updated_at = datetime.now(timezone.utc)
                    if title:
                        existing.title = title
                    await sess.commit()
                    return {
                        "session_id": existing.session_id,
                        "mode": existing.mode,
                        "title": existing.title,
                        "created_at": existing.created_at.isoformat() if existing.created_at else None,
                        "updated_at": existing.updated_at.isoformat() if existing.updated_at else None,
                    }
                else:
                    conv = ConversationModel(
                        session_id=session_id,
                        mode=mode,
                        title=title or f"会话 {session_id[:12]}",
                    )
                    sess.add(conv)
                    await sess.commit()
                    return {
                        "session_id": conv.session_id,
                        "mode": conv.mode,
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat() if conv.created_at else None,
                        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                    }
        except Exception as e:
            logger.error(f"create_or_get_conversation error: {e}")
            return None

    async def list_conversations(self) -> List[Dict[str, Any]]:
        """列出所有会话"""
        if not self._check():
            return []
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(ConversationModel).order_by(desc(ConversationModel.updated_at))
                )
                rows = result.scalars().all()
                return [
                    {
                        "session_id": r.session_id,
                        "mode": r.mode,
                        "title": r.title,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"list_conversations error: {e}")
            return []

    async def get_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self._check():
            return None
        try:
            sess = await self._get_session()
            async with sess:
                conv = await sess.get(ConversationModel, session_id)
                if not conv:
                    return None
                return {
                    "session_id": conv.session_id,
                    "mode": conv.mode,
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat() if conv.created_at else None,
                    "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
                }
        except Exception as e:
            logger.error(f"get_conversation error: {e}")
            return None

    async def clear_conversation(self, session_id: str):
        """清空指定会话的消息、摘要和会话记录"""
        if not self._check():
            return
        try:
            sess = await self._get_session()
            async with sess:
                await sess.execute(delete(MessageModel).where(MessageModel.session_id == session_id))
                await sess.execute(delete(SessionSummaryModel).where(SessionSummaryModel.session_id == session_id))
                conv = await sess.get(ConversationModel, session_id)
                if conv:
                    await sess.delete(conv)
                await sess.commit()
            logger.info(f"已清空会话: {session_id}")
        except Exception as e:
            logger.error(f"clear_conversation error: {e}")

    # ── Messages ──────────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata_: Optional[Dict[str, Any]] = None,
    ) -> str:
        """添加消息到会话历史"""
        if not self._check():
            return ""
        msg_id = str(uuid4())
        try:
            token_cnt = count_tokens(content)
            sess = await self._get_session()
            async with sess:
                sess.add(MessageModel(
                    id=msg_id,
                    session_id=session_id,
                    role=role,
                    content=content,
                    token_count=token_cnt,
                    metadata_=metadata_ or {},
                ))
                await sess.commit()

            # 同时更新会话（如果不存在则创建）
            await self.create_or_get_conversation(session_id)
            return msg_id
        except Exception as e:
            logger.error(f"add_message error: {e}")
            return ""

    async def add_user_message(self, session_id: str, content: str) -> str:
        return await self.add_message(session_id, "user", content)

    async def add_assistant_message(self, session_id: str, content: str) -> str:
        return await self.add_message(session_id, "assistant", content)

    async def get_recent_messages(
        self, session_id: str, limit: int = 12
    ) -> List[Dict[str, Any]]:
        """获取最近的 N 条消息"""
        return await self.get_history(session_id, limit=limit)

    async def get_history(
        self, session_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取会话历史消息（按时间升序）"""
        if not self._check():
            return []
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(MessageModel)
                    .where(MessageModel.session_id == session_id)
                    .order_by(MessageModel.created_at.asc())
                    .limit(limit)
                )
                rows = result.scalars().all()
                return [
                    {
                        "role": r.role,
                        "content": r.content,
                        "token_count": r.token_count,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"get_history error: {e}")
            return []

    async def count_messages(self, session_id: str) -> int:
        if not self._check():
            return 0
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(sa_func.count())
                    .select_from(MessageModel)
                    .where(MessageModel.session_id == session_id)
                )
                return result.scalar() or 0
        except Exception as e:
            logger.error(f"count_messages error: {e}")
            return 0

    async def total_tokens(self, session_id: str) -> int:
        if not self._check():
            return 0
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(sa_func.coalesce(sa_func.sum(MessageModel.token_count), 0))
                    .where(MessageModel.session_id == session_id)
                )
                return result.scalar() or 0
        except Exception as e:
            logger.error(f"total_tokens error: {e}")
            return 0

    # ── Summary ───────────────────────────────────────────────

    async def get_latest_summary(self, session_id: str) -> Optional[str]:
        """获取最新的会话摘要"""
        if not self._check():
            return None
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(SessionSummaryModel)
                    .where(SessionSummaryModel.session_id == session_id)
                    .order_by(desc(SessionSummaryModel.created_at))
                    .limit(1)
                )
                row = result.scalar_one_or_none()
                return row.summary if row else None
        except Exception as e:
            logger.error(f"get_latest_summary error: {e}")
            return None

    async def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话摘要详情"""
        summary = await self.get_latest_summary(session_id)
        summaries = await self.get_summary_list(session_id)
        if summary is None and not summaries:
            return None
        return {
            "session_id": session_id,
            "latest_summary": summary,
            "summaries": summaries,
        }

    async def save_summary(
        self,
        session_id: str,
        summary: str,
        summary_vector_id: Optional[str] = None,
    ):
        if not self._check():
            return
        try:
            token_cnt = count_tokens(summary)
            sess = await self._get_session()
            async with sess:
                sess.add(SessionSummaryModel(
                    session_id=session_id,
                    summary=summary,
                    summary_vector_id=summary_vector_id,
                    token_count=token_cnt,
                ))
                await sess.commit()
        except Exception as e:
            logger.error(f"save_summary error: {e}")

    async def get_summary_list(self, session_id: str) -> List[Dict[str, Any]]:
        if not self._check():
            return []
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(SessionSummaryModel)
                    .where(SessionSummaryModel.session_id == session_id)
                    .order_by(desc(SessionSummaryModel.created_at))
                )
                rows = result.scalars().all()
                return [
                    {
                        "summary": r.summary,
                        "token_count": r.token_count,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"get_summary_list error: {e}")
            return []

    # ── Auto-summarization ────────────────────────────────────

    MAX_ROUNDS = 12  # 超过 12 轮对话触发压缩
    MAX_TOKENS = 6000  # 超过 6000 token 触发压缩
    KEEP_ROUNDS = 6  # 保留最近 6 轮

    async def check_and_summarize(
        self, session_id: str
    ) -> Optional[str]:
        """检查是否需要摘要压缩，返回最新的摘要文本（如有）

        触发条件：
        - 消息数量超过 24 条（12 轮）
        - token_count 总数超过 6000
        """
        if not self._check():
            return None

        try:
            msg_count = await self.count_messages(session_id)
            total_tok = await self.total_tokens(session_id)

            rounds = msg_count // 2
            if rounds <= self.MAX_ROUNDS and total_tok <= self.MAX_TOKENS:
                return await self.get_latest_summary(session_id)

            logger.info(
                f"触发摘要压缩: session={session_id}, "
                f"消息数={msg_count}, token数={total_tok}"
            )

            # 读取所有消息
            history = await self.get_history(session_id, limit=200)
            if not history or len(history) < 4:
                return await self.get_latest_summary(session_id)

            # 保留最近 KEEP_ROUNDS 轮
            keep_count = self.KEEP_ROUNDS * 2
            to_summarize = history[:-keep_count] if len(history) > keep_count else history[:-4]
            to_keep = history[-keep_count:] if len(history) > keep_count else history[-4:]

            if not to_summarize or len(to_summarize) < 2:
                return await self.get_latest_summary(session_id)

            # 生成摘要
            summary_text = await self._generate_summary(to_summarize)
            if not summary_text:
                logger.warning(f"摘要生成失败，使用现有摘要")
                return await self.get_latest_summary(session_id)

            # 保存新摘要
            await self.save_summary(session_id, summary_text)

            logger.info(f"摘要压缩完成: session={session_id}, "
                        f"摘要了 {len(to_summarize)} 条历史，保留 {len(to_keep)} 条最近消息")
            return summary_text

        except Exception as e:
            logger.error(f"check_and_summarize error: {e}")
            return None

    async def build_context(
        self, session_id: str, question: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """构造带摘要和历史的上下文

        Returns:
            (summary_text, recent_messages)
        """
        summary = await self.get_latest_summary(session_id)
        history = await self.get_history(session_id, limit=20)

        # 如果问句不在历史末尾，自动写入
        if not history or history[-1].get("content") != question:
            await self.add_user_message(session_id, question)

        return summary or "", history

    async def _generate_summary(self, messages: List[Dict[str, Any]]) -> str:
        """调用 LLM 生成消息摘要"""
        try:
            from app.core.llm_factory import llm_factory

            text = "\n".join(
                f"{m['role']}: {m['content'][:500]}"
                for m in messages
            )
            prompt = (
                "请对以下对话进行简洁的中文摘要，保留关键信息和上下文：\n\n"
                f"{text}\n\n摘要："
            )
            llm = llm_factory.create_chat_model(streaming=False, temperature=0.3)
            result = llm.invoke(prompt)
            content = result.content.strip()
            if content:
                return content
        except Exception as e:
            logger.warning(f"LLM 摘要生成失败（不影响主流程）: {e}")

        # Fallback: 拼接前几条消息
        parts = [m["content"][:200] for m in messages[:6]]
        return "对话摘要：" + " | ".join(parts)

    # ── Eval Results ───────────────────────────────────────────

    async def save_eval_run(
        self,
        run_id: str,
        dataset_size: int,
        metrics: Dict[str, Any],
        report_path: Optional[str] = None,
    ) -> str:
        """保存评估运行记录"""
        if not self._check():
            return ""
        eval_id = str(uuid4())
        try:
            sess = await self._get_session()
            async with sess:
                sess.add(EvalRunModel(
                    id=eval_id,
                    run_id=run_id,
                    dataset_size=dataset_size,
                    metrics=metrics,
                    report_path=report_path,
                ))
                await sess.commit()
            return eval_id
        except Exception as e:
            logger.error(f"save_eval_run error: {e}")
            return ""

    async def save_eval_items(
        self,
        run_id: str,
        items: List[Dict[str, Any]],
    ):
        """保存评估条目"""
        if not self._check():
            return
        try:
            sess = await self._get_session()
            async with sess:
                for item in items:
                    sess.add(EvalItemModel(
                        id=str(uuid4()),
                        run_id=run_id,
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        ground_truth=item.get("ground_truth", ""),
                        contexts=item.get("contexts", []),
                        metrics=item.get("metrics", {}),
                        source=item.get("source", ""),
                    ))
                await sess.commit()
        except Exception as e:
            logger.error(f"save_eval_items error: {e}")

    async def get_eval_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取评估运行记录列表"""
        if not self._check():
            return []
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(EvalRunModel)
                    .order_by(desc(EvalRunModel.created_at))
                    .limit(limit)
                )
                rows = result.scalars().all()
                return [
                    {
                        "id": str(r.id),
                        "run_id": r.run_id,
                        "dataset_size": r.dataset_size,
                        "metrics": r.metrics,
                        "report_path": r.report_path,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"get_eval_runs error: {e}")
            return []

    async def get_eval_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """获取指定评估运行记录（含条目）"""
        if not self._check():
            return None
        try:
            sess = await self._get_session()
            async with sess:
                result = await sess.execute(
                    select(EvalRunModel).where(EvalRunModel.run_id == run_id)
                )
                run = result.scalar_one_or_none()
                if not run:
                    return None

                items_result = await sess.execute(
                    select(EvalItemModel)
                    .where(EvalItemModel.run_id == run_id)
                    .order_by(EvalItemModel.created_at.asc())
                )
                items = items_result.scalars().all()

                return {
                    "id": str(run.id),
                    "run_id": run.run_id,
                    "dataset_size": run.dataset_size,
                    "metrics": run.metrics,
                    "report_path": run.report_path,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "details": [
                        {
                            "question": item.question,
                            "answer": item.answer,
                            "ground_truth": item.ground_truth,
                            "contexts": item.contexts,
                            "metrics": item.metrics,
                            "source": item.source,
                        }
                        for item in items
                    ],
                }
        except Exception as e:
            logger.error(f"get_eval_run error: {e}")
            return None

    async def get_latest_eval_run(self) -> Optional[Dict[str, Any]]:
        """获取最新评估运行"""
        runs = await self.get_eval_runs(limit=1)
        if runs:
            return await self.get_eval_run(runs[0]["run_id"])
        return None


# 全局单例
memory_service = MemoryService()
