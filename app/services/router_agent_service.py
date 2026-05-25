"""统一双模式 LangGraph 路由 Agent

构建一个 LangGraph 状态机，支持：
- route 节点：判断请求属于 general_chat / rag_qa / aiops_diagnosis
- rag_agent 节点：走现有 RAG ReAct 工具调用
- aiops_agent 节点：走现有 Plan-Execute-Replan
- final 节点：统一整理响应
"""

import json
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from loguru import logger
from typing_extensions import TypedDict

from app.config import config
from app.core.checkpointer import create_checkpointer
from app.services.memory_service import memory_service
from app.tools import DEFAULT_LOCAL_AGENT_TOOLS


class RouterState(TypedDict):
    """路由状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    session_id: str
    question: str
    mode: str  # auto | chat | rag | aiops
    route: str  # general_chat | rag_qa | aiops_diagnosis
    route_reason: str
    rag_answer: str
    aiops_answer: str
    aiops_events: List[Dict[str, Any]]
    final_answer: str
    error: str


def _build_classify_prompt() -> str:
    return """你是一个问题分类器，需要判断用户的问题是以下哪种类型：

1. general_chat - 一般聊天、问候、闲聊、不需要查询知识库或诊断
2. rag_qa - 需要查询知识库/文档来回答问题，例如询问系统功能、配置说明、运维文档等
3. aiops_diagnosis - 需要故障诊断、排查告警、分析系统问题

请只返回一个单词：general_chat / rag_qa / aiops_diagnosis
"""


async def route_node(state: RouterState) -> Dict[str, Any]:
    """路由节点：判断请求类型"""
    question = state.get("question", "")
    mode = state.get("mode", "auto")

    logger.info(f"[路由] 开始路由判断: mode={mode}, question='{question[:50]}...'")

    # 如果用户手动指定了模式
    if mode == "chat":
        return {"route": "general_chat", "route_reason": "用户手动指定 chat 模式"}
    if mode == "rag":
        return {"route": "rag_qa", "route_reason": "用户手动指定 rag 模式"}
    if mode == "aiops":
        return {"route": "aiops_diagnosis", "route_reason": "用户手动指定 aiops 模式"}

    # auto 模式：用 LLM 判断
    try:
        from app.core.llm_factory import llm_factory
        llm = llm_factory.create_chat_model(streaming=False, temperature=0.1)
        result = llm.invoke([
            SystemMessage(content=_build_classify_prompt()),
            HumanMessage(content=question)
        ])
        route = result.content.strip().lower()
        if route not in ("general_chat", "rag_qa", "aiops_diagnosis"):
            route = "rag_qa"  # 默认走 RAG
        logger.info(f"[路由] 判断结果: {route}")
        return {"route": route, "route_reason": f"LLM 分类: {route}"}
    except Exception as e:
        logger.error(f"[路由] LLM 分类失败: {e}，默认 rag_qa")
        return {"route": "rag_qa", "route_reason": f"分类异常，默认: {e}"}


async def rag_agent_node(state: RouterState) -> Dict[str, Any]:
    """RAG Agent 节点"""
    from app.services.rag_agent_service import rag_agent_service

    question = state.get("question", "")
    session_id = state.get("session_id", "default")

    logger.info(f"[RAG Agent] 开始处理: '{question[:50]}...'")
    try:
        answer = await rag_agent_service.query(question, session_id)
        return {"rag_answer": answer, "final_answer": answer}
    except Exception as e:
        err_msg = f"RAG Agent 处理失败: {e}"
        logger.error(err_msg)
        return {"rag_answer": "", "final_answer": f"抱歉，知识库查询遇到问题: {e}"}


async def aiops_agent_node(state: RouterState) -> Dict[str, Any]:
    """AIOps Agent 节点"""
    from app.services.aiops_service import aiops_service

    session_id = state.get("session_id", "default")

    logger.info(f"[AIOps Agent] 开始诊断: session={session_id}")
    try:
        events = []
        async for event in aiops_service.diagnose(session_id=session_id):
            events.append(event)

        # 从事件中提取最终报告
        final_report = ""
        for ev in events:
            if ev.get("type") == "complete":
                diagnosis = ev.get("diagnosis", {})
                final_report = diagnosis.get("report", "")
            elif ev.get("type") == "report":
                final_report = ev.get("report", "")

        return {
            "aiops_answer": final_report,
            "aiops_events": events,
            "final_answer": final_report or "诊断完成，但未生成报告",
        }
    except Exception as e:
        err_msg = f"AIOps Agent 处理失败: {e}"
        logger.error(err_msg)
        return {
            "aiops_answer": "",
            "aiops_events": [],
            "final_answer": f"抱歉，智能运维诊断遇到问题: {e}",
        }


async def general_chat_node(state: RouterState) -> Dict[str, Any]:
    """普通对话节点（不调用工具）"""
    from app.core.llm_factory import llm_factory

    question = state.get("question", "")
    session_id = state.get("session_id", "default")

    logger.info(f"[General Chat] 开始处理: '{question[:50]}...'")
    try:
        llm = llm_factory.create_chat_model(streaming=False, temperature=0.7)
        result = llm.invoke([
            SystemMessage(content="你是一个智能 OnCall 助手。请用友好、专业的态度回答用户的问题。"),
            HumanMessage(content=question)
        ])
        answer = result.content
        await memory_service.add_user_message(session_id, question)
        await memory_service.add_assistant_message(session_id, answer)
        return {"final_answer": answer}
    except Exception as e:
        err_msg = f"General Chat 处理失败: {e}"
        logger.error(err_msg)
        return {"final_answer": f"抱歉，我遇到了一些问题: {e}"}


async def final_node(state: RouterState) -> Dict[str, Any]:
    """最终节点：确保有 final_answer"""
    answer = state.get("final_answer", "")
    if not answer:
        route = state.get("route", "unknown")
        answer = f"处理完成（路由: {route}），但未生成回答。"
    return {}


def should_route(state: RouterState) -> str:
    """路由条件边"""
    route = state.get("route", "rag_qa")
    logger.info(f"[条件边] 路由到: {route}")
    if route == "general_chat":
        return "general_chat"
    elif route == "aiops_diagnosis":
        return "aiops_agent"
    else:
        return "rag_agent"


class RouterAgentService:
    """统一路由 Agent 服务"""

    def __init__(self):
        self.checkpointer = create_checkpointer()
        self.graph = self._build_graph()
        logger.info("RouterAgentService 初始化完成")

    def _build_graph(self):
        """构建路由状态图"""
        workflow = StateGraph(RouterState)

        workflow.add_node("route", route_node)
        workflow.add_node("general_chat", general_chat_node)
        workflow.add_node("rag_agent", rag_agent_node)
        workflow.add_node("aiops_agent", aiops_agent_node)
        workflow.add_node("final", final_node)

        workflow.set_entry_point("route")

        workflow.add_conditional_edges(
            "route",
            should_route,
            {
                "general_chat": "general_chat",
                "rag_agent": "rag_agent",
                "aiops_agent": "aiops_agent",
            }
        )

        workflow.add_edge("general_chat", "final")
        workflow.add_edge("rag_agent", "final")
        workflow.add_edge("aiops_agent", "final")
        workflow.add_edge("final", END)

        return workflow.compile(checkpointer=self.checkpointer)

    async def query(
        self,
        question: str,
        session_id: str = "default",
        mode: str = "auto",
    ) -> Dict[str, Any]:
        """非流式处理"""
        try:
            # 创建会话
            await memory_service.create_or_get_conversation(session_id, mode)

            initial_state: RouterState = {
                "messages": [],
                "session_id": session_id,
                "question": question,
                "mode": mode,
                "route": "",
                "route_reason": "",
                "rag_answer": "",
                "aiops_answer": "",
                "aiops_events": [],
                "final_answer": "",
                "error": "",
            }

            config_dict = {"configurable": {"thread_id": f"router_{session_id}"}}

            result = await self.graph.ainvoke(
                input=initial_state,
                config=config_dict,
            )

            return {
                "session_id": session_id,
                "mode": mode,
                "route": result.get("route", ""),
                "route_reason": result.get("route_reason", ""),
                "answer": result.get("final_answer", ""),
            }

        except Exception as e:
            logger.error(f"RouterAgent query error: {e}")
            return {
                "session_id": session_id,
                "mode": mode,
                "route": "",
                "route_reason": "",
                "answer": f"处理失败: {e}",
            }

    async def query_stream(
        self,
        question: str,
        session_id: str = "default",
        mode: str = "auto",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """流式处理"""
        try:
            await memory_service.create_or_get_conversation(session_id, mode)

            # 先做路由判断
            route_result = await route_node({
                "messages": [],
                "session_id": session_id,
                "question": question,
                "mode": mode,
                "route": "",
                "route_reason": "",
                "rag_answer": "",
                "aiops_answer": "",
                "aiops_events": [],
                "final_answer": "",
                "error": "",
            })

            route = route_result.get("route", "rag_qa")
            reason = route_result.get("route_reason", "")

            yield {"type": "route", "data": {"route": route, "reason": reason}}

            if route == "general_chat":
                yield {"type": "content", "data": "【普通对话模式】\n\n"}
                async for chunk in self._stream_general_chat(question, session_id):
                    yield chunk

            elif route == "rag_qa":
                yield {"type": "content", "data": "【知识库问答模式】\n\n"}
                async for chunk in self._stream_rag(question, session_id):
                    yield chunk

            elif route == "aiops_diagnosis":
                yield {"type": "content", "data": "【智能运维诊断模式】\n\n"}
                async for chunk in self._stream_aiops(session_id):
                    yield chunk

            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"RouterAgent stream error: {e}")
            yield {"type": "error", "data": str(e)}

    async def _stream_general_chat(self, question: str, session_id: str):
        """流式普通对话"""
        from app.core.llm_factory import llm_factory

        await memory_service.add_user_message(session_id, question)
        llm = llm_factory.create_chat_model(streaming=True, temperature=0.7)

        full = []
        async for chunk in llm.astream([
            SystemMessage(content="你是一个智能 OnCall 助手。请用友好、专业的态度回答用户的问题。"),
            HumanMessage(content=question)
        ]):
            if chunk.content:
                full.append(chunk.content)
                yield {"type": "content", "data": chunk.content}

        await memory_service.add_assistant_message(session_id, "".join(full))

    async def _stream_rag(self, question: str, session_id: str):
        """流式 RAG（无重复写消息，由下游 rag_agent_service.query_stream 负责写入）"""
        from app.services.rag_agent_service import rag_agent_service

        async for chunk in rag_agent_service.query_stream(question, session_id):
            yield chunk

    async def _stream_aiops(self, session_id: str):
        """流式 AIOps"""
        from app.services.aiops_service import aiops_service

        async for event in aiops_service.diagnose(session_id=session_id):
            ev_type = event.get("type", "")
            if ev_type in ("plan", "step_complete", "report", "complete", "error", "status"):
                yield {"type": "event", "data": event}
            elif ev_type == "complete":
                yield {"type": "event", "data": event}
                break


# 全局单例
router_agent_service = RouterAgentService()
