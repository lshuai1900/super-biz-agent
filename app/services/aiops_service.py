"""
通用 Plan-Execute-Replan 服务
基于 LangGraph 官方教程实现
"""

import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from loguru import logger

from app.agent.aiops import PlanExecuteState, planner, executor, replanner
from app.core.checkpointer import create_checkpointer
from app.services.memory_service import memory_service


# 节点名称常量
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"


class AIOpsService:
    """通用 Plan-Execute-Replan 服务"""

    def __init__(self):
        """初始化服务"""
        self.checkpointer = create_checkpointer()
        self.graph = self._build_graph()
        logger.info("Plan-Execute-Replan Service 初始化完成")

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流"""
        logger.info("构建工作流图...")

        workflow = StateGraph(PlanExecuteState)

        workflow.add_node(NODE_PLANNER, planner)
        workflow.add_node(NODE_EXECUTOR, executor)
        workflow.add_node(NODE_REPLANNER, replanner)

        workflow.set_entry_point(NODE_PLANNER)

        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)

        def should_continue(state: PlanExecuteState) -> str:
            if state.get("response"):
                logger.info("已生成最终响应，结束流程")
                return END
            plan = state.get("plan", [])
            if plan:
                logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
                return NODE_EXECUTOR
            logger.info("计划执行完毕，生成最终响应")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            should_continue,
            {
                NODE_EXECUTOR: NODE_EXECUTOR,
                END: END,
            }
        )

        compiled_graph = workflow.compile(checkpointer=self.checkpointer)
        logger.info("工作流图构建完成")
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行 Plan-Execute-Replan 流程（流式）

        Yields:
            Dict[str, Any]: tool_call_start, tool_call_result, plan, step_complete, report, complete, error
        """
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        try:
            initial_state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": ""
            }

            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            # 心跳任务
            async def heartbeat():
                while True:
                    await asyncio.sleep(15)
                    yield {"type": "heartbeat", "message": "keep-alive"}

            async for event in self.graph.astream(
                input=initial_state,
                config=config_dict,
                stream_mode="updates"
            ):
                for node_name, node_output in event.items():
                    if node_name == NODE_PLANNER:
                        yield self._format_planner_event(node_output)
                    elif node_name == NODE_EXECUTOR:
                        yield self._format_executor_event(node_output)
                    elif node_name == NODE_REPLANNER:
                        yield self._format_replanner_event(node_output)

            # 获取最终状态
            final_state = self.graph.get_state(config_dict)
            final_response = ""
            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")

            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response
            }

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}"
            }

    async def diagnose(
        self,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        AIOps 诊断接口（流式）

        Yields:
            Dict: 诊断过程的流式事件（兼容旧接口 + 新事件类型）
        """
        from textwrap import dedent
        aiops_task = dedent("""诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 📋 活跃告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 🔍 告警根因分析

                ### 告警详情
                ### 症状描述
                ### 日志证据
                ### 根因结论

                ---

                ## 🛠️ 处理方案

                ### 已执行的排查步骤
                ### 处理建议

                ---

                ## 📊 结论

                ### 整体评估
                ### 关键发现
                ### 后续建议
                ### 风险评估
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                """)

        # 写入用户消息
        await memory_service.add_user_message(session_id, aiops_task)

        async for event in self.execute(aiops_task, session_id):
            ev_type = event.get("type")

            # 转换事件格式
            if ev_type == "complete":
                response = event.get("response", "")
                await memory_service.add_assistant_message(session_id, response)
                await memory_service.check_and_summarize(session_id)

                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "诊断流程完成",
                    "diagnosis": {
                        "status": "completed",
                        "report": response
                    }
                }
            elif ev_type in ("plan", "step_complete", "report", "status", "error", "heartbeat"):
                yield event
            elif ev_type == "tool_call":
                # 透传工具调用事件
                yield event

    def _format_planner_event(self, state: Optional[Dict]) -> Dict:
        """格式化 Planner 节点事件"""
        if not state:
            return {"type": "status", "stage": "planner", "message": "规划节点执行中"}

        plan = state.get("plan", [])
        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {len(plan)} 个步骤",
            "plan": plan,
            "tool_calls_planned": len(plan),
        }

    def _format_executor_event(self, state: Optional[Dict]) -> Dict:
        """格式化 Executor 节点事件"""
        if not state:
            return {"type": "status", "stage": "executor", "message": "执行节点运行中"}

        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if past_steps:
            last_step, last_result = past_steps[-1]
            total = len(past_steps) + len(plan)

            # 检测工具调用信息
            event = {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成 ({len(past_steps)}/{total})",
                "current_step": last_step,
                "result_preview": last_result[:200] if last_result else "",
                "remaining_steps": len(plan),
                "tool_calls": self._extract_tool_calls(last_result),
            }

            # 发送工具调用事件
            tool_calls = self._extract_tool_calls(last_step)
            if tool_calls:
                event["tool_call_start"] = {
                    "tool": tool_calls[0],
                    "input": last_step,
                }

            return event
        else:
            return {"type": "status", "stage": "executor", "message": "开始执行步骤"}

    def _format_replanner_event(self, state: Optional[Dict]) -> Dict:
        """格式化 Replanner 节点事件"""
        if not state:
            return {"type": "status", "stage": "replanner", "message": "评估节点运行中"}

        response = state.get("response", "")
        plan = state.get("plan", [])

        if response:
            return {
                "type": "report",
                "stage": "final_report",
                "message": "最终报告已生成",
                "report": response
            }
        else:
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"评估完成，{'继续执行剩余步骤' if plan else '准备生成最终响应'}",
                "remaining_steps": len(plan),
                "should_replan": bool(plan),
            }

    def _extract_tool_calls(self, text: str) -> list:
        """从文本中提取工具调用信息"""
        import re
        patterns = [
            r'使用\s*([\w_]+)\s*工具',
            r'调用\s*([\w_]+)',
            r'([\w_]+)\s*查询',
            r'通过\s*([\w_]+)\s*获取',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return [match.group(1)]
        return []


# 全局单例
aiops_service = AIOpsService()
