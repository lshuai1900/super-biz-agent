"""可配置的 LangGraph Checkpointer

支持 memory / sqlite / postgres 三种模式。

配置：
- AGENT_CHECKPOINTER=memory  使用内存（默认，重启丢失）
- AGENT_CHECKPOINTER=sqlite  使用 SQLite 持久化
- AGENT_SQLITE_DB_PATH=data/agent_state.sqlite3
"""

from pathlib import Path

from loguru import logger

from app.config import config


def create_checkpointer():
    """根据配置创建 LangGraph checkpointer

    Returns:
        LangGraph checkpointer 实例
    """
    checkpoint_type = getattr(config, "agent_checkpointer", "memory")

    if checkpoint_type == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            db_path = getattr(config, "agent_sqlite_db_path", "data/agent_state.sqlite3")
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"使用 SQLite Checkpointer: {db_path}")
            return SqliteSaver.from_conn_string(str(db_file.resolve()))
        except ImportError:
            logger.warning("langgraph-checkpoint-sqlite 未安装，降级为 MemorySaver")
        except Exception as e:
            logger.warning(f"SQLite Checkpointer 创建失败: {e}，降级为 MemorySaver")

    elif checkpoint_type == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            db_url = getattr(config, "database_url", "")
            if db_url:
                logger.info("使用 PostgreSQL Checkpointer")
                return PostgresSaver.from_conn_string(db_url)
            else:
                logger.warning("DATABASE_URL 未配置，降级为 MemorySaver")
        except ImportError:
            logger.warning("langgraph-checkpoint-postgres 未安装，降级为 MemorySaver")
        except Exception as e:
            logger.warning(f"PostgreSQL Checkpointer 创建失败: {e}，降级为 MemorySaver")

    # 默认：MemorySaver
    logger.info("使用 MemorySaver Checkpointer（会话状态不持久化）")
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
