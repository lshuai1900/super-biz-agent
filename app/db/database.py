"""数据库引擎与会话管理"""

from datetime import datetime, timezone
from typing import AsyncGenerator, Optional

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config import config

_engine = None
_async_session_factory = None
_connected = False


def get_database_url() -> str:
    """获取 PostgreSQL 连接 URL"""
    url = getattr(config, "database_url", "")
    if url:
        return url
    host = getattr(config, "postgres_host", "localhost")
    port = getattr(config, "postgres_port", 5432)
    db = getattr(config, "postgres_db", "super_biz_agent")
    user = getattr(config, "postgres_user", "postgres")
    password = getattr(config, "postgres_password", "postgres")
    return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"


async def init_db():
    """初始化数据库连接并创建所有表"""
    global _engine, _async_session_factory, _connected

    if _connected:
        return

    try:
        dsn = get_database_url()
        logger.info(f"Connecting to PostgreSQL: {dsn.split('@')[-1]}")

        _engine = create_async_engine(dsn, echo=False, pool_size=5, max_overflow=10)
        _async_session_factory = async_sessionmaker(
            _engine, class_=AsyncSession, expire_on_commit=False
        )

        # 创建所有表
        from app.db.models import Base
        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        _connected = True
        logger.info("PostgreSQL 数据库初始化成功")
    except Exception as e:
        logger.warning(f"PostgreSQL 数据库初始化失败（服务以降级模式运行）: {e}")
        _connected = False


async def close_db():
    """关闭数据库连接"""
    global _engine, _connected
    if _engine:
        await _engine.dispose()
        _engine = None
        _connected = False
        logger.info("PostgreSQL 连接已关闭")


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话（用于 FastAPI Depends）"""
    if not _connected or not _async_session_factory:
        raise ConnectionError("PostgreSQL 不可用")
    async with _async_session_factory() as session:
        yield session


def get_session_sync() -> Optional[AsyncSession]:
    """同步方式获取数据库会话"""
    if not _connected or not _async_session_factory:
        return None
    return _async_session_factory()


def is_connected() -> bool:
    return _connected
