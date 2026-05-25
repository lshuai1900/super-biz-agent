"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3
    rag_model: str = "qwen-max"  # 使用快速响应模型，不带扩展思考

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置（transport: stdio | sse | streamable-http）
    # 腾讯云托管 MCP 的 URL 通常含 /sse/，需使用 sse；本地 FastMCP 使用 streamable-http
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    # PostgreSQL 配置
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/super_biz_agent"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "super_biz_agent"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"

    # RAG 增强配置
    rag_chunk_size: int = 1600
    rag_chunk_overlap: int = 100
    rag_candidate_top_k: int = 20
    rag_final_top_k: int = 3
    rag_min_similarity_score: float = 0.0
    rag_max_l2_distance: float = 2.0
    rag_enable_rerank: bool = False
    rag_rerank_model: str = ""

    # Ragas 评估
    ragas_enable: bool = False

    # CORS
    allowed_origins: str = "http://localhost:9900,http://127.0.0.1:9900,http://localhost:3000,http://127.0.0.1:3000"

    # API Key 鉴权（为空时不启用）
    api_key: str = ""

    # 上传安全
    upload_dir: str = "uploads"
    max_upload_size_mb: int = 20
    allowed_upload_extensions: str = ".txt,.md,.pdf,.docx,.csv,.html"

    # Agent checkpointer
    agent_checkpointer: str = "memory"  # memory | sqlite
    agent_sqlite_db_path: str = "data/agent_state.sqlite3"

    # Hybrid Search
    enable_hybrid_search: bool = False
    hybrid_vector_weight: float = 0.7
    hybrid_bm25_weight: float = 0.3
    hybrid_rrf_k: int = 60

    # Prometheus
    prometheus_base_url: str = "http://127.0.0.1:9090"
    prometheus_request_timeout: float = 10.0

    # 腾讯云 CLS
    tencentcloud_secret_id: str = ""
    tencentcloud_secret_key: str = ""
    tencentcloud_region: str = "ap-beijing"
    tencentcloud_topic_id: str = ""

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
