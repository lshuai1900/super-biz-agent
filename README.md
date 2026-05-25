# 智能 OnCall Agent 系统

> 通过 AI Agent 解决团队真实痛点，整合知识库、对话、运维三大核心能力，实现问题自动应答和故障智能排查的一体化服务。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)](https://www.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.0.40+-red.svg)](https://langchain-ai.github.io/langgraph/)

## 核心能力

1. **双模态多智能体架构** - 对话侧 ReAct 动态检索，运维侧 Plan-Execute-Replan 有向图，基于 LangGraph 构建混合路由状态机
2. **长程记忆与上下文管理** - PostgreSQL 存精准历史，Milvus 存语义摘要；超阈值自动摘要压缩 + 滑动窗口截断
3. **多路径混合 RAG** - Markdown 按层级 + 滑窗分块，Milvus L2 召回，双阈值过滤 + 重排
4. **MCP 端到端闭环** - 接入 MCP 标准，集成 CLS 日志检索和监控指标查询工具，SSE 流式输出
5. **LLM-as-a-Judge 自动评估** - Ragas 评估，Context Precision / Recall / Faithfulness

## 技术栈

| 组件 | 技术 |
|------|------|
| 框架 | FastAPI + LangChain + LangGraph |
| LLM | 阿里云 DashScope (通义千问系列) |
| 向量库 | Milvus (Docker) |
| 记忆库 | PostgreSQL (Docker) |
| 工具协议 | MCP (Model Context Protocol) |
| 评估 | Ragas |
| 前端 | 纯静态 HTML/CSS/JS |

## 快速开始

### 环境要求
- Python 3.11+
- Docker & Docker Compose
- 阿里云 DashScope API Key

### 安装步骤

```bash
# 1. 安装依赖
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 3. 启动 Docker（Milvus + PostgreSQL）
docker compose -f vector-database.yml up -d

# 4. 等待服务启动
sleep 10

# 5. 启动主服务
uvicorn app.main:app --host 0.0.0.0 --port 9900 --reload

# 6. 上传运维文档
python -c "
import requests, os;
for f in os.listdir('aiops-docs'):
    if f.endswith('.md'):
        requests.post('http://localhost:9900/api/upload', files={'file': open(f'aiops-docs/{f}', 'rb')})
"
```

### 访问服务
- **Web 界面**: http://localhost:9900
- **API 文档**: http://localhost:9900/docs

## Docker 服务说明

`vector-database.yml` 包含以下必要组件：
- **Milvus Standalone** - 向量数据库（端口 19530）
- **etcd** - Milvus 元数据存储
- **MinIO** - Milvus 数据存储（端口 9000/9001）
- **PostgreSQL** - 长期记忆存储（端口 5432）
- **Attu** - Milvus Web 管理（端口 8000，可选）

## 架构说明

```
┌─────────────┐     ┌─────────────────────────────────────────────┐
│  静态前端    │     │              FastAPI 服务                    │
│  HTML/CSS/JS │────▶│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
└─────────────┘     │  │ Chat API │ │Agent API │ │ Eval API │    │
                    │  └────┬─────┘ └────┬─────┘ └────┬─────┘    │
                    │       │            │             │          │
                    │  ┌────▼────────────▼──────────────▼─────┐   │
                    │  │        LangGraph 路由状态机           │   │
                    │  │  route → rag_agent / aiops_agent     │   │
                    │  └────┬────────────┬──────────────┬─────┘   │
                    │       │            │              │         │
                    │  ┌────▼────┐ ┌────▼────┐ ┌───────▼──────┐  │
                    │  │ ReAct   │ │Plan-Exec│ │ Memory      │  │
                    │  │ Agent   │ │Replan   │ │ Service     │  │
                    │  └────┬────┘ └────┬────┘ └───────┬──────┘  │
                    └───────┼───────────┼───────────────┼─────────┘
                            │           │               │
                    ┌───────▼───┐ ┌────▼────┐    ┌─────▼──────┐
                    │  Milvus   │ │MCP工具集│    │ PostgreSQL │
                    │  向量库    │ │CLS/Mon  │    │ 会话历史   │
                    └───────────┘ └─────────┘    └────────────┘
```

## API 接口

### 核心接口

| 功能 | 方法 | 路径 | 说明 |
|------|------|------|------|
| 统一 Agent | POST | `/api/agent` | 自动路由模式 |
| 统一 Agent 流式 | POST | `/api/agent_stream` | SSE 流式 |
| 普通对话 | POST | `/api/chat` | RAG 对话 |
| 流式对话 | POST | `/api/chat_stream` | SSE 流式 |
| AIOps 诊断 | POST | `/api/aiops` | 故障诊断（流式） |
| 文件上传 | POST | `/api/upload` | 上传并索引文档 |
| 会话列表 | GET | `/api/chat/sessions` | 所有会话 |
| 会话历史 | GET | `/api/chat/session/{id}` | 指定会话 |
| 会话摘要 | GET | `/api/chat/session/{id}/summary` | 摘要查看 |
| 清空会话 | POST | `/api/chat/clear` | 清空历史 |
| 生成测试集 | POST | `/api/evaluation/generate_dataset` | QA 生成 |
| 运行评估 | POST | `/api/evaluation/run` | RAG 评估 |
| 评估结果 | GET | `/api/evaluation/results` | 查看结果 |
| MCP 状态 | GET | `/api/mcp/status` | 服务状态 |
| MCP 工具 | GET | `/api/mcp/tools` | 工具列表 |
| 健康检查 | GET | `/api/health` | 服务状态 |

### 使用示例

```bash
# 统一 Agent（自动路由）
curl -X POST http://localhost:9900/api/agent \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-001","mode":"auto","question":"data-sync-service 出现 CPU 告警，帮我排查"}'

# 知识库问答
curl -X POST http://localhost:9900/api/chat \
  -H "Content-Type: application/json" \
  -d '{"Id":"test-002","Question":"系统如何处理告警？"}'

# AIOps 诊断
curl -X POST http://localhost:9900/api/aiops \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-003"}' --no-buffer

# RAG 评估
curl -X POST http://localhost:9900/api/evaluation/run

# 查看评估结果
curl http://localhost:9900/api/evaluation/results
```

## MCP 工具

MCP 服务通过本地 FastMCP 进程启动，支持真实 API 和 mock 降级：

### CLS 日志服务（端口 8003）
- `search_log` - 搜索日志（真实腾讯云 CLS / mock）
- `search_topic_by_service_name` - 按服务名搜索日志主题
- `get_current_timestamp` - 获取时间戳

### Monitor 监控服务（端口 8004）
- `query_cpu_metrics` - CPU 监控（真实 Prometheus / mock）
- `query_memory_metrics` - 内存监控（真实 Prometheus / mock）

**配置真实 API**：在 `.env` 中配置 `TENCENTCLOUD_SECRET_ID/KEY` 或 `PROMETHEUS_BASE_URL`，
不配置时自动使用 mock 数据。

## RAG 评估

基于 Ragas 的自动质量评估系统：

### 指标说明
- **Context Precision** - 检索到的上下文中有多少是相关的
- **Context Recall** - 所有相关上下文有多少被检索到
- **Faithfulness** - 回答是否忠实于检索到的上下文
- **Answer Relevancy**（可选）- 回答是否相关

### 使用方法
```bash
# 生成 QA 测试集
python scripts/generate_eval_dataset.py --source aiops-docs --count 10

# 运行评估
python scripts/run_ragas_eval.py --count 5

# 启动 Milvus Lite（本地模式）
python scripts/start_milvus_lite.py

# 或通过 API
curl -X POST http://localhost:9900/api/evaluation/run
```

## RAG 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| RAG_CHUNK_SIZE | 1600 | 文档分块大小（字符） |
| RAG_CHUNK_OVERLAP | 100 | 分块重叠大小 |
| RAG_CANDIDATE_TOP_K | 20 | 粗召回候选数 |
| RAG_FINAL_TOP_K | 3 | 最终返回数 |
| RAG_MIN_SIMILARITY_SCORE | 0.0 | 最低相似度阈值 |
| RAG_MAX_L2_DISTANCE | 2.0 | 最大 L2 距离阈值 |
| RAG_ENABLE_RERANK | false | 是否启用重排 |

## 项目结构

```
├── app/
│   ├── api/               # API 路由层
│   │   ├── chat.py        # 对话接口
│   │   ├── aiops.py       # AIOps 诊断接口
│   │   ├── agent.py       # 统一 Agent 接口
│   │   ├── evaluation.py  # RAG 评估接口
│   │   ├── file.py        # 文件上传接口
│   │   └── health.py      # 健康检查
│   ├── services/          # 业务服务层
│   │   ├── rag_agent_service.py     # RAG Agent 服务
│   │   ├── aiops_service.py         # AIOps 服务
│   │   ├── router_agent_service.py  # 统一路由 Agent
│   │   ├── memory_service.py        # PostgreSQL 记忆服务
│   │   ├── vector_store_manager.py  # 向量存储管理
│   │   ├── vector_index_service.py  # 向量索引服务
│   │   ├── vector_search_service.py # 向量检索服务
│   │   ├── rerank_service.py        # 重排服务
│   │   └── document_splitter_service.py # 文档分割
│   ├── evaluation/        # RAG 评估模块
│   │   ├── dataset_generator.py     # QA 测试集生成
│   │   ├── ragas_evaluator.py       # Ragas 评估器
│   │   └── eval_models.py          # 评估数据模型
│   ├── agent/             # Agent 模块
│   │   ├── mcp_client.py  # MCP 客户端
│   │   └── aiops/         # AIOps 核心逻辑
│   ├── core/              # 核心组件
│   ├── tools/             # Agent 工具集
│   ├── models/            # 数据模型
│   └── utils/             # 工具类
├── mcp_servers/           # MCP 服务
│   ├── cls_server.py      # CLS 日志服务
│   └── monitor_server.py  # 监控服务
├── static/                # Web 前端
├── tests/                 # 测试
├── aiops-docs/            # 运维知识库示例文档
├── scripts/               # 启动与工具脚本
│   ├── start_milvus_lite.py      # Milvus Lite 启动
│   ├── start-windows.bat         # Windows 启动
│   ├── stop-windows.bat          # Windows 停止
│   ├── generate_eval_dataset.py  # QA 测试集生成
│   └── run_ragas_eval.py         # Ragas 评估
├── .env.example           # 环境变量模板（提交 Git）
├── vector-database.yml    # Docker Compose 向量库
├── docker-compose.dev.yml # Docker Compose 开发环境
├── pyproject.toml         # 项目配置
├── Makefile               # 常用命令
└── README.md
```

### 不提交 Git 的目录

以下目录和文件是运行时生成的，已通过 `.gitignore` 排除，不会提交到 Git：

| 目录/文件 | 说明 |
|-----------|------|
| `.env` | 本地环境变量配置（含密钥），使用 `.env.example` 作为模板 |
| `.venv/` | Python 虚拟环境，通过 `uv sync` 重建 |
| `logs/` | 运行日志 |
| `reports/` | 评估报告输出 |
| `uploads/` | 用户上传文件 |
| `volumes/` | 向量库和数据库运行数据 |
| `*.log` | 日志文件 |
| `.claude/` | Claude Code 本地配置 |
| `.pytest_cache/` | 测试缓存 |

## 常见问题

### Milvus 连接失败
```bash
# 确保 Docker 运行中
docker ps | grep milvus
docker compose -f vector-database.yml restart standalone
```

### PostgreSQL 连接失败
```bash
docker ps | grep postgres
# 检查配置：默认用户 oncall，密码 oncall123，数据库 oncall
```

### 端口被占用
```bash
# 检查端口
lsof -i :9900  # FastAPI
lsof -i :19530 # Milvus
lsof -i :5432  # PostgreSQL
lsof -i :8003  # CLS MCP
lsof -i :8004  # Monitor MCP
```

## License

MIT License
