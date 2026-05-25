# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Dual-mode multi-agent architecture with ReAct and Plan-Execute-Replan
- MCP (Model Context Protocol) integration for CLS log and monitor tools
- RAG pipeline with Milvus vector store, hybrid chunking, and reranking
- PostgreSQL-backed long-term memory with session management
- LLM-as-a-Judge evaluation with Ragas metrics
- Unified Agent API with automatic routing between RAG and AIOps modes
- Streaming SSE response support
- Docker Compose setup for Milvus, PostgreSQL, etcd, MinIO

### Changed

### Fixed

## [0.1.0] - 2025-05-25

### Added
- Initial project release
- FastAPI-based backend with chat, AIOps, and evaluation APIs
- Static HTML/CSS/JS frontend
- Mock MCP tool fallback for offline development
