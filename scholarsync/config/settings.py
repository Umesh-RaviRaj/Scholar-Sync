"""
Application settings — loaded from environment variables / .env file.
Uses Pydantic Settings for typed, validated config.
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for ScholarSync."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────
    app_name: str = "ScholarSync"
    debug: bool = False

    # ── Groq LLM ────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_api_keys: list[str] = []  # Extra keys for rotation
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.1
    groq_max_tokens: int = 4096  # Safe default; synthesizer uses its own limit

    # ── Per-agent token limits (right-sized to avoid waste) ──────
    manager_max_tokens: int = 1024     # Only needs a short JSON task list
    worker_max_tokens: int = 3072      # Structured extraction output
    checker_max_tokens: int = 1536     # Validation JSON is compact
    synthesizer_max_tokens: int = 8000 # Extended literature review (research-grade depth)
    worker_chunk_count: int = 8        # Chunks retrieved per worker call

    # ── Pipeline budget ──────────────────────────────────────────
    max_pipeline_tokens: int = 120000  # Hard ceiling per pipeline run
    llm_cache_ttl_seconds: int = 3600  # 1 hour cache for LLM responses

    # ── Mode Defaults ────────────────────────────────────────────────
    normal_mode_top_k: int = 5
    normal_mode_graph_depth: int = 1
    deep_research_top_k: int = 10
    deep_research_graph_depth: int = 3

    # ── Neo4j (GraphRAG) ────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── ChromaDB (Vector Store) ─────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma_db"
    chroma_collection_name: str = "scholarsync_papers"

    # ── Embeddings ──────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── Document Processing ─────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_papers: int = 10
    upload_dir: str = "./data/uploads"

    # ── Validation ──────────────────────────────────────────────────
    validation_threshold: float = 0.7
    max_correction_loops: int = 1  # Feedback-augmented correction is higher quality; 1 loop suffices

    # ── API ──────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── Reports ─────────────────────────────────────────────────────
    reports_dir: str = "./data/reports"

    # ── Web Search (optional) ────────────────────────────────────────
    tavily_api_key: str = ""  # Optional: enables Tavily search; falls back to DuckDuckGo

    # ── MongoDB (Chat Persistence) — LEGACY, NOT USED ───────────────
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "scholarsync_db"

    # ── Firebase Auth ───────────────────────────────────────────────
    firebase_credentials_path: str = "./firebase-service-account.json"

    # ── Local Auth (SQLite + JWT) ───────────────────────────────────
    jwt_secret_key: str = "change-me-to-a-secure-random-string-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 1440  # 24 hours
    auth_db_path: str = "./data/auth.db"


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton to avoid re-reading .env on every call."""
    return Settings()
