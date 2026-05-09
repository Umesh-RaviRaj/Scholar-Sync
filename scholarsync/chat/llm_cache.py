"""
LLM Response Cache — hash-based in-memory cache with TTL expiry.
Prevents duplicate LLM calls for identical prompts within the same
pipeline run or within the configured TTL window.
"""

from __future__ import annotations

import hashlib
import json
import time
import threading
from dataclasses import dataclass, field

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _CacheEntry:
    response: str
    created_at: float
    tokens_used: int = 0
    hit_count: int = 0


class LLMCache:
    """Thread-safe in-memory LLM response cache with TTL."""

    _instance: "LLMCache | None" = None
    _class_lock = threading.Lock()

    def __new__(cls) -> "LLMCache":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._store: dict[str, _CacheEntry] = {}
                    obj._mu = threading.Lock()
                    cls._instance = obj
        return cls._instance

    @staticmethod
    def _make_key(messages: list[dict], model: str, max_tokens: int) -> str:
        """Deterministic cache key from the request parameters."""
        raw = json.dumps(
            {"m": messages, "model": model, "mt": max_tokens},
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
    ) -> str | None:
        """Return cached response or None if miss/expired."""
        settings = get_settings()
        key = self._make_key(messages, model, max_tokens)
        with self._mu:
            entry = self._store.get(key)
            if entry is None:
                return None
            age = time.time() - entry.created_at
            if age > settings.llm_cache_ttl_seconds:
                del self._store[key]
                return None
            entry.hit_count += 1
            logger.info(
                "LLM cache HIT (key=%s… hits=%d age=%.0fs)",
                key[:12], entry.hit_count, age,
            )
            return entry.response

    def put(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int,
        response: str,
        tokens_used: int = 0,
    ) -> None:
        """Store a response in the cache."""
        key = self._make_key(messages, model, max_tokens)
        with self._mu:
            self._store[key] = _CacheEntry(
                response=response,
                created_at=time.time(),
                tokens_used=tokens_used,
            )

    def clear(self) -> None:
        with self._mu:
            self._store.clear()

    def stats(self) -> dict:
        with self._mu:
            total = len(self._store)
            hits = sum(e.hit_count for e in self._store.values())
        return {"entries": total, "total_hits": hits}


def get_llm_cache() -> LLMCache:
    """Return the singleton LLM cache."""
    return LLMCache()
"""
Pipeline-level token budget tracker.
Tracks cumulative token usage across an entire pipeline run and
raises an error if the budget is exceeded.
"""


@dataclass
class PipelineBudget:
    """Tracks token usage for a single pipeline run."""
    max_tokens: int
    tokens_used: int = 0
    call_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, tokens: int) -> None:
        with self._lock:
            self.tokens_used += tokens
            self.call_count += 1

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.tokens_used)

    @property
    def exhausted(self) -> bool:
        return self.tokens_used >= self.max_tokens

    def summary(self) -> str:
        pct = (self.tokens_used / self.max_tokens * 100) if self.max_tokens else 0
        return (
            f"{self.tokens_used:,}/{self.max_tokens:,} tokens "
            f"({pct:.0f}%) across {self.call_count} calls"
        )


# Per-session budget registry
_budgets: dict[str, PipelineBudget] = {}
_budgets_lock = threading.Lock()


def get_pipeline_budget(session_id: str) -> PipelineBudget:
    """Get or create a pipeline budget for a session."""
    with _budgets_lock:
        if session_id not in _budgets:
            settings = get_settings()
            _budgets[session_id] = PipelineBudget(
                max_tokens=settings.max_pipeline_tokens
            )
        return _budgets[session_id]


def clear_pipeline_budget(session_id: str) -> None:
    with _budgets_lock:
        _budgets.pop(session_id, None)
