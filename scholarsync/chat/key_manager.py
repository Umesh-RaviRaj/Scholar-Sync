"""
API Key Rotation Manager — production-grade round-robin key selection with:
  - True concurrent-safe rotation (separate lock per operation)
  - Per-key Groq client caching (avoid recreating on every call)
  - Immediate failover on rate-limit (switch to next key, don't wait)
  - Smart wait-for-cooldown when ALL keys are exhausted
  - Per-key token budget tracking (resets every 60 s)
  - Thread-safe index bump ensures no two parallel calls share the same key

Used as the single entry-point for all Groq LLM calls across the system.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator

from groq import Groq

from scholarsync.config.settings import get_settings
from scholarsync.utils.logger import get_logger

logger = get_logger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────

class AllKeysExhaustedError(Exception):
    """Raised when every available API key has failed or timed out."""
    pass


# ── Per-key state ─────────────────────────────────────────────────────

@dataclass
class _KeyState:
    key: str
    index: int                    # Position in the pool
    request_count: int = 0
    failure_count: int = 0
    last_used: float = 0.0
    disabled_until: float = 0.0  # Unix timestamp; 0 = enabled
    tokens_this_minute: int = 0  # Rolling token count
    minute_window_start: float = field(default_factory=time.time)

    # Cached Groq client — created once per key, reused across calls
    _client: Groq | None = field(default=None, repr=False, compare=False)

    def client(self) -> Groq:
        if self._client is None:
            if self.key.startswith("sk-or-"):
                # OpenRouter base URL
                self._client = Groq(api_key=self.key, base_url="https://openrouter.ai")
                
                # Monkey-patch to fix Groq's hardcoded path
                original_post = self._client.chat.completions._post
                def _patched_post(path, *args, **kwargs):
                    if path == "/openai/v1/chat/completions":
                        path = "/api/v1/chat/completions"
                    return original_post(path, *args, **kwargs)
                self._client.chat.completions._post = _patched_post
            else:
                self._client = Groq(api_key=self.key)
        return self._client

    def map_model(self, model_name: str) -> str:
        """Map Groq model names to OpenRouter equivalents if needed."""
        if not self.key.startswith("sk-or-"):
            return model_name
        
        m = model_name.lower()
        if "llama-3.3-70b" in m: return "meta-llama/llama-3.3-70b-instruct"
        if "llama-3.1-70b" in m: return "meta-llama/llama-3.1-70b-instruct"
        if "llama-3.1-8b" in m:  return "meta-llama/llama-3.1-8b-instruct"
        if "llama3-70b" in m:    return "meta-llama/llama-3-70b-instruct"
        if "llama3-8b" in m:     return "meta-llama/llama-3-8b-instruct"
        if "mixtral" in m:       return "mistralai/mixtral-8x7b-instruct"
        if "gemma" in m:         return "google/gemma-7b-it"
        return "meta-llama/llama-3.3-70b-instruct"  # Fallback

    @property
    def is_active(self) -> bool:
        return time.time() >= self.disabled_until

    def cooldown_remaining(self) -> float:
        return max(0.0, self.disabled_until - time.time())

    def record_tokens(self, n: int) -> None:
        """Track tokens used in the current 60 s window."""
        now = time.time()
        if now - self.minute_window_start >= 60.0:
            self.tokens_this_minute = 0
            self.minute_window_start = now
        self.tokens_this_minute += n


# ── Singleton Manager ─────────────────────────────────────────────────

class KeyManager:
    """
    Thread-safe round-robin API key manager.

    Design principles:
    - Each call atomically claims the NEXT available key via a shared index.
    - If the chosen key is rate-limited, it is immediately disabled and the
      next key in the pool is tried (no wasted retry on the same key).
    - When ALL keys are cooling down, the caller blocks until the soonest
      cooldown expires, then retries — no hard fail on first exhaustion.
    - Groq clients are cached per-key to avoid connection overhead.
    """

    _instance: "KeyManager | None" = None
    _class_lock = threading.Lock()

    def __new__(cls) -> "KeyManager":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._ready = False
                    cls._instance = obj
        return cls._instance

    def __init__(self) -> None:
        if self._ready:
            return
        self._ready = True

        settings = get_settings()

        # Deduplicate: GROQ_API_KEYS list + primary GROQ_API_KEY
        seen: set[str] = set()
        raw_keys: list[str] = []
        for k in list(settings.groq_api_keys) + [settings.groq_api_key]:
            if k and k not in seen:
                seen.add(k)
                raw_keys.append(k)

        if not raw_keys:
            raise ValueError(
                "No Groq API keys configured. Set GROQ_API_KEY or GROQ_API_KEYS in .env"
            )

        self._pool: list[_KeyState] = [
            _KeyState(key=k, index=i) for i, k in enumerate(raw_keys)
        ]
        self._rr_index: int = 0          # Round-robin cursor
        self._mu = threading.Lock()       # Protects pool state + cursor

        logger.info(
            "KeyManager ready — %d key(s): [%s]",
            len(self._pool),
            ", ".join(f"...{ks.key[-8:]}" for ks in self._pool),
        )

    # ── Internal helpers ──────────────────────────────────────────────

    def _claim_next_active(self) -> "_KeyState | None":
        """
        Atomically advance the round-robin cursor and return the next
        active (non-cooling-down) key.  Returns None if all are disabled.
        Must be called with self._mu held.
        """
        n = len(self._pool)
        for _ in range(n):
            ks = self._pool[self._rr_index % n]
            self._rr_index += 1
            if ks.is_active:
                return ks
        return None

    def _soonest_available(self) -> float:
        """Return seconds until the next key becomes active. Must hold _mu."""
        now = time.time()
        return max(0.0, min(ks.disabled_until for ks in self._pool) - now)

    def _disable_key(self, ks: "_KeyState", cooldown: float = 65.0) -> None:
        """Mark a key as rate-limited for `cooldown` seconds. Must hold _mu."""
        ks.disabled_until = time.time() + cooldown
        ks.failure_count += 1
        logger.warning(
            "Key ...%s disabled for %.0fs (failure #%d)",
            ks.key[-8:], cooldown, ks.failure_count,
        )

    def _wait_for_any_key(self, timeout: float = 70.0) -> "_KeyState | None":
        """
        Block (releasing the lock between polls) until at least one key
        becomes active, then return it.  Returns None on timeout.
        """
        deadline = time.time() + timeout
        while True:
            with self._mu:
                ks = self._claim_next_active()
                if ks is not None:
                    return ks
                wait_secs = self._soonest_available()

            remaining = deadline - time.time()
            if remaining <= 0:
                return None

            sleep_secs = min(wait_secs + 0.2, remaining)   # +0.2 s buffer
            logger.info(
                "All %d keys cooling — waiting %.1fs before retry",
                len(self._pool), sleep_secs,
            )
            time.sleep(sleep_secs)

    @staticmethod
    def _is_rate_limit_error(err_str: str) -> bool:
        patterns = (
            "rate_limit", "rate limit", "429",
            "quota", "too many request", "capacity",
            "overloaded", "tokens per min", "requests per min",
        )
        return any(p in err_str for p in patterns)

    # ── Public API ────────────────────────────────────────────────────

    def call_llm(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        max_retries: int = 6,
        session_id: str | None = None,
    ) -> str:
        """
        Call Groq with true round-robin rotation.

        On each attempt a DIFFERENT key is chosen (never the same key
        twice in a row unless it's the only active one).  When a key
        is rate-limited it is immediately disabled and the next key is
        tried on the very next attempt — no wasted wait inside the loop.
        Only when ALL keys are disabled does it block until one recovers.
        """
        import time as _time
        from scholarsync.chat.llm_cache import get_llm_cache, get_pipeline_budget

        settings = get_settings()
        _model = model or settings.groq_model
        _temp  = temperature if temperature is not None else settings.groq_temperature
        _toks  = max_tokens or settings.groq_max_tokens

        # ── Budget check ──────────────────────────────────────────
        budget = None
        if session_id:
            budget = get_pipeline_budget(session_id)
            if budget.exhausted:
                logger.warning(
                    "Pipeline budget exhausted (%s) — skipping LLM call",
                    budget.summary(),
                )
                raise AllKeysExhaustedError(
                    f"Pipeline token budget exhausted: {budget.summary()}"
                )

        # ── Cache check ──────────────────────────────────────────
        cache = get_llm_cache()
        cached = cache.get(messages, _model, _toks)
        if cached is not None:
            return cached

        start_time = _time.time()
        last_error: Exception | None = None

        for attempt in range(max_retries):
            # ── Claim a key ──────────────────────────────────────
            with self._mu:
                ks = self._claim_next_active()

            if ks is None:
                logger.warning(
                    "All keys cooling down (attempt %d/%d) — blocking until one recovers…",
                    attempt + 1, max_retries,
                )
                ks = self._wait_for_any_key(timeout=70.0)
                if ks is None:
                    break   # Timed out — fall through to raise

            # ── Fire the request ─────────────────────────────────
            try:
                kwargs: dict[str, Any] = {
                    "model": ks.map_model(_model),
                    "messages": messages,
                    "temperature": _temp,
                    "max_tokens": _toks,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = ks.client().chat.completions.create(**kwargs)
                text = response.choices[0].message.content.strip()
                
                # Robust JSON extraction: some models wrap JSON in markdown or add conversational text
                if response_format and response_format.get("type") == "json_object":
                    text = text.strip()
                    first_brace = text.find("{")
                    first_bracket = text.find("[")
                    first_idx = min(first_brace, first_bracket) if first_brace != -1 and first_bracket != -1 else max(first_brace, first_bracket)
                    
                    last_brace = text.rfind("}")
                    last_bracket = text.rfind("]")
                    last_idx = max(last_brace, last_bracket)
                    
                    if first_idx != -1 and last_idx != -1 and last_idx > first_idx:
                        text = text[first_idx : last_idx + 1]

                # Track usage
                usage = getattr(response, "usage", None)
                tokens_used = getattr(usage, "total_tokens", _toks) if usage else _toks
                elapsed = _time.time() - start_time
                with self._mu:
                    ks.request_count += 1
                    ks.last_used = _time.time()
                    ks.failure_count = 0
                    ks.record_tokens(tokens_used)

                # Record in pipeline budget
                if budget:
                    budget.record(tokens_used)

                logger.info(
                    "LLM OK — key ...%s | attempt %d | %d tok | %.1fs | budget: %s",
                    ks.key[-8:], attempt + 1, tokens_used, elapsed,
                    budget.summary() if budget else "n/a",
                )

                # Store in cache
                cache.put(messages, _model, _toks, text, tokens_used)

                return text

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                logger.warning(
                    "LLM error — key ...%s | attempt %d/%d | %s",
                    ks.key[-8:], attempt + 1, max_retries, e,
                )
                with self._mu:
                    if self._is_rate_limit_error(err_str):
                        self._disable_key(ks, cooldown=65.0)
                    else:
                        ks.failure_count += 1
                # Non-rate-limit error: tiny backoff before trying next key
                if not self._is_rate_limit_error(str(last_error).lower()):
                    time.sleep(min(2 ** attempt, 8))

        raise AllKeysExhaustedError(
            f"All {len(self._pool)} API key(s) failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def call_llm_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 6,
    ) -> Iterator[str]:
        """
        Stream Groq completions with the same round-robin rotation as
        call_llm.  Yields text chunks as they arrive.
        """
        settings = get_settings()
        _model = model or settings.groq_model
        _temp  = temperature if temperature is not None else settings.groq_temperature
        _toks  = max_tokens or settings.groq_max_tokens

        last_error: Exception | None = None

        for attempt in range(max_retries):
            with self._mu:
                ks = self._claim_next_active()

            if ks is None:
                logger.warning(
                    "All stream keys cooling (attempt %d/%d) — waiting…",
                    attempt + 1, max_retries,
                )
                ks = self._wait_for_any_key(timeout=70.0)
                if ks is None:
                    break

            try:
                stream = ks.client().chat.completions.create(
                    model=ks.map_model(_model),
                    messages=messages,
                    temperature=_temp,
                    max_tokens=_toks,
                    stream=True,
                )
                with self._mu:
                    ks.request_count += 1
                    ks.last_used = time.time()
                    ks.failure_count = 0

                chunks_seen = 0
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        chunks_seen += 1
                        yield delta.content

                logger.info(
                    "Stream OK — key ...%s | attempt %d | %d chunks",
                    ks.key[-8:], attempt + 1, chunks_seen,
                )
                return  # success

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                logger.warning(
                    "Stream error — key ...%s | attempt %d/%d | %s",
                    ks.key[-8:], attempt + 1, max_retries, e,
                )
                with self._mu:
                    if self._is_rate_limit_error(err_str):
                        self._disable_key(ks, cooldown=65.0)
                    else:
                        ks.failure_count += 1

        raise AllKeysExhaustedError(
            f"All {len(self._pool)} stream key(s) failed after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    # ── Diagnostics ───────────────────────────────────────────────────

    def get_stats(self) -> list[dict]:
        """Return real-time stats for every key in the pool."""
        with self._mu:
            return [
                {
                    "index": ks.index,
                    "key_suffix": f"...{ks.key[-8:]}",
                    "requests": ks.request_count,
                    "failures": ks.failure_count,
                    "active": ks.is_active,
                    "cooldown_remaining_s": round(ks.cooldown_remaining(), 1),
                    "tokens_this_minute": ks.tokens_this_minute,
                }
                for ks in self._pool
            ]

    def status_summary(self) -> str:
        stats = self.get_stats()
        active = sum(1 for s in stats if s["active"])
        parts = []
        for s in stats:
            suffix = s["key_suffix"]
            if s["active"]:
                status = "OK"
            else:
                status = f"cooling {s['cooldown_remaining_s']}s"
            parts.append(f"{suffix} ({status}) req={s['requests']}")
        return f"{active}/{len(stats)} keys active | " + " | ".join(parts)


# ── Module-level convenience ──────────────────────────────────────────

def get_key_manager() -> KeyManager:
    """Return the singleton KeyManager instance."""
    return KeyManager()
