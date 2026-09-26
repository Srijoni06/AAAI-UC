"""LLM backend abstraction: local (Ollama) or Gemini, with response caching.

Backend is chosen by the ``LLM_BACKEND`` env var / ``.env`` entry:
  - ``local``  (default) -> Ollama at ``OLLAMA_HOST`` (default localhost:11434),
                             model ``llama3.1:8b``. Used for all dev/testing.
  - ``gemini``            -> google-genai, gemini-2.5-flash for agent and judge
                             (override the judge with ``GEMINI_JUDGE_MODEL``).
                             For final verification runs.

The *judge role* (contradiction NLI in ``memory/detector.py`` and the reconciler)
can run on a different backend from the agents: set ``JUDGE_BACKEND=gemini`` (or
``local``) and use :func:`make_judge_llm`. Unset / ``same`` = follow ``LLM_BACKEND``.

Every ``generate`` call checks the on-disk cache (``common.cache.LLMCache``)
before hitting either backend. Disable with ``LLM_CACHE=0``.
"""

from __future__ import annotations

import abc
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from common.cache import LLMCache
from common.env import load_dotenv

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
# Generously exceeds Ollama's own OLLAMA_LOAD_TIMEOUT (5m default), so a cold
# model load on a slow/CPU machine doesn't spuriously time out client-side.
OLLAMA_REQUEST_TIMEOUT = 600
OLLAMA_AGENT_MODEL = "llama3.1:8b"
OLLAMA_JUDGE_MODEL = "llama3.1:8b"  # only one local model for now
GEMINI_AGENT_MODEL = "gemini-2.5-flash"  # gemini-2.0-flash is retired on the API
# gemini-2.5-pro is closed to new API users and gemini-3.1-pro-preview has no
# free-tier quota, so the judge defaults to flash; override with GEMINI_JUDGE_MODEL.
GEMINI_JUDGE_MODEL = "gemini-2.5-flash"

_PLACEHOLDER_KEYS = {"", "your-gemini-api-key"}


class LLMClient(abc.ABC):
    backend: str
    agent_model: str
    judge_model: str

    def __init__(self, cache: LLMCache | None):
        self._cache = cache

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        model: str | None = None,
        sample_id: int | str | None = None,
    ) -> str:
        """Generate a completion, using the on-disk cache when one is attached.

        ``sample_id`` is for deliberate repeated sampling: leave it ``None`` for
        normal caching (one answer per identical request), or pass distinct ids
        (``0, 1, 2, ...``) to draw and cache several independent completions for
        the same prompt. Re-passing a used ``sample_id`` returns that cached
        draw. Diversity across draws still requires ``temperature > 0``.
        """
        model = model or self.agent_model
        system = system or ""
        key = None
        if self._cache is not None:
            key = LLMCache.key(
                backend=self.backend,
                model=model,
                system=system,
                temperature=temperature,
                prompt=prompt,
                sample_id=sample_id,
            )
            hit = self._cache.get(key)
            if hit is not None:
                return hit

        text = self._raw_generate(
            prompt, system=system, temperature=temperature, model=model
        ).strip()
        if not text:
            raise RuntimeError(f"empty response from {self.backend}:{model}")

        if self._cache is not None and key is not None:
            meta = {"backend": self.backend, "model": model}
            if sample_id is not None:
                meta["sample_id"] = sample_id
            self._cache.set(key, text, meta)
        return text

    @abc.abstractmethod
    def _raw_generate(
        self, prompt: str, *, system: str, temperature: float, model: str
    ) -> str: ...


class OllamaBackend(LLMClient):
    backend = "local"
    agent_model = OLLAMA_AGENT_MODEL
    judge_model = OLLAMA_JUDGE_MODEL

    def __init__(self, cache: LLMCache | None = None, host: str | None = None):
        super().__init__(cache)
        self.host = (host or os.environ.get("OLLAMA_HOST") or OLLAMA_DEFAULT_HOST).rstrip(
            "/"
        )

    def _raw_generate(
        self, prompt: str, *, system: str, temperature: float, model: str
    ) -> str:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            body["system"] = system
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=OLLAMA_REQUEST_TIMEOUT) as resp:
                    payload = json.loads(resp.read())
                    return payload.get("response", "")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")
                if e.code == 500 and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"Ollama HTTP {e.code} for model {model!r}: {detail}. "
                    f"Pull it with `ollama pull {model}`."
                ) from e
            except (urllib.error.URLError, TimeoutError) as e:
                # A bare socket TimeoutError (e.g. a slow cold model load on CPU)
                # is not a urllib.error.URLError and was previously uncaught here.
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                reason = getattr(e, "reason", e)
                raise RuntimeError(
                    f"Ollama at {self.host} is unreachable or timed out ({reason}). "
                    f"Start it with `ollama serve` and pull `{model}`. A large model's "
                    f"first call after (re)starting Ollama can be slow while it loads "
                    f"into memory, especially on CPU."
                ) from e
        return ""


class GeminiBackend(LLMClient):
    backend = "gemini"
    agent_model = GEMINI_AGENT_MODEL
    judge_model = GEMINI_JUDGE_MODEL

    def __init__(self, cache: LLMCache | None = None, max_retries: int = 5):
        super().__init__(cache)
        from google import genai  # imported lazily so `local` runs without the dep

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key in _PLACEHOLDER_KEYS:
            raise RuntimeError(
                "GEMINI_API_KEY is not set to a real key. Put it in .env "
                "(GEMINI_API_KEY=...) or export it in this shell."
            )
        self._client = genai.Client(api_key=api_key)
        self._max_retries = max_retries
        self.judge_model = _gemini_judge_model()

    def _raw_generate(
        self, prompt: str, *, system: str, temperature: float, model: str
    ) -> str:
        from google.genai import errors as genai_errors
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system or None, temperature=temperature
        )
        for attempt in range(self._max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
                return resp.text or ""
            except (genai_errors.ServerError, genai_errors.ClientError) as e:
                rate_limited = getattr(e, "code", None) == 429
                # a per-DAY quota will not clear within a retry window; fail fast
                daily_quota = rate_limited and "PerDay" in str(e)
                retriable = isinstance(e, genai_errors.ServerError) or (
                    rate_limited and not daily_quota
                )
                if not retriable or attempt == self._max_retries - 1:
                    raise
                # free-tier quotas are per-minute windows, so back off much longer on 429
                time.sleep((15 if rate_limited else 2) * (attempt + 1))
        return ""  # unreachable


def _cache_enabled() -> bool:
    return os.environ.get("LLM_CACHE", "1").strip().lower() not in {"0", "false", "no"}


DEFAULT_BACKEND = "local"


@dataclass
class LLMResolution:
    """Everything that determined which backend/model a run will use."""

    backend: str
    backend_source: str
    agent_model: str
    judge_model: str
    cache_enabled: bool
    cache_path: str
    dotenv_path: str
    dotenv_exists: bool
    ollama_host: str | None = None
    gemini_key_status: str | None = None
    raw_backend_value: str | None = None
    judge_backend: str | None = None  # None = judge follows ``backend``

    def banner(self) -> str:
        lines = [
            "LLM config",
            f"  .env            : {self.dotenv_path} "
            f"({'found' if self.dotenv_exists else 'NOT FOUND'})",
            f"  LLM_BACKEND     : {self.raw_backend_value!r} -> {self.backend!r}",
            f"  resolved from   : {self.backend_source}",
            f"  agent model     : {self.agent_model}",
            f"  judge model     : {self.judge_model}",
            f"  cache           : {'on' if self.cache_enabled else 'off'} "
            f"({self.cache_path})",
        ]
        if self.backend == "local":
            lines.append(f"  OLLAMA_HOST     : {self.ollama_host}")
        if self.judge_backend:
            lines.append(f"  JUDGE_BACKEND   : {self.judge_backend} (judge role only)")
        if self.gemini_key_status is not None:
            lines.append(f"  GEMINI_API_KEY  : {self.gemini_key_status}")
        return "\n".join(lines)


def _gemini_judge_model() -> str:
    return (os.environ.get("GEMINI_JUDGE_MODEL") or "").strip() or GEMINI_JUDGE_MODEL


def _judge_model_for(backend: str) -> str:
    if backend == "local":
        return OLLAMA_JUDGE_MODEL
    if backend == "gemini":
        return _gemini_judge_model()
    return "?"


def _gemini_key_status() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    return "placeholder / missing" if key in _PLACEHOLDER_KEYS else f"set (len {len(key)})"


def resolve_config() -> LLMResolution:
    """Load .env and report the backend/model that ``make_llm`` would pick,
    plus the judge backend when ``JUDGE_BACKEND`` sends the judge role elsewhere."""
    res = _resolve_main_config()
    raw_judge = (os.environ.get("JUDGE_BACKEND") or "").strip().lower()
    if raw_judge not in {"", "same"} and raw_judge != res.backend:
        res.judge_backend = raw_judge
        res.judge_model = _judge_model_for(raw_judge)
        if raw_judge == "gemini":
            res.gemini_key_status = _gemini_key_status()
    return res


def _resolve_main_config() -> LLMResolution:
    report = load_dotenv()
    raw = os.environ.get("LLM_BACKEND")
    backend = (raw or DEFAULT_BACKEND).strip().lower()
    if raw is None:
        source = f"code default ({DEFAULT_BACKEND!r}); LLM_BACKEND unset"
    else:
        source = report.source_of("LLM_BACKEND")

    cache_on = _cache_enabled()
    common = dict(
        backend=backend,
        backend_source=source,
        cache_enabled=cache_on,
        cache_path=str(LLMCache().path),
        dotenv_path=str(report.path),
        dotenv_exists=report.exists,
        raw_backend_value=raw,
    )
    if backend == "local":
        host = (
            os.environ.get("OLLAMA_HOST") or OLLAMA_DEFAULT_HOST
        ).rstrip("/")
        return LLMResolution(
            agent_model=OLLAMA_AGENT_MODEL,
            judge_model=OLLAMA_JUDGE_MODEL,
            ollama_host=host,
            **common,
        )
    if backend == "gemini":
        return LLMResolution(
            agent_model=GEMINI_AGENT_MODEL,
            judge_model=_gemini_judge_model(),
            gemini_key_status=_gemini_key_status(),
            **common,
        )
    return LLMResolution(
        agent_model="?", judge_model="?", **common
    )


def make_llm(*, cache: bool = True) -> LLMClient:
    """Build the LLM client for the configured backend (default: local/Ollama)."""
    res = resolve_config()
    shared_cache = LLMCache() if (cache and res.cache_enabled) else None
    if res.backend == "local":
        return OllamaBackend(cache=shared_cache)
    if res.backend == "gemini":
        return GeminiBackend(cache=shared_cache)
    raise ValueError(
        f"LLM_BACKEND must be 'local' or 'gemini' (got {res.raw_backend_value!r})"
    )


def make_judge_llm(*, cache: bool = True) -> LLMClient:
    """Build the client for the judge role (detector NLI + reconciler).

    Follows ``LLM_BACKEND`` unless ``JUDGE_BACKEND`` names a different backend,
    so agents can stay on local Ollama while a stronger model judges.
    """
    res = resolve_config()
    if res.judge_backend is None:
        return make_llm(cache=cache)
    shared_cache = LLMCache() if (cache and res.cache_enabled) else None
    if res.judge_backend == "local":
        return OllamaBackend(cache=shared_cache)
    if res.judge_backend == "gemini":
        return GeminiBackend(cache=shared_cache)
    raise ValueError(
        f"JUDGE_BACKEND must be 'same', 'local' or 'gemini' (got {res.judge_backend!r})"
    )
