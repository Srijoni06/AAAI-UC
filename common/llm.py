"""LLM backend abstraction: local (Ollama) or Gemini, with response caching.

Backend is chosen by the ``LLM_BACKEND`` env var / ``.env`` entry:
  - ``local``  (default) -> Ollama at ``OLLAMA_HOST`` (default localhost:11434),
                             model ``llama3.1:8b``. Used for all dev/testing.
  - ``gemini``            -> google-genai, gemini-2.5-flash (agent) /
                             gemini-2.5-pro (judge). For final verification runs.

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
OLLAMA_AGENT_MODEL = "llama3.1:8b"
OLLAMA_JUDGE_MODEL = "llama3.1:8b"  # only one local model for now
GEMINI_AGENT_MODEL = "gemini-2.5-flash"  # gemini-2.0-flash is retired on the API
GEMINI_JUDGE_MODEL = "gemini-2.5-pro"

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
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"Ollama HTTP {e.code} for model {model!r}: {detail}. "
                f"Pull it with `ollama pull {model}`."
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama at {self.host} is unreachable ({e.reason}). "
                f"Start it with `ollama serve` and pull `{model}`."
            ) from e
        return payload.get("response", "")


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
            except genai_errors.ServerError:
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
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
        if self.backend == "gemini":
            lines.append(f"  GEMINI_API_KEY  : {self.gemini_key_status}")
        return "\n".join(lines)


def resolve_config() -> LLMResolution:
    """Load .env and report the backend/model that ``make_llm`` would pick."""
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
        key = os.environ.get("GEMINI_API_KEY", "")
        status = (
            "placeholder / missing" if key in _PLACEHOLDER_KEYS else f"set (len {len(key)})"
        )
        return LLMResolution(
            agent_model=GEMINI_AGENT_MODEL,
            judge_model=GEMINI_JUDGE_MODEL,
            gemini_key_status=status,
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
