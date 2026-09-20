"""Tests for the LLM response cache and backend routing (no network)."""

from __future__ import annotations

import pytest

from pathlib import Path

from common.cache import LLMCache
from common.env import DotenvReport, load_dotenv
from common.llm import (
    GEMINI_JUDGE_MODEL,
    GeminiBackend,
    LLMClient,
    make_judge_llm,
    make_llm,
    resolve_config,
)


def _no_dotenv(*_a, **_k):
    """Stand-in for load_dotenv: behaves as if no .env file exists."""
    return DotenvReport(path=Path("nonexistent.env"), exists=False, override=True)


class CountingBackend(LLMClient):
    backend = "fake"
    agent_model = "fake-agent"
    judge_model = "fake-judge"

    def __init__(self, cache):
        super().__init__(cache)
        self.calls = 0

    def _raw_generate(self, prompt, *, system, temperature, model):
        self.calls += 1
        return f"reply#{self.calls} to {prompt}"


def test_cache_hit_skips_backend(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    be = CountingBackend(cache)

    first = be.generate("hello", system="sys", temperature=0.2)
    second = be.generate("hello", system="sys", temperature=0.2)

    assert first == second
    assert be.calls == 1


def test_cache_key_sensitive_to_prompt_model_system_temp(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    be = CountingBackend(cache)

    be.generate("p1", system="s", temperature=0.2)
    be.generate("p2", system="s", temperature=0.2)  # different prompt
    be.generate("p1", system="s", temperature=0.9)  # different temperature
    be.generate("p1", system="other", temperature=0.2)  # different system
    be.generate("p1", system="s", temperature=0.2, model="fake-judge")  # diff model

    assert be.calls == 5


def test_cache_key_no_sample_id_is_backward_compatible():
    """Omitting sample_id must give the exact key the 5-field version produced."""
    common = dict(
        backend="local", model="m", system="s", temperature=0.2, prompt="p"
    )
    assert LLMCache.key(**common) == LLMCache.key(**common, sample_id=None)


def test_cache_key_distinct_per_sample_id():
    common = dict(
        backend="local", model="m", system="s", temperature=0.2, prompt="p"
    )
    k_none = LLMCache.key(**common)
    k0 = LLMCache.key(**common, sample_id=0)
    k1 = LLMCache.key(**common, sample_id=1)
    k0_str = LLMCache.key(**common, sample_id="0")
    assert len({k_none, k0, k1}) == 3
    assert k0 != k0_str  # int 0 and str "0" are different samples


def test_repeated_sampling_rehits_backend_then_caches_each_draw(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    be = CountingBackend(cache)

    s0 = be.generate("hello", system="s", temperature=0.7, sample_id=0)
    s1 = be.generate("hello", system="s", temperature=0.7, sample_id=1)
    s2 = be.generate("hello", system="s", temperature=0.7, sample_id=2)
    assert be.calls == 3
    assert len({s0, s1, s2}) == 3  # three independent draws

    # re-asking a used sample_id is served from cache (no new call)
    again = be.generate("hello", system="s", temperature=0.7, sample_id=1)
    assert be.calls == 3
    assert again == s1


def test_default_call_still_single_answer_alongside_samples(tmp_path):
    cache = LLMCache(tmp_path / "c.json")
    be = CountingBackend(cache)

    first = be.generate("hello", system="s", temperature=0.2)
    be.generate("hello", system="s", temperature=0.2, sample_id=0)  # separate key
    second = be.generate("hello", system="s", temperature=0.2)
    assert first == second  # the no-sample_id request is still cached
    assert be.calls == 2  # one for the default key, one for sample_id=0


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.json"
    be1 = CountingBackend(LLMCache(path))
    be1.generate("hello", system="s", temperature=0.2)

    be2 = CountingBackend(LLMCache(path))
    be2.generate("hello", system="s", temperature=0.2)
    assert be2.calls == 0


def test_cache_disabled_always_calls_backend(tmp_path):
    be = CountingBackend(None)
    be.generate("hello", system="s", temperature=0.2)
    be.generate("hello", system="s", temperature=0.2)
    assert be.calls == 2


def test_make_llm_defaults_to_local(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.setattr("common.llm.load_dotenv", _no_dotenv)
    llm = make_llm(cache=False)
    assert llm.backend == "local"
    assert llm.agent_model == "llama3.1:8b"


def test_make_llm_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setattr("common.llm.load_dotenv", _no_dotenv)
    with pytest.raises(ValueError):
        make_llm(cache=False)


def test_dotenv_overrides_shell_backend(monkeypatch, tmp_path):
    """A shell LLM_BACKEND=gemini must lose to LLM_BACKEND=local in .env."""
    monkeypatch.setenv("LLM_BACKEND", "gemini")
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_BACKEND=local\n", encoding="utf-8")
    monkeypatch.setattr("common.llm.load_dotenv", lambda *a, **k: load_dotenv(env_file))
    res = resolve_config()
    assert res.backend == "local"
    assert ".env" in res.backend_source


# --------------------------------------------------------------------------- #
# judge-role routing (JUDGE_BACKEND) - agents stay on the main backend
# --------------------------------------------------------------------------- #
def test_judge_follows_main_backend_when_unset(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("JUDGE_BACKEND", raising=False)
    monkeypatch.setattr("common.llm.load_dotenv", _no_dotenv)
    assert resolve_config().judge_backend is None
    assert make_judge_llm(cache=False).backend == "local"


def test_judge_backend_same_is_a_noop(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("JUDGE_BACKEND", "same")
    monkeypatch.setattr("common.llm.load_dotenv", _no_dotenv)
    assert resolve_config().judge_backend is None


def test_judge_backend_gemini_only_moves_the_judge_role(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("JUDGE_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr("common.llm.load_dotenv", _no_dotenv)

    agents = make_llm(cache=False)
    judge = make_judge_llm(cache=False)

    assert agents.backend == "local" and agents.agent_model == "llama3.1:8b"
    assert judge.backend == "gemini" and judge.judge_model == GEMINI_JUDGE_MODEL
    res = resolve_config()
    assert res.judge_backend == "gemini" and res.judge_model == GEMINI_JUDGE_MODEL
    assert res.agent_model == "llama3.1:8b"  # agents unchanged
    assert "JUDGE_BACKEND   : gemini" in res.banner()


def test_judge_backend_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("JUDGE_BACKEND", "openai")
    monkeypatch.setattr("common.llm.load_dotenv", _no_dotenv)
    with pytest.raises(ValueError, match="JUDGE_BACKEND"):
        make_judge_llm(cache=False)


# --------------------------------------------------------------------------- #
# Gemini retry policy (free-tier 429s and 5xx retry; other 4xx do not)
# --------------------------------------------------------------------------- #
class _FlakyModels:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def generate_content(self, *, model, contents, config):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)

        class _Resp:
            text = "ok"

        return _Resp()


def _gemini_with(models, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setattr("common.llm.time.sleep", lambda *_: None)
    be = GeminiBackend(cache=None, max_retries=3)
    be._client = type("C", (), {"models": models})()
    return be


def _api_error(cls_name, code):
    from google.genai import errors

    return getattr(errors, cls_name)(code, {"error": {"code": code, "message": "x", "status": "X"}})


def test_gemini_retries_rate_limit_then_succeeds(monkeypatch):
    models = _FlakyModels([_api_error("ClientError", 429), _api_error("ServerError", 503)])
    be = _gemini_with(models, monkeypatch)
    assert be.generate("hi", system="s", temperature=0.0, model="m") == "ok"
    assert models.calls == 3


def test_gemini_does_not_retry_other_client_errors(monkeypatch):
    from google.genai import errors

    models = _FlakyModels([_api_error("ClientError", 400)])
    be = _gemini_with(models, monkeypatch)
    with pytest.raises(errors.ClientError):
        be.generate("hi", system="s", temperature=0.0, model="m")
    assert models.calls == 1


def test_gemini_gives_up_after_max_retries(monkeypatch):
    from google.genai import errors

    models = _FlakyModels([_api_error("ClientError", 429)] * 5)
    be = _gemini_with(models, monkeypatch)
    with pytest.raises(errors.ClientError):
        be.generate("hi", system="s", temperature=0.0, model="m")
    assert models.calls == 3  # max_retries


def test_gemini_judge_model_env_override(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("JUDGE_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_JUDGE_MODEL", "gemini-3.5-flash")
    monkeypatch.setattr("common.llm.load_dotenv", _no_dotenv)
    assert make_judge_llm(cache=False).judge_model == "gemini-3.5-flash"
    assert resolve_config().judge_model == "gemini-3.5-flash"


def test_gemini_fails_fast_on_daily_quota(monkeypatch):
    from google.genai import errors

    daily = errors.ClientError(
        429,
        {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                   "message": "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier"}},
    )
    models = _FlakyModels([daily])
    be = _gemini_with(models, monkeypatch)
    with pytest.raises(errors.ClientError):
        be.generate("hi", system="s", temperature=0.0, model="m")
    assert models.calls == 1  # no pointless backoff on a per-day quota
