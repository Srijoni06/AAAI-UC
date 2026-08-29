"""Tests for the LLM response cache and backend routing (no network)."""

from __future__ import annotations

import pytest

from pathlib import Path

from common.cache import LLMCache
from common.env import DotenvReport, load_dotenv
from common.llm import LLMClient, make_llm, resolve_config


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
