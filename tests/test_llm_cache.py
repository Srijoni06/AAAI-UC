"""Tests for the LLM response cache and backend routing (no network)."""

from __future__ import annotations

import pytest

from common.cache import LLMCache
from common.llm import LLMClient, make_llm


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
    monkeypatch.setattr("common.llm.load_dotenv", lambda *a, **k: None)
    llm = make_llm(cache=False)
    assert llm.backend == "local"
    assert llm.agent_model == "llama3.1:8b"


def test_make_llm_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setattr("common.llm.load_dotenv", lambda *a, **k: None)
    with pytest.raises(ValueError):
        make_llm(cache=False)
