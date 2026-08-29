# Trusting Agreement

Peer-correlation-aware reliability for conflict resolution in shared multi-agent memory.

Research prototype for an AAAI Undergraduate Consortium submission. The idea:
use an **online, peer-correlation-aware reliability signal** (Sigma-Mem-style) as
the credibility-weighting mechanism inside a **conflict-aware shared memory**
(LatticeMind-style `PROPOSED / CONFIRMED / CONTESTED / SUPERSEDED` structure),
replacing the static evidence-type weight table. The peer-correlation signal is
used to catch a failure mode neither prior system handles: two agents agreeing
because they share a bias, not because they independently verified something.

Evaluation domain: a multi-agent literature-summarization pipeline.

## Status: milestone 1 (end-to-end skeleton)

What works now:

- **`memory/store.py`** — memory-item schema (`id, agent_id, topic, content,
  embedding, timestamp, status, source_doc_id, metadata`) and an
  add / get / list / `list_conflicts` API over a swappable backend
  (`JsonMemoryStore` today; `MemoryStore` ABC so SQLite can slot in).
  `list_conflicts` is a **naive** same-topic / differing-content scan for now.
- **`agents/orchestrator.py`** — 2 summarizer agents each read a different
  excerpt of the same document and write a one-sentence claim to shared memory.
  3 hand-built documents, each seeding one genuine contradiction (factual /
  magnitude / staleness).
- **`common/llm.py`** — LLM backend abstraction: `local` (Ollama, `llama3.1:8b`,
  default for all dev/testing) or `gemini` (`gemini-2.5-flash` agent /
  `gemini-2.5-pro` judge — requested `gemini-2.0-flash` is retired on the API).
  Selected by `LLM_BACKEND`. Every call is cached on disk first
  (`.cache/llm_cache.json`, keyed by SHA-256 of backend+model+system+temp+prompt;
  disable with `LLM_CACHE=0`).
- **`common/env.py`** — minimal `.env` loader (no `python-dotenv` dep).
- **`baselines/last_write_wins.py`** — newest write wins; the trivial resolver
  that makes the pipeline end-to-end.
- **`demo.py`** — runs the agents, prints the contradiction in memory, then
  last-write-wins "resolving" it.
- **`tests/`** — pytest coverage of the store + conflict scan + last-write-wins
  (`test_store.py`) and the LLM cache + backend routing (`test_llm_cache.py`).

Everything else in the tree is a **stub with a TODO** describing its milestone.

## Setup

```
python -m venv venv
venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env               # then edit
```

**Local backend (default, used for all dev/testing):** install Ollama, then

```
ollama serve                      # runs the daemon at localhost:11434
ollama pull llama3.1:8b
```

**Gemini backend (final verification runs only):** set in `.env`

```
LLM_BACKEND=gemini
GEMINI_API_KEY=AIza...
```

`.env` values override the shell environment; a real env var only wins if the
loader is called as `load_dotenv(override=False)`. `demo.py` prints a config
banner at startup (which `.env` was read, the raw `LLM_BACKEND` value, whether a
shell var was overridden, resolved model, cache path) — check it if a run uses
the wrong backend. `python -c "from common.llm import resolve_config; print(resolve_config().banner())"`
shows the same without running anything.

## Run

```
python demo.py        # uses LLM_BACKEND (default local); writes demo_memory.json
pytest                # store, conflict-scan, and LLM-cache tests (no network)
```

## Layout

```
memory/
  store.py         # [done]  schema + add/read/list-conflicts
  detector.py      # [stub]  embedding clustering + LLM-judge verification
  reconciler.py    # [stub]  CREDIBILITY vs COORDINATION classification
reliability/
  peer_memory.py   # [stub]  online competence + pairwise correlation
  resolver.py      # [stub]  reliability-weighted credibility resolver (our contribution)
baselines/
  base.py                # [done]  Resolution / Resolver contract + apply_resolution
  last_write_wins.py     # [done]
  majority_vote.py       # [stub]
  static_confidence.py   # [stub]  fixed weights, no learning (primary comparison baseline)
agents/
  orchestrator.py  # [done]  minimal 2-agent loop + hardcoded documents
common/
  env.py           # [done]  .env loader
  llm.py           # [done]  local (Ollama) / gemini backend + selection
  cache.py         # [done]  on-disk LLM response cache
domain/
  seed_conflicts.py # [stub]  15-20 seeded contradictions w/ gold labels
eval/
  run_comparison.py # [stub]  four-condition comparison + logging
tests/
```

## Next

1. `domain/seed_conflicts.py` — the 15-20 seed suite with gold labels.
2. `memory/detector.py` — sentence-transformers embedding clustering +
   `gemini-2.5-pro` judge to replace the naive `list_conflicts` scan
   (needs `pip install sentence-transformers`).
3. `baselines/majority_vote.py`, `baselines/static_confidence.py`.
4. `memory/reconciler.py` + `reliability/peer_memory.py` +
   `reliability/resolver.py` — the actual contribution.
5. `eval/run_comparison.py` — resolution accuracy + downstream task accuracy
   across all four conditions.
