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
  magnitude / staleness). Agent model is `AGENT_MODEL` in that file — set to
  `gemini-2.5-flash` because the requested `gemini-2.0-flash` is retired on the
  API; judge stays `gemini-2.5-pro`.
- **`common/env.py`** — minimal `.env` loader (no `python-dotenv` dep).
- **`baselines/last_write_wins.py`** — newest write wins; the trivial resolver
  that makes the pipeline end-to-end.
- **`demo.py`** — runs the agents (real Gemini calls), prints the contradiction
  in memory, then last-write-wins "resolving" it.
- **`tests/test_store.py`** — pytest coverage of the store + conflict scan +
  last-write-wins.

Everything else in the tree is a **stub with a TODO** describing its milestone.

## Setup

```
python -m venv venv
venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
```

Then put your key in a `.env` file at the repo root (gitignored):

```
GEMINI_API_KEY=AIza...
```

(or export `GEMINI_API_KEY` in the shell — a real env var wins over `.env`
only if you call `load_dotenv(override=False)`).

## Run

```
python demo.py        # real gemini-2.0-flash calls; writes demo_memory.json
pytest                # store / conflict-scan tests (no API calls)
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
