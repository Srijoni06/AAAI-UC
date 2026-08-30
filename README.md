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

- **`memory/store.py`** — the conflict-aware item schema and store API:
  - `MemoryItem` — `id, agent_id, topic, content, embedding, timestamp,
    status`, plus provenance: `source_type` (`SourceType`), `authority`
    (`Authority`, ordered), `origin` (`Origin`: tool/retrieval vs model
    inference), `evidence_span`, `version`, `source_doc_id`, `metadata`.
  - `Status` — `PROPOSED / CONFIRMED / CONTESTED / SUPERSEDED` with an
    **enforced state machine**: PROPOSED is entry-only, SUPERSEDED is terminal;
    an illegal move (e.g. `SUPERSEDED -> PROPOSED`) raises
    `InvalidTransitionError` instead of silently applying.
  - `MemoryStore` ABC (`add / get / list / update_status / clear` + a concrete
    naive `list_conflicts` same-topic / differing-content scan) and `Conflict`
    (>=2 live items on one topic that disagree).
  - Backends: **`SqliteMemoryStore` (WAL mode) is the default** — via
    `open_store(...)`, used by `demo.py` — for safe concurrent writes;
    `JsonMemoryStore` is kept only as a legacy/compatibility option.
- **`baselines/base.py`** — the `Resolution` / `Resolver` contract and
  `apply_resolution`. `Resolution` is now **multi-outcome**: it assigns
  `CONFIRMED` / `CONTESTED` / `SUPERSEDED` per item (via `ItemOutcome`), so a
  resolver can keep several claims that validly coexist, or leave a conflict
  unresolved — not just crown one winner. `winner_id` / `superseded_ids` remain
  as back-compat views.
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
- **`demo.py`** — runs the agents, prints the contradiction in shared memory
  (with a per-item provenance line: `source_type, authority, origin, version,
  source_doc_id`), then last-write-wins "resolving" it. Writes `demo_memory.db`
  (SQLite).
- **`tests/`** — pytest coverage of the store (`test_store.py`, 48 tests:
  schema + provenance round-trip, the status state machine incl.
  **invalid-transition blocking**, **concurrent SQLite writes**, the naive
  conflict scan, multi-outcome `Resolution`, and last-write-wins) and the LLM
  cache + backend routing (`test_llm_cache.py`).

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

### Troubleshooting (Windows)

- **`Activate.ps1` fails with a script-execution error:** run
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` once in that
  terminal, then retry activation.
- **Run everything from inside the project folder** (the one directly containing
  `demo.py` and `.env`). From a parent directory the tools pick up a different or
  missing `.env` and you get confusing config errors. Verify with `Get-Location`,
  and confirm the `.env` path in the config banner `demo.py` prints at startup.
- **Commands seem to run but changes don't show up, or `git log` looks stale:**
  you may be in a different copy of the repo than you think — a nested or
  duplicate git checkout one level up makes your terminal and editor tools
  silently disagree about which files are real. Check `Get-Location`, confirm
  you're in *this exact* project folder, and verify with `git status` and
  `git log --oneline` before trusting any "done" report.
- **`ollama` not recognized after install:** open a new terminal (PATH only
  refreshes in new sessions), or call it directly:
  `& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list`
- **`ollama pull llama3.1:8b` fails mid-download** (~4.9 GB; e.g. a DNS error):
  just rerun it — it resumes rather than restarting from zero.
- **Never paste a real `GEMINI_API_KEY` anywhere but your own `.env`** (not chat,
  commits, or issues); treat any key pasted elsewhere as compromised and
  regenerate it at aistudio.google.com/apikey.

## Run

```
python demo.py        # uses LLM_BACKEND (default local); writes demo_memory.db (SQLite)
pytest                # store, conflict-scan, and LLM-cache tests (no network)
```

`demo_memory.db` (and its `-wal` / `-shm` sidecars) is gitignored, as
`demo_memory.json` was before.

## Layout

```
memory/
  store.py         # [done]  schema + status state machine + SQLite(WAL)/JSON backends + naive list-conflicts
  detector.py      # [stub]  embedding clustering + LLM-judge verification
  reconciler.py    # [stub]  CREDIBILITY vs COORDINATION classification
reliability/
  peer_memory.py   # [stub]  online competence + pairwise correlation
  resolver.py      # [stub]  reliability-weighted credibility resolver (our contribution)
baselines/
  base.py                # [done]  multi-outcome Resolution / Resolver contract + apply_resolution
  last_write_wins.py     # [done]
  majority_vote.py       # [stub]
  static_confidence.py   # [stub]  fixed weights, no learning (primary comparison baseline)
agents/
  orchestrator.py  # [done]  minimal 2-agent loop + hardcoded documents (unchanged; 3-5 agent + provenance pass is next)
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

`agents/orchestrator.py` is still the untouched 2-agent milestone-1 loop. Its
next change is a single combined pass: scale to 3-5 agents (including
deliberately correlated ones that share source/prompt/model) **and** wire real
provenance (`source_type`, `origin`, `evidence_span`, `source_doc_id`) into the
items it writes.

1. `domain/seed_conflicts.py` — the 15-20 seed suite with gold labels.
2. `memory/detector.py` — sentence-transformers embedding clustering +
   `gemini-2.5-pro` judge to replace the naive `list_conflicts` scan
   (needs `pip install sentence-transformers`).
3. `baselines/majority_vote.py`, `baselines/static_confidence.py`.
4. `memory/reconciler.py` + `reliability/peer_memory.py` +
   `reliability/resolver.py` — the actual contribution.
5. `eval/run_comparison.py` — resolution accuracy + downstream task accuracy
   across all four conditions.
