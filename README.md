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

## Status: milestone 2 (agents, seed suite, provenance)

Milestone 1 built the end-to-end skeleton (store + state machine + 2-agent loop).
Milestone 2 replaces the hardcoded testbed with a real seed suite, scales the
agent loop to five agents (some **deliberately correlated**), wires real
provenance into every write, and makes the LLM cache safe for repeated sampling.

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
- **`domain/seed_conflicts.py`** — the seeded contradiction suite: **20 source
  documents**, each with 2+ excerpts that describe the same fact differently, and
  a **gold label** (`gold_excerpt_id` / `gold_answer`, `conflict_type` ∈
  `factual | magnitude | staleness | provenance`, `difficulty` ∈
  `obvious | moderate | subtle`). One seed's gold is `COEXIST` (both claims are
  legitimately true). The original 3 milestone-1 documents are seeds 1–3.
- **`agents/orchestrator.py`** — **5 summarizer agents** over the seed suite.
  `agent_A/C/D` form a correlated group `grp_A` (same model, near-identical
  prompt, and they read the *same* excerpt) so their agreement is shared bias,
  not independent verification; `agent_B` and `agent_E` are independent
  (different prompts, spread over the other excerpts). The grouping is
  configurable — pass a custom `roster` of `AgentSpec` to `run(...)`; `agent_
  groups()` / `correlated_groups()` report it so eval can compare "N independent
  agree" vs "N correlated agree". Every write carries **real provenance**:
  `source_type=RETRIEVAL`, `origin=TOOL`, `evidence_span` = the verbatim excerpt
  slice the agent read, `source_doc_id`, and `metadata` (`excerpt_id`, `section`,
  `source_id = "<doc>#<excerpt>"`, `agent_group`).
- **`common/llm.py`** — LLM backend abstraction: `local` (Ollama, `llama3.1:8b`,
  default for all dev/testing) or `gemini` (`gemini-2.5-flash` agent /
  `gemini-2.5-pro` judge — requested `gemini-2.0-flash` is retired on the API).
  Selected by `LLM_BACKEND`. Every call is cached on disk first
  (`.cache/llm_cache.json`, keyed by SHA-256 of backend+model+system+temp+prompt;
  disable with `LLM_CACHE=0`). `generate(..., sample_id=...)` widens the key so
  the same prompt can be **deliberately re-sampled** (`0, 1, 2, ...`) and each
  draw cached separately; with no `sample_id` the key and behaviour are
  unchanged (one cached answer per identical request).
- **`common/env.py`** — minimal `.env` loader (no `python-dotenv` dep).
- **`baselines/last_write_wins.py`** — newest write wins; the trivial resolver
  that makes the pipeline end-to-end.
- **`demo.py`** — runs the 5 agents over a 4-document subset of the suite (one
  per conflict type; `--all` for all 20), prints every agent's write **grouped
  by correlated group vs independent** (marking which agents landed on the gold
  excerpt), then the
  detected contradictions and last-write-wins "resolving" them, with a per-item
  provenance line. Writes `demo_memory.db` (SQLite).
- **`tests/`** — pytest, no network, 93 tests:
  - `test_store.py` (48) — schema + provenance round-trip, the status state
    machine incl. **invalid-transition blocking**, **concurrent SQLite writes**,
    the naive conflict scan, multi-outcome `Resolution`, last-write-wins.
  - `test_seed_conflicts.py` (17) — suite is well-formed: 2+ excerpts each,
    valid `gold_excerpt_id`, unique doc ids / topics, all four conflict types
    present, difficulty spread, the original 3 seeds retained.
  - `test_orchestrator.py` (17) — 5-agent run with a fake offline LLM: grouping
    + configurable roster, excerpt assignment, correlated agents produce
    identical claims, and **real provenance on every write**.
  - `test_llm_cache.py` (11) — cache + backend routing, incl. the `sample_id`
    key widening and repeated-sampling behaviour.

`memory/detector.py`, `memory/reconciler.py`, `reliability/*`,
`baselines/majority_vote.py`, `baselines/static_confidence.py`, and
`eval/run_comparison.py` are still **stubs with a TODO** describing their
milestone.

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
python demo.py        # 5 agents on a 5-doc subset; uses LLM_BACKEND (default local)
python demo.py --all   # ... on all 20 seed documents
pytest                 # 93 tests: store, seed suite, orchestration, LLM cache (no network)
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
  orchestrator.py  # [done]  5-agent loop (grp_A correlated + B/E independent) + real provenance
common/
  env.py           # [done]  .env loader
  llm.py           # [done]  local (Ollama) / gemini backend + selection; generate(sample_id=...)
  cache.py         # [done]  on-disk LLM response cache; optional sample_id key widening
domain/
  seed_conflicts.py # [done]  20 seeded contradictions w/ gold labels (type + difficulty)
eval/
  run_comparison.py # [stub]  four-condition comparison + logging
tests/
```

## Next

1. `memory/detector.py` — sentence-transformers embedding clustering +
   `gemini-2.5-pro` judge to replace the naive `list_conflicts` scan
   (needs `pip install sentence-transformers`).
2. `baselines/majority_vote.py`, `baselines/static_confidence.py`.
3. `memory/reconciler.py` + `reliability/peer_memory.py` +
   `reliability/resolver.py` — the actual contribution.
4. `eval/run_comparison.py` — resolution accuracy + downstream task accuracy
   across all four conditions.
