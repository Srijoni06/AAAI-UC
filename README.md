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

## Status

**Done.** The full pipeline, including the novel contribution, is working end-to-end:
- Shared memory store with enforced status state machine + SQLite/WAL persistence + rich provenance.
- Two-stage contradiction detection: embedding candidate clustering + LLM-judge NLI classification.
- Conflict classification: deterministic scope check (COORDINATION vs CREDIBILITY) + LLM fallback.
- Baselines: last-write-wins, majority vote, static confidence.
- Reliability engine (the novel contribution): online per-agent competence + pairwise
  peer-correlation tracking, used as the credibility-weighting function for CREDIBILITY conflicts.
- Evaluation harness: runs all conditions — including the reliability-aware resolver — against the
  20-document seeded suite, produces detection P/R/F1, resolution accuracy, per-type breakdowns,
  and escalation rate.
- 150 offline tests (no network).

**Not built at the module level.** What's left is evaluation quality, not code: running the
harness against a real LLM backend so the reliability engine has genuine per-agent competence
variance to learn from, rather than the fake backend's deterministic claims (see **Next**).

### What works now

- **`memory/store.py`** — the conflict-aware memory store:
  - **Enums.** `Status` (`PROPOSED` entry state → `CONFIRMED` / `CONTESTED` /
    `SUPERSEDED`); `SourceType` (`retrieval` / `tool` / `user` / `model` /
    `unknown`); `Authority` (ordered: `UNKNOWN < LOW < MEDIUM < HIGH <
    AUTHORITATIVE`); `Origin` (`tool` = externally grounded vs `inference` =
    the model's own synthesis).
  - **`MemoryItem`** — one claim written by one agent: `id, agent_id, topic,
    content, embedding, timestamp, status`, plus provenance `source_type,
    authority, origin, evidence_span, version, source_doc_id, metadata`.
  - **`Conflict`** — ≥2 live (non-`SUPERSEDED`) items on one topic whose
    normalized content disagrees.
  - **Enforced status lifecycle.** The transitions are a small state machine:
    `PROPOSED` is entry-only, `SUPERSEDED` is terminal. An illegal move (e.g.
    `SUPERSEDED → PROPOSED`) raises **`InvalidTransitionError`** instead of
    silently mutating the item.
  - **`MemoryStore` ABC** with two backends: **`SqliteMemoryStore` — WAL
    mode, the default** (safe for concurrent writers); **`JsonMemoryStore`**
    kept only for small fixtures / legacy.

- **`memory/detector.py`** — two-stage contradiction detection:
  - **Stage 1:** Semantic similarity filtering via sentence-transformers.
    Items on the same topic are embedded and compared pairwise via cosine
    similarity. Pairs exceeding a similarity threshold (default 0.5) become
    candidate conflicts.
  - **Stage 2:** LLM-judge NLI classification. Each candidate pair is
    classified into exactly one of ENTAILMENT (paraphrase), CONTRADICTION
    (genuine conflict), or NEUTRAL (same topic, different aspects).

- **`memory/reconciler.py`** — conflict classification before resolution:
  - **Deterministic layer:** `pairwise_scopes_agree` — when both claims cite
    the same evidence_span, the disagreement is scope confusion → COORDINATION.
  - **LLM layer:** For pairs with genuinely different evidence, the judge LLM
    classifies CREDIBILITY (one should win) vs COORDINATION (both valid).
  - **Fallback:** Unparseable LLM response → CREDIBILITY (decisive default).

- **`baselines/`** — three baseline resolvers:
  - **`last_write_wins.py`** — newest claim wins (trivial).
  - **`majority_vote.py`** — cluster claims by meaning (provenance-aware);
    largest cluster wins. Ties → all CONTESTED. **Key finding:** treats
    correlated group agreement as independent confirmation — the signature
    failure mode our contribution targets.
  - **`static_confidence.py`** — fixed weight table over authority, origin,
    source_type, capped corroboration bonus per distinct agent group, and
    recency tie-break. Stateless. **Key finding:** when all agents carry
    identical provenance, corroboration by distinct groups + recency = same as
    LWW.

- **`reliability/`** — the novel contribution:
  - **`peer_memory.py`** — `PeerMemory`: per-agent competence (EMA from a 0.5
    prior toward the empirical confirmation rate) and pairwise correlation
    (do two agents tend to agree/disagree together, i.e. share a bias?),
    both updated from resolution outcomes.
  - **`resolver.py`** — `ReliabilityResolver`: scores each claim as provenance
    base weight + competence boost, then applies a correlation discount
    across agents who share an answer, so agreement between historically
    correlated agents counts for less than agreement between independents.
    Updates `PeerMemory` after every decision; auto-plugs into
    `eval/run_comparison.py` when present.

- **`agents/orchestrator.py`** — the **5-agent** summarization loop over the
  seed suite. `DEFAULT_ROSTER`:
  - `agent_A`, `agent_C`, `agent_D` — deliberately correlated group (`grp_A`).
  - `agent_B`, `agent_E` — independent.

- **`domain/seed_conflicts.py`** — the **20-document seeded-contradiction
  suite** with gold labels (4 conflict types × difficulty spread).

- **`eval/run_comparison.py`** — the comparison harness:
  - Runs agents once, detects contradictions once, replays per condition.
  - Conditions: null (no resolution), LWW, majority vote, static confidence.
  - Auto-plugs in reliability resolver when available.
  - Metrics: detection P/R/F1, resolution accuracy, per-type breakdown,
    escalation rate.
  - Outputs `results/run_comparison.json` + `results/summary.md`.

- **`eval/fake_backend.py`** — deterministic offline LLM backends + hash-based
  embedder. No network, no model download. `ScopedFakeLLM` (agent + judge),
  `FakeEmbedder`.

- **`tests/`** — 150 tests, all offline:
  - `test_store.py` (48) — schema, provenance, state machine, concurrent writes.
  - `test_seed_conflicts.py` (17) — suite structure and gold labels.
  - `test_orchestrator.py` (17) — 5-agent offline run, grouping, provenance.
  - `test_detector.py` (18) — two-stage detection, paraphrase rejection.
  - `test_baselines.py` (14) — majority vote clustering, static confidence scoring.
  - `test_reconciler.py` (8) — deterministic scope check, LLM classification.
  - `test_reliability.py` (17) — competence EMA, correlation tracking, resolver scoring.
  - `test_run_comparison.py` (8) — full pipeline, accuracy, detection metrics.
  - `test_llm_cache.py` (11) — cache + backend routing.

## Evaluation Results (offline, fake backend, 20 docs)

| Condition          | Accuracy | Correct | Contested | Escalation |
|--------------------|----------|---------|-----------|------------|
| null (control)     | 100.0%   | 1/1     | 19        | 95.0%      |
| last_write_wins    | 65.0%    | 13/20   | 0         | 0.0%       |
| majority_vote      | 40.0%    | 8/20    | 0         | 0.0%       |
| static_confidence  | 65.0%    | 13/20   | 0         | 0.0%       |
| reliability_aware  | 60.0%    | 12/20   | 0         | 0.0%       |

Captured after the anchor-rotation fix (`agents/orchestrator.rotated_anchor_index`,
below) via `python -m eval.run_comparison` on the fake backend. Rotating the anchor
per seed removed the artificial 100% ceiling LWW/static-confidence used to sit at —
their real accuracy on this suite is 65%.

**Detection:** P=0.950, R=1.000, F1=0.974 (6 false positives from COEXIST seed).

**Why these numbers matter for the paper — and what they don't show yet:**
- **Majority vote at 40%** (up from a pre-fix 5%) still clearly loses to every
  other condition: the 3-agent correlated group's block vote wins the
  plurality regardless of whether it's on the gold excerpt, so majority vote
  is wrong whenever the group happens to be wrong. This is the exact failure
  mode the reliability engine targets.
- **reliability_aware at 60%** beats majority_vote by 20 points, but on *this*
  run it trails LWW/static-confidence (65%) rather than leading them. That is
  an honest result, not a setback to hide: the fake backend is a fixed
  deterministic function of excerpt text, so there is no real per-agent
  competence signal for `PeerMemory` to learn from across resolutions — it can
  only exploit the correlation signal, not the competence one. The paper's
  actual claim (reliability-aware beats static provenance weighting) needs
  the real-LLM run in **Next**, where agents genuinely vary in how well they
  read an excerpt.
- **Null at 100%/1 decisive** shows the reconciler correctly identifies
  doc-languages as COORDINATION (both claims true) and keeps both; it is not
  a meaningful comparison point since it only ever scores that one case.

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
python demo.py                     # 5 agents over a 4-doc subset (one per conflict type)
python demo.py --all               # ... over all 20 seed documents
pytest                             # 150 tests: all modules, no network
python -m eval.run_comparison      # offline: all 20 docs, fake backend
python -m eval.run_comparison --limit 4   # quick smoke test
python -m eval.run_comparison --backend real   # Ollama/Gemini + real embeddings
```

## Layout

```
memory/
  store.py         # [done]  schema + state machine + SQLite(WAL) + naive list-conflicts
  detector.py      # [done]  two-stage: embedding candidate clustering + LLM-judge NLI
  reconciler.py    # [done]  CREDIBILITY vs COORDINATION classification (deterministic + LLM)
reliability/
  peer_memory.py   # [done]  online per-agent competence + pairwise correlation
  resolver.py      # [done]  reliability-weighted resolver (our contribution — Srijoni)
baselines/
  base.py                # [done]  multi-outcome Resolution / Resolver contract
  last_write_wins.py     # [done]
  majority_vote.py       # [done]  meaning-cluster voting; largest cluster wins
  static_confidence.py   # [done]  fixed provenance weight table + capped corroboration
agents/
  orchestrator.py  # [done]  5-agent loop (grp_A correlated + B/E independent) + real provenance
common/
  env.py           # [done]  .env loader
  llm.py           # [done]  local (Ollama) / gemini backend + selection; generate(sample_id=...)
  cache.py         # [done]  on-disk LLM response cache; optional sample_id key widening
domain/
  seed_conflicts.py # [done]  20 seeded contradictions w/ gold labels (type + difficulty)
eval/
  fake_backend.py      # [done]  ScopedFakeLLM + FakeEmbedder (offline, deterministic)
  run_comparison.py    # [done]  baselines vs full model: resolution accuracy + detection P/R/F1
tests/
  test_store.py             # [done]  48 tests
  test_seed_conflicts.py    # [done]  17 tests
  test_orchestrator.py      # [done]  17 tests
  test_detector.py          # [done]  18 tests
  test_baselines.py         # [done]  14 tests
  test_reconciler.py        # [done]   8 tests
  test_reliability.py       # [done]  17 tests
  test_run_comparison.py    # [done]   8 tests
  test_llm_cache.py         # [done]  11 tests
```

## Next

1. **Real LLM evaluation** — run the harness with `--backend real` (Ollama or
   Gemini) to measure performance when the judge and agent LLMs produce
   non-trivial claims. This is now the priority: on the fake backend
   `reliability_aware` trails LWW/static-confidence (see Evaluation Results)
   because a deterministic fake LLM gives `PeerMemory` no real competence
   signal to learn from, only the correlation signal. The paper's central
   claim needs agents that actually vary in reading-comprehension quality.
2. **AAAI UC paper** — finalize figures, draft the 2-page summary, and prepare
   the poster + supplementary code/checkpoint release.
