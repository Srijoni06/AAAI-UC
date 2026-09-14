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

**Done.** The full evaluation pipeline is working end-to-end:
- Shared memory store with enforced status state machine + SQLite/WAL persistence + rich provenance.
- Two-stage contradiction detection: embedding candidate clustering + LLM-judge NLI classification.
- Conflict classification: deterministic scope check (COORDINATION vs CREDIBILITY) + LLM fallback.
- Baselines: last-write-wins, majority vote, static confidence.
- Evaluation harness: runs all conditions against the 20-document seeded suite, produces detection
  P/R/F1, resolution accuracy, per-type breakdowns, and escalation rate.
- 133 offline tests (no network).

**Not built yet.** The reliability engine — `reliability/peer_memory.py` and
`reliability/resolver.py` — is the novel contribution (Srijoni's lead).

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

- **`tests/`** — 133 tests, all offline:
  - `test_store.py` (48) — schema, provenance, state machine, concurrent writes.
  - `test_seed_conflicts.py` (17) — suite structure and gold labels.
  - `test_orchestrator.py` (17) — 5-agent offline run, grouping, provenance.
  - `test_detector.py` (18) — two-stage detection, paraphrase rejection.
  - `test_baselines.py` (14) — majority vote clustering, static confidence scoring.
  - `test_reconciler.py` (8) — deterministic scope check, LLM classification.
  - `test_run_comparison.py` (8) — full pipeline, accuracy, detection metrics.
  - `test_llm_cache.py` (11) — cache + backend routing.

## Evaluation Results (offline, fake backend, 20 docs)

| Condition          | Accuracy | Correct | Contested | Escalation |
|--------------------|----------|---------|-----------|------------|
| null (control)     | 100.0%   | 1/1     | 19        | 95.0%      |
| last_write_wins    | 100.0%   | 20/20   | 0         | 0.0%       |
| majority_vote      | 5.0%     | 1/20    | 0         | 0.0%       |
| static_confidence  | 100.0%   | 20/20   | 0         | 0.0%       |

**Detection:** P=0.950, R=1.000, F1=0.974 (6 false positives from COEXIST seed).

**Why these numbers matter for the paper:**
- **Majority vote at 5%** is the strawman — the 3-agent correlated group
  always outvotes the 2 independent agents, confirming the wrong answer in
  19/20 cases. This is the exact failure mode the reliability engine targets.
- **LWW and static confidence at 100%** is an honest baseline ceiling on this
  suite — agent_E (last writer) happens to always be on the gold excerpt, and
  static confidence reduces to recency because all writes share identical
  provenance. The novel contribution's job is to match or exceed this on
  harder suites where last-writer luck doesn't hold.
- **Null at 100%/1 decisive** shows the reconciler correctly identifies
  doc-languages as COORDINATION (both claims true) and keeps both.

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
pytest                             # 133 tests: all modules, no network
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
  peer_memory.py   # [stub]  online per-agent competence + pairwise correlation
  resolver.py      # [stub]  reliability-weighted resolver (our contribution — Srijoni)
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
  test_run_comparison.py    # [done]   8 tests
  test_llm_cache.py         # [done]  11 tests
```

## Next

1. **The novel contribution** — `reliability/peer_memory.py` (online per-agent
   competence + pairwise correlation estimates) and `reliability/resolver.py`
   (a resolver that weights each claim by its source's reliability *and
   discounts agreement between correlated agents*). This is Srijoni's lead.
2. **Real LLM evaluation** — run the harness with `--backend real` (Ollama or
   Gemini) to measure performance when the judge and agent LLMs produce
   non-trivial claims. The fake backend validates pipeline correctness;
   real LLM runs produce the research numbers.
3. **Rotate anchor excerpt index** per seed in the orchestrator so the
   correlated group is wrong ~half the time (currently always reads excerpt 0).
   This makes the majority-vote failure rate less uniform and gives the
   reliability engine a fairer comparison surface.
4. **AAAI UC paper** — finalize figures, draft the 2-page summary, and prepare
   the poster + supplementary code/checkpoint release.
