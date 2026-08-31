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

**Done.** The shared memory store is hardened (conflict-aware schema + an
*enforced* status state machine + SQLite/WAL persistence + rich provenance), the
resolution contract is multi-outcome, and the agent loop is a real **5-agent**
orchestration — including a deliberately **correlated** group — running over a
**20-document seeded-contradiction suite** with gold labels, writing real
provenance on every claim.

**Not built yet.** Conflict *detection* is still a naive same-topic /
differing-content scan (`list_conflicts`), not semantic contradiction detection.
The reliability signal and the peer-correlation-aware resolver — the novel
contribution — do not exist yet, nor do the other baselines or the evaluation
harness.

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
  - **`MemoryStore` ABC** (`add / get / list / update_status / clear` + a
    concrete `list_conflicts`) with two backends: **`SqliteMemoryStore` — WAL
    mode, the default** (via `open_store(...)`, used by `demo.py`), safe for
    concurrent writers; **`JsonMemoryStore`** kept only for small fixtures /
    legacy.
- **`baselines/base.py`** — the `Resolution` / `Resolver` contract. `Resolution`
  is **multi-outcome**: it assigns one of `CONFIRMED` / `CONTESTED` /
  `SUPERSEDED` to *every* item in a conflict (via `ItemOutcome`), so a resolver
  can keep several claims that legitimately coexist, or leave a conflict
  unresolved — not just crown a single winner. `apply_resolution` writes each
  outcome back through `update_status`, so the state machine still applies;
  `winner_id` / `superseded_ids` remain as back-compat views.
  `baselines/last_write_wins.py` is the trivial resolver built on it.
- **`agents/orchestrator.py`** — the **5-agent** summarization loop over the seed
  suite. Each agent reads one excerpt of a document and writes a one-sentence
  claim to shared memory. `DEFAULT_ROSTER`:
  - **`agent_A`, `agent_C`, `agent_D` — a deliberately correlated group
    (`grp_A`)**: the same underlying model, near-identical system prompts, and
    they all read the **same excerpt**. When `agent_A` misreads that excerpt,
    `agent_C` / `agent_D` tend to misread it the same way — three of them
    agreeing is shared bias, not independent verification.
  - **`agent_B`, `agent_E` — independent**: different system prompts, spread
    across the *other* excerpts, so they can genuinely contradict the group.
  - Grouping is configurable — pass a `roster` of `AgentSpec` to `run(...)`;
    `agent_groups()` / `correlated_groups()` / `independent_agents()` expose it
    so evaluation can contrast "N independent agents agree" against "N
    correlated agents agree" as different situations.
  - Every `store.add(...)` writes **real provenance**, not schema defaults:
    `source_type=RETRIEVAL`, `origin=TOOL` (the claim is grounded in a retrieved
    excerpt), `evidence_span` = the verbatim excerpt slice the agent actually
    read, `source_doc_id`, and `metadata["source_id"] = "<doc_id>#<excerpt_id>"`
    (with `excerpt_id`, `section`, `agent_group` alongside).
- **`domain/seed_conflicts.py`** — the **20-document seeded-contradiction
  suite**. Each `SeedConflict` has 2+ `Excerpt`s that state the same fact
  differently, plus a **gold label**: `gold_excerpt_id` (which excerpt is
  correct, or the `COEXIST` sentinel when both legitimately hold), `gold_answer`,
  `conflict_type`, `difficulty` (`obvious` / `moderate` / `subtle`), and `notes`
  explaining the call. Conflict-type breakdown:
  - `factual` (7) — a discrete fact two excerpts state differently (primary
    benchmark: CoNLL-2003 vs OntoNotes 5.0).
  - `magnitude` (5) — the same quantity at a materially different size (speedup
    "up to 3×" on accepted tokens vs "1.4×" wall-clock).
  - `staleness` (5) — an older claim overturned by a newer correction (a SOTA
    result retracted by a v2 erratum after test-set contamination was found).
  - `provenance` (3) — a result or method credited to the wrong source (an
    88.5 EM table row claimed as "ours" vs credited to Chen et al. (2021)).

  The 3 original hand-built documents are seeds 1–3.
- **`common/cache.py` / `common/llm.py`** — on-disk LLM response cache
  (`.cache/llm_cache.json`), keyed by SHA-256 of
  `backend + model + system + temperature + prompt`; disable with `LLM_CACHE=0`.
  `LLMCache.key(..., sample_id=...)` and `LLMClient.generate(..., sample_id=...)`
  add an **optional sample id** to the key: with no `sample_id` the cache is
  unchanged (one answer per identical request, forever); passing
  `sample_id=0, 1, 2, ...` lets the *same* prompt be re-sampled with each draw
  cached under its own key — so future self-consistency / repeated-sampling
  logic isn't silently handed one frozen answer. Backends: `local` (Ollama,
  `llama3.1:8b`, default for all dev/testing) or `gemini` (`gemini-2.5-flash`
  agent / `gemini-2.5-pro` judge), selected by `LLM_BACKEND`.
  `common/env.py` is the minimal `.env` loader (no `python-dotenv` dep).
- **`tests/`** — 93 tests, no network:
  - `test_store.py` (48) — schema + provenance round-trip, the status state
    machine incl. **invalid-transition blocking**, **concurrent SQLite writes**,
    the naive conflict scan, multi-outcome `Resolution`, last-write-wins.
  - `test_seed_conflicts.py` (17) — the suite is well-formed: 2+ excerpts each,
    valid `gold_excerpt_id`, unique doc ids / topics, all four conflict types
    present, a difficulty spread, the original 3 seeds retained.
  - `test_orchestrator.py` (17) — the 5-agent run with a fake offline LLM:
    grouping + configurable roster, excerpt assignment, correlated agents
    produce identical claims, and **real provenance on every write**.
  - `test_llm_cache.py` (11) — cache + backend routing, incl. `sample_id` key
    widening and repeated-sampling behaviour.

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
python demo.py         # 5 agents over a 4-doc subset (one per conflict type)
python demo.py --all    # ... over all 20 seed documents
pytest                  # 93 tests: store, seed suite, orchestration, LLM cache (no network)
```

`demo.py` uses `LLM_BACKEND` (default `local`) and prints, in order:

1. the resolved LLM-config banner;
2. the **agent roster** — which agents are in the correlated group `grp_A` and
   which are independent;
3. **every agent's write, grouped by correlated group vs. independent** — the
   excerpt each agent read and whether it landed on the gold excerpt
   (`(gold)` / `(off-gold)`);
4. **naive conflict detection** — the same-topic / differing-content scan, one
   block per topic, each claim tagged with its agent group and excerpt id;
5. **resolution** with `last_write_wins`, then the full shared-memory dump with a
   per-item provenance line (`source_type`, `origin`, `authority`, `version`,
   `source_id`, `evidence_span`).

The correlated group's shared-bias behaviour is visible directly in step 3:
`agent_A/C/D` produce the same answer — right or wrong — because they read the
same excerpt with near-identical prompts, while `agent_B/E` on the other excerpt
can disagree.

`demo_memory.db` (and its `-wal` / `-shm` sidecars) is gitignored, as
`demo_memory.json` was before.

## Layout

```
memory/
  store.py         # [done]  schema + status state machine + SQLite(WAL)/JSON backends + naive list-conflicts
  detector.py      # [stub]  embedding candidate clustering + LLM NLI (entailment/contradiction/neutral)
  reconciler.py    # [stub]  CREDIBILITY vs COORDINATION classification
reliability/
  peer_memory.py   # [stub]  online per-agent competence + pairwise correlation
  resolver.py      # [stub]  reliability-weighted resolver, discounts correlated agreement (our contribution)
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
  run_comparison.py # [stub]  baselines vs full model: resolution accuracy + detection P/R/F1
tests/
```

## Next

In order:

1. **Real contradiction detection** (`memory/detector.py`) — replace the naive
   same-topic `list_conflicts` scan with (a) embedding-based candidate
   clustering to group claims that are *about* the same thing, then (b) an
   LLM-based NLI step classifying each candidate pair as **entailment /
   contradiction / neutral**. Only genuine contradictions become `Conflict`s.
   (Needs `pip install sentence-transformers`.)
2. **The remaining baselines** — `baselines/majority_vote.py` and
   `baselines/static_confidence.py` (fixed evidence-type weights, no learning;
   the primary comparison baseline).
3. **The novel contribution** — `reliability/peer_memory.py` (online per-agent
   competence + pairwise correlation estimates) and `reliability/resolver.py`
   (a resolver that weights each claim by its source's reliability *and
   discounts agreement between correlated agents*).
4. **Evaluation** (`eval/run_comparison.py`) — run every baseline and the full
   model over the seed suite and report real metrics: resolution accuracy
   against the gold labels, and detection **precision / recall / F1** against
   the known seeded contradictions.
