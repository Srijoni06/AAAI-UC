"""Seeded contradiction suite for the literature-summarization testbed.

Each :class:`SeedConflict` is one source document that has been written so that
two (or more) of its excerpts describe the *same fact* differently. The
contradiction is in the source material, not model noise, so any disagreement an
agent produces is attributable to which excerpt it read (and to correlated bias
between agents that read the same one).

Every seed carries a **gold label**:

* ``gold_excerpt_id`` - the excerpt whose claim is actually correct (or the
  sentinel ``COEXIST`` when both claims are legitimately true at once);
* ``gold_answer``     - a short canonical statement of the correct answer;
* ``conflict_type``   - :class:`ConflictType` (factual / magnitude / staleness /
  provenance);
* ``difficulty``      - :class:`Difficulty` (obvious / moderate / subtle).

``eval/run_comparison.py`` scores a resolver by comparing the claim it keeps
against ``gold_excerpt_id`` / ``gold_answer``.

The first three seeds (``doc-benchmark``, ``doc-gain``, ``doc-sota``) are the
hand-built documents milestone 1 kept inline in ``agents/orchestrator.py``; the
rest extend the suite to 20 across a spread of conflict types and difficulty.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

COEXIST = "COEXIST"  # gold_excerpt_id sentinel: both claims are validly true


class ConflictType(str, Enum):
    FACTUAL = "factual"        # a discrete fact two excerpts state differently
    MAGNITUDE = "magnitude"    # same quantity, materially different size
    STALENESS = "staleness"    # an older claim overturned by a newer correction
    PROVENANCE = "provenance"  # a result/method attributed to the wrong source


class Difficulty(str, Enum):
    OBVIOUS = "obvious"    # flat contradiction, visible without domain knowledge
    MODERATE = "moderate"  # needs a careful read of both excerpts
    SUBTLE = "subtle"      # both excerpts sound authoritative; easy to get wrong


@dataclass(frozen=True)
class Excerpt:
    """One passage an agent can be handed. ``text`` is the verbatim slice the
    agent reads and later records as its ``evidence_span``."""

    excerpt_id: str  # unique within its SeedConflict
    section: str     # human-facing location label, e.g. "Section 5 (Results)"
    text: str


@dataclass(frozen=True)
class SeedConflict:
    doc_id: str
    title: str
    topic: str      # shared grouping key in memory (must be globally unique here)
    question: str
    excerpts: tuple[Excerpt, ...]
    gold_excerpt_id: str
    gold_answer: str
    conflict_type: ConflictType
    difficulty: Difficulty
    notes: str = ""  # why the gold label is what it is

    def excerpt(self, excerpt_id: str) -> Excerpt:
        for ex in self.excerpts:
            if ex.excerpt_id == excerpt_id:
                return ex
        raise KeyError(f"{self.doc_id!r} has no excerpt {excerpt_id!r}")

    @property
    def excerpt_ids(self) -> list[str]:
        return [ex.excerpt_id for ex in self.excerpts]


# --------------------------------------------------------------------------- #
# The suite. Excerpt order matters: the correlated agent group in
# agents/orchestrator.py reads ``excerpts[0]`` by default, so seeds are ordered
# to put the *wrong* claim first in roughly half the cases - that is what makes
# "three correlated agents agree" sometimes mean "agree on a mistake".
# --------------------------------------------------------------------------- #
SEED_CONFLICTS: list[SeedConflict] = [
    # -- 1. original milestone-1 seed -------------------------------------- #
    SeedConflict(
        doc_id="doc-benchmark",
        title="Neural Entity Tagger: Setup and Results",
        topic="primary_benchmark",
        question="What is the primary benchmark dataset used to evaluate the method?",
        excerpts=(
            Excerpt(
                "intro",
                "Section 1 (Introduction)",
                "In this work we primarily evaluate our tagger on the CoNLL-2003 "
                "English named-entity benchmark, the standard testbed for this "
                "task, and use it for all ablations.",
            ),
            Excerpt(
                "results",
                "Section 5 (Main Results)",
                "Our headline numbers are reported on OntoNotes 5.0, which we "
                "treat as the primary benchmark for this paper because of its "
                "broader entity inventory; CoNLL-2003 is used only as a small "
                "secondary check.",
            ),
        ),
        gold_excerpt_id="results",
        gold_answer="OntoNotes 5.0 is the primary benchmark; CoNLL-2003 is secondary.",
        conflict_type=ConflictType.FACTUAL,
        difficulty=Difficulty.SUBTLE,
        notes=(
            "The intro's 'standard testbed' framing is a genre convention; the "
            "results section is explicit that the headline evaluation and the "
            "'primary benchmark' label belong to OntoNotes 5.0."
        ),
    ),
    # -- 2. original milestone-1 seed ------------------------------------- #
    SeedConflict(
        doc_id="doc-gain",
        title="Adaptive Reranking for Retrieval QA",
        topic="improvement_over_baseline",
        question="How large is the method's Exact Match improvement over the DPR baseline?",
        excerpts=(
            Excerpt(
                "abstract",
                "Abstract",
                "Our adaptive reranker delivers a substantial +4.2 EM "
                "improvement over the strong DPR baseline on the target task.",
            ),
            Excerpt(
                "discussion",
                "Section 6 (Discussion)",
                "After controlling for the retriever budget, the adaptive "
                "reranker yields only a modest +0.5 EM gain over the DPR "
                "baseline, within one standard deviation across seeds.",
            ),
        ),
        gold_excerpt_id="discussion",
        gold_answer="About +0.5 EM once the retriever budget is held fixed (not significant).",
        conflict_type=ConflictType.MAGNITUDE,
        difficulty=Difficulty.MODERATE,
        notes=(
            "The abstract compares against an under-resourced baseline; the "
            "budget-controlled comparison in the discussion is the fair one and "
            "shows the gain is within noise."
        ),
    ),
    # -- 3. original milestone-1 seed ----------------------------------- #
    SeedConflict(
        doc_id="doc-sota",
        title="Long-Context Compression with LSH Attention (v2 with erratum)",
        topic="beats_prior_sota",
        question="Does the method outperform the prior state of the art on the long-document suite?",
        excerpts=(
            Excerpt(
                "body",
                "Section 4 (original text)",
                "On the long-document suite our method sets a new state of the "
                "art, surpassing the previous best system by 1.8 ROUGE-L.",
            ),
            Excerpt(
                "erratum",
                "Erratum (added in v2)",
                "A post-publication audit found test-set contamination in the "
                "long-document suite. With the cleaned split, the method does "
                "NOT exceed the prior state of the art and trails it by 0.3 "
                "ROUGE-L.",
            ),
        ),
        gold_excerpt_id="erratum",
        gold_answer="No - after fixing test-set contamination it trails prior SOTA by 0.3 ROUGE-L.",
        conflict_type=ConflictType.STALENESS,
        difficulty=Difficulty.MODERATE,
        notes="The erratum is a dated, explicit correction of the Section 4 claim.",
    ),
    # -- 4. factual: obvious flat contradiction ------------------------- #
    SeedConflict(
        doc_id="doc-supervision",
        title="Self-Aligning Sentence Encoders",
        topic="training_supervision",
        question="Does the method require human-labeled training data?",
        excerpts=(
            Excerpt(
                "abstract",
                "Abstract",
                "We present a fully unsupervised sentence encoder that requires "
                "no labeled data at any stage of training.",
            ),
            Excerpt(
                "method",
                "Section 3 (Training)",
                "We fine-tune the encoder on 10,000 human-annotated sentence "
                "pairs, which we found essential for the alignment objective to "
                "converge.",
            ),
        ),
        gold_excerpt_id="method",
        gold_answer="Yes - it fine-tunes on 10k human-annotated sentence pairs.",
        conflict_type=ConflictType.FACTUAL,
        difficulty=Difficulty.OBVIOUS,
        notes="The method section describes a labeled fine-tuning stage the abstract denies.",
    ),
    # -- 5. factual: which optimizer -------------------------------------- #
    SeedConflict(
        doc_id="doc-optimizer",
        title="Scaling Sparse Mixture-of-Experts LMs",
        topic="optimizer_used",
        question="Which optimizer was used to train the released models?",
        excerpts=(
            Excerpt(
                "intro",
                "Section 1 (Introduction)",
                "Training uses a standard Adam-style optimizer with a cosine "
                "learning-rate schedule.",
            ),
            Excerpt(
                "appendix",
                "Appendix B (Training Details)",
                "All released checkpoints were trained with AdamW (weight decay "
                "0.1, beta2 = 0.95); plain Adam was used only in the early "
                "prototype and is not part of the release.",
            ),
        ),
        gold_excerpt_id="appendix",
        gold_answer="AdamW (weight decay 0.1, beta2 = 0.95).",
        conflict_type=ConflictType.FACTUAL,
        difficulty=Difficulty.MODERATE,
        notes=(
            "The intro rounds 'AdamW' to 'Adam-style'; the training-details "
            "appendix is precise and scoped to the released checkpoints."
        ),
    ),
    # -- 6. factual: which metric is the headline number ---------------- #
    SeedConflict(
        doc_id="doc-humaneval",
        title="CodeLlama-Tuned: A Compact Code Model",
        topic="humaneval_pass_at_1",
        question="What is the model's pass@1 score on HumanEval?",
        excerpts=(
            Excerpt(
                "abstract",
                "Abstract",
                "Our 3B model reaches 67% on HumanEval, competitive with models "
                "twice its size.",
            ),
            Excerpt(
                "table",
                "Table 2 (HumanEval breakdown)",
                "HumanEval: pass@1 = 41.2, pass@10 = 67.0, pass@100 = 78.5. "
                "Unless noted, the headline metric elsewhere in the paper is "
                "pass@10.",
            ),
        ),
        gold_excerpt_id="table",
        gold_answer="pass@1 is 41.2 (the 67% figure is pass@10).",
        conflict_type=ConflictType.FACTUAL,
        difficulty=Difficulty.SUBTLE,
        notes=(
            "The abstract's bare '67%' is pass@10; the question asks for pass@1, "
            "which the table gives as 41.2."
        ),
    ),
    # -- 7. factual: base model it was initialized from ---------------- #
    SeedConflict(
        doc_id="doc-init",
        title="MedQA-Expert: Domain-Adapted Clinical QA",
        topic="base_model_init",
        question="Which base model were the released weights initialized from?",
        excerpts=(
            Excerpt(
                "related",
                "Section 2 (Related Work)",
                "Building directly on LLaMA-2-7B, we adapt it to the clinical "
                "domain through continued pretraining.",
            ),
            Excerpt(
                "repro",
                "Section 6 (Reproducibility Checklist)",
                "Base model: Mistral-7B-v0.1. (An earlier draft used LLaMA-2-7B; "
                "the released weights and all reported numbers use Mistral-7B.)",
            ),
        ),
        gold_excerpt_id="repro",
        gold_answer="Mistral-7B-v0.1.",
        conflict_type=ConflictType.FACTUAL,
        difficulty=Difficulty.MODERATE,
        notes=(
            "Related Work carries a stale sentence from an earlier draft; the "
            "reproducibility checklist names the base model of the actual "
            "release."
        ),
    ),
    # -- 8. factual: which split the numbers are on ------------------- #
    SeedConflict(
        doc_id="doc-split",
        title="A Benchmark for Table-to-Text Faithfulness",
        topic="reported_eval_split",
        question="Which data split are the paper's reported scores computed on?",
        excerpts=(
            Excerpt(
                "setup",
                "Section 4 (Experimental Setup)",
                "We report results on the test split of the benchmark.",
            ),
            Excerpt(
                "footnote",
                "Section 4, footnote 3",
                "Because the official test labels are held out behind a "
                "leaderboard, every number in Tables 3-6 is computed on the "
                "public validation split; test-set entries are marked 'n/a'.",
            ),
        ),
        gold_excerpt_id="footnote",
        gold_answer="The public validation split (test labels are withheld).",
        conflict_type=ConflictType.FACTUAL,
        difficulty=Difficulty.MODERATE,
        notes="The footnote qualifies the setup sentence and matches the actual tables.",
    ),
    # -- 9. factual: coexisting claims (gold = COEXIST) --------------- #
    SeedConflict(
        doc_id="doc-languages",
        title="PolyParse: A Multilingual Dependency Parser",
        topic="language_coverage_claims",
        question="How many languages does the parser support?",
        excerpts=(
            Excerpt(
                "abstract",
                "Abstract",
                "PolyParse is trained jointly on 90 languages from Universal "
                "Dependencies v2.11.",
            ),
            Excerpt(
                "eval",
                "Section 5 (Evaluation)",
                "We evaluate on the 46 languages that have a held-out test "
                "treebank of at least 1,000 sentences; the other 44 are "
                "train-only.",
            ),
        ),
        gold_excerpt_id=COEXIST,
        gold_answer=(
            "Both: trained on 90 languages, evaluated on the 46 with a large "
            "enough test treebank."
        ),
        conflict_type=ConflictType.FACTUAL,
        difficulty=Difficulty.SUBTLE,
        notes=(
            "Not a contradiction: 'trained on 90' and 'evaluated on 46' are "
            "both true and describe different stages. A good resolver keeps "
            "both, scoped."
        ),
    ),
    # -- 10. magnitude: inference speedup ---------------------------- #
    SeedConflict(
        doc_id="doc-speedup",
        title="Cascade Decoding for Faster Generation",
        topic="inference_speedup",
        question="How much faster is inference with the proposed method?",
        excerpts=(
            Excerpt(
                "abstract",
                "Abstract",
                "Cascade decoding makes generation up to 3x faster.",
            ),
            Excerpt(
                "analysis",
                "Section 5.3 (Wall-clock Analysis)",
                "Once the extra draft-verification passes are included, the "
                "end-to-end wall-clock speedup is 1.4x on the same hardware; "
                "the 3x figure counts only accepted tokens.",
            ),
        ),
        gold_excerpt_id="analysis",
        gold_answer="About 1.4x end-to-end wall-clock (3x counts accepted tokens only).",
        conflict_type=ConflictType.MAGNITUDE,
        difficulty=Difficulty.MODERATE,
        notes="'Up to 3x' is a best-case token-level figure; 1.4x is the honest wall-clock number.",
    ),
    # -- 11. magnitude: training cost, large gap (obvious) ---------- #
    SeedConflict(
        doc_id="doc-cost",
        title="TinyPretrain: Efficient Pretraining on a Budget",
        topic="pretraining_compute_cost",
        question="How much compute did pretraining the released model take?",
        excerpts=(
            Excerpt(
                "intro",
                "Section 1 (Introduction)",
                "The entire model was pretrained in under 24 GPU-hours.",
            ),
            Excerpt(
                "compute",
                "Appendix D (Compute)",
                "Pretraining used 64 A100s for 7.5 days, i.e. approximately "
                "11,500 A100-hours. The '24 GPU-hours' in the introduction "
                "refers to a single fine-tuning run and is a drafting error.",
            ),
        ),
        gold_excerpt_id="compute",
        gold_answer="Roughly 11,500 A100-hours (64 A100s for 7.5 days).",
        conflict_type=ConflictType.MAGNITUDE,
        difficulty=Difficulty.OBVIOUS,
        notes="Two orders of magnitude apart; the appendix explicitly labels the intro figure an error.",
    ),
    # -- 12. magnitude: training-set size after filtering ---------- #
    SeedConflict(
        doc_id="doc-dataset-size",
        title="WebPairs: A Corpus for Contrastive Pretraining",
        topic="training_set_size",
        question="How many training examples are actually used for the experiments?",
        excerpts=(
            Excerpt(
                "intro",
                "Section 1 (Introduction)",
                "We assemble a corpus of over 1,000,000 sentence pairs from web "
                "crawl.",
            ),
            Excerpt(
                "data",
                "Section 3 (Data)",
                "After near-duplicate removal and quality filtering, 312,000 "
                "pairs remain; all experiments in this paper use this filtered "
                "set.",
            ),
        ),
        gold_excerpt_id="data",
        gold_answer="312,000 pairs (the 1M figure is the pre-filtering crawl).",
        conflict_type=ConflictType.MAGNITUDE,
        difficulty=Difficulty.MODERATE,
        notes="The 1M is raw crawl; the experiments run on the 312k filtered set.",
    ),
    # -- 13. magnitude: human preference win rate ---------------- #
    SeedConflict(
        doc_id="doc-winrate",
        title="Assistant-RLAIF: Preference-Tuned Dialogue",
        topic="human_preference_win_rate",
        question="How often do human raters prefer the model over the baseline?",
        excerpts=(
            Excerpt(
                "abstract",
                "Abstract",
                "Human raters prefer our responses 90% of the time.",
            ),
            Excerpt(
                "study",
                "Section 6 (Human Study)",
                "In the blind pairwise study the win rate over the baseline is "
                "55% (95% CI: 51-59; n = 1,200), i.e. a modest but significant "
                "preference. The 90% figure is the rate at which our response "
                "was rated 'acceptable', not the head-to-head win rate.",
            ),
        ),
        gold_excerpt_id="study",
        gold_answer="55% head-to-head (95% CI 51-59); 90% was a different 'acceptable' metric.",
        conflict_type=ConflictType.MAGNITUDE,
        difficulty=Difficulty.MODERATE,
        notes="The abstract quotes an acceptability rate as if it were the win rate.",
    ),
    # -- 14. staleness: leaderboard rank moved (obvious) -------- #
    SeedConflict(
        doc_id="doc-leaderboard",
        title="RankFusion: Ensemble Retrieval (camera-ready)",
        topic="leaderboard_rank",
        question="Where does the system currently rank on the public leaderboard?",
        excerpts=(
            Excerpt(
                "body",
                "Section 5 (June submission text)",
                "At submission time RankFusion ranks #1 on the public "
                "leaderboard.",
            ),
            Excerpt(
                "cr_note",
                "Camera-ready note (October)",
                "Update for the camera-ready: as of October our entry has been "
                "overtaken by three later systems and now ranks #4.",
            ),
        ),
        gold_excerpt_id="cr_note",
        gold_answer="#4 as of the October camera-ready.",
        conflict_type=ConflictType.STALENESS,
        difficulty=Difficulty.OBVIOUS,
        notes="Dated camera-ready note explicitly supersedes the June '#1' claim.",
    ),
    # -- 15. staleness: scoring bug fixed in a later version --- #
    SeedConflict(
        doc_id="doc-scoring-bug",
        title="SpanBERT-QA Revisited (v3)",
        topic="reported_f1",
        question="What is the model's F1 on the QA benchmark?",
        excerpts=(
            Excerpt(
                "table",
                "Table 2 (v1)",
                "Our model reaches 82.3 F1, a new best on the benchmark.",
            ),
            Excerpt(
                "changelog",
                "Changelog (v3)",
                "v3: fixed an off-by-one in the span-scoring script that "
                "inflated F1. Corrected F1 is 79.1; the model no longer holds "
                "the top spot. All tables have been updated.",
            ),
        ),
        gold_excerpt_id="changelog",
        gold_answer="79.1 F1 after the v3 scoring-bug fix.",
        conflict_type=ConflictType.STALENESS,
        difficulty=Difficulty.MODERATE,
        notes="The changelog is a later, explicit correction of the Table 2 number.",
    ),
    # -- 16. staleness: deprecated model version regenerated --- #
    SeedConflict(
        doc_id="doc-api-version",
        title="LLM-as-Judge Agreement Study",
        topic="judge_model_version",
        question="Which judge-model version produced the numbers in the paper?",
        excerpts=(
            Excerpt(
                "setup",
                "Section 3 (Setup)",
                "All automatic judgements are produced by gpt-4-0314.",
            ),
            Excerpt(
                "revision",
                "Section 3 (revision note)",
                "Revision: gpt-4-0314 was retired before the camera-ready, so "
                "every judgement was regenerated with gpt-4-0613. Reported "
                "agreement scores changed by less than 0.5 points but all "
                "numbers now reflect gpt-4-0613.",
            ),
        ),
        gold_excerpt_id="revision",
        gold_answer="gpt-4-0613 (regenerated after gpt-4-0314 was retired).",
        conflict_type=ConflictType.STALENESS,
        difficulty=Difficulty.MODERATE,
        notes="Revision note states the reported numbers were regenerated with the newer version.",
    ),
    # -- 17. staleness: dataset version switched --------------- #
    SeedConflict(
        doc_id="doc-dataset-version",
        title="Answerability-Aware Reading Comprehension",
        topic="evaluation_dataset_version",
        question="Which version of SQuAD is used for evaluation?",
        excerpts=(
            Excerpt(
                "arxiv_v1",
                "Section 4 (arXiv v1)",
                "We evaluate on SQuAD 1.1.",
            ),
            Excerpt(
                "arxiv_v2",
                "Section 4 (arXiv v2, camera-ready)",
                "For the camera-ready we moved all evaluation to SQuAD 2.0 so "
                "that unanswerable questions are included; SQuAD 1.1 numbers "
                "have been removed.",
            ),
        ),
        gold_excerpt_id="arxiv_v2",
        gold_answer="SQuAD 2.0 (switched from 1.1 for the camera-ready).",
        conflict_type=ConflictType.STALENESS,
        difficulty=Difficulty.MODERATE,
        notes="The v2 text is the later revision and removes the 1.1 results entirely.",
    ),
    # -- 18. provenance: whose number is in the table -------- #
    SeedConflict(
        doc_id="doc-attribution",
        title="Dense Retrieval with Hard Negatives",
        topic="who_scored_88_5_em",
        question="Whose system produced the 88.5 EM entry in Table 1?",
        excerpts=(
            Excerpt(
                "claim",
                "Section 5 (text near Table 1)",
                "As Table 1 shows, we reach 88.5 EM, the best result on the "
                "benchmark.",
            ),
            Excerpt(
                "tablenote",
                "Table 1 caption",
                "Rows marked with a dagger are reproduced from prior work. The "
                "88.5 EM row (dagger) is Chen et al. (2021); our system (final "
                "row) reaches 86.9 EM.",
            ),
        ),
        gold_excerpt_id="tablenote",
        gold_answer="Chen et al. (2021) - the paper's own system scores 86.9 EM.",
        conflict_type=ConflictType.PROVENANCE,
        difficulty=Difficulty.SUBTLE,
        notes=(
            "The body text claims a borrowed row as the authors' own; the table "
            "caption attributes 88.5 EM to Chen et al. (2021)."
        ),
    ),
    # -- 19. provenance: origin of a method component ------- #
    SeedConflict(
        doc_id="doc-rope",
        title="A Long-Context Transformer for Code",
        topic="rope_contribution",
        question="Did this paper introduce rotary positional embeddings (RoPE)?",
        excerpts=(
            Excerpt(
                "contrib",
                "Section 1 (Contributions)",
                "We introduce rotary positional embeddings to let attention "
                "generalize to longer contexts.",
            ),
            Excerpt(
                "method",
                "Section 3 (Architecture)",
                "For positional information we adopt rotary positional "
                "embeddings (RoPE) from Su et al. (2021) unchanged; our "
                "contribution is the block-sparse attention pattern in 3.2.",
            ),
        ),
        gold_excerpt_id="method",
        gold_answer="No - RoPE is from Su et al. (2021); the paper's contribution is block-sparse attention.",
        conflict_type=ConflictType.PROVENANCE,
        difficulty=Difficulty.MODERATE,
        notes="The contributions list overclaims; the method section correctly cites RoPE's origin.",
    ),
    # -- 20. provenance: who built the dataset -------------- #
    SeedConflict(
        doc_id="doc-corpus-origin",
        title="IntentBank: A Corpus for Dialogue Intent Detection",
        topic="corpus_origin",
        question="Who created and annotated the IntentBank corpus?",
        excerpts=(
            Excerpt(
                "abstract",
                "Abstract",
                "We collect and annotate IntentBank, a new 12,000-utterance "
                "corpus for intent detection.",
            ),
            Excerpt(
                "data",
                "Section 3 (Dataset)",
                "IntentBank re-releases the utterances and intent labels of "
                "Nguyen et al. (2019) under a permissive licence; our "
                "contribution is a new stratified train/dev/test split and "
                "annotation-error fixes for 3% of examples.",
            ),
        ),
        gold_excerpt_id="data",
        gold_answer=(
            "Nguyen et al. (2019) - this paper contributes a new split and minor "
            "label fixes, not the annotations."
        ),
        conflict_type=ConflictType.PROVENANCE,
        difficulty=Difficulty.MODERATE,
        notes="The abstract claims original annotation; the dataset section credits Nguyen et al. (2019).",
    ),
]


# --------------------------------------------------------------------------- #
# convenience accessors
# --------------------------------------------------------------------------- #
SEEDS_BY_ID: dict[str, SeedConflict] = {s.doc_id: s for s in SEED_CONFLICTS}


def get(doc_id: str) -> SeedConflict:
    return SEEDS_BY_ID[doc_id]


def by_type(conflict_type: ConflictType) -> list[SeedConflict]:
    return [s for s in SEED_CONFLICTS if s.conflict_type == conflict_type]


def by_difficulty(difficulty: Difficulty) -> list[SeedConflict]:
    return [s for s in SEED_CONFLICTS if s.difficulty == difficulty]


def type_counts() -> dict[str, int]:
    out: dict[str, int] = {t.value: 0 for t in ConflictType}
    for s in SEED_CONFLICTS:
        out[s.conflict_type.value] += 1
    return out
