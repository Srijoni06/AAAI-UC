# Evaluation Summary

**Backend:** real  
**Documents:** 20  
**Agents:** 5  
**Similarity threshold:** -1.0  
**Agent LLM:** local:llama3.1:8b  
**Judge LLM:** local:qwen2.5:14b  

## Detection Metrics (pair-level)

| Metric | Value |
|--------|-------|
| TP | 94 |
| FP | 2 |
| FN | 20 |
| precision | 0.9792 |
| recall | 0.8246 |
| f1 | 0.8952 |

## Missed contradictions (false negatives): 20

All judged pairs: `detection_pairs.json`.

- **doc-optimizer** (factual/moderate) - judged ENTAILMENT, sim 0.5329
  - agent_C [appendix]: "The released checkpoints were trained with AdamW."
  - agent_E [intro]: "The released models were trained using a standard Adam-style optimizer."
- **doc-optimizer** (factual/moderate) - judged ENTAILMENT, sim 0.5003
  - agent_A [appendix]: "The released checkpoints were trained with AdamW, which includes weight decay of 0.1 and beta2 of 0.95."
  - agent_E [intro]: "The released models were trained using a standard Adam-style optimizer."
- **doc-optimizer** (factual/moderate) - judged ENTAILMENT, sim 0.7476
  - agent_D [appendix]: "The released models were trained with AdamW."
  - agent_E [intro]: "The released models were trained using a standard Adam-style optimizer."
- **doc-optimizer** (factual/moderate) - judged ENTAILMENT, sim 0.4561
  - agent_B [intro]: "The released models were trained using a standard Adam-style optimizer with a cosine learning-rate schedule."
  - agent_C [appendix]: "The released checkpoints were trained with AdamW."
- **doc-optimizer** (factual/moderate) - judged ENTAILMENT, sim 0.4249
  - agent_A [appendix]: "The released checkpoints were trained with AdamW, which includes weight decay of 0.1 and beta2 of 0.95."
  - agent_B [intro]: "The released models were trained using a standard Adam-style optimizer with a cosine learning-rate schedule."
- **doc-optimizer** (factual/moderate) - judged ENTAILMENT, sim 0.6439
  - agent_B [intro]: "The released models were trained using a standard Adam-style optimizer with a cosine learning-rate schedule."
  - agent_D [appendix]: "The released models were trained with AdamW."
- **doc-dataset-size** (magnitude/moderate) - judged NEUTRAL, sim 0.3906
  - agent_B [data]: "312,000 pairs of training examples are used for the experiments."
  - agent_C [intro]: "The corpus assembled contains over 1,000,000 sentence pairs."
- **doc-dataset-size** (magnitude/moderate) - judged NEUTRAL, sim 0.6365
  - agent_A [intro]: "The corpus contains over 1,000,000 sentence pairs, which are used as training examples for the experiments."
  - agent_B [data]: "312,000 pairs of training examples are used for the experiments."
- **doc-dataset-size** (magnitude/moderate) - judged NEUTRAL, sim 0.3906
  - agent_B [data]: "312,000 pairs of training examples are used for the experiments."
  - agent_D [intro]: "The corpus assembled contains over 1,000,000 sentence pairs."
- **doc-dataset-size** (magnitude/moderate) - judged NEUTRAL, sim 0.3135
  - agent_C [intro]: "The corpus assembled contains over 1,000,000 sentence pairs."
  - agent_E [data]: "312,000 pairs remain after near-duplicate removal and quality filtering."
- **doc-dataset-size** (magnitude/moderate) - judged NEUTRAL, sim 0.2302
  - agent_A [intro]: "The corpus contains over 1,000,000 sentence pairs, which are used as training examples for the experiments."
  - agent_E [data]: "312,000 pairs remain after near-duplicate removal and quality filtering."
- **doc-dataset-size** (magnitude/moderate) - judged NEUTRAL, sim 0.3135
  - agent_D [intro]: "The corpus assembled contains over 1,000,000 sentence pairs."
  - agent_E [data]: "312,000 pairs remain after near-duplicate removal and quality filtering."
- **doc-attribution** (provenance/subtle) - judged ENTAILMENT, sim 0.8633
  - agent_D [claim]: "The system being referred to produced the 88.5 EM entry in Table 1."
  - agent_E [tablenote]: "Chen et al. (2021) produced the 88.5 EM entry in Table 1."
- **doc-attribution** (provenance/subtle) - judged ENTAILMENT, sim 0.8207
  - agent_A [claim]: "The system being referred to in the excerpt produced the 88.5 EM entry in Table 1."
  - agent_E [tablenote]: "Chen et al. (2021) produced the 88.5 EM entry in Table 1."
- **doc-corpus-origin** (provenance/moderate) - judged ENTAILMENT, sim 0.8402
  - agent_C [abstract]: "The authors of the paper collected and annotated IntentBank, a 12,000-utterance corpus for intent detection."
  - agent_E [data]: "Nguyen et al. (2019) created and annotated the IntentBank corpus."
- **doc-corpus-origin** (provenance/moderate) - judged ENTAILMENT, sim 0.8141
  - agent_B [data]: "The IntentBank corpus was created and annotated by Nguyen et al. (2019), with some additional annotation-error fixes made by the current authors."
  - agent_C [abstract]: "The authors of the paper collected and annotated IntentBank, a 12,000-utterance corpus for intent detection."
- **doc-corpus-origin** (provenance/moderate) - judged ENTAILMENT, sim 0.9292
  - agent_D [abstract]: "The authors of the paper collected and annotated the IntentBank corpus."
  - agent_E [data]: "Nguyen et al. (2019) created and annotated the IntentBank corpus."
- **doc-corpus-origin** (provenance/moderate) - judged ENTAILMENT, sim 0.9292
  - agent_A [abstract]: "The authors of the paper collected and annotated the IntentBank corpus."
  - agent_E [data]: "Nguyen et al. (2019) created and annotated the IntentBank corpus."
- **doc-corpus-origin** (provenance/moderate) - judged ENTAILMENT, sim 0.8887
  - agent_B [data]: "The IntentBank corpus was created and annotated by Nguyen et al. (2019), with some additional annotation-error fixes made by the current authors."
  - agent_D [abstract]: "The authors of the paper collected and annotated the IntentBank corpus."
- **doc-corpus-origin** (provenance/moderate) - judged ENTAILMENT, sim 0.8887
  - agent_A [abstract]: "The authors of the paper collected and annotated the IntentBank corpus."
  - agent_B [data]: "The IntentBank corpus was created and annotated by Nguyen et al. (2019), with some additional annotation-error fixes made by the current authors."

## Resolution Accuracy

| Condition | Accuracy | Decisive | Correct | Contested | Escalation |
|-----------|----------|----------|---------|-----------|------------|
| null | 0.0% | 0 | 0 | 16 | 100.0% |
| last_write_wins | 68.8% | 16 | 11 | 0 | 0.0% |
| majority_vote | 37.5% | 16 | 6 | 0 | 0.0% |
| static_confidence | 62.5% | 16 | 10 | 0 | 0.0% |
| reliability_aware | 50.0% | 16 | 8 | 0 | 0.0% |

## Per-Type Accuracy

| Condition | factual | magnitude | provenance | staleness |
|-----------|----------|----------|----------|----------|
| null | 0.0% | 0.0% | 0.0% | 0.0% |
| last_write_wins | 100.0% | 50.0% | 50.0% | 60.0% |
| majority_vote | 60.0% | 0.0% | 50.0% | 40.0% |
| static_confidence | 40.0% | 100.0% | 50.0% | 60.0% |
| reliability_aware | 40.0% | 75.0% | 50.0% | 40.0% |
