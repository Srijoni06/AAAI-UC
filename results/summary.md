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
| FN | 8 |
| precision | 0.9792 |
| recall | 0.9216 |
| f1 | 0.9495 |

## Missed contradictions (false negatives): 8

All judged pairs: `detection_pairs.json`.

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

Two different questions, reported side by side: **resolution-only** isolates resolver quality alone (accuracy given that detection already found the conflict); **end-to-end** is the whole system's accuracy, including the documents where detection found nothing to resolve at all (those count as incorrect, since the system as a whole failed to produce the right answer for them).

| Condition | Resolution-only | (basis) | End-to-end | (basis) | Contested | Escalation |
|-----------|------------------|---------|------------|---------|-----------|------------|
| null | 0.0% | 0/0 decisive | 0.0% | 0/20 end-to-end | 16 | 100.0% |
| last_write_wins | 68.8% | 11/16 decisive | 55.0% | 11/20 end-to-end | 0 | 0.0% |
| majority_vote | 37.5% | 6/16 decisive | 30.0% | 6/20 end-to-end | 0 | 0.0% |
| static_confidence | 62.5% | 10/16 decisive | 50.0% | 10/20 end-to-end | 0 | 0.0% |
| reliability_aware | 56.2% | 9/16 decisive | 45.0% | 9/20 end-to-end | 0 | 0.0% |

## Per-Type Accuracy

| Condition | factual | magnitude | provenance | staleness |
|-----------|----------|----------|----------|----------|
| null | 0.0% | 0.0% | 0.0% | 0.0% |
| last_write_wins | 100.0% | 50.0% | 50.0% | 60.0% |
| majority_vote | 60.0% | 0.0% | 50.0% | 40.0% |
| static_confidence | 40.0% | 100.0% | 50.0% | 60.0% |
| reliability_aware | 60.0% | 75.0% | 50.0% | 40.0% |
