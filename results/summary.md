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
| TP | 95 |
| FP | 5 |
| FN | 19 |
| precision | 0.95 |
| recall | 0.8333 |
| f1 | 0.8879 |

## Resolution Accuracy

| Condition | Accuracy | Decisive | Correct | Contested | Escalation |
|-----------|----------|----------|---------|-----------|------------|
| null | 100.0% | 2 | 2 | 16 | 88.9% |
| last_write_wins | 72.2% | 18 | 13 | 0 | 0.0% |
| majority_vote | 38.9% | 18 | 7 | 0 | 0.0% |
| static_confidence | 72.2% | 18 | 13 | 0 | 0.0% |
| reliability_aware | 66.7% | 18 | 12 | 0 | 0.0% |

## Per-Type Accuracy

| Condition | factual | magnitude | provenance | staleness |
|-----------|----------|----------|----------|----------|
| null | 16.7% | 0.0% | 0.0% | 20.0% |
| last_write_wins | 50.0% | 100.0% | 50.0% | 80.0% |
| majority_vote | 66.7% | 0.0% | 50.0% | 40.0% |
| static_confidence | 50.0% | 100.0% | 50.0% | 80.0% |
| reliability_aware | 66.7% | 80.0% | 50.0% | 60.0% |
