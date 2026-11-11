# LexiSync evaluation (heldout)

8 questions · LLM ollama:qwen2.5:7b · 2026-11-09 20:41

## Retrieval (paragraph level; article level in brackets)

| mode | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| bm25 | 0.60 (0.60) | 0.70 (0.80) | 0.70 (0.80) | 1.00 (1.00) | 0.72 (0.72) |
| dense | 0.50 (0.60) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.80 (0.80) |
| hybrid | 0.70 (0.80) | 0.70 (0.80) | 1.00 (1.00) | 1.00 (1.00) | 0.85 (0.85) |
| hybrid_rerank | 0.70 (0.80) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.90 (0.90) |
| hybrid_rerank_mq | 0.70 (0.80) | 1.00 (1.00) | 1.00 (1.00) | 1.00 (1.00) | 0.90 (0.90) |

## Abstention threshold (best cross-encoder relevance)

| threshold | unanswerable abstained | answerable wrongly abstained |
|---|---|---|
| 0.01 | 2/3 | 0/5 |
| 0.02 | 2/3 | 0/5 |
| 0.05 | 2/3 | 0/5 |
| 0.1 | 2/3 | 0/5 |
| 0.15 | 2/3 | 0/5 |
| 0.2 | 2/3 | 0/5 |
| 0.3 | 2/3 | 0/5 |

## End-to-end

| config | correct (auto) | correct (manual) | citation precision | abstention acc. | false abstention | latency (s) | tokens |
|---|---|---|---|---|---|---|---|
| full | 88% | - | 80% | 100% | 0% | 9.877 | 1026 |
