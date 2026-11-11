# LexiSync evaluation (fix)

20 questions · LLM ollama:qwen2.5:7b · 2026-11-09 20:29

## Retrieval (paragraph level; article level in brackets)

| mode | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| bm25 | 0.52 (0.65) | 0.61 (0.77) | 0.77 (0.83) | 0.89 (0.92) | 0.76 (0.80) |
| dense | 0.53 (0.72) | 0.77 (0.90) | 0.83 (0.90) | 0.93 (0.96) | 0.76 (0.86) |
| hybrid | 0.44 (0.56) | 0.71 (0.80) | 0.93 (0.96) | 0.95 (0.98) | 0.68 (0.77) |
| hybrid_rerank | 0.62 (0.69) | 0.83 (0.93) | 0.86 (0.93) | 0.92 (0.98) | 0.83 (0.88) |
| hybrid_rerank_mq | 0.71 (0.77) | 0.93 (0.93) | 0.93 (0.93) | 0.93 (0.96) | 0.93 (0.94) |

## Abstention threshold (best cross-encoder relevance)

| threshold | unanswerable abstained | answerable wrongly abstained |
|---|---|---|
| 0.01 | 4/4 | 0/16 |
| 0.02 | 4/4 | 0/16 |
| 0.05 | 4/4 | 0/16 |
| 0.1 | 4/4 | 0/16 |
| 0.15 | 4/4 | 0/16 |
| 0.2 | 4/4 | 0/16 |
| 0.3 | 4/4 | 0/16 |

## End-to-end

| config | correct (auto) | correct (manual) | citation precision | abstention acc. | false abstention | latency (s) | tokens |
|---|---|---|---|---|---|---|---|
| full | 80% | - | 83% | 100% | 6% | 32.404 | 1496.5 |
