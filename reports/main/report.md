# LexiSync evaluation (main)

20 questions · LLM ollama:qwen2.5:7b · 2026-11-09 19:50

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
| dense_only | 60% | - | 76% | 100% | 19% | 32.513 | 1587.9 |
| full | 75% | - | 82% | 100% | 12% | 28.488 | 1290.1 |
| full_k4 | 70% | - | 81% | 100% | 25% | 29.314 | 1133.05 |
| full_no_planner | 70% | - | 76% | 100% | 12% | 30.576 | 1413.95 |
| full_no_verifier | 75% | - | 75% | 100% | 6% | 23.749 | 1271.6 |
| no_retrieval | 35% | - | - | 75% | 44% | 15.666 | 360 |
