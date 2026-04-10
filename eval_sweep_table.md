## K-Sweep Evaluation Results

| K | Faithfulness | Recall@K | MRR@K | Avg Faith | Time (s) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | No | 0.593 | 0.593 | — | 1.9 |  |
| 1 | Yes | 0.593 | 0.593 | 3.85 | 133.4 |  |
| 3 | No | 0.630 | 0.611 | — | 0.5 | ⚡ Best time |
| 3 | Yes | 0.630 | 0.611 | 4.11 | 308.5 | 🏆 Best result |
| 5 | No | 0.630 | 0.611 | — | 0.9 |  |
| 5 | Yes | 0.630 | 0.611 | 3.74 | 398.7 |  |
| 10 | No | 0.630 | 0.611 | — | 0.6 |  |
| 10 | Yes | 0.630 | 0.611 | 3.93 | 641.2 |  |

**⚡ Best time:** K=3 (no faithfulness) — 0.5s, Recall=0.630
**🏆 Best result:** K=3 (with faithfulness) — composite 0.707 (0.6×Recall + 0.4×Faith/5)
**⚖️ Best balanced:** K=3 (no faithfulness) — highest recall-per-second

![K-Sweep Chart](eval/k_sweep_chart.png)