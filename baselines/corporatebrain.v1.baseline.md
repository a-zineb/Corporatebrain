# Corporate Brain v1 — Approved Baseline

Certification status: **accepted Phase 2 reference**. This deterministic run
used the versioned `corporatebrain.v1` benchmark with judge mode disabled.
Raw run artifacts remain in ignored `evaluation_runs/`.

## Certified runtime

- Corpus chunks: 1,072
- Corpus fingerprint: `f5a471db7123789f118a604d4614aa386e8e717e8d1cc61c0cf7fb21ea6e7e75`
- Metadata fingerprint: `a3be7899397f74722d5fa8f4b7cd1e76396bf4498e77987eb8f6b777d21fd0d0`
- Runtime fingerprint: `a2f1ef6963f439d9cb956c2dbbc904d0323a5338025df857af722121437f6e4d`
- Embedding model: `paraphrase-multilingual-MiniLM-L12-v2`
- Generation model: `qwen3:8b`
- Hybrid retrieval: vector candidates 10, BM25 candidates 10, RRF k=60,
  production top-k 15, fallback threshold 3
- Evaluator generation policy: 256 output tokens, 120-second stage limit

## Deterministic metrics

| Metric | Value |
| --- | ---: |
| Recall@K | 0.8750 |
| Precision@K | 0.0528 |
| Hit Rate@K | 0.7500 |
| MRR | 0.5118 |
| NDCG@K | 0.6974 |
| Citation validity | 1.0000 |
| Expected-source match | 0.1111 |
| Refusal correctness | 0.3333 |
| Mean latency | 186,587.58 ms |

## Generation and forensics

- Successful generations: 19
- Generation timeouts: 0
- Forensic reports: 11
  - Unexpected citation source: 6
  - Expected source missing: 3
  - Refusal incorrect: 2

This baseline is the required comparison point for any future optimization.
