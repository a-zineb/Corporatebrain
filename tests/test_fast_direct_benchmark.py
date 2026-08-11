from fast_direct_benchmark import run_catalog_benchmark, run_fast_benchmark


def test_100_case_zero_llm_direct_benchmark_meets_targets():
    metrics = run_fast_benchmark()
    assert metrics["total"] == 100
    assert metrics["wrong"] == 0
    assert metrics["false_no_evidence"] <= 5
    assert metrics["correct"] >= 95
    assert metrics["cross_document_leakage"] == 0
    assert metrics["secret_leakage"] == 0
    assert metrics["ollama_calls"] == 0
    assert metrics["chroma_calls"] == 0
    assert metrics["warm_p50_ms"] < 1000
    assert metrics["warm_p95_ms"] < 3000
    assert metrics["warm_max_ms"] < 5000


def test_catalog_100_query_benchmark_is_local_and_fast():
    metrics = run_catalog_benchmark()
    assert metrics["queries"] == 100
    assert metrics["ollama_calls"] == 0
    assert metrics["embedding_calls"] == 0
    assert metrics["p95_ms"] < 2000
