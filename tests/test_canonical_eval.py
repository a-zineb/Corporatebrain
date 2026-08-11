from canonical_eval import run_benchmark


def test_end_to_end_canonical_benchmark_meets_reliability_gate():
    metrics = run_benchmark()
    assert metrics["total_cases"] == 54
    assert metrics["wrong"] == 0
    assert metrics["cross_document_leakage"] == 0
    assert metrics["secret_leakage"] == 0
    assert metrics["unsupported_generated_answer"] == 0
    assert metrics["supported_factual_recall"] >= 0.95
    assert metrics["ambiguous_refusal"] == 1
    assert set(metrics["per_format"]) == {"docx", "pdf", "xlsx", "csv"}
