"""RAG benchmark RAGAS evaluation node."""

from __future__ import annotations

from typing import Any

from autorag_pipeline.states.rag_bench_state import RAGBenchState


def evaluate_ragas(state: RAGBenchState) -> dict[str, Any]:
    """Evaluate benchmark results using RAGAS metrics.

    Wraps ExtendedRAGEvaluator from autorag_rag_eval.
    """
    from autorag_rag_eval.evaluator import ExtendedRAGEvaluator
    from autorag_rag_eval.metrics import MetricPreset

    bench_results = state["bench_results"]
    qa_pairs = state["qa_pairs"]

    evaluator = ExtendedRAGEvaluator(preset=MetricPreset.CORE_ONLY)

    ground_truths = [qa.get("answer", "") for qa in qa_pairs]
    questions = [qa["query"] for qa in qa_pairs]

    for result_entry in bench_results:
        run_results = result_entry["run_results"]
        for strategy_name, query_results in run_results.items():
            contexts = [
                [doc.get("page_content", "") for doc in qr.get("docs", [])]
                for qr in query_results
            ]
            answers = [qr.get("answer", "") for qr in query_results]

            report = evaluator.evaluate(
                questions=questions[: len(answers)],
                contexts=contexts,
                answers=answers,
                ground_truths=ground_truths[: len(answers)],
            )
            result_entry["ragas_report"] = {
                "aggregate": report.aggregate_dict,
                "weighted_score": report.weighted_score,
            }

    return {"bench_results": bench_results, "bench_done": True}
