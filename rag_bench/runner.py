"""
BenchmarkRunner — 전략 비교 벤치마크 실행기.

여러 RAG 전략을 동일한 쿼리 세트로 실행하고 결과를 비교한다.
"""

import time
from typing import Dict, List, Optional

from rag_bench.base import BaseRAGStrategy

try:
    from rag_bench.evaluation import RAGEvaluator
except ImportError:
    RAGEvaluator = None


class BenchmarkRunner:
    """
    RAG 전략 비교 벤치마크 러너.

    Usage:
        strategies = [
            DenseSparseStrategy(combo_id=1),
            DenseSparseStrategy(combo_id=2),
        ]
        queries = ["쿠버네티스 Pod란?", "Docker와 VM의 차이"]
        runner = BenchmarkRunner(strategies, queries)
        results = runner.run()
        runner.compare()
    """

    def __init__(
        self,
        strategies: List[BaseRAGStrategy],
        queries: List[str],
        k: int = 3,
        evaluator: Optional["RAGEvaluator"] = None,
    ):
        self.strategies = strategies
        self.queries = queries
        self.k = k
        self.evaluator = evaluator
        self._results: Dict[str, List[dict]] = {}

        # Generation model for RAGAS (if needed)
        try:
            import httpx
            from langchain_openai import ChatOpenAI

            self._generator = ChatOpenAI(
                model="gpt-3.5-turbo", http_client=httpx.Client(verify=False)
            )
        except Exception:
            self._generator = None

    def run(self) -> Dict[str, List[dict]]:
        """
        모든 전략에 대해 쿼리를 실행하고 결과를 수집한다.

        Returns:
            전략 이름 → 쿼리별 결과 목록.
        """
        self._results = {}

        for strategy in self.strategies:
            strategy_name = strategy.name
            print(f"\n{'=' * 60}")
            print(f"전략: {strategy_name}")
            print(f"설명: {strategy.description}")
            print(f"{'=' * 60}")

            query_results = []
            for query in self.queries:
                result = self._run_single(strategy, query)
                query_results.append(result)

            self._results[strategy_name] = query_results

        return self._results

    def _run_single(self, strategy: BaseRAGStrategy, query: str) -> dict:
        """단일 전략 + 쿼리 실행."""
        start = time.time()
        try:
            docs = strategy.retrieve(query, k=self.k)
            elapsed = (time.time() - start) * 1000
            print(f"  [{elapsed:7.1f}ms] {query[:50]}... → {len(docs)} results")
            return {
                "query": query,
                "num_results": len(docs),
                "latency_ms": round(elapsed, 1),
                "results": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in docs
                ],
                "error": None,
            }
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            print(f"  [{elapsed:7.1f}ms] {query[:50]}... → ERROR: {e}")
            return {
                "query": query,
                "num_results": 0,
                "latency_ms": round(elapsed, 1),
                "results": [],
                "error": str(e),
            }

    def evaluate(self, ground_truths: Optional[List[List[str]]] = None):
        """
        수집된 결과에 대해 RAGAS 평가를 수행한다.
        """
        if not self.evaluator:
            print("Evaluator가 설정되지 않았습니다.")
            return None

        if not self._results:
            print("run()을 먼저 실행하세요.")
            return None

        import pandas as pd

        print(f"\n{'=' * 60}")
        print("RAGAS 평가 시작...")
        print(f"{'=' * 60}")

        all_scores = []

        for name, query_results in self._results.items():
            print(f"Evaluating {name}...")

            questions = [r["query"] for r in query_results]
            contexts = [[d["content"] for d in r["results"]] for r in query_results]

            # Generate answers if not present
            answers = []
            for i, r in enumerate(query_results):
                if "answer" in r:
                    answers.append(r["answer"])
                else:
                    # Generate simple answer for evaluation
                    if self._generator:
                        context_str = "\n\n".join(contexts[i])
                        prompt = f"Context:\n{context_str}\n\nQuestion: {questions[i]}\nAnswer:"
                        try:
                            ans = self._generator.invoke(prompt).content
                        except Exception:
                            ans = "Generation failed."
                        answers.append(ans)
                        # Store back in results for dataframe
                        r["answer"] = ans
                    else:
                        answers.append("No generator available.")

            # Prepare ground truths
            gts = None
            if ground_truths:
                gts = ground_truths

            # Evaluate
            try:
                scores = self.evaluator.evaluate(
                    questions=questions,
                    contexts=contexts,
                    answers=answers,
                    ground_truths=gts,
                )

                # scores is a dict of averages
                scores_dict = {
                    k: round(v, 4)
                    for k, v in scores.items()
                    if isinstance(v, (int, float))
                }
                scores_dict["strategy"] = name
                all_scores.append(scores_dict)

                print(f"  -> {scores_dict}")

            except Exception as e:
                print(f"Evaluation failed for {name}: {e}")

        return pd.DataFrame(all_scores)

    def compare(self) -> None:
        """전략 간 성능 비교 요약 출력."""
        if not self._results:
            print("run()을 먼저 실행하세요.")
            return

        print(f"\n{'=' * 70}")
        print("전략 비교 요약")
        print(f"{'=' * 70}")
        print(f"{'전략':<40} {'평균 레이턴시':>12} {'평균 결과수':>10}")
        print(f"{'-' * 40} {'-' * 12} {'-' * 10}")

        for name, query_results in self._results.items():
            valid = [r for r in query_results if r["error"] is None]
            if valid:
                avg_latency = sum(r["latency_ms"] for r in valid) / len(valid)
                avg_results = sum(r["num_results"] for r in valid) / len(valid)
            else:
                avg_latency = 0
                avg_results = 0
            errors = len(query_results) - len(valid)
            suffix = f" ({errors} errors)" if errors > 0 else ""
            print(f"{name:<40} {avg_latency:>9.1f} ms {avg_results:>9.1f}{suffix}")

    def to_dataframe(self):
        """
        결과를 pandas DataFrame으로 변환한다.

        Returns:
            pd.DataFrame: 전략, 쿼리, 레이턴시, 결과수 등.
        """
        try:
            import pandas as pd
        except ImportError:
            print("pandas가 필요합니다: pip install pandas")
            return None

        rows = []
        for name, query_results in self._results.items():
            for r in query_results:
                rows.append(
                    {
                        "strategy": name,
                        "query": r["query"],
                        "num_results": r["num_results"],
                        "latency_ms": r["latency_ms"],
                        "error": r["error"],
                    }
                )
                if "answer" in r:
                    rows[-1]["answer"] = r["answer"]
        return pd.DataFrame(rows)
