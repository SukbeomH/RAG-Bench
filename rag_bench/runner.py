"""
BenchmarkRunner — 전략 비교 벤치마크 실행기.

여러 RAG 전략을 동일한 쿼리 세트로 실행하고 결과를 비교한다.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from rag_bench.evaluation.evaluator import ExtendedRAGEvaluator

from rag_bench.base import BaseRAGStrategy
from rag_bench.config import DEFAULT_ANSWER_LLM, DEFAULT_LLM_WORKERS
from rag_bench.evaluation.evaluator import EvaluationReport


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
        evaluator: Optional["ExtendedRAGEvaluator"] = None,
        parallel_queries: int = 0,
        parallel_strategies: int = 0,
    ):
        self.strategies = strategies
        self.queries = queries
        self.k = k
        self.evaluator = evaluator
        self.parallel_queries = parallel_queries or int(os.environ.get("RAG_BENCH_PARALLEL", "0"))
        self.parallel_strategies = parallel_strategies or int(os.environ.get("RAG_BENCH_PARALLEL_STRATEGIES", "0"))
        self._results: Dict[str, List[dict]] = {}
        self._reports: Dict[str, "EvaluationReport"] = {}
        self._generator = None  # lazy 초기화

    def _ensure_generator(self):
        """LLM 클라이언트를 필요할 때만 초기화."""
        if self._generator is not None:
            return
        try:
            import httpx
            from langchain_openai import ChatOpenAI

            self._generator = ChatOpenAI(
                model=DEFAULT_ANSWER_LLM, http_client=httpx.Client(verify=False)
            )
        except Exception:
            self._generator = None

    def run(self) -> Dict[str, List[dict]]:
        """
        모든 전략에 대해 쿼리를 실행하고 결과를 수집한다.

        parallel_strategies > 1이면 여러 전략을 동시에 실행한다.
        parallel_queries > 0이면 전략 내 쿼리를 병렬 실행한다.

        Returns:
            전략 이름 → 쿼리별 결과 목록.
        """
        self._results = {}

        if self.parallel_strategies > 1:
            return self._run_strategies_parallel()

        for strategy in self.strategies:
            strategy_name = strategy.name
            print(f"\n{'=' * 60}")
            print(f"전략: {strategy_name}")
            print(f"설명: {strategy.description}")
            print(f"{'=' * 60}")

            if self.parallel_queries > 1:
                query_results = self._run_parallel(strategy)
            else:
                query_results = []
                for query in self.queries:
                    result = self._run_single(strategy, query)
                    query_results.append(result)

            self._results[strategy_name] = query_results

        return self._results

    def _run_strategy_all_queries(self, strategy: BaseRAGStrategy) -> tuple:
        """단일 전략의 모든 쿼리를 실행하고 (name, results) 반환."""
        strategy_name = strategy.name
        print(f"\n{'=' * 60}")
        print(f"전략: {strategy_name}")
        print(f"설명: {strategy.description}")
        print(f"{'=' * 60}")

        if self.parallel_queries > 1:
            query_results = self._run_parallel(strategy)
        else:
            query_results = []
            for query in self.queries:
                result = self._run_single(strategy, query)
                query_results.append(result)

        return strategy_name, query_results

    def _run_strategies_parallel(self) -> Dict[str, List[dict]]:
        """여러 전략을 ThreadPool으로 병렬 실행."""
        results: Dict[str, List[dict]] = {}
        workers = min(self.parallel_strategies, len(self.strategies))
        print(f"\n  [병렬 모드] {len(self.strategies)}개 전략을 {workers}개 워커로 실행")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self._run_strategy_all_queries, strategy): strategy
                for strategy in self.strategies
            }
            for future in as_completed(futures):
                try:
                    name, query_results = future.result()
                    results[name] = query_results
                except Exception as e:
                    strategy = futures[future]
                    print(f"  [오류] {strategy.name}: {e}")
                    results[strategy.name] = []

        # strategies 순서 유지
        ordered: Dict[str, List[dict]] = {}
        for strategy in self.strategies:
            if strategy.name in results:
                ordered[strategy.name] = results[strategy.name]
        self._results = ordered
        return ordered

    def _run_parallel(self, strategy: BaseRAGStrategy) -> List[dict]:
        """전략 내 쿼리들을 ThreadPool으로 병렬 실행."""
        results: List[Optional[dict]] = [None] * len(self.queries)
        with ThreadPoolExecutor(max_workers=self.parallel_queries) as executor:
            futures = {
                executor.submit(self._run_single, strategy, query): i
                for i, query in enumerate(self.queries)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results  # type: ignore[return-value]

    def inject_results(self, results: Dict[str, List[dict]]) -> None:
        """외부에서 수집된 검색 결과를 주입하여 재검색을 방지한다."""
        strategy_names = {s.name for s in self.strategies}
        self._results = {
            name: qr for name, qr in results.items()
            if name in strategy_names
        }
        injected = len(self._results)
        print(f"  결과 주입 완료: {injected}개 전략 (재검색 생략)")

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

            # Generate answers if not present (병렬 LLM 호출)
            self._ensure_generator()
            answers: List[Optional[str]] = [None] * len(query_results)
            pending: List[tuple] = []  # (index, prompt) 튜플

            for i, r in enumerate(query_results):
                if "answer" in r:
                    answers[i] = r["answer"]
                elif self._generator:
                    context_str = "\n\n".join(contexts[i])
                    prompt = f"Context:\n{context_str}\n\nQuestion: {questions[i]}\nAnswer:"
                    pending.append((i, prompt))
                else:
                    answers[i] = "No generator available."

            if pending and self._generator:
                def _invoke(prompt):
                    try:
                        return self._generator.invoke(prompt).content
                    except Exception:
                        return "Generation failed."

                with ThreadPoolExecutor(max_workers=DEFAULT_LLM_WORKERS) as executor:
                    futures = {
                        executor.submit(_invoke, prompt): idx
                        for idx, prompt in pending
                    }
                    for future in as_completed(futures):
                        idx = futures[future]
                        ans = future.result()
                        answers[idx] = ans
                        query_results[idx]["answer"] = ans

            # Prepare ground truths
            gts = None
            if ground_truths:
                gts = ground_truths

            # Evaluate
            try:
                result = self.evaluator.evaluate(
                    questions=questions,
                    contexts=contexts,
                    answers=answers,  # type: ignore[arg-type]
                    ground_truths=gts,  # type: ignore[arg-type]
                )

                result.strategy_name = name
                scores_dict: Dict[str, Any] = {
                    k: round(v, 4) for k, v in result.aggregate_dict.items()
                    if isinstance(v, (int, float))
                }
                self._reports[name] = result

                scores_dict["strategy"] = name
                all_scores.append(scores_dict)

                print(f"  -> {scores_dict}")

            except Exception as e:
                print(f"Evaluation failed for {name}: {e}")

        return pd.DataFrame(all_scores)

    @property
    def reports(self) -> Dict[str, "EvaluationReport"]:
        """ExtendedRAGEvaluator 사용 시 per-sample 리포트 접근."""
        return self._reports

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
