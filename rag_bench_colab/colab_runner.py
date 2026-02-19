"""
ColabBenchmarkRunner — 체크포인트 지원 Colab 벤치마크 러너.

run_all_combos.py의 2-Pass 벤치마크를 Colab 환경에 맞게 래핑:
- Google Drive 체크포인트로 중단/재개
- tqdm.notebook 진행률
- CUDA 디바이스 자동 감지
- GraphRAG 통합
"""

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from rag_bench_colab.colab_config import (
    DRIVE_BENCHDATA_DIR,
    DRIVE_CHECKPOINTS_DIR,
    DRIVE_RESULTS_DIR,
    get_qdrant_path,
    release_memory,
)


# ---------------------------------------------------------------------------
# 체크포인트 매니저
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Google Drive 기반 체크포인트 관리."""

    def __init__(self, session_id: str, checkpoint_dir: Optional[Path] = None):
        self.session_id = session_id
        self.base_dir = checkpoint_dir or DRIVE_CHECKPOINTS_DIR
        self.session_dir = self.base_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, data: Any) -> None:
        """체크포인트 저장."""
        path = self.session_dir / f"{key}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")

    def load(self, key: str) -> Optional[Any]:
        """체크포인트 로드. 없으면 None."""
        path = self.session_dir / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def exists(self, key: str) -> bool:
        return (self.session_dir / f"{key}.json").exists()

    def save_metadata(self, metadata: dict) -> None:
        self.save("_metadata", metadata)

    def list_completed(self, prefix: str) -> List[str]:
        """완료된 체크포인트 키 목록."""
        return [
            p.stem for p in self.session_dir.glob(f"{prefix}*.json")
            if not p.stem.startswith("_")
        ]


# ---------------------------------------------------------------------------
# ColabBenchmarkRunner
# ---------------------------------------------------------------------------


class ColabBenchmarkRunner:
    """Colab 전용 2-Pass 벤치마크 러너.

    Usage:
        runner = ColabBenchmarkRunner(preset="quick", k=3, top_n=4)
        child_chunks, parent_pairs, queries, ground_truths = runner.prepare_data()
        combos = runner.generate_combos()
        lat_df = runner.run_pass1(combos, queries, child_chunks, parent_pairs)
        ragas_df = runner.run_pass2(lat_df, combos, queries, ground_truths, child_chunks, parent_pairs)
        runner.export_results(lat_df, ragas_df)
    """

    def __init__(
        self,
        preset: str = "quick",
        k: int = 3,
        top_n: int = 10,
        qdrant_mode: str = "ephemeral",
        device: Optional[str] = None,
        session_id: Optional[str] = None,
        parallel_queries: int = 0,
        reindex: bool = False,
    ):
        self.preset = preset
        self.k = k
        self.top_n = top_n
        self.qdrant_mode = qdrant_mode
        self.parallel_queries = parallel_queries
        self.reindex = reindex

        if device is None:
            from rag_bench_colab.colab_config import get_device
            device = get_device()
        self.device = device

        self.session_id = session_id or f"{preset}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint = CheckpointManager(self.session_id)

        # 세션 메타데이터 저장
        self.checkpoint.save_metadata({
            "preset": preset,
            "k": k,
            "top_n": top_n,
            "qdrant_mode": qdrant_mode,
            "device": device,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        self._index_cache = None
        self._tqdm = None
        self._pass1_results = {}  # Pass 1 검색 결과 (Pass 2 재사용용)

    def _get_tqdm(self):
        """노트북 tqdm 또는 표준 tqdm 반환."""
        if self._tqdm is None:
            try:
                from tqdm.notebook import tqdm
                self._tqdm = tqdm
            except ImportError:
                from tqdm import tqdm
                self._tqdm = tqdm
        return self._tqdm

    # ------------------------------------------------------------------
    # 데이터 준비
    # ------------------------------------------------------------------

    def prepare_data(self) -> Tuple[list, list, list, list]:
        """QA 데이터셋 로드 + Parent-Child 청킹.

        Returns:
            (child_chunks, parent_pairs, queries, ground_truths)
        """
        from rag_bench.config import BENCH_DATA_DIR, BENCH_DOCS_DIR
        from rag_bench.indexing.chunker import create_parent_child_chunks

        # QA 데이터셋 로드
        qa_path = BENCH_DATA_DIR / "qa_dataset.json"
        if not qa_path.exists():
            # Colab data 디렉토리에서 복사
            from rag_bench_colab.colab_config import COLAB_DATA_DIR
            src = COLAB_DATA_DIR / "qa_dataset.json"
            if src.exists():
                import shutil
                qa_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, qa_path)
            else:
                raise FileNotFoundError(f"QA 데이터셋을 찾을 수 없습니다: {qa_path}")

        dataset = json.loads(qa_path.read_text(encoding="utf-8"))
        qa_pairs = dataset["qa_pairs"]
        queries = [qa["question"] for qa in qa_pairs]
        ground_truths = [qa["ground_truth"] for qa in qa_pairs]
        print(f"[Data] QA 데이터셋: {len(queries)}개 쿼리")

        # Parent-Child 청킹
        parent_store_path = BENCH_DATA_DIR / "parent_store"
        parent_pairs, child_chunks = create_parent_child_chunks(
            markdown_dir=str(BENCH_DOCS_DIR),
            parent_store_path=str(parent_store_path),
        )
        print(f"[Data] 청킹 완료: Parent {len(parent_pairs)}개, Child {len(child_chunks)}개")

        return child_chunks, parent_pairs, queries, ground_truths

    # ------------------------------------------------------------------
    # 조합 생성
    # ------------------------------------------------------------------

    def generate_combos(self) -> list:
        """프리셋 기반 ComboSpec 목록 생성."""
        from rag_bench.scripts.run_all_combos import PRESETS, generate_valid_combinations

        if self.preset not in PRESETS:
            raise ValueError(f"알 수 없는 프리셋: {self.preset}. 사용 가능: {list(PRESETS.keys())}")

        config = PRESETS[self.preset]
        combos = generate_valid_combinations(config)
        print(f"[Combos] 프리셋 '{self.preset}': {len(combos)}개 조합 생성")
        return combos

    # ------------------------------------------------------------------
    # Pass 1: 레이턴시 벤치마크
    # ------------------------------------------------------------------

    def run_pass1(
        self,
        combos: list,
        queries: list,
        child_chunks: list,
        parent_pairs: list,
    ) -> pd.DataFrame:
        """Pass 1: 전체 조합 레이턴시 측정 (체크포인트 지원).

        Returns:
            레이턴시 결과 DataFrame.
        """
        from rag_bench.runner import BenchmarkRunner
        from rag_bench.scripts.run_all_combos import (
            IndexCacheManager,
            build_strategy_from_spec,
        )

        tqdm = self._get_tqdm()

        if self._index_cache is None:
            self._index_cache = IndexCacheManager()

        strategies = []
        completed_labels = set(self.checkpoint.list_completed("pass1_"))
        all_latency_rows = []

        # 이전 체크포인트 결과 로드
        for label in completed_labels:
            data = self.checkpoint.load(label)
            if data:
                all_latency_rows.extend(data)

        print(f"\n[Pass 1] 레이턴시 벤치마크 시작")
        print(f"  조합: {len(combos)}개, 쿼리: {len(queries)}개")
        print(f"  이미 완료: {len(completed_labels)}개")

        for i, spec in enumerate(tqdm(combos, desc="Pass 1: 전략 빌드 & 검색")):
            ckpt_key = f"pass1_{spec.label}"

            if ckpt_key in completed_labels:
                continue

            strategy = None
            try:
                # Qdrant 경로 오버라이드
                strategy = self._build_strategy(
                    spec, child_chunks, parent_pairs
                )

                if strategy is None:
                    continue

                # 쿼리별 레이턴시 측정
                runner = BenchmarkRunner(
                    strategies=[strategy],
                    queries=queries,
                    k=self.k,
                    evaluator=None,
                    parallel_queries=self.parallel_queries,
                )
                runner.run()

                # Pass 1 결과 저장 (Pass 2 재사용용)
                self._pass1_results.update(runner._results)

                df = runner.to_dataframe()

                if df is not None:
                    rows = df.to_dict("records")
                    all_latency_rows.extend(rows)

                    # 체크포인트 저장
                    self.checkpoint.save(ckpt_key, rows)

                strategies.append((spec, strategy))

            except Exception as e:
                print(f"\n  [Error] {spec.label}: {e}")

            # Reranker 래핑 전략만 cleanup (base strategy는 캐시에 보존)
            if strategy is not None and hasattr(strategy, '_base_strategy'):
                try:
                    strategy.cleanup()
                except Exception:
                    pass

            release_memory()

        lat_df = pd.DataFrame(all_latency_rows)

        # 전략별 평균 레이턴시 계산
        if not lat_df.empty and "latency_ms" in lat_df.columns:
            summary = (
                lat_df.groupby("strategy")["latency_ms"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            summary.columns = ["strategy", "avg_latency_ms", "std_latency_ms", "query_count"]
            summary["avg_latency"] = summary["avg_latency_ms"] / 1000
            summary = summary.sort_values("avg_latency")
            print(f"\n[Pass 1] 완료: {len(summary)}개 전략 측정")
            return summary

        return lat_df

    # ------------------------------------------------------------------
    # Pass 2: RAGAS 평가
    # ------------------------------------------------------------------

    def run_pass2(
        self,
        latency_df: pd.DataFrame,
        combos: list,
        queries: list,
        ground_truths: list,
        child_chunks: list,
        parent_pairs: list,
    ) -> pd.DataFrame:
        """Pass 2: 상위 N개 전략 RAGAS 평가 (체크포인트 지원).

        Returns:
            RAGAS 점수 DataFrame.
        """
        from rag_bench.evaluation import ExtendedRAGEvaluator
        from rag_bench.runner import BenchmarkRunner
        from rag_bench.scripts.run_all_combos import IndexCacheManager

        tqdm = self._get_tqdm()

        # 상위 N개 선별
        if "avg_latency" in latency_df.columns:
            top_strategies = latency_df.nsmallest(self.top_n, "avg_latency")["strategy"].tolist()
        else:
            top_strategies = latency_df["strategy"].head(self.top_n).tolist()

        # 전략명 → ComboSpec 매핑
        spec_map = {spec.label: spec for spec in combos}

        print(f"\n[Pass 2] RAGAS 평가 시작")
        print(f"  상위 {len(top_strategies)}개 전략:")
        for i, name in enumerate(top_strategies, 1):
            lat = latency_df.loc[latency_df["strategy"] == name, "avg_latency"].values
            lat_str = f"{lat[0]:.3f}s" if len(lat) > 0 else "N/A"
            print(f"    {i}. {name} ({lat_str})")

        evaluator = ExtendedRAGEvaluator(llm_model="gpt-4o-mini")
        completed_labels = set(self.checkpoint.list_completed("pass2_"))
        all_scores = []

        # 이전 체크포인트 결과 로드
        for label in completed_labels:
            data = self.checkpoint.load(label)
            if data:
                all_scores.append(data)

        if self._index_cache is None:
            self._index_cache = IndexCacheManager()

        for strategy_name in tqdm(top_strategies, desc="Pass 2: RAGAS 평가"):
            ckpt_key = f"pass2_{strategy_name}"
            if ckpt_key in completed_labels:
                continue

            # 전략 빌드 (캐시 활용)
            spec = None
            for s in combos:
                if s.label == strategy_name or self._strategy_name_from_spec(s) == strategy_name:
                    spec = s
                    break

            if spec is None:
                print(f"  [Skip] {strategy_name}: ComboSpec 매칭 실패")
                continue

            try:
                strategy = self._build_strategy(spec, child_chunks, parent_pairs)
                if strategy is None:
                    continue

                runner = BenchmarkRunner(
                    strategies=[strategy],
                    queries=queries,
                    k=self.k,
                    evaluator=None,
                    parallel_queries=self.parallel_queries,
                )

                # Pass 1 결과 재사용 (재검색 방지)
                strategy_name_candidate = strategy.name
                if self._pass1_results and strategy_name_candidate in self._pass1_results:
                    runner.inject_results({strategy_name_candidate: self._pass1_results[strategy_name_candidate]})
                else:
                    runner.run()  # 폴백: Pass 1 결과 없으면 재검색

                # 검색 결과에서 RAGAS 평가
                results = runner._results
                for name, query_results in results.items():
                    questions = [r["query"] for r in query_results]
                    contexts = [[d["content"] for d in r["results"]] for r in query_results]

                    # Answer 생성은 evaluator 내부에서 처리하지 않으므로 별도 생성
                    answers = self._generate_answers(questions, contexts)

                    report = evaluator.evaluate_strategy(
                        strategy_name=name,
                        questions=questions,
                        contexts=contexts,
                        answers=answers,
                        ground_truths=ground_truths,
                    )

                    score_dict = {"strategy": name}
                    score_dict.update(report.aggregate_dict)
                    all_scores.append(score_dict)

                    # 체크포인트 저장
                    self.checkpoint.save(ckpt_key, score_dict)

                    print(f"  [{name}] {report.aggregate_dict}")

            except Exception as e:
                print(f"  [Error] {strategy_name}: {e}")

            release_memory()

        ragas_df = pd.DataFrame(all_scores)
        print(f"\n[Pass 2] 완료: {len(all_scores)}개 전략 평가")
        return ragas_df

    # ------------------------------------------------------------------
    # GraphRAG 실행
    # ------------------------------------------------------------------

    def run_graphrag(
        self,
        parent_pairs: list,
        queries: list,
        ground_truths: list,
    ) -> Optional[Dict]:
        """GraphRAG 별도 실행.

        Returns:
            {'latency': dict, 'ragas': dict} 또는 None.
        """
        from rag_bench.config import BENCH_DATA_DIR
        from rag_bench.runner import BenchmarkRunner
        from rag_bench.strategies.graph_rag import GraphRAGStrategy

        print(f"\n[GraphRAG] LightRAG 실행 시작")

        working_dir = str(BENCH_DATA_DIR / "lightrag_graphrag")
        strategy = GraphRAGStrategy(
            mode="hybrid",
            working_dir=working_dir,
            llm_model="gpt-4.1-nano",
            top_k=60,
        )

        parent_docs = [doc for _, doc in parent_pairs]

        try:
            strategy.index(parent_docs)

            runner = BenchmarkRunner(
                strategies=[strategy],
                queries=queries,
                k=self.k,
                evaluator=None,
            )
            runner.run()
            runner.compare()

            lat_df = runner.to_dataframe()
            result = {"latency": lat_df.to_dict() if lat_df is not None else {}}

            # RAGAS 평가
            from rag_bench.evaluation import ExtendedRAGEvaluator
            evaluator = ExtendedRAGEvaluator(llm_model="gpt-4o-mini")

            for name, query_results in runner._results.items():
                questions = [r["query"] for r in query_results]
                contexts = [[d["content"] for d in r["results"]] for r in query_results]
                answers = self._generate_answers(questions, contexts)

                report = evaluator.evaluate_strategy(
                    strategy_name=name,
                    questions=questions,
                    contexts=contexts,
                    answers=answers,
                    ground_truths=ground_truths,
                )
                result["ragas"] = {"strategy": name, **report.aggregate_dict}

            self.checkpoint.save("graphrag", result)
            print(f"[GraphRAG] 완료")
            return result

        except Exception as e:
            print(f"[GraphRAG] 실패: {e}")
            return None
        finally:
            release_memory()

    # ------------------------------------------------------------------
    # 결과 Export
    # ------------------------------------------------------------------

    def export_results(
        self,
        latency_df: Optional[pd.DataFrame] = None,
        ragas_df: Optional[pd.DataFrame] = None,
        graphrag_result: Optional[Dict] = None,
    ) -> Path:
        """결과를 Google Drive에 저장.

        Returns:
            결과 디렉토리 경로.
        """
        output_dir = DRIVE_RESULTS_DIR / self.session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if latency_df is not None:
            lat_path = output_dir / "latency.csv"
            latency_df.to_csv(lat_path, index=False, encoding="utf-8-sig")
            print(f"  레이턴시: {lat_path}")

        if ragas_df is not None:
            ragas_path = output_dir / "ragas.csv"
            ragas_df.to_csv(ragas_path, index=False, encoding="utf-8-sig")
            print(f"  RAGAS: {ragas_path}")

        if graphrag_result is not None:
            grag_path = output_dir / "graphrag.json"
            grag_path.write_text(
                json.dumps(graphrag_result, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            print(f"  GraphRAG: {grag_path}")

        # Markdown 리포트 생성
        report = self._generate_report(latency_df, ragas_df, graphrag_result)
        report_path = output_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"  리포트: {report_path}")

        print(f"\n[Export] 결과 저장 완료: {output_dir}")
        return output_dir

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _build_strategy(self, spec, child_chunks, parent_pairs):
        """ComboSpec에서 전략 인스턴스 생성 (Qdrant 경로 오버라이드)."""
        from rag_bench.scripts.run_all_combos import (
            IndexCacheManager,
            build_strategy_from_spec,
        )

        # Qdrant 경로를 Colab 모드에 맞게 오버라이드
        if self.qdrant_mode != "ephemeral":
            # BENCH_DATA_DIR의 qdrant 경로를 사용
            pass  # config가 이미 패치됨

        try:
            strategy = build_strategy_from_spec(
                spec, self._index_cache, child_chunks, parent_pairs, reindex=self.reindex
            )
            return strategy
        except Exception as e:
            print(f"  [Build Error] {spec.label}: {e}")
            return None

    def _strategy_name_from_spec(self, spec) -> str:
        """ComboSpec에서 전략 이름 추론."""
        dense_short = spec.dense
        if "/" in dense_short:
            dense_short = dense_short.split("/")[-1]
        return f"DS({dense_short}+{spec.sparse})"

    def _generate_answers(self, questions: list, contexts: list) -> list:
        """LLM으로 답변 생성."""
        try:
            import httpx
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                http_client=httpx.Client(verify=False),
            )

            def _invoke(prompt):
                try:
                    return llm.invoke(prompt).content
                except Exception:
                    return "Generation failed."

            prompts = []
            for i in range(len(questions)):
                ctx = "\n\n".join(contexts[i])
                prompts.append(f"Context:\n{ctx}\n\nQuestion: {questions[i]}\nAnswer:")

            answers = [None] * len(questions)
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {
                    executor.submit(_invoke, p): i for i, p in enumerate(prompts)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    answers[idx] = future.result()

            return answers

        except Exception as e:
            print(f"  [Warning] 답변 생성 실패: {e}")
            return ["No answer available."] * len(questions)

    def _generate_report(
        self,
        latency_df: Optional[pd.DataFrame],
        ragas_df: Optional[pd.DataFrame],
        graphrag_result: Optional[Dict],
    ) -> str:
        """Markdown 리포트 생성."""
        lines = [
            f"# RAG Benchmark Report — {self.session_id}",
            "",
            f"**프리셋**: {self.preset}",
            f"**k**: {self.k}",
            f"**Top-N**: {self.top_n}",
            f"**디바이스**: {self.device}",
            f"**생성 시각**: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ]

        if latency_df is not None and not latency_df.empty:
            lines.append("## Pass 1: 레이턴시 결과")
            lines.append("")
            lines.append("| # | 전략 | 평균 레이턴시 |")
            lines.append("|---|------|:----------:|")

            sort_col = "avg_latency" if "avg_latency" in latency_df.columns else "avg_latency_ms"
            if sort_col in latency_df.columns:
                sorted_df = latency_df.sort_values(sort_col)
                for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
                    val = row.get("avg_latency", row.get("avg_latency_ms", 0) / 1000)
                    lines.append(f"| {i} | {row['strategy']} | {val:.3f}s |")
            lines.append("")

        if ragas_df is not None and not ragas_df.empty:
            lines.append("## Pass 2: RAGAS 평가 결과")
            lines.append("")
            metric_cols = [c for c in ragas_df.columns if c != "strategy"]
            header = "| 전략 | " + " | ".join(metric_cols) + " |"
            sep = "|------|" + "|".join(":---:" for _ in metric_cols) + "|"
            lines.append(header)
            lines.append(sep)
            for _, row in ragas_df.iterrows():
                vals = " | ".join(
                    f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c])
                    for c in metric_cols
                )
                lines.append(f"| {row['strategy']} | {vals} |")
            lines.append("")

        if graphrag_result and "ragas" in graphrag_result:
            lines.append("## GraphRAG 결과")
            lines.append("")
            ragas = graphrag_result["ragas"]
            for k, v in ragas.items():
                if k != "strategy":
                    lines.append(f"- **{k}**: {v}")
            lines.append("")

        return "\n".join(lines)
