"""
ExtendedRAGEvaluator — RAGAS v0.4+ per-sample 평가 + CostTracker + 스코어링.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from rag_bench.evaluation.metrics import MetricPreset, create_metrics


# ---------------------------------------------------------------------------
# Scoring Profiles
# ---------------------------------------------------------------------------

SCORING_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": {
        "faithfulness": 0.25,
        "answer_relevancy": 0.25,
        "context_precision": 0.25,
        "llm_context_recall": 0.25,
    },
    "precision_critical": {
        "faithfulness": 0.4,
        "context_precision": 0.3,
        "answer_relevancy": 0.3,
    },
    "speed_critical": {
        "answer_relevancy": 0.5,
        "faithfulness": 0.5,
    },
    "comprehensive": {
        "faithfulness": 0.15,
        "answer_relevancy": 0.15,
        "context_precision": 0.15,
        "llm_context_recall": 0.15,
        "factual_correctness": 0.15,
        "context_entity_recall": 0.10,
        "response_relevancy": 0.15,
    },
}


# ---------------------------------------------------------------------------
# CostTracker (내부 클래스)
# ---------------------------------------------------------------------------


@dataclass
class CostTracker:
    """API 비용 추적기."""

    total_tokens: int = 0
    total_cost_usd: float = 0.0
    eval_count: int = 0
    start_time: float = field(default_factory=time.time)

    def record(self, tokens: int = 0, cost: float = 0.0):
        self.total_tokens += tokens
        self.total_cost_usd += cost
        self.eval_count += 1

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "eval_count": self.eval_count,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
        }


# ---------------------------------------------------------------------------
# EvaluationReport
# ---------------------------------------------------------------------------


@dataclass
class EvaluationReport:
    """단일 전략의 평가 결과."""

    strategy_name: str
    per_sample_df: pd.DataFrame
    aggregate_dict: Dict[str, float]
    cost_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> Dict[str, float]:
        """프로파일별 가중 점수 계산."""
        scores = {}
        for profile_name, weights in SCORING_PROFILES.items():
            total = 0.0
            for metric, weight in weights.items():
                total += self.aggregate_dict.get(metric, 0.0) * weight
            scores[profile_name] = round(total, 4)
        return scores


# ---------------------------------------------------------------------------
# ExtendedRAGEvaluator
# ---------------------------------------------------------------------------


class ExtendedRAGEvaluator:
    """RAGAS v0.4+ 기반 확장 평가기.

    기능:
    - class-based 메트릭 + llm_factory (RAGAS 네이티브 LLM)
    - per-sample 점수 반환
    - CostTracker 내장
    - reference 유무에 따라 메트릭 자동 필터링
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        preset: MetricPreset = MetricPreset.CORE_ONLY,
        embeddings: Optional[Any] = None,
    ):
        self.llm_model = llm_model
        self.preset = preset
        self._evaluator_llm = None
        self._embeddings = embeddings
        self._metrics: Optional[List[Any]] = None
        self.cost_tracker = CostTracker()

    def _ensure_initialized(self, has_reference: bool = True):
        """LLM 및 메트릭 lazy 초기화."""
        if self._evaluator_llm is None:
            try:
                import os

                import httpx
                from openai import AsyncOpenAI
                from ragas.llms import llm_factory

                openai_client = AsyncOpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    http_client=httpx.AsyncClient(verify=False),
                )
                self._evaluator_llm = llm_factory(
                    model=self.llm_model,
                    client=openai_client,
                )
            except Exception as e:
                print(f"Warning: LLM 초기화 실패: {e}")
                return

        self._metrics = create_metrics(
            preset=self.preset,
            evaluator_llm=self._evaluator_llm,
            embeddings=self._embeddings,
            has_reference=has_reference,
        )

    def evaluate(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> EvaluationReport:
        """RAGAS 평가 실행. per-sample 결과를 포함한 EvaluationReport 반환."""
        from ragas import EvaluationDataset, SingleTurnSample, evaluate

        has_reference = ground_truths is not None and len(ground_truths) > 0
        self._ensure_initialized(has_reference=has_reference)

        if not self._metrics:
            return EvaluationReport(
                strategy_name="unknown",
                per_sample_df=pd.DataFrame(),
                aggregate_dict={},
            )

        # SingleTurnSample 리스트 생성
        samples = []
        for i in range(len(questions)):
            kwargs = {
                "user_input": questions[i],
                "response": answers[i],
                "retrieved_contexts": contexts[i],
            }
            if has_reference and ground_truths is not None:
                kwargs["reference"] = ground_truths[i]
            samples.append(SingleTurnSample(**kwargs))

        dataset = EvaluationDataset(samples=samples)

        t0 = time.time()
        results = evaluate(dataset=dataset, metrics=self._metrics)
        elapsed = time.time() - t0

        self.cost_tracker.record()

        # per-sample DataFrame
        per_sample_df = results.to_pandas()

        # aggregate (평균)
        aggregate = {}
        for col in per_sample_df.columns:
            if col not in ("user_input", "response", "retrieved_contexts", "reference"):
                vals = per_sample_df[col].dropna()
                if len(vals) > 0:
                    aggregate[col] = round(vals.mean(), 4)

        return EvaluationReport(
            strategy_name="",
            per_sample_df=per_sample_df,
            aggregate_dict=aggregate,
            cost_info={"elapsed_seconds": round(elapsed, 1)},
        )

    def evaluate_strategy(
        self,
        strategy_name: str,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> EvaluationReport:
        """전략 이름을 포함한 평가."""
        report = self.evaluate(questions, contexts, answers, ground_truths)
        report.strategy_name = strategy_name
        return report


# ---------------------------------------------------------------------------
# rank_strategies — 스코어링 유틸리티
# ---------------------------------------------------------------------------


def rank_strategies(
    reports: List[EvaluationReport],
    profile: str = "balanced",
) -> pd.DataFrame:
    """여러 전략의 EvaluationReport를 프로파일별 가중 점수로 순위 매김.

    Args:
        reports: 전략별 EvaluationReport 리스트
        profile: SCORING_PROFILES 키

    Returns:
        순위 DataFrame (strategy, weighted_score, 개별 메트릭 점수)
    """
    if profile not in SCORING_PROFILES:
        raise ValueError(f"Unknown profile: {profile}. 사용 가능: {list(SCORING_PROFILES.keys())}")

    weights = SCORING_PROFILES[profile]
    rows = []
    for report in reports:
        weighted = 0.0
        for metric, weight in weights.items():
            weighted += report.aggregate_dict.get(metric, 0.0) * weight

        row = {"strategy": report.strategy_name, "weighted_score": round(weighted, 4)}
        row.update(report.aggregate_dict)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("weighted_score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based 순위
    df.index.name = "rank"
    return df
