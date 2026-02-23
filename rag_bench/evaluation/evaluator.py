"""
ExtendedRAGEvaluator — RAGAS v0.4+ per-sample 평가 + CostTracker + 스코어링.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField

from rag_bench.config import DEFAULT_EVAL_LLM
from rag_bench.evaluation.metrics import MetricPreset, create_metrics


# ---------------------------------------------------------------------------
# 다양한 관점 역질문 생성 — Pydantic 구조화 출력 모델
# ---------------------------------------------------------------------------


class _ReverseQuestion(PydanticBaseModel):
    """단일 역질문 (RAGAS AnswerRelevancy/ResponseRelevancy 호환 형식)."""

    question: str = PydanticField(description="역질문 텍스트")
    noncommittal: int = PydanticField(
        default=0,
        description="0=확정적 답변에서 유추 가능한 질문, 1=모호한 답변",
    )


class _MultiPerspectiveOutput(PydanticBaseModel):
    """다양한 관점의 역질문 n개를 단일 API 호출로 생성하기 위한 구조화 출력 스키마."""

    questions: List[_ReverseQuestion] = PydanticField(
        description="서로 다른 관점에서 생성된 역질문 목록"
    )


_MULTI_PERSPECTIVE_SYSTEM = (
    "당신은 RAG 평가를 위한 역질문 생성 전문가입니다. "
    "주어진 답변과 컨텍스트를 분석하여, 해당 답변이 원래 어떤 질문에 대한 것인지 "
    "다양한 시각(목적/방법/비교/한계/배경/응용 등)에서 유추하여 역질문을 생성합니다."
)

_MULTI_PERSPECTIVE_USER_SUFFIX = (
    "\n\n위 내용을 바탕으로 **{n}개의 서로 다른 관점**에서 역질문을 생성하세요. "
    "각 질문은 독립적이어야 하며 다양한 시각(목적/방법/비교/한계/배경/응용 등)을 반영해야 합니다."
)


class _MultiPerspectiveLLM:
    """단일 API 호출로 n개의 다양한 관점 역질문을 생성하는 RAGAS LLM 래퍼.

    RAGAS가 agenerate_text(prompt, n>1)를 호출할 때:
      1. ChatOpenAI.with_structured_output(_MultiPerspectiveOutput) 단일 호출
      2. 응답 JSON에서 n개의 _ReverseQuestion 파싱
      3. 각각을 RAGAS Generation(text=json) 형태로 변환하여 반환

    n=1인 경우 base_llm(llm_factory 기반)에 위임 → 기존 instructor 동작 유지.
    """

    def __init__(self, base_llm: Any, model: str, verify_ssl: bool = False) -> None:
        self._base_llm = base_llm
        self._model = model
        self._verify_ssl = verify_ssl
        self._structured_llm: Optional[Any] = None  # lazy init

    def _get_structured_llm(self) -> Any:
        """ChatOpenAI.with_structured_output 인스턴스를 lazy 초기화한다."""
        if self._structured_llm is None:
            import httpx
            from langchain_openai import ChatOpenAI

            chat_llm = ChatOpenAI(
                model=self._model,
                http_client=httpx.Client(verify=self._verify_ssl),
                http_async_client=httpx.AsyncClient(verify=self._verify_ssl),
                temperature=0.7,
            )
            # json_schema: OpenAI Structured Outputs — 스키마 준수 보장
            self._structured_llm = chat_llm.with_structured_output(
                _MultiPerspectiveOutput,
                method="json_schema",
            )
        return self._structured_llm

    async def agenerate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 1e-8,
        stop: Optional[List[str]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Any:
        """n>1이면 단일 구조화 호출로 n개 역질문을 생성한다."""
        callbacks = callbacks or []
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.outputs import Generation, LLMResult

        if n > 1:
            prompt_str = (
                prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
            )
            user_content = prompt_str + _MULTI_PERSPECTIVE_USER_SUFFIX.format(n=n)

            structured_llm = self._get_structured_llm()
            result: _MultiPerspectiveOutput = await structured_llm.ainvoke(
                [
                    SystemMessage(content=_MULTI_PERSPECTIVE_SYSTEM),
                    HumanMessage(content=user_content),
                ]
            )

            items = list(result.questions)
            # n개 맞추기 (부족 시 패딩, 초과 시 슬라이스)
            while len(items) < n:
                items.append(_ReverseQuestion(question="", noncommittal=0))
            items = items[:n]

            gens = [
                Generation(
                    text=json.dumps(
                        {"question": q.question, "noncommittal": q.noncommittal},
                        ensure_ascii=False,
                    )
                )
                for q in items
            ]
            return LLMResult(generations=[gens])

        # n=1: instructor 기반 base_llm에 위임
        return await self._base_llm.agenerate_text(
            prompt, n, temperature, stop, callbacks
        )

    def generate_text(
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 1e-8,
        stop: Optional[List[str]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Any:
        """동기 래퍼 — 이벤트 루프 중첩 방지."""
        callbacks = callbacks or []
        import asyncio
        import concurrent.futures

        try:
            asyncio.get_running_loop()
            # 이미 실행 중인 루프가 있으면 별도 스레드에서 실행
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.agenerate_text(prompt, n, temperature, stop, callbacks),
                ).result()
        except RuntimeError:
            return asyncio.run(
                self.agenerate_text(prompt, n, temperature, stop, callbacks)
            )

    async def generate(
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Any:
        """RAGAS BaseRagasLLM.generate() 인터페이스 구현 — n 파라미터 지원."""
        callbacks = callbacks or []
        return await self.agenerate_text(prompt, n, temperature or 1e-8, stop, callbacks)

    def __getattr__(self, name: str) -> Any:
        """그 외 RAGAS LLM 인터페이스는 base_llm에 위임."""
        return getattr(self._base_llm, name)


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
        llm_model: str = DEFAULT_EVAL_LLM,
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
                import warnings

                import httpx
                from langchain_openai import ChatOpenAI
                from ragas.llms import LangchainLLMWrapper

                chat_llm = ChatOpenAI(
                    model=self.llm_model,
                    http_client=httpx.Client(verify=False),
                    http_async_client=httpx.AsyncClient(verify=False),
                )
                # LangchainLLMWrapper는 BaseRagasLLM을 구현하므로 agenerate_text() 지원
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    base_llm = LangchainLLMWrapper(chat_llm)
                # n>1 요청을 단일 구조화 호출로 처리하는 래퍼 적용
                self._evaluator_llm = _MultiPerspectiveLLM(
                    base_llm=base_llm,
                    model=self.llm_model,
                    verify_ssl=False,
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
        from ragas import RunConfig
        results = evaluate(
            dataset=dataset,
            metrics=self._metrics,
            run_config=RunConfig(max_workers=16, timeout=180),
        )
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
