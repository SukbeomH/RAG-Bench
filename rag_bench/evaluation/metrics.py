"""
MetricRegistry — RAGAS v0.4+ class-based 메트릭 레지스트리.

프리셋에 따라 적절한 메트릭 인스턴스 리스트를 생성한다.
"""

from enum import Enum
from typing import Any, Dict, List, Optional


class MetricTier(Enum):
    CORE = "core"
    EXTENDED = "extended"
    LIGHTWEIGHT = "lightweight"


class MetricPreset(Enum):
    CORE_ONLY = "core_only"
    FULL = "full"
    REFERENCE_FREE = "reference_free"
    COMPREHENSIVE = "comprehensive"  # Core + 핵심 Extended


METRIC_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Core 4개 (LLM 필요)
    "faithfulness": {
        "cls": "Faithfulness",
        "tier": MetricTier.CORE,
        "requires_reference": False,
        "requires_llm": True,
    },
    "answer_relevancy": {
        "cls": "AnswerRelevancy",
        "tier": MetricTier.CORE,
        "requires_reference": False,
        "requires_llm": True,
    },
    "context_precision": {
        "cls": "ContextPrecision",
        "tier": MetricTier.CORE,
        "requires_reference": True,
        "requires_llm": True,
    },
    "llm_context_recall": {
        "cls": "LLMContextRecall",
        "tier": MetricTier.CORE,
        "requires_reference": True,
        "requires_llm": True,
    },
    # Extended (LLM 필요)
    "answer_correctness": {
        "cls": "AnswerCorrectness",
        "tier": MetricTier.EXTENDED,
        "requires_reference": True,
        "requires_llm": True,
    },
    "factual_correctness": {
        "cls": "FactualCorrectness",
        "tier": MetricTier.EXTENDED,
        "requires_reference": True,
        "requires_llm": True,
    },
    "noise_sensitivity": {
        "cls": "NoiseSensitivity",
        "tier": MetricTier.EXTENDED,
        "requires_reference": True,
        "requires_llm": True,
    },
    # Extended 추가 (RAGAS v0.4+)
    "context_entity_recall": {
        "cls": "ContextEntityRecall",
        "tier": MetricTier.EXTENDED,
        "requires_reference": True,
        "requires_llm": True,
    },
    "response_relevancy": {
        "cls": "ResponseRelevancy",
        "tier": MetricTier.EXTENDED,
        "requires_reference": False,
        "requires_llm": True,
    },
    # Lightweight (LLM 불필요)
    "string_presence": {
        "cls": "StringPresence",
        "tier": MetricTier.LIGHTWEIGHT,
        "requires_reference": True,
        "requires_llm": False,
    },
    "exact_match": {
        "cls": "ExactMatch",
        "tier": MetricTier.LIGHTWEIGHT,
        "requires_reference": True,
        "requires_llm": False,
    },
    "non_llm_string_similarity": {
        "cls": "NonLLMStringSimilarity",
        "tier": MetricTier.LIGHTWEIGHT,
        "requires_reference": True,
        "requires_llm": False,
    },
    "semantic_similarity": {
        "cls": "SemanticSimilarity",
        "tier": MetricTier.LIGHTWEIGHT,
        "requires_reference": True,
        "requires_llm": False,
    },
    "bleu_score": {
        "cls": "BleuScore",
        "tier": MetricTier.LIGHTWEIGHT,
        "requires_reference": True,
        "requires_llm": False,
    },
    "rouge_score": {
        "cls": "RougeScore",
        "tier": MetricTier.LIGHTWEIGHT,
        "requires_reference": True,
        "requires_llm": False,
    },
}


def _get_metrics_for_preset(preset: MetricPreset) -> List[str]:
    """프리셋에 해당하는 메트릭 키 리스트 반환."""
    if preset == MetricPreset.CORE_ONLY:
        return [k for k, v in METRIC_REGISTRY.items() if v["tier"] == MetricTier.CORE]
    elif preset == MetricPreset.FULL:
        return list(METRIC_REGISTRY.keys())
    elif preset == MetricPreset.REFERENCE_FREE:
        return [k for k, v in METRIC_REGISTRY.items() if not v["requires_reference"]]
    elif preset == MetricPreset.COMPREHENSIVE:
        # Core 4 + 핵심 Extended (noise_sensitivity 제외 — 비용 대비 효과 낮음)
        core = [k for k, v in METRIC_REGISTRY.items() if v["tier"] == MetricTier.CORE]
        extended_keys = ["factual_correctness", "context_entity_recall", "response_relevancy"]
        return core + [k for k in extended_keys if k in METRIC_REGISTRY]
    return []


def create_metrics(
    preset: MetricPreset = MetricPreset.CORE_ONLY,
    evaluator_llm: Optional[Any] = None,
    embeddings: Optional[Any] = None,
    has_reference: bool = True,
) -> list:
    """프리셋에 따라 RAGAS v0.4 메트릭 인스턴스 리스트 반환.

    Args:
        preset: 메트릭 프리셋
        evaluator_llm: RAGAS llm_factory LLM 인스턴스 (InstructorBaseRagasLLM)
        embeddings: LangchainEmbeddingsWrapper 인스턴스
        has_reference: reference(ground_truth)가 있는지 여부

    Returns:
        RAGAS 메트릭 인스턴스 리스트
    """
    import ragas.metrics as ragas_metrics

    metric_keys = _get_metrics_for_preset(preset)

    if not has_reference:
        metric_keys = [
            k for k in metric_keys if not METRIC_REGISTRY[k]["requires_reference"]
        ]

    instances = []
    for key in metric_keys:
        info = METRIC_REGISTRY[key]
        cls_name = info["cls"]
        cls = getattr(ragas_metrics, cls_name, None)
        if cls is None:
            continue

        kwargs = {}
        if info["requires_llm"] and evaluator_llm is not None:
            kwargs["llm"] = evaluator_llm
        if not info["requires_llm"] and embeddings is not None:
            if cls_name == "SemanticSimilarity":
                kwargs["embeddings"] = embeddings

        try:
            metric = cls(**kwargs)
            # strictness > 1은 RAGAS 내부적으로 n>1 API 호출을 요청하지만
            # instructor 라이브러리 패치로 인해 OpenAI가 n=1만 반환함.
            # strictness=1로 고정하여 "LLM returned 1 generations instead of 3" 경고 제거.
            if hasattr(metric, "strictness") and metric.strictness > 1:
                metric.strictness = 1
            instances.append(metric)
        except Exception:
            pass

    return instances
