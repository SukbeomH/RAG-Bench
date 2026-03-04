"""
RAGAS 평가 가중치 및 메트릭 컬럼 상수.

ranker / generate_k8s_report 등에서 공유하는 평가 기준.
"""

RAGAS_WEIGHTS = {
    "context_recall": 0.35,
    "context_precision": 0.30,
    "faithfulness": 0.20,
    "answer_relevancy": 0.15,
}

RAGAS_COLS = list(RAGAS_WEIGHTS.keys())
