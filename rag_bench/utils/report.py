"""RAGAS 결과 출력 유틸리티."""
from typing import Optional

import pandas as pd


def print_ragas_table(
    scores_df: Optional[pd.DataFrame],
    scoring_profile: str = "balanced",
) -> None:
    """RAGAS 평가 결과를 콘솔에 포맷팅하여 출력한다.

    Args:
        scores_df: 전략별 메트릭 DataFrame ('strategy' 컬럼 + 메트릭 컬럼들).
        scoring_profile: SCORING_PROFILES 키 (balanced, precision_critical 등).
    """
    if scores_df is None or scores_df.empty:
        print("RAGAS 평가 결과가 없습니다.")
        return

    from rag_bench.evaluation.evaluator import SCORING_PROFILES

    print(f"\n{'═' * 100}")
    print(f" RAGAS 평가 결과 비교 (scoring: {scoring_profile})")
    print(f"{'═' * 100}")

    metric_cols = [c for c in scores_df.columns if c != "strategy"]

    # 가중 점수 계산
    weights = SCORING_PROFILES.get(scoring_profile, SCORING_PROFILES["balanced"])
    weighted_scores = []
    for _, row in scores_df.iterrows():
        ws = 0.0
        for metric, weight in weights.items():
            val = row.get(metric, 0.0)
            if isinstance(val, (int, float)):
                ws += val * weight
        weighted_scores.append(round(ws, 4))

    display_cols = metric_cols + ["weighted"]
    header = f"  {'전략':<45}"
    for col in display_cols:
        header += f" {col:>14}"
    print(header)
    print(f"  {'─' * 45} " + " ".join("─" * 14 for _ in display_cols))

    for i, (_, row) in enumerate(scores_df.iterrows()):
        line = f"  {row['strategy']:<45}"
        for col in metric_cols:
            val = row.get(col, "N/A")
            if isinstance(val, float):
                line += f" {val:>14.4f}"
            else:
                line += f" {str(val):>14}"
        line += f" {weighted_scores[i]:>14.4f}"
        print(line)
