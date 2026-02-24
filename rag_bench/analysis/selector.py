"""
selector — 카테고리별 최종 모델 선정 근거 생성.

출력:
  - 카테고리별 1순위 조합 + 선정 이유
  - 공통 우승 조합 (여러 카테고리에서 1위)
  - 기본 추천 1개 ("모르면 이걸 써라")
  - 선정 불가 카테고리 (데이터 부족)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from rag_bench.analysis.insight import rank_strategies_overall


@dataclass
class CategoryRecommendation:
    """카테고리별 추천 결과."""
    category: str
    winner: str                    # 1위 조합
    composite_score: float         # 복합 점수
    pass_rate: float               # pass_rate (%)
    avg_latency_ms: Optional[float]
    reason: str                    # 선정 이유
    runner_up: Optional[str] = None  # 2위 (참고)


@dataclass
class SelectionReport:
    """최종 선정 보고서 데이터 구조."""
    per_category: Dict[str, CategoryRecommendation] = field(default_factory=dict)
    common_winners: List[str] = field(default_factory=list)
    default_recommendation: Optional[str] = None
    default_reason: str = ""
    skipped_categories: List[str] = field(default_factory=list)
    overall_ranking: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "per_category": {
                cat: {
                    "winner": r.winner,
                    "composite_score": r.composite_score,
                    "pass_rate": r.pass_rate,
                    "avg_latency_ms": r.avg_latency_ms,
                    "reason": r.reason,
                    "runner_up": r.runner_up,
                }
                for cat, r in self.per_category.items()
            },
            "common_winners": self.common_winners,
            "default_recommendation": self.default_recommendation,
            "default_reason": self.default_reason,
            "skipped_categories": self.skipped_categories,
            "overall_ranking": self.overall_ranking,
        }


def generate_selection_report(
    ranked: Dict[str, pd.DataFrame],
    insights: Dict[str, dict],
    compressed: Dict[str, pd.DataFrame],
) -> SelectionReport:
    """
    분석 결과를 종합하여 최종 선정 보고서를 생성한다.

    Args:
        ranked: rank_by_doc_type() 반환값
        insights: analyze_strengths_weaknesses() 반환값
        compressed: compress_similar_results() 반환값

    Returns:
        SelectionReport
    """
    report = SelectionReport()

    # 1. 카테고리별 1위 선정
    winner_counts: Dict[str, int] = {}

    for category, df in compressed.items():
        if df.empty:
            report.skipped_categories.append(category)
            continue

        # 동점 그룹 0 (1위 그룹)의 winner
        top_group = df[df["tie_group"] == 0]
        winner_row = top_group[top_group["tie_winner"]].iloc[0] if not top_group[top_group["tie_winner"]].empty else top_group.iloc[0]
        winner = winner_row["strategy"]
        composite = float(winner_row["composite"])
        pass_rate = float(winner_row.get("pass_rate", 0.0))
        avg_lat = float(winner_row["avg_latency_ms"]) if pd.notna(winner_row.get("avg_latency_ms")) else None

        # 2위
        runner_up = None
        if len(df) > 1:
            runner_up = df.iloc[1]["strategy"]

        # 선정 이유 생성
        reason = _generate_reason(winner, category, df, insights.get(winner, {}))

        report.per_category[category] = CategoryRecommendation(
            category=category,
            winner=winner,
            composite_score=composite,
            pass_rate=pass_rate,
            avg_latency_ms=avg_lat,
            reason=reason,
            runner_up=runner_up,
        )
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    # 2. 공통 우승자 (2개 이상 카테고리에서 1위)
    report.common_winners = [
        s for s, cnt in sorted(winner_counts.items(), key=lambda x: -x[1])
        if cnt >= 2
    ]

    # 3. 기본 추천 (전체 평균 1위)
    overall = rank_strategies_overall(insights)
    report.overall_ranking = overall

    if overall:
        best = overall[0]["strategy"]
        report.default_recommendation = best
        best_info = insights.get(best, {})
        strengths = best_info.get("strengths", [])
        if strengths:
            report.default_reason = (
                f"전체 카테고리 평균 복합 점수 1위 "
                f"(avg={overall[0]['avg_composite']:.3f}). "
                f"강점: {', '.join(strengths[:3])}"
            )
        else:
            report.default_reason = (
                f"전체 카테고리 평균 복합 점수 1위 "
                f"(avg={overall[0]['avg_composite']:.3f})"
            )

    # common_winners가 있으면 default_recommendation 업데이트
    if report.common_winners:
        top_common = report.common_winners[0]
        cnt = winner_counts[top_common]
        report.default_recommendation = top_common
        report.default_reason = (
            f"{cnt}개 카테고리에서 1위 — 다중 도메인 적용 시 최적"
        )

    return report


def _generate_reason(
    winner: str,
    category: str,
    df: pd.DataFrame,
    insight: dict,
) -> str:
    """선정 이유를 구체적으로 생성한다."""
    parts = []

    # 복합 점수
    row = df[df["strategy"] == winner]
    if not row.empty:
        composite = float(row.iloc[0]["composite"])
        recall = float(row.iloc[0].get("context_recall", 0.0))
        parts.append(f"복합 점수 {composite:.3f} (Context Recall {recall:.3f})")

    # 동점 여부
    top_group = df[df["tie_group"] == 0]
    if len(top_group) > 1:
        tied = [s for s in top_group["strategy"].tolist() if s != winner]
        parts.append(f"동점 그룹({len(top_group)}개) 중 레이턴시 기준 선택")

    # 인사이트 패턴
    pattern = insight.get("pattern", "")
    if pattern and "데이터 없음" not in pattern:
        parts.append(pattern)

    return " | ".join(parts) if parts else f"{category} 카테고리 최고 성능"
