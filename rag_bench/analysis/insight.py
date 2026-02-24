"""
insight — 조합별 강점/약점 프로파일 도출.

카테고리 간 비교로 각 조합의 패턴을 파악한다.
"""

from typing import Dict, List

import pandas as pd


def analyze_strengths_weaknesses(
    ranked: Dict[str, pd.DataFrame],
) -> Dict[str, dict]:
    """
    각 조합(strategy)의 강점/약점을 카테고리 간 비교로 도출한다.

    Returns:
        Dict[strategy_name, {
            "strengths": List[str],   # 1~2위 카테고리
            "weaknesses": List[str],  # 최하위 카테고리
            "scores": Dict[category, composite],
            "avg_rank": float,
            "avg_composite": float,
            "pattern": str,
        }]
    """
    if not ranked:
        return {}

    # 전체 전략 집합
    all_strategies: set = set()
    for df in ranked.values():
        all_strategies.update(df["strategy"].tolist())

    # 카테고리 수
    n_categories = len(ranked)

    insights: Dict[str, dict] = {}

    for strategy in all_strategies:
        strengths: List[str] = []
        weaknesses: List[str] = []
        scores: Dict[str, float] = {}
        ranks: List[int] = []

        for category, df in ranked.items():
            row = df[df["strategy"] == strategy]
            if row.empty:
                continue
            r = int(row.iloc[0]["rank"])
            composite = float(row.iloc[0]["composite"])
            scores[category] = composite
            ranks.append(r)

            total = len(df)
            if r == 1:
                strengths.append(f"{category}(1위)")
            elif r == 2:
                strengths.append(f"{category}(2위)")
            if r == total and total > 2:
                weaknesses.append(f"{category}(최하위)")
            elif r >= max(3, total - 1) and total > 3:
                weaknesses.append(f"{category}({r}위/{total})")

        avg_rank = sum(ranks) / len(ranks) if ranks else float("nan")
        avg_composite = sum(scores.values()) / len(scores) if scores else 0.0

        # 패턴 요약 생성
        pattern = _generate_pattern(strategy, scores, strengths, weaknesses)

        insights[strategy] = {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "scores": scores,
            "avg_rank": round(avg_rank, 2),
            "avg_composite": round(avg_composite, 4),
            "pattern": pattern,
        }

    return insights


def _generate_pattern(
    strategy: str,
    scores: Dict[str, float],
    strengths: List[str],
    weaknesses: List[str],
) -> str:
    """조합의 강점/약점 패턴을 자연어로 요약한다."""
    if not scores:
        return "데이터 없음"

    top_cats = [s.split("(")[0] for s in strengths if "1위" in s or "2위" in s]
    weak_cats = [s.split("(")[0] for s in weaknesses]

    parts = []
    if top_cats:
        parts.append(f"{'+'.join(top_cats)} 카테고리 강세")
    if weak_cats:
        parts.append(f"{'+'.join(weak_cats)} 카테고리 약세")
    if not parts:
        avg = sum(scores.values()) / len(scores)
        parts.append(f"전체 평균 {avg:.3f} (균형형)")

    return " / ".join(parts)


def rank_strategies_overall(
    insights: Dict[str, dict],
) -> List[dict]:
    """
    전체 카테고리 평균 복합 점수로 전략 순위를 반환한다.

    Returns:
        List[{"strategy": str, "avg_composite": float, "avg_rank": float, ...}]
    """
    rows = []
    for strategy, info in insights.items():
        rows.append({
            "strategy": strategy,
            "avg_composite": info["avg_composite"],
            "avg_rank": info["avg_rank"],
            "n_strengths": len([s for s in info["strengths"] if "1위" in s]),
            "pattern": info["pattern"],
        })
    rows.sort(key=lambda x: (-x["avg_composite"], x["avg_rank"]))
    return rows
