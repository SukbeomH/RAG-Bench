"""
deduplication — 통계적으로 유사한 조합 그룹 압축.

점수 차 5% 이내 조합 → 동점 그룹으로 통합.
동점 그룹 내에서는 레이턴시(속도) 기준으로 우선순위 결정.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd


def compress_similar_results(
    ranked: Dict[str, pd.DataFrame],
    similarity_threshold: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """
    카테고리별로 복합 점수 차이가 similarity_threshold 이내인 조합을 동점 그룹으로 묶는다.

    Args:
        ranked: rank_by_doc_type() 반환값
        similarity_threshold: 동점 판정 임계값 (기본 0.05 = 5%)

    Returns:
        Dict[category_name, DataFrame]
        추가 컬럼: tie_group (int, 0-based), tie_winner (bool)
    """
    compressed: Dict[str, pd.DataFrame] = {}

    for category, df in ranked.items():
        df = df.copy()
        df["tie_group"] = _assign_tie_groups(df["composite"].tolist(), similarity_threshold)
        df["tie_winner"] = _mark_tie_winners(df)
        compressed[category] = df

    return compressed


def _assign_tie_groups(composites: List[float], threshold: float) -> List[int]:
    """복합 점수 리스트를 동점 그룹 인덱스로 변환한다."""
    if not composites:
        return []

    groups = [0]
    current_group = 0
    reference = composites[0]  # 그룹의 기준점 (1위 점수)

    for score in composites[1:]:
        # 상위 그룹의 최고 점수 대비 차이
        if abs(reference - score) > threshold:
            current_group += 1
            reference = score  # 새 그룹의 기준점
        groups.append(current_group)

    return groups


def _mark_tie_winners(df: pd.DataFrame) -> List[bool]:
    """
    동점 그룹 내에서 승자를 표시한다.
    승자 기준: avg_latency_ms 최소 (레이턴시 없으면 composite 최대).
    """
    winners = [False] * len(df)

    for group_id in df["tie_group"].unique():
        group_mask = df["tie_group"] == group_id
        group_df = df[group_mask]

        if len(group_df) == 1:
            idx = group_df.index[0]
            winners[df.index.get_loc(idx)] = True
        else:
            # 레이턴시로 정렬
            has_lat = group_df["avg_latency_ms"].notna().any()
            if has_lat:
                best_idx = group_df["avg_latency_ms"].idxmin()
            else:
                best_idx = group_df["composite"].idxmax()
            winners[df.index.get_loc(best_idx)] = True

    return winners


def format_tie_groups_summary(
    compressed: Dict[str, pd.DataFrame],
) -> Dict[str, List[dict]]:
    """
    동점 그룹을 사람이 읽기 쉬운 형식으로 변환한다.

    Returns:
        Dict[category, List[{"group": int, "strategies": List[str], "winner": str, "note": str}]]
    """
    summary: Dict[str, List[dict]] = {}

    for category, df in compressed.items():
        groups = []
        for group_id in sorted(df["tie_group"].unique()):
            group_df = df[df["tie_group"] == group_id].copy()
            strategies = group_df["strategy"].tolist()
            winner_rows = group_df[group_df["tie_winner"]]
            winner = winner_rows.iloc[0]["strategy"] if not winner_rows.empty else strategies[0]

            if len(strategies) > 1:
                scores = group_df.set_index("strategy")["composite"].to_dict()
                score_range = f"{min(scores.values()):.3f}~{max(scores.values()):.3f}"
                note = f"동점 그룹 (복합 점수 {score_range}) — 속도 기준 {winner} 권장"
            else:
                score = float(group_df.iloc[0]["composite"])
                note = f"복합 점수 {score:.3f}"

            groups.append({
                "group": group_id,
                "strategies": strategies,
                "winner": winner,
                "note": note,
            })
        summary[category] = groups

    return summary
