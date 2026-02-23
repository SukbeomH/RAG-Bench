"""RAGAS 결과 출력 유틸리티."""
from typing import Optional

import pandas as pd


def print_combo_timing_table(timing_df: pd.DataFrame) -> None:
    """조합별 전체 소요 시간을 콘솔에 출력한다.

    Args:
        timing_df: _build_combo_timing_df()가 반환한 DataFrame.
                   컬럼: label, dense, sparse, reranker, llm_support,
                         build_s, pass1_s, pass2_s, total_s
    """
    if timing_df is None or timing_df.empty:
        return

    print(f"\n{'═' * 100}")
    print(" 조합별 전체 소요 시간")
    print(f"{'═' * 100}")
    print(f"  {'조합':<48} {'빌드(s)':>8} {'Pass1(s)':>9} {'Pass2(s)':>9} {'합계(s)':>9}")
    print(f"  {'─' * 48} {'─' * 8} {'─' * 9} {'─' * 9} {'─' * 9}")

    for _, row in timing_df.sort_values("total_s", ascending=False).iterrows():
        print(
            f"  {row['label']:<48}"
            f" {row['build_s']:>8.1f}"
            f" {row['pass1_s']:>9.1f}"
            f" {row['pass2_s']:>9.1f}"
            f" {row['total_s']:>9.1f}"
        )

    # 레이어별 평균
    for layer_col, layer_name in [
        ("dense", "Dense Model"),
        ("sparse", "Sparse"),
        ("reranker", "Reranker"),
        ("llm_support", "LLM Support"),
    ]:
        print(f"\n  ── {layer_name} 별 평균 소요 시간 ──")
        print(f"  {'값':<20} {'빌드(s)':>8} {'Pass1(s)':>9} {'Pass2(s)':>9} {'합계(s)':>9} {'조합 수':>7}")
        print(f"  {'─' * 20} {'─' * 8} {'─' * 9} {'─' * 9} {'─' * 9} {'─' * 7}")
        for val, grp in timing_df.groupby(layer_col):
            print(
                f"  {str(val):<20}"
                f" {grp['build_s'].mean():>8.1f}"
                f" {grp['pass1_s'].mean():>9.1f}"
                f" {grp['pass2_s'].mean():>9.1f}"
                f" {grp['total_s'].mean():>9.1f}"
                f" {len(grp):>7}"
            )


def print_qa_scaling_table(
    timing_df: pd.DataFrame,
    n_strategies: int,
    n_eval_strategies: int,
) -> None:
    """QA 수 변화에 따른 예상 소요 시간 스케일링 테이블을 출력한다.

    빌드 시간은 QA 수와 무관하게 고정이므로 별도 표시한다.
    Pass 1 / Pass 2 는 QA 수에 비례하여 선형 추정한다.

    Args:
        timing_df:         _build_combo_timing_df() 반환 DataFrame.
        n_strategies:      전체 실행 전략 수 (Pass 1 대상).
        n_eval_strategies: RAGAS 평가 전략 수 (Pass 2 대상). 0이면 Pass 2 미포함.
    """
    if timing_df is None or timing_df.empty:
        return

    # QA당 평균 Pass 1 시간 (전략 전체 합산)
    # pass1_s_per_qa: 해당 전략의 QA 1개 검색 평균 시간
    # 전체 Pass 1 속도 = 모든 전략의 pass1_s_per_qa 합 (순차 실행 기준)
    pass1_rate = timing_df["pass1_s_per_qa"].sum()  # QA 1개당 전체 전략 합산 시간

    # QA당 평균 Pass 2 시간 (RAGAS 평가 전략만)
    eval_rows = timing_df[timing_df["pass2_s_per_qa"] > 0]
    pass2_rate = eval_rows["pass2_s_per_qa"].sum() if not eval_rows.empty else 0.0

    # 빌드 시간 (QA 수 무관, 고정)
    # contextual retrieval 포함 조합은 문서 수에 비례하지만 QA 수와 무관
    total_build_s = timing_df["build_s"].sum()

    # 현재 실행된 QA 수
    current_n = int(timing_df["n_queries"].max()) if "n_queries" in timing_df.columns else 0

    qa_candidates = [5, 10, 20, 30, 50, 75, 100, 150, 200]
    # 현재 QA 수가 후보에 없으면 삽입
    if current_n > 0 and current_n not in qa_candidates:
        qa_candidates = sorted(set(qa_candidates + [current_n]))

    print(f"\n{'═' * 85}")
    print(" QA 수별 예상 소요 시간 (빌드 고정 + 검색/RAGAS 선형 추정)")
    print(f"{'═' * 85}")
    print(f"  기준: Pass1 전략 {n_strategies}개, Pass2(RAGAS) 전략 {n_eval_strategies}개")
    print(f"  QA당 Pass1 합산: {pass1_rate:.2f}s  |  QA당 Pass2 합산: {pass2_rate:.2f}s  |  빌드(고정): {total_build_s:.0f}s")
    print()
    print(f"  {'QA 수':>6}  {'빌드(s)':>9}  {'Pass1(s)':>9}  {'Pass2(s)':>9}  {'합계(s)':>9}  {'합계(분)':>9}  {'비고':>6}")
    print(f"  {'─' * 6}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 6}")

    for n in qa_candidates:
        p1 = pass1_rate * n
        p2 = pass2_rate * n
        total = total_build_s + p1 + p2
        note = "◀ 현재" if n == current_n else ""
        print(
            f"  {n:>6}  {total_build_s:>9.0f}  {p1:>9.0f}  {p2:>9.0f}"
            f"  {total:>9.0f}  {total / 60:>9.1f}  {note}"
        )

    print()
    print("  * 빌드 시간은 인덱스 캐시 유무에 따라 크게 달라질 수 있습니다.")
    print("  * Pass2 시간은 RAGAS 평가 전략이 없는 경우 0으로 표시됩니다.")


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
