"""
reporter — W&B Horangi v3 패턴 6섹션 보고서 생성기.

Markdown + JSON 두 가지 형식으로 출력한다.

섹션 구조:
  Section 1: 평가 개요
  Section 2: 종합 성능 리더보드
  Section 3: 카테고리별 상세 비교
  Section 4: 조합별 강점/약점 프로파일
  Section 5: 동점 그룹 압축 결과
  Section 6: 최종 선정 가이드

CLI:
    python -m rag_bench.analysis.reporter --run_dir _benchdata/service_run
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from rag_bench.analysis.ranker import load_results, rank_by_doc_type
from rag_bench.analysis.insight import analyze_strengths_weaknesses, rank_strategies_overall
from rag_bench.analysis.deduplication import compress_similar_results, format_tie_groups_summary
from rag_bench.analysis.selector import generate_selection_report, SelectionReport


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------

def generate_report(
    run_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    similarity_threshold: float = 0.05,
    verbose: bool = True,
) -> SelectionReport:
    """
    run_dir 의 벤치마크 결과를 분석하고 Markdown + JSON 보고서를 생성한다.

    Args:
        run_dir: run_service_bench.py 의 출력 디렉토리
        output_dir: 보고서 저장 위치 (기본: run_dir)
        similarity_threshold: 동점 판정 임계값 (기본 0.05 = 5%)
        verbose: 진행 상태 출력 여부

    Returns:
        SelectionReport
    """
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str):
        if verbose:
            print(msg)

    _log(f"\n{'=' * 60}")
    _log(" RAG 모델 선정 분석 시작")
    _log(f"{'=' * 60}")
    _log(f"  run_dir: {run_dir}")

    # 1. 데이터 로드
    _log("\n[1/5] 결과 로드 중...")
    raw_results = load_results(run_dir)
    if not raw_results:
        print(f"오류: {run_dir} 에서 result.json 파일을 찾을 수 없습니다.")
        return SelectionReport()

    _log(f"  로드된 카테고리: {list(raw_results.keys())}")

    # 2. 순위 계산
    _log("\n[2/5] 순위 계산 중...")
    ranked = rank_by_doc_type(raw_results, latency_dir=run_dir)
    _log(f"  순위 완료: {list(ranked.keys())}")

    # 3. 인사이트 분석
    _log("\n[3/5] 강점/약점 분석 중...")
    insights = analyze_strengths_weaknesses(ranked)
    _log(f"  분석 완료: {len(insights)}개 조합")

    # 4. 동점 그룹 압축
    _log("\n[4/5] 동점 그룹 압축 중...")
    compressed = compress_similar_results(ranked, similarity_threshold)
    tie_summary = format_tie_groups_summary(compressed)

    # 5. 선정 보고서
    _log("\n[5/5] 선정 보고서 생성 중...")
    selection = generate_selection_report(ranked, insights, compressed)

    # 6. 파일 생성
    md_path = output_dir / "selection_report.md"
    json_path = output_dir / "selection_report.json"

    md_content = _render_markdown(
        run_dir=run_dir,
        raw_results=raw_results,
        ranked=ranked,
        insights=insights,
        compressed=compressed,
        tie_summary=tie_summary,
        selection=selection,
    )
    md_path.write_text(md_content, encoding="utf-8")

    json_content = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dir": str(run_dir),
        "categories": list(ranked.keys()),
        "selection": selection.to_dict(),
        "ranked": {
            cat: df.to_dict(orient="records")
            for cat, df in ranked.items()
        },
        "insights": insights,
        "tie_groups": tie_summary,
    }
    json_path.write_text(
        json.dumps(json_content, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    _log(f"\n  보고서 저장:")
    _log(f"    Markdown : {md_path}")
    _log(f"    JSON     : {json_path}")
    _log(f"\n{'=' * 60}")
    _log(" 분석 완료")
    _log(f"{'=' * 60}")

    # 콘솔 요약 출력
    _print_summary(selection, ranked)

    return selection


# ---------------------------------------------------------------------------
# Markdown 렌더링
# ---------------------------------------------------------------------------

def _render_markdown(
    run_dir: Path,
    raw_results: Dict[str, dict],
    ranked: Dict[str, pd.DataFrame],
    insights: Dict[str, dict],
    compressed: Dict[str, pd.DataFrame],
    tie_summary: Dict[str, List[dict]],
    selection: SelectionReport,
) -> str:
    lines: List[str] = []

    lines += _section1_overview(run_dir, raw_results, ranked)
    lines += _section2_leaderboard(ranked, insights)
    lines += _section3_category_detail(ranked, raw_results)
    lines += _section4_strength_weakness(insights, ranked)
    lines += _section5_tie_groups(tie_summary)
    lines += _section6_final_guide(selection, ranked)

    return "\n".join(lines)


def _section1_overview(
    run_dir: Path,
    raw_results: Dict[str, dict],
    ranked: Dict[str, pd.DataFrame],
) -> List[str]:
    categories = list(ranked.keys())
    total_qa = sum(raw_results[c].get("n_qa", 0) for c in categories)
    # 전략 수 (첫 카테고리 기준)
    n_combos = len(next(iter(ranked.values()))) if ranked else 0

    lines = [
        "# RAG 모델 선정 보고서",
        "",
        "---",
        "## Section 1: 평가 개요",
        "",
        "| 항목 | 내용 |",
        "|------|------|",
        "| 고정 파이프라인 | [Dense] + [Sparse] + ColBERT Reranker + Contextual Retrieval |",
        f"| 비교 변수 | {n_combos}개 조합 (4 Dense × 2 Sparse) |",
        f"| 평가 카테고리 | {len(categories)}개 ({', '.join(c.upper() for c in categories)}) |",
        f"| 총 QA 수 | {total_qa:,}개 |",
        "| 주요 지표 | RAGAS 복합 점수 (Recall×0.35 + Precision×0.30 + Faith×0.20 + Relevancy×0.15) |",
        "| 보조 지표 | recall_pct (context_recall 기반 %) |",
        f"| 결과 디렉토리 | `{run_dir}` |",
        "",
        "### RAGAS 가중치 근거",
        "",
        "| 지표 | 가중치 | 이유 |",
        "|------|--------|------|",
        "| Context Recall | **0.35** | 누락이 오탐보다 치명적 (서비스 신뢰도) |",
        "| Context Precision | **0.30** | 불필요 컨텍스트 → 답변 품질 저하 |",
        "| Faithfulness | **0.20** | 할루시네이션 방지 |",
        "| Answer Relevancy | **0.15** | 최종 사용자 만족도 |",
        "",
        "---",
    ]
    return lines


def _section2_leaderboard(
    ranked: Dict[str, pd.DataFrame],
    insights: Dict[str, dict],
) -> List[str]:
    if not ranked:
        return ["## Section 2: 종합 성능 리더보드", "", "데이터 없음", "", "---"]

    categories = list(ranked.keys())
    overall = rank_strategies_overall(insights)

    lines = [
        "## Section 2: 종합 성능 리더보드",
        "",
        "복합 점수 = Context Recall×0.35 + Precision×0.30 + Faithfulness×0.20 + Relevancy×0.15",
        "",
    ]

    # 헤더
    header_cats = " | ".join(c.upper() for c in categories)
    sep_cats = " | ".join("------" for _ in categories)
    lines.append(f"| 순위 | 조합 | {header_cats} | **평균** |")
    lines.append(f"|------|------|{sep_cats}|---------|")

    for i, row in enumerate(overall[:10], 1):  # 최대 10개
        strategy = row["strategy"]
        avg = row["avg_composite"]
        cat_scores = []
        for cat in categories:
            df = ranked[cat]
            s_row = df[df["strategy"] == strategy]
            if s_row.empty:
                cat_scores.append("—")
            else:
                score = s_row.iloc[0]["composite"]
                rank = int(s_row.iloc[0]["rank"])
                marker = "**" if rank == 1 else ""
                cat_scores.append(f"{marker}{score:.3f}{marker}")
        cat_str = " | ".join(cat_scores)
        lines.append(f"| {i} | `{strategy}` | {cat_str} | **{avg:.3f}** |")

    lines += ["", "---"]
    return lines


def _section3_category_detail(
    ranked: Dict[str, pd.DataFrame],
    raw_results: Dict[str, dict],
) -> List[str]:
    lines = [
        "## Section 3: 카테고리별 상세 비교",
        "",
    ]

    ragas_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    for cat_idx, (category, df) in enumerate(ranked.items(), 1):
        n_qa = raw_results.get(category, {}).get("n_qa", "?")
        lines += [
            f"### 3-{cat_idx}. {category.upper()}",
            "",
            f"평가 QA 수: {n_qa}개",
            "",
        ]

        # 상세 테이블
        col_headers = "| 순위 | 조합 | Recall | Precision | Faithfulness | Relevancy | **복합** | Recall(%) |"
        col_sep = "|------|------|--------|-----------|--------------|-----------|----------|-----------|"
        lines += [col_headers, col_sep]

        for _, row in df.iterrows():
            rank = int(row["rank"])
            strategy = row["strategy"]
            recall = f"{row['context_recall']:.3f}"
            precision = f"{row['context_precision']:.3f}"
            faith = f"{row['faithfulness']:.3f}"
            relev = f"{row['answer_relevancy']:.3f}"
            composite = f"**{row['composite']:.3f}**" if rank == 1 else f"{row['composite']:.3f}"
            recall_pct = f"{row.get('recall_pct', row.get('pass_rate', 0.0)):.1f}%"
            lines.append(f"| {rank} | `{strategy}` | {recall} | {precision} | {faith} | {relev} | {composite} | {recall_pct} |")

        lines += [""]

    lines += ["---"]
    return lines


def _section4_strength_weakness(
    insights: Dict[str, dict],
    ranked: Dict[str, pd.DataFrame],
) -> List[str]:
    lines = [
        "## Section 4: 조합별 강점/약점 프로파일",
        "",
    ]

    # 전체 평균으로 정렬
    overall = rank_strategies_overall(insights)

    for item in overall:
        strategy = item["strategy"]
        info = insights.get(strategy, {})
        strengths = info.get("strengths", [])
        weaknesses = info.get("weaknesses", [])
        pattern = info.get("pattern", "")
        avg_composite = item["avg_composite"]

        lines += [
            f"#### `{strategy}`",
            "",
            f"- **전체 평균 복합 점수**: {avg_composite:.3f}",
        ]

        if strengths:
            lines.append(f"- **강점**: {', '.join(strengths)}")
        if weaknesses:
            lines.append(f"- **약점**: {', '.join(weaknesses)}")
        if pattern:
            lines.append(f"- **패턴**: {pattern}")

        # 카테고리별 점수 미니 테이블
        scores = info.get("scores", {})
        if scores:
            score_parts = [f"{cat}={v:.3f}" for cat, v in scores.items()]
            lines.append(f"- **카테고리 점수**: {' | '.join(score_parts)}")

        lines.append("")

    lines += ["---"]
    return lines


def _section5_tie_groups(
    tie_summary: Dict[str, List[dict]],
) -> List[str]:
    lines = [
        "## Section 5: 동점 그룹 압축 결과",
        "",
        "점수 차이 5% 이내 조합은 통계적으로 동등하며, 속도/비용 기준으로 우선순위를 결정합니다.",
        "",
    ]

    has_ties = False
    for category, groups in tie_summary.items():
        tie_groups = [g for g in groups if len(g["strategies"]) > 1]
        if not tie_groups:
            continue

        has_ties = True
        lines.append(f"### {category.upper()}")
        lines.append("")

        for g in tie_groups:
            strategies_str = " vs ".join(f"`{s}`" for s in g["strategies"])
            lines += [
                f"- **동점 그룹 {g['group']+1}**: {strategies_str}",
                f"  - {g['note']}",
                "",
            ]

    if not has_ties:
        lines += ["카테고리별 명확한 성능 차이가 있어 동점 그룹이 발생하지 않았습니다.", ""]

    lines += ["---"]
    return lines


def _section6_final_guide(
    selection: SelectionReport,
    ranked: Dict[str, pd.DataFrame],
) -> List[str]:
    lines = [
        "## Section 6: 최종 선정 가이드",
        "",
        "### 카테고리별 추천 조합",
        "",
        "| 문서 타입 | 1순위 조합 | 복합 점수 | Pass% | 선정 이유 |",
        "|-----------|-----------|----------|-------|---------|",
    ]

    for category, rec in selection.per_category.items():
        lines.append(
            f"| {category.upper()} | `{rec.winner}` | {rec.composite_score:.3f} | {rec.pass_rate:.1f}% | {rec.reason} |"
        )

    lines += [""]

    if selection.common_winners:
        lines += [
            "### 공통 우승 조합",
            "",
        ]
        for w in selection.common_winners:
            cats = [cat for cat, rec in selection.per_category.items() if rec.winner == w]
            lines.append(f"- `{w}`: {', '.join(c.upper() for c in cats)} 카테고리 1위")
        lines.append("")

    if selection.default_recommendation:
        lines += [
            "### 기본 추천 (문서 타입 혼용 또는 불명확 시)",
            "",
            f"> **`{selection.default_recommendation}`**",
            f"> {selection.default_reason}",
            "",
        ]

    if selection.skipped_categories:
        lines += [
            "### 선정 불가 카테고리",
            "",
            f"데이터 부족으로 분석 불가: {', '.join(selection.skipped_categories)}",
            "",
        ]

    # 전체 랭킹
    if selection.overall_ranking:
        lines += [
            "### 전체 순위 요약",
            "",
            "| 순위 | 조합 | 평균 복합 점수 | 평균 순위 | 패턴 |",
            "|------|------|-------------|----------|------|",
        ]
        for i, row in enumerate(selection.overall_ranking[:8], 1):
            avg_rank_str = f"{row['avg_rank']:.1f}"
            lines.append(
                f"| {i} | `{row['strategy']}` | {row['avg_composite']:.3f} | {avg_rank_str} | {row['pattern']} |"
            )
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# 콘솔 요약
# ---------------------------------------------------------------------------

def _print_summary(selection: SelectionReport, ranked: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 60)
    print(" 최종 선정 요약")
    print("=" * 60)

    for category, rec in selection.per_category.items():
        lat_str = f"{rec.avg_latency_ms:.0f}ms" if rec.avg_latency_ms else "—"
        print(f"  {category.upper():10s} -> {rec.winner}")
        print(f"              복합점수={rec.composite_score:.3f} | Pass%={rec.pass_rate:.1f}% | 레이턴시={lat_str}")

    if selection.default_recommendation:
        print(f"\n  기본 추천: {selection.default_recommendation}")
        print(f"  이유     : {selection.default_reason}")


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="서비스 벤치마크 결과 분석 + 보고서 생성"
    )
    parser.add_argument("--run_dir", required=True, help="run_service_bench.py 출력 디렉토리")
    parser.add_argument("--output_dir", default=None, help="보고서 저장 위치 (기본: run_dir)")
    parser.add_argument("--threshold", type=float, default=0.05, help="동점 판정 임계값 (기본: 0.05)")
    args = parser.parse_args()

    report = generate_report(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        similarity_threshold=args.threshold,
    )
    if not report.per_category:
        sys.exit(1)


if __name__ == "__main__":
    main()
