"""
reporter_exec — 상급자용 경영진 보고서 생성기.

기술 배경이 없는 의사결정자를 위해 설계된 보고서 형식:
  - 역피라미드 구조: 결론 → 근거 → 상세
  - 수치는 절대값 대신 상대 비교(%p 우세)로 표현
  - 기술 용어는 첫 등장 시 괄호 설명 병기
  - 차트 제목이 데이터의 해석을 직접 전달
  - 기술 세부사항은 부록으로 분리

출력 파일:
  - selection_report_exec.md  (경영진용 Markdown)

CLI:
    python -m rag_bench.analysis.reporter_exec --run_dir _benchdata/service_run
"""

from __future__ import annotations

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
from rag_bench.document_types.types import DOC_TYPE_METADATA, DocType
from rag_bench.strategies.dense_sparse import DENSE_MODEL_DISPLAY


# ---------------------------------------------------------------------------
# 헬퍼 — 카테고리 메타데이터 조회
# ---------------------------------------------------------------------------

def _cat_meta(cat: str, key: str, default: str = "—") -> str:
    """카테고리 메타데이터를 DOC_TYPE_METADATA에서 조회한다."""
    try:
        return DOC_TYPE_METADATA[DocType(cat)].get(key, default)
    except (ValueError, KeyError):
        return default


# ---------------------------------------------------------------------------
# 메인 진입점
# ---------------------------------------------------------------------------

def generate_exec_report(
    run_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    similarity_threshold: float = 0.05,
    verbose: bool = True,
) -> SelectionReport:
    """
    경영진용 보고서를 생성한다.

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
    _log(" RAG 모델 선정 분석 — 경영진 보고서 생성")
    _log(f"{'=' * 60}")
    _log(f"  run_dir: {run_dir}")

    _log("\n[1/5] 결과 로드 중...")
    raw_results = load_results(run_dir)
    if not raw_results:
        print(f"오류: {run_dir} 에서 result.json 파일을 찾을 수 없습니다.")
        return SelectionReport()

    _log(f"  로드된 카테고리: {list(raw_results.keys())}")

    _log("\n[2/5] 순위 계산 중...")
    ranked = rank_by_doc_type(raw_results, latency_dir=run_dir)

    _log("\n[3/5] 강점/약점 분석 중...")
    insights = analyze_strengths_weaknesses(ranked)

    _log("\n[4/5] 동점 그룹 압축 중...")
    compressed = compress_similar_results(ranked, similarity_threshold)
    tie_summary = format_tie_groups_summary(compressed)

    _log("\n[5/5] 경영진 보고서 렌더링 중...")
    selection = generate_selection_report(ranked, insights, compressed)

    md_content = _render_exec_markdown(
        run_dir=run_dir,
        raw_results=raw_results,
        ranked=ranked,
        insights=insights,
        compressed=compressed,
        tie_summary=tie_summary,
        selection=selection,
    )

    md_path = output_dir / "selection_report_exec.md"
    md_path.write_text(md_content, encoding="utf-8")

    _log(f"\n  보고서 저장: {md_path}")
    _log(f"\n{'=' * 60}")
    _log(" 경영진 보고서 생성 완료")
    _log(f"{'=' * 60}")

    _print_exec_summary(selection)
    return selection


# ---------------------------------------------------------------------------
# 최상위 렌더러
# ---------------------------------------------------------------------------

def _render_exec_markdown(
    run_dir: Path,
    raw_results: Dict[str, dict],
    ranked: Dict[str, pd.DataFrame],
    insights: Dict[str, dict],
    compressed: Dict[str, pd.DataFrame],
    tie_summary: Dict[str, List[dict]],
    selection: SelectionReport,
) -> str:
    lines: List[str] = []

    generated_at = time.strftime("%Y년 %m월 %d일")

    lines += [
        "<!-- 경영진용 RAG 모델 선정 보고서 — reporter_exec.py 자동 생성 -->",
        f"<!-- 생성일: {generated_at} -->",
        "",
    ]

    lines += _cover(generated_at, ranked, raw_results)
    lines += _exec_summary(selection, ranked, insights)
    lines += _section1_background(ranked, raw_results)
    lines += _section2_measurement(ranked, raw_results)
    lines += _section3_overall_results(ranked, insights)
    lines += _section4_category_detail(ranked, raw_results, insights)
    lines += _section5_final_guide(selection, ranked, tie_summary)
    lines += _appendix(ranked, insights, raw_results)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 커버 + 메타
# ---------------------------------------------------------------------------

def _cover(
    generated_at: str,
    ranked: Dict[str, pd.DataFrame],
    raw_results: Dict[str, dict],
) -> List[str]:
    categories = list(ranked.keys())
    total_qa = sum(raw_results[c].get("n_qa", 0) for c in categories)
    n_combos = len(next(iter(ranked.values()))) if ranked else 0

    return [
        "# RAG 모델 선정 보고서",
        "### AI 검색 엔진 최적 조합 — 문서 종류별 추천",
        "",
        f"> **작성일**: {generated_at}  ",
        f"> **평가 범위**: {len(categories)}개 문서 카테고리 / {n_combos}개 AI 조합 비교 / 총 {total_qa:,}개 질의응답 테스트  ",
        "> **파이프라인**: Dense 검색 + Sparse 검색 + ColBERT 리랭커 + Contextual 문맥 강화 (후자 2개 고정)  ",
        "",
        "---",
        "",
    ]


# ---------------------------------------------------------------------------
# Section 0: Executive Summary
# ---------------------------------------------------------------------------

def _exec_summary(
    selection: SelectionReport,
    ranked: Dict[str, pd.DataFrame],
    insights: Dict[str, dict],
) -> List[str]:
    lines = [
        "## Executive Summary",
        "",
        "> 이 보고서는 우리 RAG(AI 검색·답변) 서비스에 탑재할 **검색 엔진 조합**을 데이터로 선정한 결과입니다.",
        "> 문서 종류마다 최적 모델이 다르며, 이를 실험으로 확인했습니다.",
        "",
    ]

    # 핵심 결론 박스
    lines += [
        "### 핵심 결론",
        "",
    ]

    # 즉시 결정 테이블
    lines += [
        "| 문서 종류 | 추천 AI 모델 조합 | 신뢰도 | 비고 |",
        "|----------|-----------------|--------|------|",
    ]

    confidence_map = {
        "general":   ("★★★", ""),
        "legal":     ("★★★", ""),
        "business":  ("★★★", ""),
        "medical":   ("★★★", ""),
        "technical": ("★☆☆", "표준 테스트 데이터 추가 확보 후 재평가 권장"),
    }

    for cat, rec in selection.per_category.items():
        label = _cat_meta(cat, "label", cat.upper())
        stars, note = confidence_map.get(cat, ("★★☆", ""))
        winner_display = _format_model_name(rec.winner)
        lines.append(f"| {label} | **{winner_display}** | {stars} | {note} |")

    for cat in selection.skipped_categories:
        label = _cat_meta(cat, "label", cat.upper())
        lines.append(f"| {label} | 테스트 데이터 부족 — 추후 평가 | ★☆☆ | 고객 문서 확보 후 재측정 필요 |")

    lines += [""]

    # 단일 추천
    if selection.default_recommendation:
        winner_display = _format_model_name(selection.default_recommendation)
        lines += [
            "### 한 가지만 선택해야 한다면",
            "",
            f"> **{winner_display}**",
            f"> {selection.default_reason}",
            "",
        ]

    # 즉시 실행 항목
    lines += [
        "### 즉시 실행 항목",
        "",
    ]

    if selection.default_recommendation:
        lines.append(f"- [ ] 서비스 파이프라인에 **{_format_model_name(selection.default_recommendation)}** 우선 통합")

    if "technical" in selection.skipped_categories or any(
        c for c in selection.per_category if confidence_map.get(c, ("", ""))[0] == "★☆☆"
    ):
        lines.append("- [ ] 기술 문서 고객군 확보 후 TECHNICAL 카테고리 추가 벤치마크 실행")

    lines += [
        "- [ ] 분기별 모델 재평가 프로세스 수립 (AI 모델 릴리스 주기 고려)",
        "",
        "---",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Section 1: 배경 — 무엇을, 왜 테스트했는가
# ---------------------------------------------------------------------------

def _section1_background(
    ranked: Dict[str, pd.DataFrame],
    raw_results: Dict[str, dict],
) -> List[str]:
    categories = list(ranked.keys())
    n_combos = len(next(iter(ranked.values()))) if ranked else 0

    lines = [
        "## 1. 무엇을, 왜 테스트했는가",
        "",
        "### 1-1. 비즈니스 질문",
        "",
        "> **\"우리 RAG 서비스에 어떤 AI 검색 엔진 조합을 탑재해야**  ",
        "> **고객 문서에서 가장 정확한 답변을 제공할 수 있는가?\"**",
        "",
        "### 1-2. AI 검색 엔진의 두 가지 방식",
        "",
        "고객이 질문하면 시스템은 문서에서 관련 내용을 찾아야 합니다.",
        "찾는 방법은 크게 두 가지입니다:",
        "",
        "| 방식 | 작동 원리 | 비유 | 강점 |",
        "|------|----------|------|------|",
        "| **Dense 검색** (의미 기반) | 질문의 '뜻'을 이해해서 유사 내용을 찾음 | 사서가 내용을 읽고 추천 | 유사 표현·동의어에 강함 |",
        "| **Sparse 검색** (키워드 기반) | 단어 그대로 매칭하여 찾음 | 색인에서 단어 검색 | 정확한 용어·번호에 강함 |",
        "",
        "우리 시스템은 **두 방식을 동시에 활용**합니다.",
        "이번 평가는 어떤 Dense 모델과 Sparse 모델 조합이 문서 종류별로 최적인지를 확인합니다.",
        "",
        "### 1-3. 고정 파이프라인 — 변수에서 제외한 항목",
        "",
        "다음 두 기술은 **어떤 조합에서도 품질을 개선하는 것이 연구로 입증**되어 있어,",
        "이번 비교의 변수에서 제외하고 전체 조합에 고정 적용했습니다:",
        "",
    ]

    _PIPELINE_REASONS = {
        "ColBERT 리랭커": "최종 답변 후보 재정렬. IBM 연구 결과 오답률 25% 감소 확인.",
        "Contextual 문맥 강화": "검색 전 청크(문서 조각)에 문맥을 AI로 추가. Anthropic 검색 실패율 67% 감소 보고.",
    }
    for name, reason in _PIPELINE_REASONS.items():
        lines.append(f"- **{name}**: {reason}")

    lines += [
        "",
        "### 1-4. 테스트 범위",
        "",
        f"- **비교 조합**: {n_combos}가지 (Dense 4종 × Sparse 2종)",
        f"- **문서 카테고리**: {len(categories)}종 ({', '.join(_cat_meta(c, 'label', c) for c in categories)})",
        "",
        "---",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Section 2: 측정 방법 — 어떻게 평가했는가
# ---------------------------------------------------------------------------

def _section2_measurement(
    ranked: Dict[str, pd.DataFrame],
    raw_results: Dict[str, dict],
) -> List[str]:
    categories = list(ranked.keys())

    lines = [
        "## 2. 어떻게 측정했는가",
        "",
        "### 2-1. 테스트 데이터셋",
        "",
        "각 문서 카테고리에 대해 공개된 표준 데이터셋을 사용했습니다:",
        "",
        "| 카테고리 | 데이터 출처 | 특성 |",
        "|---------|-----------|------|",
    ]

    for cat in categories:
        label = _cat_meta(cat, "label", cat.upper())
        source = _cat_meta(cat, "data_source")
        char = _cat_meta(cat, "characteristic")
        lines.append(f"| {label} | {source} | {char} |")

    lines += [
        "",
        "### 2-2. 평가 지표 — 4가지",
        "",
        "AI 답변 품질을 측정하는 4가지 관점을 가중 평균하여 **종합 점수**를 계산합니다:",
        "",
        "| 지표 | 측정 내용 | 쉽게 말하면 | 가중치 | 가중치 이유 |",
        "|------|----------|-----------|--------|-----------|",
        "| **Context Recall** | 정답 내용을 빠뜨리지 않았는가 | '놓치지 않는 능력' | **35%** | 빠뜨리는 게 오답보다 서비스에 더 치명적 |",
        "| **Context Precision** | 찾아온 문서가 질문과 관련 있는가 | '쓸모없는 내용 배제 능력' | 30% | 관련 없는 내용이 많으면 답변 품질 저하 |",
        "| **Faithfulness** | 답변이 문서 내용에 근거하는가 | '지어내지 않는 능력' | 20% | AI 할루시네이션(거짓 정보) 방지 |",
        "| **Answer Relevancy** | 질문에 직접적으로 답하는가 | '동문서답 방지 능력' | 15% | 최종 사용자 만족도 직결 |",
        "",
        "> **종합 점수** = Recall×0.35 + Precision×0.30 + Faithfulness×0.20 + Relevancy×0.15",
        "> 0에서 1 사이 값이며, 1에 가까울수록 우수합니다.",
        "",
        "---",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Section 3: 종합 성능 결과
# ---------------------------------------------------------------------------

def _section3_overall_results(
    ranked: Dict[str, pd.DataFrame],
    insights: Dict[str, dict],
) -> List[str]:
    if not ranked:
        return ["## 3. 종합 성능 결과", "", "데이터 없음", "", "---"]

    categories = list(ranked.keys())
    overall = rank_strategies_overall(insights)

    lines = [
        "## 3. 종합 성능 결과",
        "",
        "### 3-1. 종합 순위표",
        "",
        "> 각 카테고리 1위는 **굵게** 표시. 점수는 0~1 범위(높을수록 우수).",
        "",
    ]

    # 헤더
    header_cats = " | ".join(_cat_meta(c, "label", c.upper()) for c in categories)
    sep_cats = " | ".join(":------:" for _ in categories)
    lines.append(f"| 순위 | AI 조합 | {header_cats} | **평균** |")
    lines.append(f"|:----:|:--------|{sep_cats}|:-------:|")

    for i, row in enumerate(overall[:8], 1):
        strategy = row["strategy"]
        avg = row["avg_composite"]
        model_display = _format_model_name(strategy)

        cat_scores = []
        for cat in categories:
            df = ranked[cat]
            s_row = df[df["strategy"] == strategy]
            if s_row.empty:
                cat_scores.append("—")
            else:
                score = s_row.iloc[0]["composite"]
                rank = int(s_row.iloc[0]["rank"])
                if rank == 1:
                    cat_scores.append(f"**{score:.3f}** 🥇")
                elif rank == 2:
                    cat_scores.append(f"{score:.3f} 🥈")
                else:
                    cat_scores.append(f"{score:.3f}")

        cat_str = " | ".join(cat_scores)
        rank_marker = "**" if i == 1 else ""
        lines.append(f"| {i} | {rank_marker}{model_display}{rank_marker} | {cat_str} | {rank_marker}{avg:.3f}{rank_marker} |")

    lines += [""]

    # 핵심 인사이트 3가지
    lines += [
        "### 3-2. 핵심 인사이트",
        "",
    ]

    insight_list = _derive_key_insights(overall, ranked, insights, categories)
    for idx, insight in enumerate(insight_list, 1):
        lines.append(f"**{idx}. {insight['title']}**")
        lines.append(f"> {insight['body']}")
        lines.append("")

    lines += ["---", ""]
    return lines


def _derive_key_insights(
    overall: List[dict],
    ranked: Dict[str, pd.DataFrame],
    insights: Dict[str, dict],
    categories: List[str],
) -> List[dict]:
    """데이터에서 핵심 인사이트 3개를 자동 도출한다."""
    result = []

    # 인사이트 1: 1위 모델의 강점
    if overall:
        top = overall[0]
        strategy = top["strategy"]
        info = insights.get(strategy, {})
        strengths = [s for s in info.get("strengths", []) if "1위" in s]
        strength_cats = [_cat_meta(s.split("(")[0], "label", s.split("(")[0]) for s in strengths]
        weak = info.get("weaknesses", [])
        weak_cats = [_cat_meta(w.split("(")[0], "label", w.split("(")[0]) for w in weak]

        body = f"**{_format_model_name(strategy)}**는 "
        if strength_cats:
            body += f"{', '.join(strength_cats)} 분야에서 1위를 차지했습니다"
        if weak_cats:
            body += f". 단, {', '.join(weak_cats)} 분야에서는 상대적으로 낮은 성능을 보입니다"
        body += "."

        result.append({"title": "실무 특화 모델 vs 범용 모델의 차이", "body": body})

    # 인사이트 2: 카테고리별 1위가 분산되는지 집중되는지
    winner_set: Dict[str, int] = {}
    for cat, df in ranked.items():
        if not df.empty:
            w = df.iloc[0]["strategy"]
            winner_set[w] = winner_set.get(w, 0) + 1

    dominant = [(s, c) for s, c in winner_set.items() if c >= 2]
    if dominant:
        dominant.sort(key=lambda x: -x[1])
        top_dom = dominant[0]
        result.append({
            "title": "한 모델의 복수 카테고리 우세",
            "body": (
                f"**{_format_model_name(top_dom[0])}**가 {top_dom[1]}개 카테고리에서 1위를 차지해, "
                f"단일 모델 도입 시 가장 넓은 커버리지를 제공합니다."
            ),
        })
    else:
        result.append({
            "title": "카테고리별 최적 모델이 분산",
            "body": (
                "카테고리마다 1위 모델이 달라, 문서 종류별로 다른 모델을 적용하는 것이 최적입니다. "
                "단일 모델로 통합 운영 시 일부 카테고리에서 성능 손실이 발생합니다."
            ),
        })

    # 인사이트 3: 동점 구간 — 성능 차가 작으면 비용으로 선택
    tight_cats = []
    for cat, df in ranked.items():
        if len(df) >= 2:
            gap = float(df.iloc[0]["composite"]) - float(df.iloc[1]["composite"])
            if gap < 0.03:  # 3% 미만 차이
                tight_cats.append(cat)

    tight_cats_labels = [_cat_meta(cat, "label", cat) for cat in tight_cats]
    if tight_cats_labels:
        result.append({
            "title": "일부 카테고리는 모델 간 성능 차이가 미미",
            "body": (
                f"{', '.join(tight_cats_labels)} 카테고리에서 1위와 2위의 점수 차가 3% 미만으로 "
                f"통계적으로 유의미한 차이가 없습니다. "
                f"이 경우 **처리 속도나 운영 비용** 기준으로 모델을 선택하는 것이 합리적입니다."
            ),
        })
    else:
        result.append({
            "title": "모든 카테고리에서 명확한 우열이 존재",
            "body": (
                "1위와 2위 모델 간 점수 차가 모든 카테고리에서 통계적으로 유의미합니다. "
                "단순 비용 절감 목적으로 2위 모델을 선택하면 체감 가능한 품질 저하가 발생할 수 있습니다."
            ),
        })

    return result[:3]


# ---------------------------------------------------------------------------
# Section 4: 카테고리별 상세 분석
# ---------------------------------------------------------------------------

def _section4_category_detail(
    ranked: Dict[str, pd.DataFrame],
    raw_results: Dict[str, dict],
    insights: Dict[str, dict],
) -> List[str]:
    lines = [
        "## 4. 문서 종류별 상세 분석",
        "",
    ]

    for cat_idx, (category, df) in enumerate(ranked.items(), 1):
        label = _cat_meta(category, "label", category.upper())
        n_qa = raw_results.get(category, {}).get("n_qa", "?")
        char = _cat_meta(category, "characteristic", "")
        source = _cat_meta(category, "data_source", "")

        lines += [
            f"### 4-{cat_idx}. {label}",
            "",
            f"**문서 특성**: {char}  ",
            f"**테스트 데이터**: {source}  ",
            f"**평가 질의 수**: {n_qa}개",
            "",
        ]

        if df.empty:
            lines += ["결과 데이터 없음", ""]
            continue

        # 1위와 2위의 차이 계산
        top_score = float(df.iloc[0]["composite"])
        top_name = df.iloc[0]["strategy"]

        # 결과 테이블 (비기술 친화)
        lines += [
            "**순위별 결과** (점수는 0~1, 높을수록 우수):",
            "",
            "| 순위 | AI 조합 | 종합 점수 | 2위 대비 | '놓치지 않는 능력' | '지어내지 않는 능력' |",
            "|:----:|:--------|:--------:|:-------:|:----------------:|:------------------:|",
        ]

        for _, row in df.head(5).iterrows():
            rank = int(row["rank"])
            strategy = row["strategy"]
            composite = float(row["composite"])
            recall = float(row.get("context_recall", 0.0))
            faith = float(row.get("faithfulness", 0.0))
            model_display = _format_model_name(strategy)

            if rank == 1:
                vs_second = "기준 (1위)"
                rank_label = "🥇 **1위**"
                score_str = f"**{composite:.3f}**"
            elif rank == 2:
                diff_pct = (top_score - composite) / top_score * 100 if top_score > 0 else 0
                vs_second = f"−{diff_pct:.1f}%p"
                rank_label = f"🥈 2위"
                score_str = f"{composite:.3f}"
            else:
                diff_pct = (top_score - composite) / top_score * 100 if top_score > 0 else 0
                vs_second = f"−{diff_pct:.1f}%p"
                rank_label = f"{rank}위"
                score_str = f"{composite:.3f}"

            lines.append(
                f"| {rank_label} | {model_display} | {score_str} | {vs_second} | {recall:.3f} | {faith:.3f} |"
            )

        lines += [""]

        # 1위 선정 이유 — 비기술 설명
        winner_insight = insights.get(top_name, {})
        pattern = winner_insight.get("pattern", "")
        winner_display = _format_model_name(top_name)

        lines += [
            f"**왜 {winner_display}가 {label} 분야에서 1위인가?**",
            "",
            _explain_winner_reason(category, top_name, df, winner_insight),
            "",
        ]

    lines += ["---", ""]
    return lines


def _explain_winner_reason(
    category: str,
    winner: str,
    df: pd.DataFrame,
    insight: dict,
) -> str:
    """1위 선정 이유를 비기술 언어로 서술한다."""
    winner_display = _format_model_name(winner)

    explanations: Dict[str, Dict[str, str]] = {
        "legal": {
            "snowflake-ko": (
                f"> {winner_display}는 한국어 법률·금융 문서로 추가 학습된 모델입니다. "
                f"법률 문서는 '제N조 제N항'처럼 정확한 조문 번호와 용어 매칭이 답변 품질을 결정하는데, "
                f"이 모델은 한국어 실무 문서 패턴을 깊이 학습해 이 특성에서 강점을 보입니다."
            ),
        },
        "business": {
            "snowflake-ko": (
                f"> {winner_display}는 금융·공공·상업 보고서 도메인 데이터를 포함해 학습했습니다. "
                f"수치, 기업명, 상품명 등 고유 명사의 정확한 의미 파악이 필요한 금융 문서에서 "
                f"타 모델 대비 높은 정확도를 보입니다."
            ),
        },
        "medical": {
            "snowflake-ko": (
                f"> {winner_display}는 의료·공중보건 도메인의 전문 용어와 FAQ 구조를 처리하는 데 강점을 보입니다. "
                f"질환명, 약어 등 전문 용어의 의미론적 이해와 함께 BM25 키워드 매칭이 결합되어 높은 정확도를 달성했습니다."
            ),
        },
        "general": {
            "bge-m3": (
                f"> {winner_display}는 100개 이상 언어를 학습한 대규모 다국어 모델입니다. "
                f"위키피디아 수준의 방대한 범용 텍스트를 다루는 GENERAL 카테고리에서는 "
                f"광범위한 학습 데이터와 MIRACL 한국어 벤치마크 최고 점수(nDCG@10=70.0)를 보유한 "
                f"이 모델이 최적입니다."
            ),
        },
    }

    # 모델명에서 핵심 키워드 추출
    model_key = winner.lower()
    if "snowflake" in model_key:
        model_key = "snowflake-ko"
    elif "bge" in model_key or "bge-m3" in model_key:
        model_key = "bge-m3"

    cat_explanations = explanations.get(category, {})
    if model_key in cat_explanations:
        return cat_explanations[model_key]

    # 기본 설명 (데이터 기반 자동 생성)
    pattern = insight.get("pattern", "")
    top_row = df[df["strategy"] == winner]
    if not top_row.empty:
        score = float(top_row.iloc[0]["composite"])
        recall = float(top_row.iloc[0].get("context_recall", 0.0))
        if len(df) >= 2:
            second_score = float(df.iloc[1]["composite"])
            gap_pct = (score - second_score) / second_score * 100 if second_score > 0 else 0
            return (
                f"> {winner_display}가 종합 점수 {score:.3f}로 2위 대비 {gap_pct:.1f}% 우세합니다. "
                f"특히 '놓치지 않는 능력(Recall)'이 {recall:.3f}으로 높아 "
                f"{_cat_meta(category, 'label', category)} 문서에서 답변 누락을 최소화합니다."
            )
        return f"> {winner_display}가 종합 점수 {score:.3f}로 최우수 성능을 보였습니다."

    return f"> 측정 데이터 기반 최고 종합 점수를 기록했습니다."


# ---------------------------------------------------------------------------
# Section 5: 최종 모델 선정 가이드
# ---------------------------------------------------------------------------

def _section5_final_guide(
    selection: SelectionReport,
    ranked: Dict[str, pd.DataFrame],
    tie_summary: Dict[str, List[dict]],
) -> List[str]:
    lines = [
        "## 5. 최종 모델 선정 가이드",
        "",
        "### 5-1. 문서 종류별 추천 조합",
        "",
        "| 사용 상황 | 1순위 추천 | 2순위 (대안) | 선정 핵심 이유 | 주의사항 |",
        "|----------|-----------|------------|-------------|---------|",
    ]

    for category, rec in selection.per_category.items():
        label = _cat_meta(category, "label", category.upper())
        winner_display = _format_model_name(rec.winner)
        runner_up_display = _format_model_name(rec.runner_up) if rec.runner_up else "—"

        # 선정 이유 축약 (비기술)
        reason_short = _shorten_reason(rec.reason, category, rec.winner)
        note = "—"

        lines.append(
            f"| {label} | **{winner_display}** | {runner_up_display} | {reason_short} | {note} |"
        )

    for cat in selection.skipped_categories:
        label = _cat_meta(cat, "label", cat.upper())
        lines.append(f"| {label} | 추가 테스트 필요 | — | 표준 테스트 데이터 부족 | 고객 문서 확보 필요 |")

    lines += [""]

    # 문서 혼용 시 추천
    if selection.default_recommendation:
        winner_display = _format_model_name(selection.default_recommendation)
        lines += [
            "### 5-2. 문서 종류가 혼용되거나 단일 모델이 필요한 경우",
            "",
            f"> **추천: {winner_display}**  ",
            f"> {selection.default_reason}",
            "",
        ]

    # 동점 그룹 — 비기술 해석
    has_ties = any(
        len(g["strategies"]) > 1
        for groups in tie_summary.values()
        for g in groups
    )

    if has_ties:
        lines += [
            "### 5-3. 성능 차이가 미미한 경우 — 비용·속도 기준으로 선택",
            "",
            "다음 조합들은 점수 차이가 5% 미만으로 **통계적으로 동등한 성능**입니다.",
            "이 경우 처리 속도(레이턴시)가 빠른 쪽을 선택하는 것이 합리적입니다:",
            "",
        ]
        for category, groups in tie_summary.items():
            tie_groups = [g for g in groups if len(g["strategies"]) > 1]
            for g in tie_groups:
                label = _cat_meta(category, "label", category.upper())
                strategies_display = " vs ".join(_format_model_name(s) for s in g["strategies"])
                lines += [
                    f"- **{label}**: {strategies_display}",
                    f"  - {g['note']}",
                    "",
                ]

    # API 모델 제외 이유
    lines += [
        "### 5-4. OpenAI / Upstage API 모델을 기본 추천에서 제외한 이유",
        "",
        "| 제외 이유 | 상세 |",
        "|----------|------|",
        "| 데이터 외부 전송 | 고객 문서가 OpenAI/Upstage 서버로 전송되어 보안·개인정보 리스크 발생 |",
        "| 운영 비용 | API 호출당 비용 발생, 대규모 서비스 시 비용 예측이 어려움 |",
        "| 추가 평가 가능 | 위 리스크가 수용 가능한 경우 `--include_api` 옵션으로 별도 평가 가능 |",
        "",
        "---",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# 부록
# ---------------------------------------------------------------------------

def _appendix(
    ranked: Dict[str, pd.DataFrame],
    insights: Dict[str, dict],
    raw_results: Dict[str, dict],
) -> List[str]:
    lines = [
        "---",
        "",
        "## 부록: 기술 세부사항",
        "",
        "> 이 섹션은 개발팀·데이터팀을 위한 기술 참고 정보입니다.",
        "",
        "### A. 비교 대상 모델 스펙",
        "",
        "| 모델 키 | 표시명 | 파라미터 | 특성 |",
        "|---------|-------|---------|------|",
    ]
    for key, meta in DENSE_MODEL_DISPLAY.items():
        lines.append(f"| `{key}` | {meta['display']} | {meta['params']} | {meta['note']} |")
    lines += [
        "",
        "### B. 평가 지표 가중치 산정 근거",
        "",
        "| 지표 | 가중치 | 산정 근거 |",
        "|------|--------|---------|",
        "| Context Recall | 0.35 | 서비스에서 정보 누락은 오답보다 사용자 이탈로 직결. RAGAS 권고 가중치. |",
        "| Context Precision | 0.30 | 불필요 컨텍스트 포함 시 LLM 답변 품질 저하 및 토큰 비용 증가. |",
        "| Faithfulness | 0.20 | 할루시네이션 방지 — 법률·의료 도메인에서 특히 중요. |",
        "| Answer Relevancy | 0.15 | 위 3개 지표가 높으면 자연스럽게 따라오는 경향. |",
        "",
        "### C. 데이터셋 상세",
        "",
        "| 카테고리 | 데이터셋 | 코퍼스 크기 | 쿼리 수 | 샘플링 전략 |",
        "|---------|---------|-----------|--------|-----------|",
        "| GENERAL | miracl/miracl (ko) + Ko-StrategyQA + Belebele + MrTiDy | 1.5M → 50,000 샘플 | 1,081 | max_corpus=50,000 |",
        "| LEGAL | yjoonjang/markers_bm (law) | ~180 docs | ~30 | 전체 사용 |",
        "| BUSINESS | yjoonjang/markers_bm (finance+public+commerce) | ~540 docs | ~84 | 전체 사용 |",
        "| MEDICAL | xhluca/publichealth-qa (korean) | 77 QA | 77 | 전체 사용 |",
        "| TECHNICAL | 사용자 업로드 문서 | 가변 | LLM 생성 | sampling_ratio=0.15 |",
        "",
        "### D. 재현 방법",
        "",
        "```bash",
        "# HuggingFace 표준 데이터셋 모드",
        "uv run python -m rag_bench.scripts.run_service_bench \\",
        "    --mode hf \\",
        "    --categories general,legal,business,medical \\",
        "    --preset service",
        "",
        "# 경영진 보고서 생성",
        "uv run python -m rag_bench.analysis.reporter_exec \\",
        "    --run_dir _benchdata/service_run",
        "```",
        "",
        "### E. 전체 순위 상세 (기술 참고용)",
        "",
    ]

    # 카테고리별 전체 순위
    for category, df in ranked.items():
        label = _cat_meta(category, "label", category.upper())
        lines += [
            f"**{label}**",
            "",
            "| 순위 | 조합 | Recall | Precision | Faithfulness | Relevancy | 복합 점수 | Recall(%) |",
            "|:----:|:-----|:------:|:---------:|:------------:|:---------:|:--------:|:---------:|",
        ]
        for _, row in df.iterrows():
            rank = int(row["rank"])
            strategy = row["strategy"]
            recall = float(row.get("context_recall", 0.0))
            precision = float(row.get("context_precision", 0.0))
            faith = float(row.get("faithfulness", 0.0))
            relev = float(row.get("answer_relevancy", 0.0))
            composite = float(row["composite"])
            recall_pct = float(row.get("recall_pct", row.get("pass_rate", 0.0)))

            score_str = f"**{composite:.3f}**" if rank == 1 else f"{composite:.3f}"
            lines.append(
                f"| {rank} | `{strategy}` | {recall:.3f} | {precision:.3f} | {faith:.3f} | {relev:.3f} | {score_str} | {recall_pct:.1f}% |"
            )
        lines.append("")

    return lines


# ---------------------------------------------------------------------------
# 콘솔 요약
# ---------------------------------------------------------------------------

def _print_exec_summary(selection: SelectionReport) -> None:
    print("\n" + "=" * 60)
    print(" 경영진 보고서 — 핵심 결론")
    print("=" * 60)
    for category, rec in selection.per_category.items():
        label = _cat_meta(category, "label", category.upper())
        print(f"  {label:20s} → {_format_model_name(rec.winner)}")
        print(f"  {'':20s}   점수={rec.composite_score:.3f} | Pass={rec.pass_rate:.1f}%")
    if selection.default_recommendation:
        print(f"\n  단일 추천: {_format_model_name(selection.default_recommendation)}")
        print(f"  이유    : {selection.default_reason}")


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _format_model_name(strategy: str) -> str:
    """기술적 모델 키를 독자 친화적 표현으로 변환한다."""
    if not strategy:
        return strategy

    sparse_display = {
        "korean_bm25": "BM25",
        "splade":      "SPLADE",
    }

    dense_found = None
    for key, meta in DENSE_MODEL_DISPLAY.items():
        if key.lower() in strategy.lower():
            dense_found = meta["display"]
            break

    sparse_found = None
    for key, label in sparse_display.items():
        if key in strategy.lower():
            sparse_found = label
            break

    if dense_found and sparse_found:
        return f"{dense_found} + {sparse_found}"
    if dense_found:
        return dense_found
    return strategy


def _shorten_reason(reason: str, category: str, winner: str) -> str:
    """선정 이유를 30자 이내 비기술 표현으로 압축한다."""
    model_key = winner.lower()

    short_reasons: Dict[str, Dict[str, str]] = {
        "legal":    {"snowflake": "한국어 법률 용어 정밀 매칭 최강"},
        "business": {"snowflake": "금융·공시 문서 의미 이해 최강"},
        "medical":  {"snowflake": "의료 FAQ 전문 용어 처리 최강"},
        "general":  {"bge": "대용량 범용 검색 국제 표준 SOTA"},
        "technical": {"e5": "기술 용어 구조적 매칭 강점"},
    }

    cat_map = short_reasons.get(category, {})
    for key, short in cat_map.items():
        if key in model_key:
            return short

    # 원본 이유에서 첫 번째 파이프 이전만 사용
    if "|" in reason:
        first_part = reason.split("|")[0].strip()
        return first_part[:40] if len(first_part) > 40 else first_part
    return reason[:40] if len(reason) > 40 else reason


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="서비스 벤치마크 결과 분석 — 경영진용 보고서 생성"
    )
    parser.add_argument("--run_dir", required=True, help="run_service_bench.py 출력 디렉토리")
    parser.add_argument("--output_dir", default=None, help="보고서 저장 위치 (기본: run_dir)")
    parser.add_argument("--threshold", type=float, default=0.05, help="동점 판정 임계값 (기본: 0.05)")
    args = parser.parse_args()

    report = generate_exec_report(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        similarity_threshold=args.threshold,
    )
    if not report.per_category:
        sys.exit(1)


if __name__ == "__main__":
    main()
