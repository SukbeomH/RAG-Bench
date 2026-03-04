"""
K8s 벤치마크 결과 보고서 생성기.

K8s 분산 실행 결과(콤보별 개별 디렉토리)를 기존 분석 파이프라인 포맷으로
병합한 뒤, reporter_exec.py 스타일의 경영진 보고서를 생성한다.

출력 파일:
  - {run_dir}/k8s_benchmark_report.md  (Markdown 보고서)

CLI:
    python k8s_results/generate_k8s_report.py --run_dir k8s_results/20260226-0948
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# 패키지 임포트
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autorag_rag_eval.constants import RAGAS_WEIGHTS, RAGAS_COLS

# ---------------------------------------------------------------------------
# 표시명 매핑
# ---------------------------------------------------------------------------

DENSE_DISPLAY = {
    "bge-m3": {
        "short": "BGE-M3",
        "params": "570M",
        "note": "100+ 언어, MIRACL 한국어 SOTA",
        "type": "local",
    },
    "e5": {
        "short": "E5-multilingual",
        "params": "560M",
        "note": "다국어 E5, 명령어 prefix 방식",
        "type": "local",
    },
    "kosimcse": {
        "short": "KoSimCSE",
        "params": "110M",
        "note": "한국어 SimCSE 대조 학습",
        "type": "local",
    },
    "snowflake": {
        "short": "Snowflake-KO",
        "params": "600M",
        "note": "한국어 실무 문서 SOTA",
        "type": "local",
    },
    "text-embedding-3-large": {
        "short": "OpenAI (API)",
        "params": "—",
        "note": "text-embedding-3-large, 3072차원",
        "type": "api",
    },
    "embedding-query": {
        "short": "Upstage Solar (API)",
        "params": "—",
        "note": "Upstage Solar Embedding, 4096차원",
        "type": "api",
    },
}

SPARSE_DISPLAY = {
    "korean_bm25": "BM25",
    "splade": "SPLADE",
}

RERANKER_DISPLAY = {
    "colbert": "ColBERT",
    "flashrank": "FlashRank",
}


def _short_name(strategy: str) -> str:
    """기술적 strategy → 사람 친화 표시명."""
    dense = None
    for key, meta in DENSE_DISPLAY.items():
        if key.lower() in strategy.lower():
            dense = meta["short"]
            break

    sparse = None
    for key, label in SPARSE_DISPLAY.items():
        if key in strategy.lower():
            sparse = label
            break

    reranker = None
    strategy_lower = strategy.lower()
    for key, label in RERANKER_DISPLAY.items():
        if key in strategy_lower:
            reranker = label
            break

    parts = []
    if dense:
        parts.append(dense)
    if sparse:
        parts.append(sparse)
    if reranker:
        parts.append(reranker)

    if parts:
        return " + ".join(parts)
    return strategy


# ---------------------------------------------------------------------------
# 데이터 로드 — K8s 콤보별 결과 병합
# ---------------------------------------------------------------------------


def load_k8s_results(run_dir: Path) -> Dict[str, dict]:
    """K8s 콤보별 result.json을 카테고리별로 병합."""
    results: Dict[str, dict] = {}

    for cat_dir in sorted(run_dir.iterdir()):
        if not cat_dir.is_dir():
            continue

        ragas_all = []
        n_qa = 0
        category = cat_dir.name

        for combo_dir in sorted(cat_dir.iterdir()):
            if not combo_dir.is_dir():
                continue
            result_file = combo_dir / "result.json"
            if not result_file.exists():
                continue

            data = json.loads(result_file.read_text(encoding="utf-8"))
            category = data.get("category", cat_dir.name)
            n_qa = max(n_qa, data.get("n_qa", 0))
            ragas_all.extend(data.get("ragas", []))

        if ragas_all:
            results[category] = {
                "category": category,
                "n_qa": n_qa,
                "ragas": ragas_all,
            }

    return results


def load_k8s_latency(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """K8s 콤보별 latency.csv를 카테고리별로 병합."""
    latency: Dict[str, pd.DataFrame] = {}

    for cat_dir in sorted(run_dir.iterdir()):
        if not cat_dir.is_dir():
            continue

        dfs = []
        for combo_dir in sorted(cat_dir.iterdir()):
            if not combo_dir.is_dir():
                continue
            lat_file = combo_dir / "latency.csv"
            if lat_file.exists():
                try:
                    df = pd.read_csv(lat_file, encoding="utf-8-sig")
                    dfs.append(df)
                except Exception:
                    pass

        if dfs:
            latency[cat_dir.name] = pd.concat(dfs, ignore_index=True)

    return latency


# ---------------------------------------------------------------------------
# 순위 계산
# ---------------------------------------------------------------------------


def rank_combos(
    raw_results: Dict[str, dict],
    latency: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """RAGAS 가중 복합 점수 기반 순위 계산.

    레이턴시는 참고 데이터로만 병합하며, 순위 결정에는 사용하지 않는다.
    (2026-02-24 확정: LLM 추론 노이즈가 전략 간 차이를 압도하여 측정 신뢰도 낮음)
    """
    ranked: Dict[str, pd.DataFrame] = {}

    for category, data in raw_results.items():
        ragas = data.get("ragas", [])
        if not ragas:
            continue

        df = pd.DataFrame(ragas)
        for col in RAGAS_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0).clip(0.0, 1.0)

        df["composite"] = df.apply(
            lambda row: round(
                sum(row.get(c, 0) * w for c, w in RAGAS_WEIGHTS.items()), 4
            ),
            axis=1,
        )

        # 레이턴시 병합 (참고용, 순위 결정에 미사용)
        if category in latency:
            lat_df = latency[category]
            agg = lat_df.groupby("strategy")["latency_ms"].agg(["mean", "median"])
            agg.columns = ["avg_latency_ms", "median_latency_ms"]
            df = df.merge(agg, on="strategy", how="left")

        for col in ["avg_latency_ms", "median_latency_ms"]:
            if col not in df.columns:
                df[col] = float("nan")

        # 복합 점수 기준으로만 정렬 (레이턴시 미반영)
        df = df.sort_values(
            "composite",
            ascending=False,
        ).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        ranked[category] = df

    return ranked


# ---------------------------------------------------------------------------
# 보고서 렌더링
# ---------------------------------------------------------------------------


def generate_report(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Path:
    """K8s 벤치마크 결과 보고서를 생성한다."""
    output_dir = output_dir or run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str):
        if verbose:
            print(msg)

    _log(f"\n{'=' * 60}")
    _log(" K8s 벤치마크 결과 보고서 생성")
    _log(f"{'=' * 60}")

    _log(f"\n[1/3] 결과 로드 중... ({run_dir})")
    raw_results = load_k8s_results(run_dir)
    latency = load_k8s_latency(run_dir)

    if not raw_results:
        print(f"오류: {run_dir} 에서 result.json 파일을 찾을 수 없습니다.")
        sys.exit(1)

    _log(f"  카테고리: {list(raw_results.keys())}")
    _log(f"  레이턴시: {list(latency.keys())}")

    _log("\n[2/3] 순위 계산 중...")
    ranked = rank_combos(raw_results, latency)
    for cat, df in ranked.items():
        _log(f"  {cat}: {len(df)}개 조합")

    _log("\n[3/3] 보고서 렌더링 중...")
    md = _render_markdown(run_dir, raw_results, ranked)

    md_path = output_dir / "k8s_benchmark_report.md"
    md_path.write_text(md, encoding="utf-8")

    _log(f"\n  보고서 저장: {md_path}")
    _log(f"\n{'=' * 60}")
    _log(" 보고서 생성 완료")
    _log(f"{'=' * 60}")

    _print_summary(ranked)
    return md_path


# ---------------------------------------------------------------------------
# Markdown 렌더러
# ---------------------------------------------------------------------------


def _render_markdown(
    run_dir: Path,
    raw_results: Dict[str, dict],
    ranked: Dict[str, pd.DataFrame],
) -> str:
    lines: List[str] = []
    generated_at = time.strftime("%Y년 %m월 %d일")

    lines += [
        "<!-- K8s 벤치마크 결과 보고서 — generate_k8s_report.py 자동 생성 -->",
        f"<!-- 생성일: {generated_at} -->",
        "",
    ]

    lines += _cover(generated_at, raw_results, ranked)
    lines += _exec_summary(ranked)
    lines += _section1_background(ranked)
    lines += _section2_measurement(raw_results)
    lines += _section3_ranking(ranked)
    lines += _section4_metric_detail(ranked)
    lines += _section5_latency(ranked)
    lines += _section6_model_comparison(ranked)
    lines += _section7_recommendation(ranked)
    lines += _appendix(ranked, raw_results)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 커버
# ---------------------------------------------------------------------------


def _cover(
    generated_at: str,
    raw_results: Dict[str, dict],
    ranked: Dict[str, pd.DataFrame],
) -> List[str]:
    categories = list(ranked.keys())
    total_qa = sum(raw_results[c].get("n_qa", 0) for c in categories)
    n_combos = max(len(df) for df in ranked.values()) if ranked else 0

    # 사용된 리랭커 감지
    used_rerankers = set()
    for df in ranked.values():
        for strategy in df["strategy"]:
            for key, label in RERANKER_DISPLAY.items():
                if key in strategy.lower():
                    used_rerankers.add(label)

    if len(used_rerankers) == 1:
        reranker_label = next(iter(used_rerankers))
        title = f"# RAG {reranker_label} Rerank 벤치마크 결과 보고서"
        subtitle = f"### Dense × Sparse 조합별 성능 비교 — {reranker_label} Rerank + Contextual Retrieval 고정"
        pipeline = f"> **파이프라인**: Dense 검색 + Sparse 검색 + {reranker_label} 리랭커 + Contextual 문맥 강화  "
    elif len(used_rerankers) >= 2:
        reranker_list = " + ".join(sorted(used_rerankers))
        title = "# RAG 리랭커 비교 벤치마크 결과 보고서"
        subtitle = f"### Dense × Sparse × Reranker 조합별 성능 비교 — {reranker_list} 리랭커 비교"
        pipeline = f"> **파이프라인**: Dense 검색 + Sparse 검색 + 리랭커({reranker_list}) + Contextual 문맥 강화  "
    else:
        title = "# RAG 벤치마크 결과 보고서"
        subtitle = "### Dense × Sparse 조합별 성능 비교"
        pipeline = "> **파이프라인**: Dense 검색 + Sparse 검색 + Contextual 문맥 강화  "

    return [
        title,
        subtitle,
        "",
        f"> **작성일**: {generated_at}  ",
        f"> **평가 범위**: {len(categories)}개 카테고리 / {n_combos}개 AI 조합 / 총 {total_qa * n_combos:,}개 질의응답 테스트  ",
        pipeline,
        "> **실행 환경**: EKS K8s 클러스터 (management 노드, CPU-only)  ",
        "",
        "---",
        "",
    ]


# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------


def _exec_summary(ranked: Dict[str, pd.DataFrame]) -> List[str]:
    # 사용된 리랭커 감지
    used_rerankers = set()
    for df in ranked.values():
        for strategy in df["strategy"]:
            for key, label in RERANKER_DISPLAY.items():
                if key in strategy.lower():
                    used_rerankers.add(label)

    if len(used_rerankers) == 1:
        reranker_name = next(iter(used_rerankers))
        summary_desc = (
            f"> 이 보고서는 RAG 검색 파이프라인에서 {reranker_name} Rerank를 적용한 상태에서\n"
            "> **어떤 Dense + Sparse 조합이 최적인지**를 K8s 클러스터 실험으로 확인한 결과입니다."
        )
    elif len(used_rerankers) >= 2:
        reranker_list = " / ".join(sorted(used_rerankers))
        summary_desc = (
            f"> 이 보고서는 RAG 검색 파이프라인에서 {reranker_list} 리랭커를 비교하여\n"
            "> **어떤 Dense + Sparse + Reranker 조합이 최적인지**를 K8s 클러스터 실험으로 확인한 결과입니다."
        )
    else:
        summary_desc = (
            "> 이 보고서는 RAG 검색 파이프라인에서\n"
            "> **어떤 Dense + Sparse 조합이 최적인지**를 K8s 클러스터 실험으로 확인한 결과입니다."
        )

    lines = [
        "## Executive Summary",
        "",
        summary_desc,
        "",
        "### 핵심 결론",
        "",
    ]

    for category, df in ranked.items():
        if df.empty:
            continue

        label = category.upper()
        top = df.iloc[0]
        top_name = _short_name(top["strategy"])

        # 카테고리 헤더
        lines += [
            f"#### {label} 카테고리",
            "",
        ]

        # 종합 1위 박스
        lines += [
            f"**종합 1위: {top_name}** (복합 점수 {top['composite']:.4f})",
            "",
        ]

        # 즉시 결정 테이블
        lines += [
            "| 순위 | 조합 | 복합 점수 | 핵심 강점 |",
            "|:----:|:-----|:--------:|:---------|",
        ]

        for _, row in df.head(3).iterrows():
            rank = int(row["rank"])
            name = _short_name(row["strategy"])
            composite = row["composite"]

            # 개별 강점 도출
            strength = _derive_strength(row, df)

            if rank == 1:
                lines.append(
                    f"| **{rank}** | **{name}** | **{composite:.4f}** | {strength} |"
                )
            else:
                lines.append(f"| {rank} | {name} | {composite:.4f} | {strength} |")

        lines += [""]

    # 즉시 실행 항목
    lines += [
        "### 즉시 실행 항목",
        "",
    ]

    # 카테고리별 1위 요약
    for category, df in ranked.items():
        top = df.iloc[0]
        top_name = _short_name(top["strategy"])
        lines.append(
            f"- [ ] **{category.upper()}**: 서비스 파이프라인에 **{top_name}** 조합 통합"
        )

    # 공통 권고사항 1회만 출력
    has_api = any(
        any(
            k.lower() in row["strategy"].lower()
            for k, m in DENSE_DISPLAY.items()
            if m.get("type") == "api"
        )
        for df in ranked.values()
        for _, row in df.iterrows()
    )
    if has_api:
        lines.append(
            "- [ ] API 모델 사용 시 보안(데이터 외부 전송) 및 비용(건당 과금) 검토"
        )

    lines += [""]

    lines += ["---", ""]
    return lines


def _derive_strength(row: pd.Series, df: pd.DataFrame) -> str:
    """해당 조합의 핵심 강점을 데이터에서 자동 도출."""
    strategy = row["strategy"]
    strengths = []

    metrics = {
        "faithfulness": "Faithfulness 최고",
        "context_recall": "Recall 최고",
        "context_precision": "Precision 최고",
        "answer_relevancy": "Relevancy 최고",
    }

    for metric, label in metrics.items():
        best_val = df[metric].max()
        if abs(row[metric] - best_val) < 1e-6:
            strengths.append(label)

    if not strengths:
        strengths.append("균형 성능")

    return ", ".join(strengths[:2])


# ---------------------------------------------------------------------------
# Section 1: 배경
# ---------------------------------------------------------------------------


def _section1_background(ranked: Dict[str, pd.DataFrame]) -> List[str]:
    n_combos = max(len(df) for df in ranked.values()) if ranked else 0

    # 실제 사용된 Dense 모델 감지
    used_dense = set()
    for df in ranked.values():
        for strategy in df["strategy"]:
            for key in DENSE_DISPLAY:
                if key.lower() in strategy.lower():
                    used_dense.add(key)
                    break

    local_models = [k for k in used_dense if DENSE_DISPLAY[k]["type"] == "local"]
    api_models = [k for k in used_dense if DENSE_DISPLAY[k]["type"] == "api"]
    n_dense = len(used_dense)

    # 사용된 리랭커 감지
    used_rerankers = set()
    for df in ranked.values():
        for strategy in df["strategy"]:
            for key, label in RERANKER_DISPLAY.items():
                if key in strategy.lower():
                    used_rerankers.add(label)

    if len(used_rerankers) == 1:
        reranker_name = next(iter(used_rerankers))
        biz_question = (
            f'> **"{reranker_name} Rerank를 적용한 상태에서, 어떤 Dense + Sparse 조합이**  \n'
            '> **가장 정확한 검색 결과를 제공하는가?"**'
        )
        pipeline_diagram = (
            f"질문 → [Dense 검색] + [Sparse 검색] → [{reranker_name} 리랭킹] → [LLM 답변 생성]\n"
            "         ↑ 변수         ↑ 변수           ↑ 고정              ↑ 고정"
        )
        reranker_var_label = "↑ 고정"
    elif len(used_rerankers) >= 2:
        reranker_list = " / ".join(sorted(used_rerankers))
        biz_question = (
            '> **"어떤 Dense + Sparse + Reranker 조합이**  \n'
            '> **가장 정확한 검색 결과를 제공하는가?"**'
        )
        pipeline_diagram = (
            f"질문 → [Dense 검색] + [Sparse 검색] → [리랭킹({reranker_list})] → [LLM 답변 생성]\n"
            "         ↑ 변수         ↑ 변수           ↑ 변수                  ↑ 고정"
        )
    else:
        biz_question = (
            '> **"어떤 Dense + Sparse 조합이**  \n'
            '> **가장 정확한 검색 결과를 제공하는가?"**'
        )
        pipeline_diagram = (
            "질문 → [Dense 검색] + [Sparse 검색] → [리랭킹] → [LLM 답변 생성]\n"
            "         ↑ 변수         ↑ 변수         ↑ 고정       ↑ 고정"
        )

    lines = [
        "## 1. 무엇을, 왜 테스트했는가",
        "",
        "### 1-1. 비즈니스 질문",
        "",
        biz_question,
        "",
        "### 1-2. 테스트 설계",
        "",
        "고객이 질문하면 RAG 시스템은 다음 4단계를 거칩니다:",
        "",
        "```",
        pipeline_diagram,
        "```",
        "",
        "| 방식 | 작동 원리 | 비유 | 강점 |",
        "|------|----------|------|------|",
        "| **Dense 검색** (의미 기반) | 질문의 '뜻'을 이해해서 유사 내용을 찾음 | 사서가 내용을 읽고 추천 | 유사 표현·동의어에 강함 |",
        "| **Sparse 검색** (키워드 기반) | 단어 그대로 매칭하여 찾음 | 색인에서 단어 검색 | 정확한 용어·번호에 강함 |",
        "",
        f"이번 평가에서는 Dense {n_dense}종 × Sparse 2종 = **{n_combos}개 조합**을 비교했습니다.",
        "",
        "### 1-3. 고정 파이프라인 — 변수에서 제외한 항목",
        "",
        "| 고정 요소 | 적용 이유 |",
        "|----------|---------|",
    ]

    # 리랭커가 1종이면 고정 요소로, 2종이면 생략 (변수)
    if len(used_rerankers) == 1:
        reranker_name = next(iter(used_rerankers))
        if reranker_name == "ColBERT":
            lines.append(
                "| **ColBERT 리랭커** (jina-colbert-v2) | 최종 답변 후보를 토큰 수준으로 재정렬. 오답률 25% 감소 확인 (IBM). |"
            )
        elif reranker_name == "FlashRank":
            lines.append(
                "| **FlashRank 리랭커** | 경량 교차 인코더 기반 리랭킹. 속도 대비 품질 균형. |"
            )

    lines += [
        "| **Contextual 문맥 강화** | 검색 전 청크에 문맥을 AI로 추가. 검색 실패율 67% 감소 (Anthropic). |",
        "",
        "### 1-4. 비교 대상 모델",
        "",
        "| 구분 | 모델명 | 파라미터 | 특성 | 유형 |",
        "|------|-------|---------|------|:----:|",
    ]

    for key in sorted(used_dense, key=lambda k: (DENSE_DISPLAY[k]["type"], k)):
        m = DENSE_DISPLAY[key]
        model_type = "로컬(HF)" if m["type"] == "local" else "API"
        lines.append(
            f"| Dense | **{m['short']}** | {m['params']} | {m['note']} | {model_type} |"
        )

    lines += [
        "| Sparse | **BM25** (OKt) | — | 한국어 형태소 기반 키워드 매칭 | — |",
        "| Sparse | **SPLADE** | 110M | 학습된 확장 토큰으로 동의어 포착 | — |",
        "",
    ]

    if api_models:
        lines += [
            "> **API 모델 참고**: API 모델(OpenAI, Upstage)은 로컬 모델 대비 품질을 비교하기 위해 포함했습니다.",
            "> 실서비스에서는 보안(데이터 외부 전송) 및 비용(건당 과금) 관점에서 별도 검토가 필요합니다.",
            "",
        ]

    lines += ["---", ""]
    return lines


# ---------------------------------------------------------------------------
# Section 2: 측정 방법
# ---------------------------------------------------------------------------

DATASET_INFO = {
    "general": {
        "source": "MIRACL(ko) + Ko-StrategyQA + Belebele + MrTiDy",
        "note": "위키피디아 기반 범용 질의응답",
    },
    "legal": {"source": "법률 QA 데이터셋", "note": "법률 문서 질의응답"},
    "business": {"source": "비즈니스 QA 데이터셋", "note": "비즈니스 문서 질의응답"},
    "medical": {"source": "의료 QA 데이터셋", "note": "의료 문서 질의응답"},
    "technical": {"source": "기술 QA 데이터셋", "note": "기술 문서 질의응답"},
}


def _section2_measurement(raw_results: Dict[str, dict]) -> List[str]:
    lines = [
        "## 2. 어떻게 측정했는가",
        "",
        "### 2-1. 테스트 데이터셋",
        "",
        "| 카테고리 | 데이터 출처 | 쿼리 수 | 특성 |",
        "|---------|-----------|:------:|------|",
    ]

    for category, data in raw_results.items():
        cat_key = category.lower()
        info = DATASET_INFO.get(cat_key, {"source": "—", "note": "—"})
        n_qa = data.get("n_qa", "?")
        lines.append(
            f"| {category.upper()} | {info['source']} | {n_qa} | {info['note']} |"
        )

    lines += [""]
    lines += [
        "### 2-2. 평가 지표 — 4가지",
        "",
        "AI 답변 품질을 측정하는 4가지 관점을 가중 평균하여 **종합 점수**를 계산합니다:",
        "",
        "| 지표 | 측정 내용 | 쉽게 말하면 | 가중치 |",
        "|------|----------|-----------|:------:|",
        "| **Context Recall** | 정답 내용을 빠뜨리지 않았는가 | '놓치지 않는 능력' | **35%** |",
        "| **Context Precision** | 찾아온 문서가 질문과 관련 있는가 | '쓸모없는 내용 배제 능력' | 30% |",
        "| **Faithfulness** | 답변이 문서 내용에 근거하는가 | '지어내지 않는 능력' | 20% |",
        "| **Answer Relevancy** | 질문에 직접적으로 답하는가 | '동문서답 방지 능력' | 15% |",
        "",
        "> **종합 점수** = Recall×0.35 + Precision×0.30 + Faithfulness×0.20 + Relevancy×0.15",
        "> 0에서 1 사이 값이며, 1에 가까울수록 우수합니다.",
        "",
        "---",
        "",
    ]

    return lines


# ---------------------------------------------------------------------------
# Section 3: 종합 순위
# ---------------------------------------------------------------------------


def _section3_ranking(ranked: Dict[str, pd.DataFrame]) -> List[str]:
    lines = [
        "## 3. 종합 성능 결과",
        "",
    ]

    for cat_idx, (category, df) in enumerate(ranked.items(), 1):
        label = category.upper()

        lines += [
            f"### 3-{cat_idx}. {label} 카테고리",
            "",
            "#### 종합 순위표",
            "",
            "> 점수는 0~1 범위 (높을수록 우수). 1위 대비 차이(%p)를 병기합니다.",
            "",
            "| 순위 | AI 조합 | 종합 점수 | 1위 대비 | Recall | Precision | Faithfulness | Relevancy |",
            "|:----:|:--------|:--------:|:-------:|:------:|:---------:|:------------:|:---------:|",
        ]

        top_score = float(df.iloc[0]["composite"])

        for _, row in df.iterrows():
            rank = int(row["rank"])
            name = _short_name(row["strategy"])
            composite = row["composite"]
            recall = row["context_recall"]
            prec = row["context_precision"]
            faith = row["faithfulness"]
            rel = row["answer_relevancy"]

            if rank == 1:
                vs = "기준 (1위)"
                rank_label = "🥇 **1위**"
                score_str = f"**{composite:.4f}**"
            elif rank == 2:
                gap = (top_score - composite) / top_score * 100
                vs = f"−{gap:.1f}%"
                rank_label = "🥈 2위"
                score_str = f"{composite:.4f}"
            elif rank == 3:
                gap = (top_score - composite) / top_score * 100
                vs = f"−{gap:.1f}%"
                rank_label = "🥉 3위"
                score_str = f"{composite:.4f}"
            else:
                gap = (top_score - composite) / top_score * 100
                vs = f"−{gap:.1f}%"
                rank_label = f"{rank}위"
                score_str = f"{composite:.4f}"

            lines.append(
                f"| {rank_label} | {name} | {score_str} | {vs} "
                f"| {recall:.4f} | {prec:.4f} | {faith:.4f} | {rel:.4f} |"
            )

        lines += [""]

        # 핵심 인사이트
        lines += _derive_ranking_insights(df)

    lines += ["---", ""]
    return lines


def _derive_ranking_insights(df: pd.DataFrame) -> List[str]:
    """순위 데이터에서 핵심 인사이트 3가지를 도출."""
    lines = [
        "#### 핵심 인사이트",
        "",
    ]

    top = df.iloc[0]
    top_name = _short_name(top["strategy"])

    # 인사이트 1: 1위 선정 이유
    top_metrics = []
    for metric, label in [
        ("faithfulness", "Faithfulness"),
        ("context_recall", "Context Recall"),
        ("context_precision", "Context Precision"),
        ("answer_relevancy", "Answer Relevancy"),
    ]:
        if abs(top[metric] - df[metric].max()) < 1e-6:
            top_metrics.append(label)

    if top_metrics:
        lines += [
            f"**1. {top_name}가 종합 1위인 이유**",
            f"> {', '.join(top_metrics)} 지표에서 전 조합 중 최고점을 기록했습니다.",
        ]
    else:
        lines += [
            f"**1. {top_name}가 종합 1위인 이유**",
            "> 모든 지표에서 균형 있게 높은 성능을 보여 가중 합산에서 최고 점수를 달성했습니다.",
        ]

    if len(df) >= 2:
        second = df.iloc[1]
        gap = float(top["composite"]) - float(second["composite"])
        gap_pct = gap / float(top["composite"]) * 100
        lines.append(
            f"> 2위({_short_name(second['strategy'])}) 대비 {gap_pct:.1f}% 우세합니다."
        )
    lines.append("")

    # 인사이트 2: 상위 동점 구간
    if len(df) >= 2:
        gap_1_2 = abs(float(df.iloc[0]["composite"]) - float(df.iloc[1]["composite"]))
        if gap_1_2 < 0.02:
            second_name = _short_name(df.iloc[1]["strategy"])
            lines += [
                "**2. 1위와 2위 성능 차이가 미미**",
                f"> 1위와 2위의 종합 점수 차이가 {gap_1_2:.4f} ({gap_1_2 / float(df.iloc[0]['composite']) * 100:.1f}%)로 "
                "통계적으로 유의미하지 않을 수 있습니다.",
                "> 두 조합 모두 실서비스 적용이 가능하며, **운영 비용·보안 요건**(API vs 로컬)에 따라 선택하는 것이 합리적입니다.",
                "",
            ]
        else:
            lines += [
                "**2. 명확한 성능 우열 존재**",
                f"> 1위와 2위의 종합 점수 차이가 {gap_1_2:.4f} ({gap_1_2 / float(df.iloc[0]['composite']) * 100:.1f}%)로 "
                "유의미한 차이입니다.",
                "> 품질 중심 선택이 타당합니다.",
                "",
            ]

    # 인사이트 3: 로컬 vs API 모델 비교
    local_rows = []
    api_rows = []
    for _, row in df.iterrows():
        is_api = any(
            k.lower() in row["strategy"].lower()
            for k, m in DENSE_DISPLAY.items()
            if m.get("type") == "api"
        )
        if is_api:
            api_rows.append(row)
        else:
            local_rows.append(row)

    if local_rows and api_rows:
        best_local = max(local_rows, key=lambda r: r["composite"])
        best_api = max(api_rows, key=lambda r: r["composite"])
        local_name = _short_name(best_local["strategy"])
        api_name = _short_name(best_api["strategy"])
        diff = float(best_local["composite"]) - float(best_api["composite"])

        if diff > 0:
            lines += [
                "**3. 로컬 모델이 API 모델과 동등 이상의 성능**",
                f"> 로컬 최고({local_name}, {best_local['composite']:.4f})가 "
                f"API 최고({api_name}, {best_api['composite']:.4f})보다 {abs(diff):.4f} 우세합니다.",
                "> API 모델이 반드시 우수하지 않으므로, 보안·비용 측면에서 로컬 모델이 유리합니다.",
                "",
            ]
        else:
            lines += [
                "**3. API 모델이 품질 우위**",
                f"> API 최고({api_name}, {best_api['composite']:.4f})가 "
                f"로컬 최고({local_name}, {best_local['composite']:.4f})보다 {abs(diff):.4f} 우세합니다.",
                "> 다만 API 모델은 데이터 외부 전송 및 건당 과금이 발생하므로 보안·비용 검토가 필요합니다.",
                "",
            ]

    return lines


# ---------------------------------------------------------------------------
# Section 4: 지표별 상세
# ---------------------------------------------------------------------------


def _section4_metric_detail(ranked: Dict[str, pd.DataFrame]) -> List[str]:
    lines = [
        "## 4. 지표별 상세 분석",
        "",
    ]

    metric_info = [
        (
            "faithfulness",
            "Faithfulness — 지어내지 않는 능력",
            "답변이 검색된 문서 내용에 근거하는지 측정합니다. "
            "할루시네이션(거짓 정보 생성) 방지의 핵심 지표입니다.",
        ),
        (
            "context_recall",
            "Context Recall — 놓치지 않는 능력",
            "정답에 필요한 문서를 빠짐없이 검색하는지 측정합니다. "
            "정보 누락은 서비스 신뢰도를 직접 훼손합니다.",
        ),
        (
            "context_precision",
            "Context Precision — 정확하게 찾는 능력",
            "검색 결과 중 실제 관련 문서의 비율을 측정합니다. "
            "불필요한 문서가 많으면 LLM 답변 품질이 저하됩니다.",
        ),
        (
            "answer_relevancy",
            "Answer Relevancy — 질문에 답하는 능력",
            "생성된 답변이 질문에 직접적으로 대응하는지 측정합니다.",
        ),
    ]

    for cat_idx, (category, df) in enumerate(ranked.items(), 1):
        label = category.upper()
        lines += [
            f"### 4-{cat_idx}. {label} 카테고리",
            "",
        ]

        for metric, title, desc in metric_info:
            sorted_df = df.sort_values(metric, ascending=False).reset_index(drop=True)
            best = sorted_df.iloc[0]
            worst = sorted_df.iloc[-1]

            lines += [
                f"#### {title}",
                "",
                f"> {desc}",
                "",
                "| 순위 | 조합 | 점수 | 비고 |",
                "|:----:|:-----|:----:|:-----|",
            ]

            for i, (_, row) in enumerate(sorted_df.iterrows()):
                name = _short_name(row["strategy"])
                score = row[metric]
                note = ""
                if i == 0:
                    note = "**최고**"
                elif i == len(sorted_df) - 1:
                    gap = float(best[metric]) - float(row[metric])
                    note = f"최고 대비 −{gap:.4f}"
                lines.append(f"| {i + 1} | {name} | {score:.4f} | {note} |")

            gap_total = float(best[metric]) - float(worst[metric])
            gap_pct = gap_total / float(best[metric]) * 100 if best[metric] > 0 else 0

            lines += [
                "",
                f"> 전체 편차: {gap_total:.4f} ({gap_pct:.1f}%) — "
                + (
                    "조합 간 차이가 크므로 모델 선택이 중요합니다."
                    if gap_pct > 5
                    else "조합 간 차이가 작아 이 지표만으로는 우열을 가리기 어렵습니다."
                ),
                "",
            ]

    lines += ["---", ""]
    return lines


# ---------------------------------------------------------------------------
# Section 5: 레이턴시
# ---------------------------------------------------------------------------


def _section5_latency(ranked: Dict[str, pd.DataFrame]) -> List[str]:
    """레이턴시 참고 섹션. 순위 판단에는 미반영."""
    # 레이턴시 데이터가 없으면 섹션 자체를 생략
    has_latency = any(pd.notna(df["avg_latency_ms"]).any() for df in ranked.values())
    if not has_latency:
        return []

    lines = [
        "## 5. 레이턴시(속도) 참고",
        "",
        "> **레이턴시는 순위 결정에 반영하지 않습니다.**",
        "> LLM 추론 노이즈가 전략 간 차이를 압도하며, 동일 전략도 실행 시점에 따라 편차가 큽니다.",
        "> CPU-only 환경 수치이므로 GPU 환경과 직접 비교할 수 없습니다.",
        "> 아래는 실행 환경에서의 기준선 참고 데이터입니다.",
        "",
    ]

    for cat_idx, (category, df) in enumerate(ranked.items(), 1):
        valid = df[pd.notna(df["avg_latency_ms"])]
        if valid.empty:
            continue

        label = category.upper()
        sorted_df = valid.sort_values("avg_latency_ms").reset_index(drop=True)

        lines += [
            f"### {label}",
            "",
            "| 조합 | 평균 (s/query) | 중앙값 (s/query) |",
            "|:-----|:--------------:|:---------------:|",
        ]

        for _, row in sorted_df.iterrows():
            name = _short_name(row["strategy"])
            avg = row["avg_latency_ms"] / 1000
            med = (
                row["median_latency_ms"] / 1000
                if pd.notna(row.get("median_latency_ms"))
                else float("nan")
            )
            med_str = f"{med:.0f}" if pd.notna(med) else "—"
            lines.append(f"| {name} | {avg:.0f} | {med_str} |")

        lines += ["", "---", ""]

    return lines


# ---------------------------------------------------------------------------
# Section 6: 모델 비교
# ---------------------------------------------------------------------------


def _section6_model_comparison(ranked: Dict[str, pd.DataFrame]) -> List[str]:
    lines = [
        "## 6. 모델 유형별 비교",
        "",
    ]

    for cat_idx, (category, df) in enumerate(ranked.items(), 1):
        label = category.upper()
        lines += [
            f"### 6-{cat_idx}. {label} 카테고리",
            "",
        ]

        # Dense 모델 비교
        dense_groups: Dict[str, List[pd.Series]] = {}
        for _, row in df.iterrows():
            s = row["strategy"]
            for key, meta in DENSE_DISPLAY.items():
                if key.lower() in s.lower():
                    dlabel = meta["short"]
                    dense_groups.setdefault(dlabel, []).append(row)
                    break

        if len(dense_groups) > 1:
            lines += [
                "#### Dense 모델 비교 (Sparse 평균)",
                "",
                "> 각 Dense 모델의 BM25/SPLADE 결과를 평균하여 순수 Dense 모델 성능을 비교합니다.",
                "",
                "| Dense 모델 | 유형 | 평균 복합 점수 | 평균 Recall | 평균 Faithfulness |",
                "|-----------|:----:|:------------:|:----------:|:----------------:|",
            ]

            dense_summary = []
            for name, rows in dense_groups.items():
                rdf = pd.DataFrame(rows)
                model_type = "API"
                for key, meta in DENSE_DISPLAY.items():
                    if meta["short"] == name:
                        model_type = "API" if meta.get("type") == "api" else "로컬"
                        break
                dense_summary.append(
                    {
                        "name": name,
                        "type": model_type,
                        "composite": rdf["composite"].mean(),
                        "recall": rdf["context_recall"].mean(),
                        "faith": rdf["faithfulness"].mean(),
                    }
                )

            dense_summary.sort(key=lambda x: -x["composite"])
            for ds in dense_summary:
                lines.append(
                    f"| {ds['name']} | {ds['type']} | {ds['composite']:.4f} | {ds['recall']:.4f} "
                    f"| {ds['faith']:.4f} |"
                )

            lines.append("")

            # Dense 인사이트
            best_dense = dense_summary[0]
            worst_dense = dense_summary[-1]
            gap = best_dense["composite"] - worst_dense["composite"]
            lines += [
                f"> **{best_dense['name']}**가 평균 복합 점수 {best_dense['composite']:.4f}로 Dense 모델 중 1위.",
                f"> 최하위({worst_dense['name']}) 대비 {gap:.4f} ({gap / best_dense['composite'] * 100:.1f}%) 우세.",
                "",
            ]

        # Sparse 모델 비교
        sparse_groups: Dict[str, List[pd.Series]] = {}
        for _, row in df.iterrows():
            s = row["strategy"]
            for key, label in SPARSE_DISPLAY.items():
                if key in s.lower():
                    sparse_groups.setdefault(label, []).append(row)
                    break

        if len(sparse_groups) > 1:
            n_dense = len(dense_groups) if dense_groups else 0
            lines += [
                "#### Sparse 모델 비교 (Dense 평균)",
                "",
                f"> 각 Sparse 모델의 Dense {n_dense}종 결과를 평균하여 순수 Sparse 모델 효과를 비교합니다.",
                "",
                "| Sparse 모델 | 평균 복합 점수 | 평균 Recall | 평균 Precision |",
                "|------------|:------------:|:----------:|:-------------:|",
            ]

            sparse_summary = []
            for name, rows in sparse_groups.items():
                rdf = pd.DataFrame(rows)
                sparse_summary.append(
                    {
                        "name": name,
                        "composite": rdf["composite"].mean(),
                        "recall": rdf["context_recall"].mean(),
                        "precision": rdf["context_precision"].mean(),
                    }
                )

            sparse_summary.sort(key=lambda x: -x["composite"])
            for ss in sparse_summary:
                lines.append(
                    f"| {ss['name']} | {ss['composite']:.4f} | {ss['recall']:.4f} "
                    f"| {ss['precision']:.4f} |"
                )

            lines.append("")

            # Sparse 인사이트
            bm25 = next((s for s in sparse_summary if s["name"] == "BM25"), None)
            splade = next((s for s in sparse_summary if s["name"] == "SPLADE"), None)

            if bm25 and splade:
                diff = bm25["composite"] - splade["composite"]
                recall_diff = bm25["recall"] - splade["recall"]

                if abs(diff) < 0.01:
                    lines.append(
                        f"> BM25와 SPLADE의 평균 복합 점수 차이가 {abs(diff):.4f}로 미미합니다."
                    )
                elif diff > 0:
                    lines.append(
                        f"> **BM25**가 SPLADE 대비 복합 점수 +{diff:.4f} 우세합니다."
                    )
                else:
                    lines.append(
                        f"> **SPLADE**가 BM25 대비 복합 점수 +{abs(diff):.4f} 우세합니다."
                    )

                if recall_diff > 0.01:
                    lines.append(
                        f"> 특히 BM25가 Recall에서 +{recall_diff:.4f} 우세 — "
                        "한국어 형태소 분석(OKt) 기반의 정확한 키워드 매칭이 Recall에 기여합니다."
                    )
                elif recall_diff < -0.01:
                    lines.append(
                        f"> 특히 SPLADE가 Recall에서 +{abs(recall_diff):.4f} 우세 — "
                        "학습된 확장 토큰이 동의어·유사 표현까지 포착합니다."
                    )

                lines.append("")

    lines += ["---", ""]
    return lines


# ---------------------------------------------------------------------------
# Section 7: 최종 추천
# ---------------------------------------------------------------------------


def _section7_recommendation(ranked: Dict[str, pd.DataFrame]) -> List[str]:
    lines = [
        "## 7. 최종 모델 선정 가이드",
        "",
    ]

    all_categories = {"general", "legal", "business", "medical", "technical"}
    tested_categories = {c.lower() for c in ranked.keys()}

    for cat_idx, (category, df) in enumerate(ranked.items(), 1):
        label = category.upper()
        top = df.iloc[0]
        second = df.iloc[1] if len(df) > 1 else None

        top_name = _short_name(top["strategy"])
        second_name = _short_name(second["strategy"]) if second is not None else "—"

        lines += [
            f"### 7-{cat_idx}. {label} 카테고리",
            "",
            "#### 용도별 추천",
            "",
            "| 사용 상황 | 추천 조합 | 이유 |",
            "|----------|---------|------|",
            f"| **품질 최우선** | {top_name} | 종합 점수 1위 ({top['composite']:.4f}) |",
        ]

        # Recall 최우선 추천
        recall_best = df.sort_values("context_recall", ascending=False).iloc[0]
        recall_name = _short_name(recall_best["strategy"])
        if recall_name != top_name:
            lines.append(
                f"| **정보 누락 방지** | {recall_name} | Recall 최고 ({recall_best['context_recall']:.4f}) |"
            )

        # Faithfulness 최우선 추천
        faith_best = df.sort_values("faithfulness", ascending=False).iloc[0]
        faith_name = _short_name(faith_best["strategy"])
        if faith_name != top_name:
            lines.append(
                f"| **할루시네이션 방지** | {faith_name} | Faithfulness 최고 ({faith_best['faithfulness']:.4f}) |"
            )

        # 로컬 모델 최고 추천 (API 모델이 1위인 경우)
        is_top_api = any(
            k.lower() in top["strategy"].lower()
            for k, m in DENSE_DISPLAY.items()
            if m.get("type") == "api"
        )
        if is_top_api:
            local_only = df[
                ~df["strategy"].apply(
                    lambda s: any(
                        k.lower() in s.lower()
                        for k, m in DENSE_DISPLAY.items()
                        if m.get("type") == "api"
                    )
                )
            ]
            if not local_only.empty:
                local_best = local_only.iloc[0]
                local_best_name = _short_name(local_best["strategy"])
                lines.append(
                    f"| **보안·비용 우선 (로컬)** | {local_best_name} | 로컬 모델 최고 ({local_best['composite']:.4f}), 데이터 외부 전송 없음 |"
                )

        lines += [""]

        # 단일 추천
        lines += [
            "#### 한 가지만 선택해야 한다면",
            "",
        ]

        # 1위와 2위 차이 확인
        if second is not None:
            gap_1_2 = abs(float(top["composite"]) - float(second["composite"]))
            if gap_1_2 < 0.02 and is_top_api:
                # 동점 구간이고 1위가 API → 로컬 모델 추천
                local_only = df[
                    ~df["strategy"].apply(
                        lambda s: any(
                            k.lower() in s.lower()
                            for k, m in DENSE_DISPLAY.items()
                            if m.get("type") == "api"
                        )
                    )
                ]
                if not local_only.empty:
                    local_best = local_only.iloc[0]
                    local_best_name = _short_name(local_best["strategy"])
                    lines += [
                        f"> **{local_best_name}**",
                        ">",
                        f"> 종합 점수 {local_best['composite']:.4f}로 1위({top_name}, {top['composite']:.4f})와 "
                        f"{gap_1_2:.4f} ({gap_1_2 / float(top['composite']) * 100:.1f}%) 차이로 동등 수준이며, "
                        "로컬 실행으로 데이터 외부 전송 없이 보안을 확보할 수 있습니다.",
                    ]
                else:
                    lines += [
                        f"> **{top_name}**",
                        ">",
                        f"> 종합 점수 {top['composite']:.4f}로 1위입니다.",
                    ]
            else:
                lines += [
                    f"> **{top_name}**",
                    ">",
                    f"> 종합 점수 {top['composite']:.4f}로 1위이며, "
                    f"2위 대비 {gap_1_2:.4f} ({gap_1_2 / float(top['composite']) * 100:.1f}%) 우세합니다.",
                ]
        else:
            lines += [
                f"> **{top_name}**",
                ">",
                f"> 종합 점수 {top['composite']:.4f}로 1위입니다.",
            ]

        lines += [""]

    # 향후 과제 (루프 밖, 1회만 출력)
    lines += [
        f"### 7-{len(ranked) + 1}. 향후 과제",
        "",
    ]

    untested = all_categories - tested_categories
    if untested:
        untested_list = ", ".join(c.upper() for c in sorted(untested))
        lines.append(
            f"- [ ] **추가 카테고리 벤치마크**: {untested_list} 카테고리에서 동일 조합 검증"
        )

    # 사용된 리랭커 확인
    used_rerankers = set()
    for df in ranked.values():
        for strategy in df["strategy"]:
            for key, rlabel in RERANKER_DISPLAY.items():
                if key in strategy.lower():
                    used_rerankers.add(rlabel)

    if "FlashRank" not in used_rerankers:
        lines.append(
            "- [ ] **FlashRank 리랭커 비교**: ColBERT 대비 경량 리랭커의 품질-속도 트레이드오프 확인"
        )
    if "ColBERT" not in used_rerankers:
        lines.append(
            "- [ ] **ColBERT 리랭커 비교**: FlashRank 대비 토큰 수준 리랭커의 품질 확인"
        )

    lines += [
        "- [ ] **분기별 재평가**: Dense 모델 신규 릴리스에 맞춘 정기 벤치마크 실행",
        "",
    ]

    lines += ["---", ""]
    return lines


# ---------------------------------------------------------------------------
# 부록
# ---------------------------------------------------------------------------


def _appendix(
    ranked: Dict[str, pd.DataFrame],
    raw_results: Dict[str, dict],
) -> List[str]:
    lines = [
        "---",
        "",
        "## 부록: 기술 세부사항",
        "",
        "> 이 섹션은 개발팀·데이터팀을 위한 기술 참고 정보입니다.",
        "",
        "### A. 평가 지표 가중치 산정 근거",
        "",
        "| 지표 | 가중치 | 산정 근거 |",
        "|------|:------:|---------|",
        "| Context Recall | 0.35 | 서비스에서 정보 누락은 오답보다 사용자 이탈로 직결. RAGAS 권고 가중치. |",
        "| Context Precision | 0.30 | 불필요 컨텍스트 포함 시 LLM 답변 품질 저하 및 토큰 비용 증가. |",
        "| Faithfulness | 0.20 | 할루시네이션 방지 — 법률·의료 도메인에서 특히 중요. |",
        "| Answer Relevancy | 0.15 | 위 3개 지표가 높으면 자연스럽게 따라오는 경향. |",
        "",
        "> **복합 점수** = Recall×0.35 + Precision×0.30 + Faithfulness×0.20 + Relevancy×0.15",
        "",
        "### B. 파이프라인 구성 요소",
        "",
        "| 요소 | 설정 | 적용 근거 |",
        "|------|------|---------|",
        "| ColBERT Rerank | jina-colbert-v2 | 토큰 수준 재정렬, 오답률 25% 감소 (IBM 연구) |",
        "| FlashRank Rerank | FlashRank v2 | 경량 교차 인코더 리랭킹, 속도 대비 품질 균형 |",
        "| Contextual Retrieval | LLM 문맥 강화 | 검색 실패율 67% 감소 (Anthropic 보고) |",
        "| Chunking | Parent-Child (512/128 tokens) | 문맥 보존 + 세밀 검색 동시 확보 |",
        "",
        "### C. 실행 환경",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        "| 클러스터 | EKS zcp-ags-cp-eks (ap-northeast-2) |",
        "| 노드 | management (m7i/m8i.2xlarge, 8C/32G) |",
        "| GPU | 없음 (CPU-only) |",
        "| Namespace | rag-bench-test |",
        "| PVC | bench-results (EFS RWX) |",
        "| 실행 시간 | 약 137~168분/Job (ColBERT CPU rerank 병목) |",
        "",
        "### D. 전체 원시 데이터",
        "",
    ]

    for category, df in ranked.items():
        label = category.upper()
        n_qa = raw_results.get(category, {}).get("n_qa", "?")

        lines += [
            f"**{label}** (질의 {n_qa}개, {len(df)}개 조합)",
            "",
            "| Rank | Strategy | Recall | Precision | Faithfulness | Relevancy | Composite | Avg Lat (s) | Med Lat (s) |",
            "|:----:|:---------|:------:|:---------:|:------------:|:---------:|:---------:|:-----------:|:-----------:|",
        ]

        for _, row in df.iterrows():
            rank = int(row["rank"])
            lat_avg = (
                row["avg_latency_ms"] / 1000 if pd.notna(row["avg_latency_ms"]) else "—"
            )
            lat_med = (
                row["median_latency_ms"] / 1000
                if pd.notna(row.get("median_latency_ms"))
                else "—"
            )
            lat_avg_str = f"{lat_avg:.1f}" if isinstance(lat_avg, float) else lat_avg
            lat_med_str = f"{lat_med:.1f}" if isinstance(lat_med, float) else lat_med

            score_str = (
                f"**{row['composite']:.4f}**"
                if rank == 1
                else f"{row['composite']:.4f}"
            )
            lines.append(
                f"| {rank} | `{row['strategy']}` "
                f"| {row['context_recall']:.4f} | {row['context_precision']:.4f} "
                f"| {row['faithfulness']:.4f} | {row['answer_relevancy']:.4f} "
                f"| {score_str} | {lat_avg_str} | {lat_med_str} |"
            )

        lines += [""]

    lines += [
        "### E. 재현 방법",
        "",
        "```bash",
        "# K8s 벤치마크 실행",
        "uv run python -m k8s.orchestrator \\",
        "    --categories general \\",
        "    --rerankers colbert \\",
        "    --preset service",
        "",
        "# 결과 보고서 생성",
        "uv run python k8s_results/generate_k8s_report.py \\",
        "    --run_dir k8s_results/20260226-0948",
        "```",
        "",
    ]

    return lines


# ---------------------------------------------------------------------------
# 콘솔 요약 출력
# ---------------------------------------------------------------------------


def _print_summary(ranked: Dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 60)
    print(" K8s 벤치마크 — 핵심 결론")
    print("=" * 60)

    for category, df in ranked.items():
        print(f"\n  [{category.upper()}]")
        for _, row in df.head(3).iterrows():
            rank = int(row["rank"])
            name = _short_name(row["strategy"])
            composite = row["composite"]
            marker = " ★" if rank == 1 else ""
            print(f"    {rank}위  {name:25s}  점수={composite:.4f}{marker}")


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="K8s 벤치마크 결과 — 보고서 생성")
    parser.add_argument("--run_dir", required=True, help="K8s 결과 디렉토리")
    parser.add_argument(
        "--output_dir", default=None, help="보고서 저장 위치 (기본: run_dir)"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else None

    generate_report(run_dir, output_dir)


if __name__ == "__main__":
    main()
