"""
PDF 파서 벤치마크 보고서 자동 생성 모듈

rag_parser_full_report_v2.md 구조를 준수하여 Markdown 보고서를 생성합니다.

사용법:
    from autorag_pdf_eval.report import generate_report
    generate_report([Path("bench_results/run1"), Path("bench_results/run2")])
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 상수 ──────────────────────────────────────────────────────────────────────

WEIGHTS: dict[str, float] = {
    "parsing_accuracy": 0.35,
    "data_security": 0.20,
    "cost_efficiency": 0.15,
    "table_accuracy": 0.15,
    "dpi_stability": 0.10,
    "ease_of_adoption": 0.05,
}

BACKEND_ATTRS: dict[str, dict[str, int]] = {
    "pymupdf": {"security": 5, "cost": 5, "ease": 5},
    "docling": {"security": 5, "cost": 5, "ease": 2},
    "openai": {"security": 2, "cost": 1, "ease": 5},
    "openai-4.1": {"security": 2, "cost": 1, "ease": 5},
    "upstage": {"security": 2, "cost": 2, "ease": 5},
    "upstage-enhanced": {"security": 2, "cost": 2, "ease": 5},
    "paddleocr-vl": {"security": 5, "cost": 5, "ease": 3},
    "deepseek-ocr2": {"security": 5, "cost": 5, "ease": 3},
}

BACKEND_PROFILES: dict[str, dict[str, str]] = {
    "pymupdf": {
        "developer": "Artifex Software",
        "type": "로컬 (규칙 기반)",
        "model_size": "N/A (AI 없음)",
        "license": "AGPL-3.0",
        "description": "PDF 텍스트 스트림 직접 추출. 극도로 빠름. 이미지 PDF 처리 불가",
    },
    "docling": {
        "developer": "IBM Research",
        "type": "로컬 (OCR 파이프라인)",
        "model_size": "N/A (복합)",
        "license": "MIT",
        "description": "레이아웃+OCR+표 인식 통합. transformers 호환 문제 존재",
    },
    "openai": {
        "developer": "OpenAI",
        "type": "API (VLM)",
        "model_size": "~200B (추정)",
        "license": "상용 종량제",
        "description": "페이지별 이미지를 GPT-4o에 전송하여 변환",
    },
    "openai-4.1": {
        "developer": "OpenAI",
        "type": "API (VLM)",
        "model_size": "~200B (추정)",
        "license": "상용 종량제",
        "description": "GPT-4.1 최신 모델 기반 문서 변환",
    },
    "upstage": {
        "developer": "Upstage AI",
        "type": "API (전용 Document Parse)",
        "model_size": "비공개",
        "license": "상용 종량제",
        "description": "문서 파싱 전용 API. 텍스트·표에서 높은 정확도",
    },
    "upstage-enhanced": {
        "developer": "Upstage AI",
        "type": "API (VLM 정밀 모드)",
        "model_size": "비공개",
        "license": "상용 종량제",
        "description": "Upstage VLM 정밀 모드",
    },
    "paddleocr-vl": {
        "developer": "Baidu (PaddlePaddle)",
        "type": "로컬 (VLM)",
        "model_size": "0.9B",
        "license": "Apache-2.0",
        "description": "OmniDocBench SOTA. 초경량 VLM. K8s CPU/GPU 서비스",
    },
    "deepseek-ocr2": {
        "developer": "DeepSeek",
        "type": "로컬 (VLM)",
        "model_size": "비공개",
        "license": "MIT",
        "description": "DeepSeek-OCR-2 모델 기반 문서 변환",
    },
}

# ── 문서 분류 ─────────────────────────────────────────────────────────────────

_TEXT_PDFS = {"text_only.pdf"}
_TABLE_PDFS = {
    "table_native.pdf",
    "table_image.pdf",
    "table_image_200dpi.pdf",
    "table_image_150dpi.pdf",
    "table_image_72dpi.pdf",
}
_GRAPH_PDFS = {
    "graph_rich.pdf",
    "graph_rich_image.pdf",
    "graph_rich_image_200dpi.pdf",
    "graph_rich_image_150dpi.pdf",
    "graph_rich_image_72dpi.pdf",
}


def classify_doc_type(pdf_name: str) -> str:
    """PDF 파일명으로 문서 유형 분류."""
    if pdf_name in _TEXT_PDFS:
        return "text"
    if pdf_name in _TABLE_PDFS:
        return "table"
    if pdf_name in _GRAPH_PDFS:
        return "graph"
    return "unknown"


def classify_dpi(pdf_name: str) -> str:
    """PDF 파일명으로 DPI 분류."""
    if "72dpi" in pdf_name:
        return "72dpi"
    if "150dpi" in pdf_name:
        return "150dpi"
    if "200dpi" in pdf_name:
        return "200dpi"
    if "_image" in pdf_name and "dpi" not in pdf_name:
        return "image"
    return "native"


# ── 데이터 로드 ──────────────────────────────────────────────────────────────


def load_results(results_dirs: list[Path]) -> list[dict]:
    """
    metrics.json 재귀 수집, 중복(같은 backend+pdf_name) 시 최신 우선.

    Returns: list of metrics dict
    """
    all_metrics: dict[str, dict] = {}  # key: "backend|pdf_name"

    for d in results_dirs:
        for mf in sorted(d.rglob("metrics.json")):
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            if data.get("error"):
                continue

            key = f"{data.get('backend', '')}|{data.get('pdf_name', '')}"
            existing = all_metrics.get(key)
            if existing is None:
                all_metrics[key] = data
            else:
                # 최신 타임스탬프 우선
                new_ts = data.get("timestamp", "")
                old_ts = existing.get("timestamp", "")
                if new_ts > old_ts:
                    all_metrics[key] = data

    return list(all_metrics.values())


# ── 순위 계산 ────────────────────────────────────────────────────────────────


def _get_ned(m: dict) -> float | None:
    """metrics dict에서 NED(edit_dist) 추출."""
    summary = m.get("summary", {})
    # 신형 포맷
    val = summary.get("avg_edit_dist")
    if val is not None:
        return val
    # 구형 포맷 호환
    val = summary.get("avg_text_ned")
    if val is not None:
        return val
    return None


def _get_teds_html(m: dict) -> float | None:
    """metrics dict에서 TEDS-HTML 추출."""
    summary = m.get("summary", {})
    val = summary.get("avg_teds_html")
    if val is not None:
        return val
    # 구형 포맷
    omnidoc = m.get("omnidoc_summary", {})
    if omnidoc:
        return omnidoc.get("avg_teds_html")
    return None


def _get_speed(m: dict) -> float | None:
    """metrics dict에서 속도 추출."""
    summary = m.get("summary", {})
    return summary.get("avg_speed_s")


def compute_backend_averages(
    metrics: list[dict],
) -> dict[str, dict[str, Any]]:
    """
    백엔드별 종합/유형별 평균 계산.

    Returns: {backend: {
        "overall_ned": float, "text_ned": float, "table_ned": float, "graph_ned": float,
        "table_teds": float, "overall_speed": float, ...
    }}
    """
    # backend -> doc_type -> list of values
    by_backend: dict[str, dict[str, list[float]]] = {}
    teds_by_backend: dict[str, dict[str, list[float]]] = {}
    speed_by_backend: dict[str, list[float]] = {}
    # DPI 분석: backend -> doc_type -> dpi -> list of NED
    dpi_by_backend: dict[str, dict[str, dict[str, list[float]]]] = {}

    for m in metrics:
        backend = m.get("backend", "")
        pdf_name = m.get("pdf_name", "")
        doc_type = classify_doc_type(pdf_name)
        dpi = classify_dpi(pdf_name)

        ned = _get_ned(m)
        teds = _get_teds_html(m)
        speed = _get_speed(m)

        if ned is not None:
            by_backend.setdefault(backend, {}).setdefault(doc_type, []).append(ned)
            by_backend[backend].setdefault("all", []).append(ned)
            dpi_by_backend.setdefault(backend, {}).setdefault(doc_type, {}).setdefault(
                dpi, []
            ).append(ned)

        if teds is not None and teds >= 0:
            teds_by_backend.setdefault(backend, {}).setdefault(doc_type, []).append(
                teds
            )
            teds_by_backend[backend].setdefault("all", []).append(teds)

        if speed is not None:
            speed_by_backend.setdefault(backend, []).append(speed)

    result: dict[str, dict[str, Any]] = {}
    for backend in by_backend:
        neds = by_backend[backend]
        teds = teds_by_backend.get(backend, {})
        speeds = speed_by_backend.get(backend, [])

        def _avg(lst: list[float]) -> float | None:
            return round(sum(lst) / len(lst), 4) if lst else None

        result[backend] = {
            "overall_ned": _avg(neds.get("all", [])),
            "text_ned": _avg(neds.get("text", [])),
            "table_ned": _avg(neds.get("table", [])),
            "graph_ned": _avg(neds.get("graph", [])),
            "table_teds": _avg(teds.get("table", [])),
            "overall_teds": _avg(teds.get("all", [])),
            "overall_speed": _avg(speeds),
            "dpi": dpi_by_backend.get(backend, {}),
            "pdf_count": len(neds.get("all", [])),
        }

    return result


def _ned_to_score(ned: float | None) -> int:
    """NED를 1~5 점수로 변환."""
    if ned is None:
        return 1
    if ned >= 0.70:
        return 5
    if ned >= 0.65:
        return 4
    if ned >= 0.55:
        return 3
    if ned >= 0.45:
        return 2
    return 1


def _teds_to_score(teds: float | None) -> int:
    """TEDS를 1~5 점수로 변환."""
    if teds is None:
        return 1
    if teds >= 0.60:
        return 5
    if teds >= 0.50:
        return 4
    if teds >= 0.35:
        return 3
    if teds >= 0.20:
        return 2
    return 1


def _dpi_stability_score(dpi_data: dict[str, dict[str, list[float]]]) -> int:
    """DPI 안정성 점수 (1~5)."""
    max_spread = 0.0
    for doc_type in ("table", "graph"):
        dpi_vals = dpi_data.get(doc_type, {})
        if len(dpi_vals) < 2:
            continue
        avgs = [sum(v) / len(v) for v in dpi_vals.values() if v]
        if len(avgs) >= 2:
            spread = max(avgs) - min(avgs)
            max_spread = max(max_spread, spread)

    if max_spread <= 0.03:
        return 5
    if max_spread <= 0.05:
        return 4
    if max_spread <= 0.10:
        return 3
    if max_spread <= 0.20:
        return 2
    return 1


def compute_weighted_scores(
    backend_avgs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    가중 평점 매트릭스 계산.

    Returns: {backend: {"scores": {criterion: raw_score}, "weighted": {criterion: weighted},
              "total": total_weighted}}
    """
    result: dict[str, dict[str, Any]] = {}

    for backend, avgs in backend_avgs.items():
        attrs = BACKEND_ATTRS.get(backend, {"security": 3, "cost": 3, "ease": 3})

        scores = {
            "parsing_accuracy": _ned_to_score(avgs.get("overall_ned")),
            "data_security": attrs["security"],
            "cost_efficiency": attrs["cost"],
            "table_accuracy": _teds_to_score(avgs.get("table_teds")),
            "dpi_stability": _dpi_stability_score(avgs.get("dpi", {})),
            "ease_of_adoption": attrs["ease"],
        }

        weighted = {k: round(v * WEIGHTS[k], 2) for k, v in scores.items()}
        total = round(sum(weighted.values()), 2)

        result[backend] = {
            "scores": scores,
            "weighted": weighted,
            "total": total,
        }

    return result


# ── 보고서 렌더링 ────────────────────────────────────────────────────────────

_CRITERION_KR = {
    "parsing_accuracy": "파싱 정확도",
    "data_security": "데이터 보안",
    "cost_efficiency": "비용 효율성",
    "table_accuracy": "표 구조 정확도",
    "dpi_stability": "DPI 안정성",
    "ease_of_adoption": "도입 용이성",
}


def _fmt(val: float | None, fmt: str = ".4f") -> str:
    if val is None:
        return "—"
    return f"{val:{fmt}}"


def _rank_backends(
    backend_avgs: dict[str, dict[str, Any]], key: str, reverse: bool = True
) -> list[tuple[str, float | None]]:
    """key 기준으로 백엔드 정렬."""
    items = [(b, avgs.get(key)) for b, avgs in backend_avgs.items()]
    # None 값은 뒤로
    with_val = [(b, v) for b, v in items if v is not None]
    without_val = [(b, v) for b, v in items if v is None]
    with_val.sort(key=lambda x: x[1], reverse=reverse)
    return with_val + without_val


def render_report(
    metrics: list[dict],
    backend_avgs: dict[str, dict[str, Any]],
    weighted_scores: dict[str, dict[str, Any]],
) -> str:
    """v2 보고서 구조에 따라 전체 Markdown 보고서 렌더링."""
    lines: list[str] = []

    def w(text: str = "") -> None:
        lines.append(text)

    # 정렬: 총점 내림차순
    ranked = sorted(weighted_scores.items(), key=lambda x: x[1]["total"], reverse=True)
    backends_sorted = [b for b, _ in ranked]
    top_backend = backends_sorted[0] if backends_sorted else "N/A"
    top_score = ranked[0][1]["total"] if ranked else 0
    top_ned = backend_avgs.get(top_backend, {}).get("overall_ned")

    now = datetime.now().strftime("%Y-%m-%d")

    # ── 1. 헤더 + Executive Summary ─────────────────────────────────────────
    w("# PDF 파서 솔루션 비교·선정 보고서")
    w()
    w(
        f"> **문서 버전**: auto-generated | **작성일**: {now} | "
        f"**평가 백엔드**: {len(backends_sorted)}종"
    )
    w()
    w("---")
    w()
    w("## 1. Executive Summary")
    w()
    w(
        f"**{top_backend}**이 종합 가중 점수 **{top_score}/5.00**으로 1위를 차지했습니다."
    )
    if top_ned is not None:
        w(f"텍스트 정확도(NED) 종합 {top_ned:.4f}.")
    w()
    w("---")
    w()

    # ── 2. 배경 및 범위 ─────────────────────────────────────────────────────
    w("## 2. 배경 및 범위")
    w()
    total_pdfs = len({m.get("pdf_name") for m in metrics})
    w("| 항목 | 내용 |")
    w("|---|---|")
    w(f"| **평가 대상** | {len(backends_sorted)}종 파싱 솔루션 |")
    w(f"| **테스트 문서** | PDF {total_pdfs}개 |")
    w("| **평가 지표** | NED, BLEU, METEOR, TEDS-HTML (OmniDoc 5종) |")
    w()
    w("---")
    w()

    # ── 3. 평가 방법론 ──────────────────────────────────────────────────────
    w("## 3. 평가 방법론")
    w()
    w("| 지표 | 정식 명칭 | 의미 | 범위 |")
    w("|---|---|---|---|")
    w("| **NED** | Normalized Edit Distance | 텍스트 정확도 | 0~1 |")
    w("| **BLEU** | Bilingual Evaluation Understudy | n-gram 정밀도 | 0~100 |")
    w(
        "| **METEOR** | Metric for Evaluation of Translation | 정밀도+재현율+어순 | 0~100 |"
    )
    w(
        "| **TEDS-HTML** | Tree Edit Distance Similarity | HTML 기반 표 구조 정확도 | 0~1 |"
    )
    w()
    w("---")
    w()

    # ── 4. 평가 기준 및 가중치 ──────────────────────────────────────────────
    w("## 4. 평가 기준 및 가중치")
    w()
    w("| 기준 | 가중치 |")
    w("|---|:---:|")
    for k, v in WEIGHTS.items():
        w(f"| **{_CRITERION_KR[k]}** | {int(v * 100)}% |")
    w()
    w("---")
    w()

    # ── 5. 후보 솔루션 개요 ─────────────────────────────────────────────────
    w("## 5. 후보 솔루션 개요")
    w()
    w("| 솔루션 | 개발사 | 유형 | 모델 규모 | 라이선스 |")
    w("|---|---|---|---|---|")
    for b in backends_sorted:
        p = BACKEND_PROFILES.get(b, {})
        w(
            f"| **{b}** | {p.get('developer', '—')} | {p.get('type', '—')} | "
            f"{p.get('model_size', '—')} | {p.get('license', '—')} |"
        )
    w()
    w("---")
    w()

    # ── 6. 평가 결과 ────────────────────────────────────────────────────────
    w("## 6. 평가 결과")
    w()

    # 6-1. 가중 평점 매트릭스
    w("### 6-1. 가중 평점 매트릭스 (종합 결과)")
    w()
    w("> 점수 척도: 1(최하) ~ 5(최상) | 가중 점수 = 원점수 x 가중치")
    w()

    # 헤더
    header = "| 기준 (가중치) |"
    sep = "|---|"
    for b in backends_sorted:
        header += f" {b} |"
        sep += ":---:|"
    w(header)
    w(sep)

    for criterion, weight in WEIGHTS.items():
        kr_name = _CRITERION_KR[criterion]
        row = f"| {kr_name} ({int(weight * 100)}%) |"
        max_weighted = max(
            (weighted_scores[b]["weighted"][criterion] for b in backends_sorted),
            default=0,
        )
        for b in backends_sorted:
            raw = weighted_scores[b]["scores"][criterion]
            wt = weighted_scores[b]["weighted"][criterion]
            bold = "**" if wt == max_weighted else ""
            row += f" {raw} / {bold}{wt}{bold} |"
        w(row)

    # 종합
    row_total = "| **종합 가중 점수** |"
    for b in backends_sorted:
        t = weighted_scores[b]["total"]
        bold = "**" if b == top_backend else ""
        row_total += f" {bold}{t}{bold} |"
    w(row_total)

    row_rank = "| **종합 순위** |"
    for i, b in enumerate(backends_sorted, 1):
        bold = "**" if i == 1 else ""
        row_rank += f" {bold}{i}위{bold} |"
    w(row_rank)
    w()

    # 6-2. 문서 유형별 NED 순위
    w("### 6-2. 문서 유형별 NED 순위")
    w()

    for doc_type, doc_label in [
        ("text", "텍스트형 문서"),
        ("table", "표형 문서"),
        ("graph", "그래프형 문서"),
    ]:
        w(f"#### {doc_label}")
        w()

        ranking = _rank_backends(backend_avgs, f"{doc_type}_ned")
        if doc_type == "table":
            w("| 순위 | 솔루션 | NED | TEDS-HTML |")
            w("|:---:|---|:---:|:---:|")
            for i, (b, ned) in enumerate(ranking, 1):
                teds = backend_avgs[b].get("table_teds")
                bold = "**" if i == 1 else ""
                w(f"| {i} | {bold}{b}{bold} | {bold}{_fmt(ned)}{bold} | {_fmt(teds)} |")
        else:
            w("| 순위 | 솔루션 | NED |")
            w("|:---:|---|:---:|")
            for i, (b, ned) in enumerate(ranking, 1):
                bold = "**" if i == 1 else ""
                w(f"| {i} | {bold}{b}{bold} | {bold}{_fmt(ned)}{bold} |")
        w()

    # 6-3. DPI 안정성
    w("### 6-3. DPI 안정성 분석")
    w()

    for doc_type, doc_label in [("table", "표형"), ("graph", "그래프형")]:
        w(f"**{doc_label} — DPI별 NED 변화**")
        w()

        # 이 문서유형의 DPI 데이터가 있는 백엔드만
        dpi_backends = []
        for b in backends_sorted:
            dpi_data = backend_avgs.get(b, {}).get("dpi", {}).get(doc_type, {})
            if len(dpi_data) >= 2:
                dpi_backends.append(b)

        if not dpi_backends:
            w("DPI 변형 데이터 없음.")
            w()
            continue

        dpi_labels = ["native", "image", "72dpi", "150dpi", "200dpi"]

        header = "| DPI |"
        sep = "|:---:|"
        for b in dpi_backends:
            header += f" {b} |"
            sep += ":---:|"
        w(header)
        w(sep)

        for dpi in dpi_labels:
            row = f"| {dpi} |"
            for b in dpi_backends:
                vals = backend_avgs[b].get("dpi", {}).get(doc_type, {}).get(dpi, [])
                if vals:
                    avg = sum(vals) / len(vals)
                    row += f" {avg:.4f} |"
                else:
                    row += " — |"
            w(row)

        # 변동폭
        row_spread = "| **변동폭** |"
        for b in dpi_backends:
            dpi_data = backend_avgs[b].get("dpi", {}).get(doc_type, {})
            avgs = [sum(v) / len(v) for v in dpi_data.values() if v]
            if len(avgs) >= 2:
                spread = max(avgs) - min(avgs)
                row_spread += f" **±{spread * 100:.1f}%p** |"
            else:
                row_spread += " — |"
        w(row_spread)
        w()

    w("---")
    w()

    # ── 7. 비용 분석 ────────────────────────────────────────────────────────
    w("## 7. 비용 분석")
    w()
    w("| 비용 항목 | 로컬 솔루션 | API 솔루션 |")
    w("|---|---|---|")
    w("| **라이선스/API 비용** | 0원 (오픈소스) | 페이지당 종량제 과금 |")
    w("| **인프라** | K8s 노드 비용 | 해당 없음 |")
    w("| **확장 시 비용 증가** | 선형 미만 | **선형 (문서 수 비례)** |")
    w()
    w("---")
    w()

    # ── 8. 리스크 평가 ──────────────────────────────────────────────────────
    w("## 8. 리스크 평가")
    w()
    w("#### 현상 유지 (솔루션 미선정)")
    w()
    w("| 리스크 | 발생 가능성 | 영향도 | 판정 |")
    w("|---|:---:|:---:|:---:|")
    w("| RAG 시스템 구축 일정 지연 | H | H | [!!] |")
    w("| 임의 선정 시 정확도 격차 | H | H | [!!] |")
    w()
    w("---")
    w()

    # ── 9. 추천 ─────────────────────────────────────────────────────────────
    w("## 9. 추천")
    w()
    w(f"### 기본 추천: {top_backend}")
    w()
    top_avgs = backend_avgs.get(top_backend, {})
    w(f"**{top_backend}**을 RAG 시스템의 기본 PDF 파싱 백엔드로 추천합니다.")
    w()
    w(f"- **종합 NED**: {_fmt(top_avgs.get('overall_ned'))}")
    w(f"- **종합 가중 점수**: {top_score}/5.00")
    w()

    # 미선정 사유
    if len(ranked) > 1:
        w("### 미선정 사유")
        w()
        w("| 솔루션 | 종합 점수 | 종합 NED |")
        w("|---|:---:|:---:|")
        for b, ws in ranked[1:]:
            ned = backend_avgs.get(b, {}).get("overall_ned")
            w(f"| {b} | {ws['total']} | {_fmt(ned)} |")
        w()

    w("---")
    w()

    # ── 10. 다음 단계 ───────────────────────────────────────────────────────
    w("## 10. 다음 단계")
    w()
    w("| 순서 | 액션 |")
    w("|:---:|---|")
    w(f"| 1 | {top_backend} 프로덕션 배포 |")
    w("| 2 | Hybrid 라우팅 구현 |")
    w("| 3 | 추가 백엔드 벤치마크 |")
    w()
    w("---")
    w()

    # ── 부록 A. 용어 사전 ───────────────────────────────────────────────────
    w("## 부록 A. 용어 사전")
    w()
    w("| 약어 | 정식 명칭 | 의미 |")
    w("|---|---|---|")
    w("| **NED** | Normalized Edit Distance | 텍스트 정확도 (0~1) |")
    w("| **BLEU** | Bilingual Evaluation Understudy | n-gram 정밀도 (0~100) |")
    w(
        "| **METEOR** | Metric for Evaluation of Translation | 정밀도+재현율+어순 (0~100) |"
    )
    w("| **TEDS** | Tree Edit Distance Similarity | 표 구조 정확도 (0~1) |")
    w("| **VLM** | Vision-Language Model | 이미지+텍스트 AI 모델 |")
    w("| **DPI** | Dots Per Inch | 이미지 해상도 단위 |")
    w("| **RAG** | Retrieval-Augmented Generation | 검색 증강 생성 |")
    w()
    w("---")
    w()

    # ── 부록 B. 세부 결과표 ─────────────────────────────────────────────────
    w("## 부록 B. 세부 결과표")
    w()

    # 백엔드별 PDF별 결과
    # metrics를 backend->pdf_name으로 인덱싱
    by_backend_pdf: dict[str, dict[str, dict]] = {}
    for m in metrics:
        b = m.get("backend", "")
        p = m.get("pdf_name", "")
        by_backend_pdf.setdefault(b, {})[p] = m

    all_pdfs = sorted({m.get("pdf_name", "") for m in metrics})

    for doc_type, doc_label, pdfs in [
        ("text", "텍스트형", [p for p in all_pdfs if classify_doc_type(p) == "text"]),
        ("table", "표형", [p for p in all_pdfs if classify_doc_type(p) == "table"]),
        ("graph", "그래프형", [p for p in all_pdfs if classify_doc_type(p) == "graph"]),
    ]:
        if not pdfs:
            continue

        w(f"### B-{doc_type[0].upper()}. {doc_label} (NED)")
        w()

        header = "| PDF |"
        sep = "|---|"
        for b in backends_sorted:
            header += f" {b} |"
            sep += ":---:|"
        w(header)
        w(sep)

        for pdf in pdfs:
            row = f"| {pdf.replace('.pdf', '')} |"
            for b in backends_sorted:
                m = by_backend_pdf.get(b, {}).get(pdf)
                if m:
                    ned = _get_ned(m)
                    row += f" {_fmt(ned)} |"
                else:
                    row += " — |"
            w(row)
        w()

    w("---")
    w()

    # ── 부록 C. 솔루션 상세 프로필 ──────────────────────────────────────────
    w("## 부록 C. 솔루션 상세 프로필")
    w()

    for i, b in enumerate(backends_sorted, 1):
        p = BACKEND_PROFILES.get(b, {})
        w(f"### C-{i}. {b}")
        w()
        w("| 항목 | 내용 |")
        w("|---|---|")
        w(f"| 개발사 | {p.get('developer', '—')} |")
        w(f"| 유형 | {p.get('type', '—')} |")
        w(f"| 모델 규모 | {p.get('model_size', '—')} |")
        w(f"| 라이선스 | {p.get('license', '—')} |")
        w(f"| 핵심 특성 | {p.get('description', '—')} |")
        w()

    w("---")
    w()

    # ── 부록 D. 속도 참고 데이터 ────────────────────────────────────────────
    w("## 부록 D. 속도 참고 데이터")
    w()

    header = "| 솔루션 |"
    sep = "|---|"
    for doc_type_label in ["텍스트형", "표형 (평균)", "그래프형 (평균)"]:
        header += f" {doc_type_label} |"
        sep += ":---:|"
    w(header)
    w(sep)

    for b in backends_sorted:
        row = f"| {b} |"
        for doc_type in ("text", "table", "graph"):
            # 해당 doc_type의 속도 평균
            speeds = []
            for m in metrics:
                if (
                    m.get("backend") == b
                    and classify_doc_type(m.get("pdf_name", "")) == doc_type
                ):
                    s = _get_speed(m)
                    if s is not None:
                        speeds.append(s)
            if speeds:
                avg = sum(speeds) / len(speeds)
                if avg >= 60:
                    row += f" {avg / 60:.1f}분 |"
                else:
                    row += f" {avg:.1f}초 |"
            else:
                row += " — |"
        w(row)
    w()
    w("---")
    w()

    # ── 부록 E. 제약사항 ────────────────────────────────────────────────────
    w("## 부록 E. 제약사항")
    w()
    w("1. 속도 데이터는 환경에 따라 크게 달라질 수 있으며, 참고용으로만 제공")
    w("2. 일부 백엔드는 특정 PDF를 처리하지 못할 수 있음 (결과표에서 `—` 표시)")
    w("3. 보고서는 수집된 metrics.json 데이터를 기반으로 자동 생성됨")
    w()

    return "\n".join(lines)


# ── 공개 API ─────────────────────────────────────────────────────────────────


def generate_report(
    results_dirs: list[Path],
    output_path: Path | None = None,
) -> Path:
    """
    벤치마크 결과에서 Markdown 보고서 생성.

    Args:
        results_dirs: metrics.json이 포함된 결과 디렉토리 목록
        output_path: 보고서 저장 경로 (None이면 첫 디렉토리의 부모에 저장)

    Returns:
        생성된 보고서 파일 경로
    """
    metrics = load_results(results_dirs)
    if not metrics:
        raise ValueError(
            f"metrics.json을 찾을 수 없습니다: {[str(d) for d in results_dirs]}"
        )

    backend_avgs = compute_backend_averages(metrics)
    weighted_scores = compute_weighted_scores(backend_avgs)
    report_md = render_report(metrics, backend_avgs, weighted_scores)

    if output_path is None:
        output_path = results_dirs[0].parent / "report.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".md.tmp")
    tmp.write_text(report_md, encoding="utf-8")
    tmp.rename(output_path)

    print(f"\n보고서 생성 완료: {output_path}")
    return output_path
