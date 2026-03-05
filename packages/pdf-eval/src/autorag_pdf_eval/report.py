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

BACKEND_ATTRS: dict[str, dict[str, int | float | None]] = {
    "pymupdf": {"security": 5, "cost": 5, "ease": 5, "params_b": None},
    "docling": {"security": 5, "cost": 5, "ease": 2, "params_b": None},
    "openai": {"security": 2, "cost": 1, "ease": 5, "params_b": 200.0},
    "openai-4.1": {"security": 2, "cost": 1, "ease": 5, "params_b": 200.0},
    "upstage": {"security": 2, "cost": 2, "ease": 5, "params_b": None},
    "upstage-enhanced": {"security": 2, "cost": 2, "ease": 5, "params_b": None},
    "paddleocr-vl": {"security": 5, "cost": 5, "ease": 3, "params_b": 0.9},
    "deepseek-ocr2": {"security": 5, "cost": 5, "ease": 3, "params_b": None},
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

_CRITERION_DESC = {
    "parsing_accuracy": "RAG 검색 품질에 직결. 종합 NED 기반 (OmniDocBench)",
    "data_security": "기업 문서 유출 방지. 로컬 처리 가능 여부",
    "cost_efficiency": "대규모 처리 시 비용 예측 가능성. 종량제 vs 인프라 비용",
    "table_accuracy": "재무제표·사양서 등 표 구조 재현 능력. IBM TEDS 기반",
    "dpi_stability": "다양한 스캔 해상도에서 파싱 품질 일관성",
    "ease_of_adoption": "초기 셋업 복잡도와 운영 부담",
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
    w("### 3-1. OmniDocBench 프레임워크")
    w()
    w("본 벤치마크의 평가 메트릭은 **OmniDocBench** (CVPR 2025)의 방법론을 따릅니다.")
    w()
    w(
        "> **OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations**"
    )
    w("> — OpenDataLab, Shanghai AI Laboratory (CVPR 2025)")
    w("> — GitHub: opendatalab/OmniDocBench (Apache-2.0)")
    w()
    w(
        "OmniDocBench는 다양한 문서 유형(텍스트, 표, 수식, 그래프 등)에 대해 "
        "체계적인 평가를 제공하는 종합 벤치마크입니다. "
        "PDF 파싱 솔루션의 텍스트 추출 정확도와 표 구조 재현 능력을 "
        "공정하고 재현 가능한 방식으로 측정합니다."
    )
    w()
    w("**채택 이유:**")
    w()
    w("- CVPR 2025 정식 발표 논문으로 학술적 검증 완료")
    w("- 텍스트(NED, BLEU, METEOR) + 표 구조(TEDS) 메트릭을 통합 제공")
    w("- PaddleOCR, MinerU, Docling 등 주요 파싱 솔루션이 OmniDocBench 리더보드에 참여")
    w("- Apache-2.0 라이선스로 상업적 활용 가능")
    w()
    w("### 3-2. 평가 지표 요약")
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
    w("### 3-3. 지표별 상세 해설")
    w()
    w("#### NED (Normalized Edit Distance)")
    w()
    w("두 텍스트 간 **편집 거리(Levenshtein Distance)**를 정규화한 값입니다.")
    w()
    w(
        "- **편집 거리(Edit Distance)**: 한 문자열을 다른 문자열로 바꾸기 위해 "
        "필요한 최소 편집 횟수 (삽입·삭제·치환)"
    )
    w(
        "- **정규화(Normalization)**: 편집 거리를 두 문자열의 최대 길이로 나누어 "
        "0~1 범위로 변환"
    )
    w("- **산출 공식**: `NED = 1 - (편집 거리 / max(예측 길이, 정답 길이))`")
    w("- **해석**: 1.0 = 완벽 일치, 0.0 = 완전히 다름")
    w("- **전처리**: 연속 공백을 단일 공백으로 정규화 후 비교")
    w()
    w("#### BLEU (Bilingual Evaluation Understudy)")
    w()
    w("기계 번역 평가에서 유래한 **n-gram 정밀도** 기반 지표입니다.")
    w()
    w("- **n-gram**: 연속된 n개 단어의 조합 (unigram=1개, bigram=2개, ...)")
    w("- **정밀도(Precision)**: 예측 텍스트의 n-gram 중 정답에도 등장하는 비율")
    w(
        "- **BLEU-4**: 1-gram~4-gram 정밀도를 균등 가중 기하평균 "
        "(weights = 0.25, 0.25, 0.25, 0.25)"
    )
    w(
        "- **Smoothing (Method 1)**: 고차 n-gram 매칭이 0일 때 점수가 0이 되지 않도록 "
        "보정하는 기법"
    )
    w("- **해석**: 100 = 완벽 일치, 0 = 겹치는 n-gram 없음")
    w()
    w("#### METEOR (Metric for Evaluation of Translation)")
    w()
    w("BLEU의 한계를 보완한 **정밀도+재현율+어순** 통합 지표입니다.")
    w()
    w("- **정밀도(Precision)**: 예측 단어 중 정답에 포함된 비율")
    w("- **재현율(Recall)**: 정답 단어 중 예측에 포함된 비율")
    w("- **F-mean**: 정밀도와 재현율의 조화평균 (재현율에 높은 가중치)")
    w(
        "- **어순 패널티(Fragmentation Penalty)**: 일치하는 단어들이 연속되지 않고 "
        "흩어져 있으면 감점"
    )
    w("- **WordNet 동의어 매칭**: 동의어도 일치로 인정 (예: 'big'↔'large')")
    w("- **해석**: 100 = 완벽 일치, 0 = 겹치는 단어 없음")
    w()
    w("#### TEDS-HTML (Tree Edit Distance based Similarity)")
    w()
    w("IBM Research가 제안한 **HTML 트리 기반 표 구조 평가** 지표입니다.")
    w()
    w(
        "- **트리 편집 거리(Tree Edit Distance)**: 한 트리를 다른 트리로 변환하기 위한 "
        "최소 편집 비용 (노드 삽입·삭제·이름변경)"
    )
    w("- **HTML 트리 파싱**: 표의 HTML을 `<table>→<tr>→<td>` 계층 트리로 변환")
    w(
        "- **셀 토큰화(Cell Tokenization)**: `<td>` 내부의 텍스트와 태그를 "
        "토큰 시퀀스로 분해하여 비교"
    )
    w(
        "- **셀 내용 비교**: Levenshtein 거리 기반 연속 유사도 "
        "(0.0=동일, 1.0=완전히 다름)"
    )
    w("- **colspan/rowspan 지원**: 병합된 셀의 span 속성을 트리 노드에 반영")
    w(
        "- **산출 공식**: `TEDS = 1 - (트리 편집 거리 / max(예측 노드 수, 정답 노드 수))`"
    )
    w("- **해석**: 1.0 = 동일한 표 구조+내용, 0.0 = 완전히 다른 표")
    w()
    w("### 3-4. 용어 사전 (평가 방법론)")
    w()
    w("| 용어 | 설명 |")
    w("|---|---|")
    w(
        "| **Levenshtein Distance** | 두 문자열 간 최소 편집(삽입·삭제·치환) 횟수. "
        "Vladimir Levenshtein(1965) 제안 |"
    )
    w(
        "| **n-gram** | 연속된 n개 토큰(단어)의 조합. "
        '예: "the cat sat" → bigram: {"the cat", "cat sat"} |'
    )
    w(
        "| **정밀도 (Precision)** | 예측 결과 중 실제 정답과 일치하는 비율. "
        "`TP / (TP + FP)` |"
    )
    w("| **재현율 (Recall)** | 실제 정답 중 예측이 맞춘 비율. `TP / (TP + FN)` |")
    w(
        "| **조화평균 (Harmonic Mean)** | 정밀도·재현율을 균형 있게 결합하는 평균. "
        "`2PR / (P + R)` |"
    )
    w(
        "| **Tree Edit Distance** | 한 트리를 다른 트리로 변환하는 최소 비용. "
        "APTED 알고리즘 사용 |"
    )
    w("| **토큰화 (Tokenization)** | 텍스트를 최소 의미 단위(토큰)로 분할하는 과정 |")
    w("| **colspan / rowspan** | HTML 테이블에서 셀이 가로/세로로 병합되는 범위 속성 |")
    w("| **Smoothing** | n-gram 매칭 0 발생 시 점수 소실을 방지하는 보정 기법 |")
    w(
        "| **WordNet** | Princeton 대학 제작 영어 어휘 데이터베이스. 동의어·상위어 관계 제공 |"
    )
    w()
    w("### 3-5. TEDS 구현 출처")
    w()
    w(
        "TEDS(Tree Edit Distance based Similarity)는 IBM Research가 제안한 "
        "표 구조 평가 메트릭으로, OmniDocBench에서 표준 평가 방법으로 채택되었습니다. "
        "본 보고서의 TEDS 구현은 **OmniDocBench 원본 코드를 직접 벤더링**하여 사용합니다."
    )
    w()
    w("| 구현 요소 | 상세 |")
    w("|---|---|")
    w("| **원본 출처** | OmniDocBench `metrics/table_metric.py` (IBM TEDS) |")
    w("| **HTML 파싱** | `lxml` — malformed HTML 허용, colspan/rowspan 지원 |")
    w("| **트리 비교** | `apted` — Tree Edit Distance 알고리즘 |")
    w("| **셀 내용 비교** | Levenshtein 거리 기반 연속 유사도 (이진 비교 아님) |")
    w("| **라이선스** | Apache-2.0 (IBM / OpenDataLab) |")
    w()
    w("---")
    w()

    # ── 4. 평가 기준 및 가중치 ──────────────────────────────────────────────
    w("## 4. 평가 기준 및 가중치")
    w()
    w("### 4-1. 가중치 배분 원칙")
    w()
    w(
        "가중치는 **RAG 시스템의 핵심 요구사항**에 따라 설정했습니다. "
        "RAG의 품질은 검색 대상 문서의 파싱 정확도에 가장 크게 의존하며, "
        "기업 환경에서는 데이터 보안이 필수 요건입니다. "
        "비용·표 구조·DPI 안정성은 운영 단계에서 유의미한 차이를 만들고, "
        "도입 용이성은 초기 적용 속도에 영향을 주지만 장기적으로는 비중이 낮습니다."
    )
    w()
    w("| 기준 | 가중치 | 설명 |")
    w("|---|:---:|---|")
    for k, v in WEIGHTS.items():
        w(f"| **{_CRITERION_KR[k]}** | {int(v * 100)}% | {_CRITERION_DESC[k]} |")
    w()
    w("### 4-2. 기준별 상세 설명")
    w()
    w("#### 파싱 정확도 (35%)")
    w()
    w(
        "RAG 시스템에서 파싱 정확도는 **검색 품질에 직결**되는 핵심 기준입니다. "
        "PDF에서 추출한 텍스트가 부정확하면, 임베딩과 검색 결과 모두 저하됩니다. "
        "OmniDocBench NED를 기반으로 측정하며, "
        "텍스트형·표형·그래프형 문서 전체의 종합 NED 평균을 사용합니다."
    )
    w()
    w("| NED 범위 | 점수 | 의미 |")
    w("|---|:---:|---|")
    w("| ≥ 0.70 | 5 | 원문 대비 높은 충실도 |")
    w("| 0.65~0.70 | 4 | 양호 |")
    w("| 0.55~0.65 | 3 | 보통 |")
    w("| 0.45~0.55 | 2 | 미흡 |")
    w("| < 0.45 | 1 | 부적합 |")
    w()
    w("#### 데이터 보안 (20%)")
    w()
    w(
        "기업 문서에는 기밀 정보, 개인정보가 포함될 수 있습니다. "
        "외부 API로 문서 데이터를 전송하면 유출 리스크가 발생하므로, "
        "**로컬 처리 가능 여부**가 핵심입니다."
    )
    w()
    w("| 기준 | 점수 |")
    w("|---|:---:|")
    w("| 완전 로컬 처리 (오픈소스) | 5 |")
    w("| 외부 API 사용 (데이터 전송) | 2 |")
    w()
    w("#### 비용 효율성 (15%)")
    w()
    w(
        "문서 처리량이 증가할수록 API 종량제 비용은 선형으로 증가하지만, "
        "로컬 솔루션은 인프라 비용만 발생합니다. "
        "**대규모 처리 시 비용 예측 가능성**이 핵심입니다."
    )
    w()
    w("| 기준 | 점수 |")
    w("|---|:---:|")
    w("| 오픈소스 (인프라 비용만) | 5 |")
    w("| API 종량제 (중간 단가) | 2 |")
    w("| API 종량제 (고단가) | 1 |")
    w()
    w("#### 표 구조 정확도 (15%)")
    w()
    w(
        "기업 문서에는 재무제표, 사양서, 비교표 등 **표가 빈번하게 등장**합니다. "
        "표의 행·열 구조와 셀 내용을 정확하게 재현하는 능력은 "
        "RAG 검색 시 정확한 컨텍스트 제공에 필수적입니다. "
        "IBM TEDS-HTML을 기반으로 측정합니다."
    )
    w()
    w("| TEDS 범위 | 점수 | 의미 |")
    w("|---|:---:|---|")
    w("| ≥ 0.60 | 5 | 표 구조+내용 높은 충실도 |")
    w("| 0.50~0.60 | 4 | 양호 |")
    w("| 0.35~0.50 | 3 | 보통 |")
    w("| 0.20~0.35 | 2 | 미흡 |")
    w("| < 0.20 | 1 | 부적합 |")
    w()
    w("#### DPI 안정성 (10%)")
    w()
    w(
        "실제 운영 환경에서는 다양한 해상도의 스캔 PDF가 유입됩니다. "
        "**DPI가 달라져도 파싱 품질이 일정한지**가 운영 안정성의 척도입니다. "
        "동일 문서의 72dpi~native 변형 간 NED 변동폭으로 측정합니다."
    )
    w()
    w("| NED 변동폭 | 점수 | 의미 |")
    w("|---|:---:|---|")
    w("| ≤ 3%p | 5 | 매우 안정 |")
    w("| ≤ 5%p | 4 | 안정 |")
    w("| ≤ 10%p | 3 | 보통 |")
    w("| ≤ 20%p | 2 | 불안정 |")
    w("| > 20%p | 1 | 매우 불안정 |")
    w()
    w("#### 도입 용이성 (5%)")
    w()
    w(
        "프로토타입 구축 속도와 운영 복잡도를 반영합니다. "
        "API 솔루션은 설정이 간단하지만 장기 운영 비용이 높고, "
        "로컬 솔루션은 초기 셋업이 복잡하지만 안정적입니다. "
        "**장기 운영 가치 대비 가중치를 낮게(5%) 설정**했습니다."
    )
    w()
    w("| 기준 | 점수 |")
    w("|---|:---:|")
    w("| API 키만으로 즉시 사용 | 5 |")
    w("| K8s/Docker 배포 필요 | 3 |")
    w("| 의존성 충돌·특수 환경 필요 | 2 |")
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
    w(
        "각 기준별 1~5 원점수에 가중치를 곱한 후 합산하여 종합 점수를 산출합니다. "
        "원점수/가중 점수를 함께 표시하여, 어떤 기준에서 강점·약점이 있는지 파악할 수 있습니다."
    )
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
    w(
        "문서 유형별로 NED를 분리 집계하여, "
        "솔루션이 특정 유형에서 강점이나 약점을 보이는지 확인합니다. "
        "NED가 1.0에 가까울수록 원문과 파싱 결과의 일치도가 높습니다."
    )
    w()

    for doc_type, doc_label, doc_desc in [
        (
            "text",
            "텍스트형 문서",
            "본문 위주의 문서. 파싱 난이도가 가장 낮으며, 기본 텍스트 추출 능력을 측정합니다.",
        ),
        (
            "table",
            "표형 문서",
            "표가 포함된 문서. NED(텍스트 정확도)와 TEDS-HTML(표 구조 정확도)을 함께 비교합니다.",
        ),
        (
            "graph",
            "그래프형 문서",
            "차트·그래프가 포함된 문서. 이미지 내 텍스트 추출 능력과 레이아웃 이해도를 측정합니다.",
        ),
    ]:
        w(f"#### {doc_label}")
        w()
        w(doc_desc)
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
    w(
        "동일 문서를 72dpi, 150dpi, 200dpi, 원본(native) 등 다양한 해상도로 변환 후 파싱한 결과입니다. "
        "변동폭이 작을수록 해상도 변화에 강건한 솔루션이며, 실제 운영 환경에서 안정적으로 작동합니다."
    )
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

    # ── 7. 리소스 효율성 ────────────────────────────────────────────────────
    w("## 7. 리소스 효율성")
    w()
    w(
        "모델 파라미터 수 대비 달성한 점수를 비교하여, "
        "동일한 정확도를 더 적은 리소스로 달성하는 솔루션을 식별합니다. "
        "파라미터가 적을수록 추론 비용과 배포 부담이 낮습니다."
    )
    w()

    # 효율성 테이블: 파라미터 대비 NED
    w("### 7-1. 파라미터 대비 정확도 (NED/B)")
    w()
    w(
        "모델 파라미터 1B당 달성하는 NED 점수입니다. "
        "값이 클수록 적은 파라미터로 높은 정확도를 달성하는 효율적인 솔루션입니다."
    )
    w()
    w("| 솔루션 | 파라미터 | 종합 NED | NED/B | 유형 |")
    w("|---|:---:|:---:|:---:|---|")

    # 효율성 계산 & 정렬
    eff_rows: list[tuple[str, str, float | None, float | None, str]] = []
    for b in backends_sorted:
        attrs = BACKEND_ATTRS.get(b, {})
        profile = BACKEND_PROFILES.get(b, {})
        params_b = attrs.get("params_b")
        ned = backend_avgs.get(b, {}).get("overall_ned")
        b_type = profile.get("type", "—")

        if params_b is not None and ned is not None and params_b > 0:
            eff = ned / params_b
            params_str = f"{params_b}B"
            eff_rows.append((b, params_str, ned, eff, b_type))
        else:
            params_str = "비공개" if params_b is None else f"{params_b}B"
            eff_rows.append((b, params_str, ned, None, b_type))

    # NED/B 내림차순 (None은 뒤로)
    eff_with = [(r, r[3]) for r in eff_rows if r[3] is not None]
    eff_without = [r for r in eff_rows if r[3] is None]
    eff_with.sort(key=lambda x: x[1], reverse=True)
    sorted_eff = [r for r, _ in eff_with] + eff_without

    for b, params_str, ned, eff, b_type in sorted_eff:
        ned_str = _fmt(ned)
        if eff is not None:
            eff_str = f"{eff:.4f}"
        else:
            eff_str = "—"
        w(f"| {b} | {params_str} | {ned_str} | {eff_str} | {b_type} |")
    w()

    # 비용 구조 비교
    w("### 7-2. 비용 구조 비교")
    w()
    w("| 비용 항목 | 로컬 솔루션 | API 솔루션 |")
    w("|---|---|---|")
    w("| **라이선스/API 비용** | 0원 (오픈소스) | 페이지당 종량제 과금 |")
    w("| **인프라** | K8s 노드 비용 | 해당 없음 |")
    w("| **확장 시 비용 증가** | 선형 미만 | **선형 (문서 수 비례)** |")
    w("| **파라미터 효율성** | 소형 모델로 동등 품질 가능 | 대형 모델 의존 |")
    w()
    w("---")
    w()

    # ── 8. 추천 ─────────────────────────────────────────────────────────────
    w("## 8. 추천")
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

    # ── 9. 다음 단계 ───────────────────────────────────────────────────────
    w("## 9. 다음 단계")
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
    w("| **TEDS** | Tree Edit Distance Similarity | IBM 제안 표 구조 정확도 (0~1) |")
    w("| **OmniDocBench** | — | CVPR 2025 PDF 파싱 종합 벤치마크 (OpenDataLab) |")
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
