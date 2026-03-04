"""
PDF Parser 백엔드 전체 비교 보고서 생성기
Usage: python generate_report.py [--output path/to/report.md]
"""
import pathlib
import json
import argparse
from datetime import datetime

BASE = pathlib.Path(__file__).parent / "bench_results"
OUTPUT_DEFAULT = pathlib.Path(__file__).parent / "reports" / "pdf_parser_comparison.md"

# ─── Canonical run-id per backend ────────────────────────────────────────────
CANONICAL_RUNS = {
    "openai": "vlm-20260303-1550",
    "upstage": "upstage-20260303-1635",
    "upstage-enhanced": "upstage-20260303-1635",
    "paddleocr-vl": "paddleocr-20260303-1541",
}

# K8s Phase1 결과 (로컬 불가: docling rt_detr_v2 오류, pymupdf K8s 전용)
# run-id: 20260227-1118, 출처: memory/pdf-parser-results.md
K8S_PHASE1 = {
    "pymupdf": {
        "text_only.pdf":    {"avg_text_ned": 0.6577, "avg_table_teds": 0.4928, "avg_speed_s": 2.3},
        "table_native.pdf": {"avg_text_ned": 0.6269, "avg_table_teds": 0.4072, "avg_speed_s": 2.1},
        "graph_rich.pdf":   {"avg_text_ned": 0.3543, "avg_table_teds": None,   "avg_speed_s": 1.9},
        # table_image 계열: N/A (텍스트 추출 불가)
    },
    "docling": {
        "text_only.pdf":    {"avg_text_ned": 0.7431, "avg_table_teds": 0.5348, "avg_speed_s": 45.0},
        "table_native.pdf": {"avg_text_ned": 0.6995, "avg_table_teds": 0.5253, "avg_speed_s": 60.0},
        "table_image.pdf":  {"avg_text_ned": 0.27,   "avg_table_teds": 0.26,   "avg_speed_s": 70.0},
        "graph_rich.pdf":   {"avg_text_ned": 0.6242, "avg_table_teds": None,   "avg_speed_s": 55.0},
        "graph_rich_image.pdf": {"avg_text_ned": 0.14, "avg_table_teds": None, "avg_speed_s": 65.0},
    },
}

# ─── 백엔드 표시 정보 ────────────────────────────────────────────────────────
BACKEND_INFO = {
    "pymupdf":          {"label": "PyMuPDF",           "type": "local",  "note": "K8s 결과†"},
    "docling":          {"label": "Docling",            "type": "local",  "note": "K8s 결과†"},
    "openai":           {"label": "OpenAI GPT-4o",      "type": "api",    "note": ""},
    "upstage":          {"label": "Upstage",            "type": "api",    "note": ""},
    "upstage-enhanced": {"label": "Upstage Enhanced",   "type": "api",    "note": ""},
    "paddleocr-vl":     {"label": "PaddleOCR-VL",       "type": "local",  "note": ""},
}

BACKEND_ORDER = ["pymupdf", "docling", "openai", "upstage", "upstage-enhanced", "paddleocr-vl"]

# ─── PDF 유형 분류 ────────────────────────────────────────────────────────────
PDF_CATEGORY = {
    "text": ["text_only.pdf"],
    "table": [
        "table_native.pdf",
        "table_image.pdf",
        "table_image_72dpi.pdf",
        "table_image_150dpi.pdf",
        "table_image_200dpi.pdf",
    ],
    "graph": [
        "graph_rich.pdf",
        "graph_rich_image.pdf",
        "graph_rich_image_72dpi.pdf",
        "graph_rich_image_150dpi.pdf",
        "graph_rich_image_200dpi.pdf",
    ],
}

PDF_LABEL = {
    "text_only.pdf":              "텍스트 전용",
    "table_native.pdf":           "표 (네이티브)",
    "table_image.pdf":            "표 (이미지)",
    "table_image_72dpi.pdf":      "표 (이미지, 72dpi)",
    "table_image_150dpi.pdf":     "표 (이미지, 150dpi)",
    "table_image_200dpi.pdf":     "표 (이미지, 200dpi)",
    "graph_rich.pdf":             "그래프 (텍스트 혼재)",
    "graph_rich_image.pdf":       "그래프+이미지",
    "graph_rich_image_72dpi.pdf": "그래프+이미지 (72dpi)",
    "graph_rich_image_150dpi.pdf":"그래프+이미지 (150dpi)",
    "graph_rich_image_200dpi.pdf":"그래프+이미지 (200dpi)",
}


def load_local_data() -> dict[str, dict[str, dict]]:
    """canonical run에서 metrics.json 수집. {backend: {pdf: summary}}"""
    data: dict[str, dict[str, dict]] = {}
    for run_dir in BASE.iterdir():
        run_name = run_dir.name
        for mf in run_dir.rglob("metrics.json"):
            rec = json.loads(mf.read_text())
            backend = rec.get("backend")
            if backend not in CANONICAL_RUNS:
                continue
            if run_name != CANONICAL_RUNS[backend]:
                continue
            if rec.get("error"):
                continue
            pdf = rec.get("pdf_name", "?")
            s = rec["summary"]
            if backend not in data:
                data[backend] = {}
            data[backend][pdf] = s
    return data


def merge_all() -> dict[str, dict[str, dict]]:
    """로컬 + K8s Phase1 데이터 병합."""
    data = load_local_data()
    for backend, pdfs in K8S_PHASE1.items():
        if backend not in data:
            data[backend] = {}
        for pdf, summary in pdfs.items():
            if pdf not in data[backend]:
                data[backend][pdf] = summary
    return data


def fmt_ned(v) -> str:
    if v is None or v < 0:
        return "N/A"
    return f"{v:.4f}"


def fmt_teds(v) -> str:
    if v is None or v < 0:
        return "N/A"
    return f"{v:.4f}"


def fmt_speed(v) -> str:
    if v is None or v <= 0:
        return "N/A"
    if v < 60:
        return f"{v:.0f}s"
    return f"{v/60:.1f}min"


def avg_over(data: dict, backend: str, pdfs: list[str], key: str) -> float | None:
    vals = []
    for pdf in pdfs:
        s = data.get(backend, {}).get(pdf)
        if s:
            v = s.get(key)
            if v is not None and v >= 0:
                vals.append(v)
    return sum(vals) / len(vals) if vals else None


def category_avg(data: dict, backend: str, cat: str, key: str) -> float | None:
    return avg_over(data, backend, PDF_CATEGORY[cat], key)


def overall_avg(data: dict, backend: str, key: str) -> float | None:
    all_pdfs = [p for ps in PDF_CATEGORY.values() for p in ps]
    return avg_over(data, backend, all_pdfs, key)


def best_backend(data: dict, cat: str, key: str, backends: list[str]) -> str:
    scores = {}
    for b in backends:
        v = category_avg(data, b, cat, key)
        if v is not None:
            scores[b] = v
    if not scores:
        return "N/A"
    best = max(scores, key=lambda b: scores[b])
    return f"{BACKEND_INFO[best]['label']} ({scores[best]:.4f})"


def generate(output: pathlib.Path):
    data = merge_all()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    def w(s=""):
        lines.append(s)

    today = datetime.now().strftime("%Y-%m-%d")

    # ── 헤더 ──────────────────────────────────────────────────────────────────
    w(f"# PDF Parser 백엔드 전체 비교 보고서")
    w(f"> 생성일: {today}  |  총 6 백엔드  |  11 PDF 유형")
    w()
    w("---")
    w()

    # ── 1. 핵심 결론 ──────────────────────────────────────────────────────────
    w("## 1. 핵심 결론")
    w()

    # 전체 NED 순위
    ned_rank = []
    for b in BACKEND_ORDER:
        v = overall_avg(data, b, "avg_text_ned")
        if v is not None:
            ned_rank.append((b, v))
    ned_rank.sort(key=lambda x: -x[1])

    teds_rank = []
    for b in BACKEND_ORDER:
        v = overall_avg(data, b, "avg_table_teds")
        if v is not None:
            teds_rank.append((b, v))
    teds_rank.sort(key=lambda x: -x[1])

    if ned_rank:
        top_ned = ned_rank[0]
        w(f"- **텍스트 정확도(NED) 1위**: {BACKEND_INFO[top_ned[0]]['label']} ({top_ned[1]:.4f})")
    if teds_rank:
        top_teds = teds_rank[0]
        w(f"- **표 정확도(TEDS) 1위**: {BACKEND_INFO[top_teds[0]]['label']} ({top_teds[1]:.4f})")

    # 유형별 최적
    w(f"- **텍스트형 최적**: {best_backend(data, 'text', 'avg_text_ned', BACKEND_ORDER)}")
    w(f"- **표형 최적 (NED)**: {best_backend(data, 'table', 'avg_text_ned', BACKEND_ORDER)}")
    w(f"- **표형 최적 (TEDS)**: {best_backend(data, 'table', 'avg_table_teds', BACKEND_ORDER)}")
    w(f"- **그래프형 최적**: {best_backend(data, 'graph', 'avg_text_ned', BACKEND_ORDER)}")
    w()

    # upstage vs upstage-enhanced 비교
    u_ned = overall_avg(data, 'upstage', 'avg_text_ned')
    ue_ned = overall_avg(data, 'upstage-enhanced', 'avg_text_ned')
    if u_ned and ue_ned:
        winner = "Upstage" if u_ned >= ue_ned else "Upstage Enhanced"
        diff = abs(u_ned - ue_ned)
        w(f"- **Upstage vs Enhanced**: {winner} 우세 (NED 차이 {diff:.4f})")

    w()

    # ── 2. 백엔드 종합 순위표 ─────────────────────────────────────────────────
    w("## 2. 백엔드 종합 순위표")
    w()
    w("> NED: 텍스트 일치도 (0~1, 높을수록 좋음) | TEDS: 표 구조 정확도 (0~1, 높을수록 좋음)")
    w("> †K8s Phase1 결과 (run-id: 20260227-1118), 로컬 재현 불가 (docling: transformers 호환 문제)")
    w()
    w("| 순위 | 백엔드 | 유형 | 전체 평균 NED | 전체 평균 TEDS | 비고 |")
    w("|:---:|---|:---:|:---:|:---:|---|")

    rank_rows = []
    for b in BACKEND_ORDER:
        ned = overall_avg(data, b, "avg_text_ned")
        teds = overall_avg(data, b, "avg_table_teds")
        info = BACKEND_INFO[b]
        rank_rows.append((b, ned, teds, info))

    rank_rows.sort(key=lambda x: (x[1] or 0), reverse=True)
    for i, (b, ned, teds, info) in enumerate(rank_rows, 1):
        btype = "🌐 API" if info["type"] == "api" else "💻 로컬"
        note = info["note"]
        w(f"| {i} | **{info['label']}** | {btype} | {fmt_ned(ned)} | {fmt_teds(teds)} | {note} |")
    w()

    # ── 3. 문서 유형별 비교 ───────────────────────────────────────────────────
    w("## 3. 문서 유형별 비교")
    w()

    cat_names = {"text": "텍스트형", "table": "표형", "graph": "그래프형"}
    cat_icons = {"text": "📄", "table": "📊", "graph": "📈"}

    for cat, cat_label in cat_names.items():
        w(f"### 3-{list(cat_names).index(cat)+1}. {cat_icons[cat]} {cat_label}")
        w()
        w(f"| 백엔드 | NED | TEDS | 비고 |")
        w("|---|:---:|:---:|---|")

        rows = []
        for b in BACKEND_ORDER:
            ned = category_avg(data, b, cat, "avg_text_ned")
            teds = category_avg(data, b, cat, "avg_table_teds")
            rows.append((b, ned, teds))

        rows.sort(key=lambda x: (x[1] or 0), reverse=True)
        for b, ned, teds in rows:
            info = BACKEND_INFO[b]
            marker = " ★" if rows[0][0] == b else ""
            w(f"| {info['label']}{marker} | {fmt_ned(ned)} | {fmt_teds(teds)} | {info['note']} |")
        w()

    # ── 4. 세부 결과표 ────────────────────────────────────────────────────────
    w("## 4. 세부 결과표 (백엔드 × PDF)")
    w()
    w("> NED 기준 정렬 | TEDS N/A = 해당 PDF에 표 없음 (graph_rich 계열)")
    w()

    all_pdfs = [p for ps in PDF_CATEGORY.values() for p in ps]

    for cat, cat_label in cat_names.items():
        w(f"### {cat_icons[cat]} {cat_label}")
        w()

        # 헤더
        header_backends = [b for b in BACKEND_ORDER if any(
            data.get(b, {}).get(pdf) for pdf in PDF_CATEGORY[cat]
        )]
        header = "| PDF 유형 | " + " | ".join(
            f"{BACKEND_INFO[b]['label']} NED | {BACKEND_INFO[b]['label']} TEDS"
            for b in header_backends
        ) + " |"
        sep = "|---|" + ":---:|:---:|" * len(header_backends)
        w(header)
        w(sep)

        for pdf in PDF_CATEGORY[cat]:
            label = PDF_LABEL.get(pdf, pdf)
            cells = []
            for b in header_backends:
                s = data.get(b, {}).get(pdf)
                if s:
                    cells.append(fmt_ned(s.get("avg_text_ned")))
                    cells.append(fmt_teds(s.get("avg_table_teds")))
                else:
                    cells.append("—")
                    cells.append("—")
            w(f"| {label} | " + " | ".join(cells) + " |")
        w()

    # ── 5. 속도 비교 ──────────────────────────────────────────────────────────
    w("## 5. 속도 참고 (평균 처리 시간/페이지)")
    w()
    w("> ⚠️ 속도는 환경(네트워크, 서버 부하)에 따라 편차가 큼. 순위 판단에 미반영.")
    w()
    w("| 백엔드 | 텍스트형 | 표형 (avg) | 그래프형 (avg) | 비고 |")
    w("|---|:---:|:---:|:---:|---|")

    for b in BACKEND_ORDER:
        txt_spd = category_avg(data, b, "text", "avg_speed_s")
        tbl_spd = category_avg(data, b, "table", "avg_speed_s")
        gph_spd = category_avg(data, b, "graph", "avg_speed_s")
        info = BACKEND_INFO[b]
        w(f"| {info['label']} | {fmt_speed(txt_spd)} | {fmt_speed(tbl_spd)} | {fmt_speed(gph_spd)} | {info['note']} |")
    w()

    # ── 6. 추천 ───────────────────────────────────────────────────────────────
    w("## 6. 백엔드 선택 가이드")
    w()
    w("| 문서 유형 | 1순위 추천 | 2순위 추천 | 근거 |")
    w("|---|---|---|---|")

    def top2(cat, key):
        rows = [(b, category_avg(data, b, cat, key)) for b in BACKEND_ORDER]
        rows = [(b, v) for b, v in rows if v is not None]
        rows.sort(key=lambda x: -x[1])
        top = [BACKEND_INFO[b]['label'] for b, _ in rows[:2]]
        scores = [f"{v:.4f}" for _, v in rows[:2]]
        return top, scores

    for cat, cat_label in cat_names.items():
        tops_ned, scores_ned = top2(cat, "avg_text_ned")
        tops_teds, scores_teds = top2(cat, "avg_table_teds")
        first = tops_ned[0] if tops_ned else "N/A"
        second = tops_ned[1] if len(tops_ned) > 1 else "N/A"
        reason = f"NED {scores_ned[0]}" if scores_ned else ""
        if cat == "table" and tops_teds:
            reason += f", TEDS {scores_teds[0]}"
        w(f"| {cat_icons[cat]} {cat_label} | **{first}** | {second} | {reason} |")
    w()
    w("> 💡 **비용·보안 고려 시**: API 백엔드(OpenAI, Upstage) 대신 PaddleOCR-VL 활용 가능")
    w("> 💡 **Phase 5 예정**: MinerU Pipeline 추가 예정 — 현재 순위 변동 가능")
    w()

    # ── 7. 데이터 출처 ────────────────────────────────────────────────────────
    w("## 7. 데이터 출처")
    w()
    w("| 백엔드 | Run ID | PDF 수 | 상태 |")
    w("|---|---|:---:|---|")
    sources = [
        ("pymupdf",          "20260227-1118 (K8s)", "5†",  "K8s 결과 (부분)"),
        ("docling",          "20260227-1118 (K8s)", "5†",  "K8s 결과 (부분), 로컬 실행 불가"),
        ("openai",           "vlm-20260303-1550",   "11",  "완료"),
        ("upstage",          "upstage-20260303-1635","11", "완료"),
        ("upstage-enhanced", "upstage-20260303-1635","11", "완료"),
        ("paddleocr-vl",     "paddleocr-20260303-1541","11","완료"),
    ]
    for b, run_id, n, status in sources:
        w(f"| {BACKEND_INFO[b]['label']} | `{run_id}` | {n} | {status} |")
    w()
    w("† pymupdf/docling: K8s Phase1 실행 결과. 로컬 docling은 `transformers>=4.49.0` 호환 문제로 실행 불가 (Task 2 예정).")
    w()

    report = "\n".join(lines)
    output.write_text(report, encoding="utf-8")
    print(f"✅ 보고서 생성 완료: {output}")
    print(f"   총 {len(lines)}줄")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    args = parser.parse_args()
    generate(pathlib.Path(args.output))
