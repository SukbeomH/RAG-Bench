"""
서비스 벤치마크 결과 병합 스크립트.

여러 실행(카테고리, 프리셋)에서 생성된 result.json / latency.csv를
하나의 통합 리포트로 합친다.

사용법:
  # 기본 (service_run/ 자동 탐색)
  uv run python -m rag_bench.scripts.merge_service_results

  # 특정 디렉토리 지정
  uv run python -m rag_bench.scripts.merge_service_results --run_dirs service_run standard_run

  # 출력 파일 지정
  uv run python -m rag_bench.scripts.merge_service_results --output merged_report.html

출력:
  - merged_report.html  : 카테고리 × 전략 RAGAS 히트맵 + 레이턴시 표
  - merged_ragas.csv    : 전략별 카테고리 평균 점수
  - merged_latency.csv  : 전략별 카테고리 평균 레이턴시
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from rag_bench.config import BENCH_DATA_DIR


# ---------------------------------------------------------------------------
# 전략 이름 단축
# ---------------------------------------------------------------------------

def _short_strategy(name: str) -> str:
    """전략 이름을 리랭커+Dense+Sparse 형식으로 단축.

    full 이름: "ColBERT Rerank (Contextual Retrieval (DS(bge-m3+korean_bm25)))"
    short 이름: "colbert|bge-m3+korean_bm25"  ← 리랭커를 prefix로 포함하여 충돌 방지

    리랭커가 다른 전략이 같은 DS 조합을 가질 때(full 프리셋) 동일 key로 합쳐지는
    버그를 방지하기 위해 리랭커 종류를 prefix로 분리한다.
    """
    import re
    # 리랭커 종류 추출
    reranker = "unknown"
    if "ColBERT" in name:
        reranker = "colbert"
    elif "FlashRank" in name or "flashrank" in name.lower():
        reranker = "flashrank"

    m = re.search(r"DS\((.+?)\)", name)
    if m:
        inner = m.group(1)
        inner = inner.replace("KoSimCSE-roberta-multitask", "KoSimCSE")
        inner = inner.replace("multilingual-e5-large", "E5")
        return f"{reranker}|{inner}"
    return name


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------

def load_run_dir(run_dir: Path) -> Dict:
    """run_dir 하위 카테고리 폴더에서 result.json 수집."""
    results = {}
    for cat_dir in sorted(run_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        result_file = cat_dir / "result.json"
        if not result_file.exists():
            continue
        data = json.loads(result_file.read_text(encoding="utf-8"))
        # result.json의 category 필드를 우선, 없으면 디렉토리명 사용
        category = data.get("category", cat_dir.name)
        results[category] = (data, cat_dir.name)  # (data, dir_name) 튜플로 저장
    return results


def load_latency(run_dir: Path, dir_to_category: Dict[str, str]) -> Dict[str, List[dict]]:
    """카테고리별 latency.csv 로드.

    dir_to_category: 디렉토리명 → result.json category 매핑 (key 불일치 방지).
    """
    latency = {}
    for cat_dir in sorted(run_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        csv_file = cat_dir / "latency.csv"
        if not csv_file.exists():
            continue
        # result.json과 동일한 category key 사용
        category = dir_to_category.get(cat_dir.name, cat_dir.name)
        rows = []
        with open(csv_file, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 오류가 발생한 행은 레이턴시 집계에서 제외
                if row.get("error", "").strip():
                    continue
                rows.append(row)
        latency[category] = rows
    return latency


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def validate_and_collect_ragas(all_results: Dict[str, Dict]) -> None:
    """병합 전 RAGAS 지표 완전성 검증.

    카테고리 × 전략 조합 중 하나라도 지표가 누락되어 있으면
    ValueError를 발생시켜 병합을 중단한다.

    누락 기준:
      - result.json에 "ragas" 키 자체가 없는 경우
      - ragas 항목에서 RAGAS_METRICS 중 하나라도 None / 누락인 경우
      - ragas 항목이 빈 리스트인 경우 (RAGAS 평가 미실행)
    """
    errors: List[str] = []

    for category, data in all_results.items():
        ragas_list = data.get("ragas")

        if ragas_list is None:
            errors.append(f"  [{category}] 'ragas' 키가 result.json에 없음 (RAGAS 평가 미실행)")
            continue

        if len(ragas_list) == 0:
            errors.append(f"  [{category}] ragas 리스트가 비어 있음 (전략 결과 0개)")
            continue

        for entry in ragas_list:
            strategy = entry.get("strategy", "unknown")
            missing = [m for m in RAGAS_METRICS if entry.get(m) is None]
            if missing:
                errors.append(
                    f"  [{category}] 전략 '{strategy}': "
                    f"지표 누락 → {missing}"
                )

    if errors:
        raise ValueError(
            "병합 중단: 아래 항목에서 RAGAS 지표가 누락되었습니다.\n"
            + "\n".join(errors)
            + "\n\n누락된 카테고리/전략의 벤치마크를 완료한 후 다시 실행하세요."
        )


def aggregate_ragas(all_results: Dict[str, Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """strategy → category → metric → score 구조로 집계.

    사전 조건: validate_and_collect_ragas() 통과 후 호출해야 한다.
    모든 지표가 완전한 것이 보장된 상태에서 집계하므로 None 처리 불필요.
    동일 (strategy, category) 중복 시 경고 출력 후 기존 값 유지 (덮어쓰기 방지).
    """
    agg: Dict[str, Dict[str, Dict[str, float]]] = {}
    for category, data in all_results.items():
        ragas_list = data.get("ragas", [])
        for entry in ragas_list:
            strategy = _short_strategy(entry.get("strategy", "unknown"))
            if strategy not in agg:
                agg[strategy] = {}
            if category in agg[strategy]:
                print(f"[경고] 중복 감지: strategy='{strategy}', category='{category}' — 기존 값 유지")
                continue
            scores = {m: float(entry[m]) for m in RAGAS_METRICS}
            agg[strategy][category] = scores
    return agg


def aggregate_latency(all_latency: Dict[str, List[dict]]) -> Dict[str, Dict[str, float]]:
    """strategy → category → avg_latency_ms."""
    agg: Dict[str, Dict[str, float]] = {}
    for category, rows in all_latency.items():
        by_strategy: Dict[str, List[float]] = {}
        for row in rows:
            strat = _short_strategy(row.get("strategy", ""))
            try:
                lat = float(row.get("latency_ms", 0))
            except (ValueError, TypeError):
                lat = 0.0
            by_strategy.setdefault(strat, []).append(lat)
        for strat, lats in by_strategy.items():
            if strat not in agg:
                agg[strat] = {}
            agg[strat][category] = sum(lats) / len(lats) if lats else 0.0
    return agg


# ---------------------------------------------------------------------------
# CSV 출력
# ---------------------------------------------------------------------------

def save_ragas_csv(agg: Dict[str, Dict[str, Dict[str, float]]], categories: List[str], output: Path) -> None:
    """전략별 카테고리 RAGAS 점수 CSV."""
    header = ["strategy"]
    for cat in categories:
        for m in RAGAS_METRICS:
            header.append(f"{cat}_{m}")
    # 카테고리 평균 열 추가
    for m in RAGAS_METRICS:
        header.append(f"avg_{m}")

    rows = []
    for strategy, cat_scores in agg.items():
        row = {"strategy": strategy}
        all_scores: Dict[str, List[float]] = {m: [] for m in RAGAS_METRICS}
        for cat in categories:
            scores = cat_scores.get(cat, {})
            for m in RAGAS_METRICS:
                val = scores.get(m)  # None이면 누락 (0.0과 구별)
                col = f"{cat}_{m}"
                row[col] = f"{val:.4f}" if val is not None else ""
                if val is not None:
                    all_scores[m].append(val)
        for m in RAGAS_METRICS:
            vals = all_scores[m]
            row[f"avg_{m}"] = f"{sum(vals)/len(vals):.4f}" if vals else ""
        rows.append(row)

    # avg_faithfulness 기준 내림차순 정렬
    rows.sort(key=lambda r: float(r.get("avg_faithfulness", 0) or 0), reverse=True)

    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[CSV] RAGAS 결과 저장: {output}")


def save_latency_csv(agg: Dict, categories: List[str], output: Path) -> None:
    """전략별 카테고리 평균 레이턴시 CSV."""
    header = ["strategy"] + [f"{cat}_avg_ms" for cat in categories] + ["overall_avg_ms"]
    rows = []
    for strategy, cat_lats in agg.items():
        row = {"strategy": strategy}
        vals = []
        for cat in categories:
            lat = cat_lats.get(cat, None)
            row[f"{cat}_avg_ms"] = f"{lat:.1f}" if lat is not None else ""
            if lat is not None:
                vals.append(lat)
        row["overall_avg_ms"] = f"{sum(vals)/len(vals):.1f}" if vals else ""
        rows.append(row)

    rows.sort(key=lambda r: float(r.get("overall_avg_ms", 999999) or 999999))

    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[CSV] 레이턴시 결과 저장: {output}")


# ---------------------------------------------------------------------------
# HTML 리포트
# ---------------------------------------------------------------------------

def _color(val: float, lo: float = 0.7, hi: float = 1.0) -> str:
    """점수를 빨강→노랑→초록 색상으로 변환."""
    ratio = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    r = int(255 * (1 - ratio))
    g = int(200 * ratio)
    return f"rgb({r},{g},80)"


def _lat_color(val: float, lo: float = 3000, hi: float = 8000) -> str:
    """레이턴시를 초록→노랑→빨강 색상으로 변환 (낮을수록 좋음)."""
    ratio = max(0.0, min(1.0, (val - lo) / (hi - lo)))
    r = int(200 * ratio)
    g = int(200 * (1 - ratio))
    return f"rgb({r},{g},80)"


def save_html_report(
    ragas_agg: Dict,
    latency_agg: Dict,
    categories: List[str],
    all_results: Dict,
    output: Path,
) -> None:
    """카테고리 × 전략 히트맵 HTML 리포트 생성."""

    strategies = sorted(ragas_agg.keys())

    # --- RAGAS 히트맵 테이블 ---
    def ragas_table() -> str:
        cols = []
        for cat in categories:
            cols.append(f'<th colspan="4">{cat}</th>')
        cols.append('<th colspan="4">평균</th>')

        subcols = []
        for _ in categories + ["avg"]:
            for m in ["faith", "ans_rel", "ctx_prec", "ctx_rec"]:
                subcols.append(f"<th>{m}</th>")

        rows_html = []
        for strat in strategies:
            cat_scores = ragas_agg.get(strat, {})
            cells = []
            all_vals: Dict[str, List[float]] = {m: [] for m in RAGAS_METRICS}
            for cat in categories:
                scores = cat_scores.get(cat, {})
                for m in RAGAS_METRICS:
                    val = scores.get(m)
                    if val is not None:
                        bg = _color(val)
                        cells.append(f'<td style="background:{bg}">{val:.3f}</td>')
                        all_vals[m].append(val)
                    else:
                        cells.append('<td style="color:#aaa">—</td>')
            # 평균 열
            for m in RAGAS_METRICS:
                vals = all_vals[m]
                if vals:
                    avg = sum(vals) / len(vals)
                    bg = _color(avg)
                    cells.append(f'<td style="background:{bg};font-weight:bold">{avg:.3f}</td>')
                else:
                    cells.append('<td style="color:#aaa">—</td>')

            rows_html.append(f'<tr><td class="strat">{strat}</td>{"".join(cells)}</tr>')

        return f"""
<table class="heatmap">
  <thead>
    <tr><th rowspan="2">전략</th>{"".join(cols)}</tr>
    <tr>{"".join(subcols)}</tr>
  </thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>"""

    # --- 레이턴시 테이블 ---
    def latency_table() -> str:
        header = "<tr><th>전략</th>" + "".join(f"<th>{cat}</th>" for cat in categories) + "<th>전체 평균</th></tr>"
        rows_html = []
        strats_sorted = sorted(
            latency_agg.keys(),
            key=lambda s: sum(latency_agg[s].values()) / max(len(latency_agg[s]), 1)
        )
        for strat in strats_sorted:
            cat_lats = latency_agg.get(strat, {})
            cells = []
            vals = []
            for cat in categories:
                lat = cat_lats.get(cat)
                if lat is not None:
                    bg = _lat_color(lat)
                    cells.append(f'<td style="background:{bg}">{lat:.0f}ms</td>')
                    vals.append(lat)
                else:
                    cells.append('<td style="color:#aaa">—</td>')
            if vals:
                avg = sum(vals) / len(vals)
                bg = _lat_color(avg)
                cells.append(f'<td style="background:{bg};font-weight:bold">{avg:.0f}ms</td>')
            else:
                cells.append('<td>—</td>')
            rows_html.append(f'<tr><td class="strat">{strat}</td>{"".join(cells)}</tr>')
        return f'<table class="heatmap"><thead>{header}</thead><tbody>{"".join(rows_html)}</tbody></table>'

    # --- 카테고리별 n_qa 요약 및 불균등 경고 ---
    n_qa_list = [all_results[cat].get("n_qa", 0) for cat in categories if cat in all_results]
    qa_summary = " | ".join(
        f"{cat}: {all_results[cat].get('n_qa', '?')}개" for cat in categories if cat in all_results
    )
    unequal_nqa = len(set(n_qa_list)) > 1
    nqa_warning = (
        '<p style="color:#e74c3c;font-size:0.9em">⚠️ 카테고리별 n_qa가 다릅니다. '
        '"평균" 열은 n_qa 미반영 단순 평균이므로 해석에 주의하세요.</p>'
        if unequal_nqa else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>RAG 서비스 벤치마크 통합 리포트</title>
<style>
  body {{ font-family: 'Pretendard', 'Noto Sans KR', sans-serif; margin: 20px; background: #f8f9fa; }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; margin-top: 2em; border-bottom: 2px solid #3498db; padding-bottom: 6px; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 1em; }}
  table.heatmap {{ border-collapse: collapse; font-size: 0.82em; margin-bottom: 2em; }}
  table.heatmap th {{ background: #2c3e50; color: white; padding: 6px 10px; text-align: center; white-space: nowrap; }}
  table.heatmap td {{ padding: 5px 8px; text-align: center; border: 1px solid #ddd; white-space: nowrap; }}
  td.strat {{ text-align: left; font-weight: bold; background: #ecf0f1; max-width: 220px; overflow: hidden; text-overflow: ellipsis; }}
  .legend {{ display: flex; gap: 20px; margin: 10px 0; font-size: 0.85em; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .legend-box {{ width: 20px; height: 20px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>RAG 서비스 벤치마크 통합 리포트</h1>
<div class="meta">
  카테고리: {", ".join(categories)} | QA 수: {qa_summary}<br>
  전략 수: {len(strategies)} | 지표: faithfulness / answer_relevancy / context_precision / context_recall
</div>

{nqa_warning}
<h2>RAGAS 품질 지표 히트맵</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-box" style="background:rgb(255,0,80)"></div> 낮음 (&lt;0.7)</div>
  <div class="legend-item"><div class="legend-box" style="background:rgb(128,100,80)"></div> 중간</div>
  <div class="legend-item"><div class="legend-box" style="background:rgb(0,200,80)"></div> 높음 (≥1.0)</div>
</div>
{ragas_table()}

<h2>검색 레이턴시 (평균 ms)</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-box" style="background:rgb(0,200,80)"></div> 빠름 (&lt;3s)</div>
  <div class="legend-item"><div class="legend-box" style="background:rgb(200,0,80)"></div> 느림 (&gt;8s)</div>
</div>
{latency_table()}
</body>
</html>"""

    output.write_text(html, encoding="utf-8")
    print(f"[HTML] 통합 리포트 저장: {output}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="서비스 벤치마크 결과 병합")
    parser.add_argument(
        "--run_dirs", nargs="+", default=None,
        help="결과 디렉토리 목록 (기본: _benchdata/service_run)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="HTML 출력 파일 경로 (기본: _benchdata/merged_report.html)"
    )
    parser.add_argument(
        "--categories", type=str, default=None,
        help="포함할 카테고리 목록 (쉼표 구분, 기본: 자동 탐지)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 실행 디렉토리 결정
    if args.run_dirs:
        run_dirs = [Path(d) for d in args.run_dirs]
    else:
        run_dirs = [BENCH_DATA_DIR / "service_run"]

    # 결과 수집
    all_results: Dict[str, dict] = {}
    all_latency: Dict[str, list] = {}
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"[경고] 디렉토리 없음: {run_dir}")
            continue
        loaded = load_run_dir(run_dir)
        # dir_name → category 역매핑 (latency key 동기화용)
        dir_to_category = {dir_name: data["category"] if "category" in data else dir_name
                           for _, (data, dir_name) in loaded.items()}
        for category, (data, _) in loaded.items():
            if category in all_results:
                # 동일 카테고리 중복 시 n_qa 기준으로 더 많은 데이터를 우선
                existing_n = all_results[category].get("n_qa", 0)
                new_n = data.get("n_qa", 0)
                if new_n > existing_n:
                    print(f"[경고] '{category}' 중복 — n_qa 더 큰 값으로 교체 ({existing_n} → {new_n})")
                    all_results[category] = data
                else:
                    print(f"[경고] '{category}' 중복 — 기존 값 유지 (n_qa: {existing_n} ≥ {new_n})")
            else:
                all_results[category] = data
        lat = load_latency(run_dir, dir_to_category)
        for cat, rows in lat.items():
            if cat not in all_latency:
                all_latency[cat] = rows
            else:
                print(f"[경고] latency '{cat}' 중복 — 기존 값 유지")

    if not all_results:
        print("[오류] 결과 파일을 찾을 수 없습니다.")
        return

    # 카테고리 정렬
    cat_order = ["general", "legal", "business", "medical", "technical"]
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",")]
    else:
        categories = sorted(all_results.keys(), key=lambda c: cat_order.index(c) if c in cat_order else 99)

    print(f"[병합] 카테고리: {categories}")
    print(f"[병합] RAGAS 있는 카테고리: {[c for c in categories if all_results.get(c, {}).get('ragas')]}")

    # 지표 완전성 검증 — 누락 시 여기서 중단
    ragas_targets = {c: all_results[c] for c in categories if c in all_results}
    try:
        validate_and_collect_ragas(ragas_targets)
    except ValueError as e:
        print(f"\n[오류] {e}")
        return

    # 집계
    ragas_agg = aggregate_ragas(ragas_targets)
    latency_agg = aggregate_latency({c: all_latency[c] for c in categories if c in all_latency})

    # 출력 경로
    output_dir = BENCH_DATA_DIR
    html_output = Path(args.output) if args.output else output_dir / "merged_report.html"
    ragas_csv = html_output.parent / "merged_ragas.csv"
    latency_csv = html_output.parent / "merged_latency.csv"

    # 저장
    save_ragas_csv(ragas_agg, categories, ragas_csv)
    save_latency_csv(latency_agg, categories, latency_csv)
    save_html_report(ragas_agg, latency_agg, categories, all_results, html_output)

    print("\n병합 완료!")
    print(f"  HTML  : {html_output}")
    print(f"  RAGAS : {ragas_csv}")
    print(f"  레이턴시: {latency_csv}")


if __name__ == "__main__":
    main()
