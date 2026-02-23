"""
기존 벤치마크 결과 데이터로 QA 수별 소요 시간을 분석한다.

재실행 없이 다음 파일에서 데이터를 읽어 스케일링 테이블을 생성한다:
  - _benchdata/all_combos_latency.csv  → Pass 1 레이턴시
  - _benchdata/run_history/latest.json → 빌드 시간 + RAGAS 단계 시간
  - _benchdata/all_combos_ragas.csv    → RAGAS 평가 전략 수

Usage:
    python -m rag_bench.scripts.timing_report
    python -m rag_bench.scripts.timing_report --latency-csv path/to/latency.csv
    python -m rag_bench.scripts.timing_report --qa-range 10,20,50,100
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from rag_bench.config import BENCH_DATA_DIR


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------


def _load_latency_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[Warning] latency CSV 없음: {path}")
        return None
    df = pd.read_csv(path)
    required = {"strategy", "query", "latency_ms"}
    if not required.issubset(df.columns):
        print(f"[Warning] latency CSV 컬럼 불일치: {df.columns.tolist()}")
        return None
    return df


def _load_run_history(history_dir: Path) -> Optional[dict]:
    """최신 run_history JSON을 로드한다."""
    latest = history_dir / "latest.json"
    if latest.exists():
        try:
            return json.loads(latest.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Warning] latest.json 읽기 실패: {e}")

    # symlink가 없으면 가장 최신 파일 사용
    jsons = sorted(history_dir.glob("run_*.json"), reverse=True)
    if jsons:
        try:
            return json.loads(jsons[0].read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Warning] run_history JSON 읽기 실패: {e}")
    return None


def _load_ragas_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 분석
# ---------------------------------------------------------------------------


def analyze_timing(
    latency_df: pd.DataFrame,
    run_record: Optional[dict],
    ragas_df: Optional[pd.DataFrame],
) -> dict:
    """기존 데이터로 타이밍 분석 결과를 계산한다.

    Returns:
        dict with keys:
            n_queries, n_strategies, n_eval_strategies,
            pass1_rate_s         — QA 1개당 전체 Pass 1 시간 (전략 합산)
            pass1_per_strategy_s — QA 1개당 전략 평균 Pass 1 시간
            pass2_rate_s         — QA 1개당 전체 Pass 2 시간 (RAGAS 전략 합산)
            pass2_per_strategy_s — QA 1개당 RAGAS 전략 평균 Pass 2 시간
            total_build_s        — 전체 빌드 시간 (QA 무관, 고정)
            avg_build_s          — 전략 평균 빌드 시간
            layer_stats          — 레이어별 통계
            strategy_stats       — 전략별 pass1_per_qa, pass2_per_qa
    """
    valid = latency_df[latency_df["error"].isna()].copy()
    n_queries = latency_df["query"].nunique()
    n_strategies = latency_df["strategy"].nunique()

    # ── Pass 1: 전략별 평균 레이턴시 → QA당 시간 ──
    per_strat = (
        valid.groupby("strategy")["latency_ms"]
        .mean()
        .rename("avg_ms")
        .reset_index()
    )
    per_strat["pass1_per_qa_s"] = per_strat["avg_ms"] / 1000.0

    # 전체 Pass 1 rate = 모든 전략의 QA당 시간 합산 (순차 실행 기준)
    pass1_rate_s = per_strat["pass1_per_qa_s"].sum()

    # ── 빌드 시간: run_history에서 추출 ──
    total_build_s = 0.0
    build_by_label: Dict[str, float] = {}
    if run_record:
        for st in run_record.get("strategy_timings", []):
            label = st.get("label", "")
            build_s = st.get("build_time_s", 0.0)
            build_by_label[label] = build_s
            if st.get("build_success", True):
                total_build_s += build_s

    # ── Pass 2: run_history phase_times에서 pass2_ragas 추출 ──
    pass2_total_s = 0.0
    n_eval_strategies = 0
    if run_record:
        for pt in run_record.get("phase_times", []):
            if pt.get("phase") == "pass2_ragas":
                pass2_total_s = pt.get("duration_s", 0.0)
                break

    if ragas_df is not None:
        n_eval_strategies = len(ragas_df)

    # RAGAS phase 시간 전체 / n_queries = QA 1개당 전체 RAGAS 시간
    pass2_rate_s = pass2_total_s / n_queries if n_queries > 0 and pass2_total_s > 0 else 0.0

    # ── 레이어별 통계 ──
    # strategy 이름에서 레이어 정보를 추출하기 어려우므로 run_history 활용
    layer_stats: Dict[str, Dict[str, List[float]]] = {}
    if run_record:
        for st in run_record.get("strategy_timings", []):
            if not st.get("build_success", True):
                continue
            for layer_key, layer_name in [
                ("dense_model", "Dense Model"),
                ("sparse_model", "Sparse"),
                ("reranker", "Reranker"),
                ("llm_support", "LLM Support"),
            ]:
                val = st.get(layer_key) or "none"
                layer_stats.setdefault(layer_name, {}).setdefault(val, [])
                # latency CSV에서 해당 전략 찾기 (전략명이 다를 수 있으므로 라벨로 매칭 불가)
                # 대신 avg_latency_ms 사용
                if st.get("avg_latency_ms", 0) > 0:
                    layer_stats[layer_name][val].append(st["avg_latency_ms"] / 1000.0)

    return {
        "n_queries": n_queries,
        "n_strategies": n_strategies,
        "n_eval_strategies": n_eval_strategies,
        "pass1_rate_s": round(pass1_rate_s, 3),
        "pass1_per_strategy_s": round(pass1_rate_s / n_strategies, 3) if n_strategies else 0,
        "pass2_rate_s": round(pass2_rate_s, 3),
        "pass2_per_strategy_s": round(pass2_rate_s / n_eval_strategies, 3) if n_eval_strategies > 0 else 0,
        "total_build_s": round(total_build_s, 1),
        "avg_build_s": round(total_build_s / n_strategies, 1) if n_strategies else 0,
        "layer_stats": layer_stats,
        "strategy_stats": per_strat,
        "pass2_total_s": round(pass2_total_s, 1),
    }


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------


def _fmt_time(seconds: float) -> str:
    """초 → '1h 23m 45s' 형식."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m > 0:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def print_timing_report(stats: dict, qa_range: List[int]) -> None:
    """분석 결과를 콘솔에 출력한다."""
    n_q = stats["n_queries"]
    n_s = stats["n_strategies"]
    n_e = stats["n_eval_strategies"]
    p1_rate = stats["pass1_rate_s"]
    p2_rate = stats["pass2_rate_s"]
    build_s = stats["total_build_s"]

    # ── 현재 실행 요약 ──
    print(f"\n{'═' * 80}")
    print(" 현재 실행 기준 타이밍 요약")
    print(f"{'═' * 80}")
    print(f"  QA 수          : {n_q}개")
    print(f"  Pass 1 전략    : {n_s}개")
    print(f"  Pass 2 전략    : {n_e}개  (RAGAS 평가 대상)")
    print()
    print(f"  빌드 총시간    : {build_s:.0f}s  ({_fmt_time(build_s)})  ← QA 수와 무관")
    print(f"  QA 1개당 Pass1 : {p1_rate:.2f}s  ({n_s}개 전략 합산, 순차 기준)")
    print(f"    전략 평균    : {stats['pass1_per_strategy_s']:.3f}s / QA")
    if p2_rate > 0:
        print(f"  QA 1개당 Pass2 : {p2_rate:.2f}s  ({n_e}개 전략 RAGAS 합산)")
        print(f"    전략 평균    : {stats['pass2_per_strategy_s']:.3f}s / QA")
    else:
        print(f"  QA 1개당 Pass2 : (데이터 없음 — 현재 실행 진행 중이거나 Pass1-only 모드)")

    # ── QA 수별 스케일링 테이블 ──
    # 현재 QA 수가 범위에 없으면 삽입
    candidates = sorted(set(qa_range + [n_q]))

    print(f"\n{'═' * 80}")
    print(" QA 수별 예상 소요 시간")
    print(f"{'═' * 80}")
    print(f"  (빌드 {build_s:.0f}s 고정  +  Pass1 {p1_rate:.2f}s/QA × N  +  Pass2 {p2_rate:.2f}s/QA × N)")
    print()
    print(f"  {'QA':>5}  {'빌드':>8}  {'Pass1':>8}  {'Pass2':>8}  {'합계':>9}  {'(분)':>6}  {'비고'}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*6}  {'─'*6}")

    for n in candidates:
        p1 = p1_rate * n
        p2 = p2_rate * n
        total = build_s + p1 + p2
        note = "◀ 현재" if n == n_q else ""
        p2_str = f"{p2:>8.0f}" if p2_rate > 0 else f"{'N/A':>8}"
        print(f"  {n:>5}  {build_s:>8.0f}  {p1:>8.0f}  {p2_str}  {total:>9.0f}  {total/60:>6.1f}  {note}")

    # ── 전략별 QA당 소요 시간 (Pass 1) ──
    strat_df: pd.DataFrame = stats["strategy_stats"]
    print(f"\n{'═' * 80}")
    print(" 전략별 QA 1개 평균 소요 시간 (Pass 1 검색)")
    print(f"{'═' * 80}")
    print(f"  {'전략':<55} {'QA당(s)':>8}  {'QA당(ms)':>9}")
    print(f"  {'─'*55}  {'─'*8}  {'─'*9}")
    for _, row in strat_df.sort_values("pass1_per_qa_s", ascending=False).iterrows():
        print(
            f"  {row['strategy']:<55}"
            f" {row['pass1_per_qa_s']:>8.3f}"
            f"  {row['avg_ms']:>9.1f}"
        )

    # ── 레이어별 QA당 Pass 1 평균 ──
    layer_stats = stats.get("layer_stats", {})
    if layer_stats:
        print(f"\n{'═' * 80}")
        print(" 레이어별 QA 1개 평균 검색 시간 (run_history 기반)")
        print(f"{'═' * 80}")
        for layer_name, val_map in layer_stats.items():
            if not val_map:
                continue
            print(f"\n  ── {layer_name} ──")
            print(f"  {'값':<24} {'평균(s/QA)':>10}  {'조합 수':>7}")
            print(f"  {'─'*24}  {'─'*10}  {'─'*7}")
            for val, lats in sorted(val_map.items()):
                if not lats:
                    continue
                avg = sum(lats) / len(lats)
                print(f"  {str(val):<24}  {avg:>10.3f}  {len(lats):>7}")

    print()
    print("  * 빌드 시간은 인덱스 캐시 여부에 따라 크게 달라질 수 있습니다.")
    print("  * Pass 2 시간은 RAGAS 평가 전략 수 × QA 수에 비례합니다.")
    print("  * 순차 실행 기준이며, --pass1-workers 사용 시 Pass 1 시간이 단축됩니다.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="기존 벤치마크 결과에서 QA 수별 소요 시간 분석"
    )
    parser.add_argument(
        "--latency-csv", type=str, default=None,
        help=f"latency CSV 경로 (기본: {BENCH_DATA_DIR}/all_combos_latency.csv)",
    )
    parser.add_argument(
        "--ragas-csv", type=str, default=None,
        help=f"RAGAS CSV 경로 (기본: {BENCH_DATA_DIR}/all_combos_ragas.csv)",
    )
    parser.add_argument(
        "--qa-range", type=str, default="5,10,20,30,50,75,100,150,200",
        help="스케일링 테이블에 표시할 QA 수 (쉼표 구분, 기본: 5,10,20,30,50,75,100,150,200)",
    )
    args = parser.parse_args()

    latency_path = Path(args.latency_csv) if args.latency_csv else BENCH_DATA_DIR / "all_combos_latency.csv"
    ragas_path = Path(args.ragas_csv) if args.ragas_csv else BENCH_DATA_DIR / "all_combos_ragas.csv"
    history_dir = BENCH_DATA_DIR / "run_history"
    qa_range = [int(x.strip()) for x in args.qa_range.split(",") if x.strip().isdigit()]

    latency_df = _load_latency_csv(latency_path)
    if latency_df is None:
        print("latency CSV를 로드할 수 없습니다. 벤치마크를 먼저 실행하세요.")
        return

    run_record = _load_run_history(history_dir)
    ragas_df = _load_ragas_csv(ragas_path)

    stats = analyze_timing(latency_df, run_record, ragas_df)
    print_timing_report(stats, qa_range)


if __name__ == "__main__":
    main()
