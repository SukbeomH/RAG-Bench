"""
ranker — 카테고리별 조합 순위 계산.

RAGAS 가중치:
  context_recall     × 0.35  (누락이 오탐보다 치명적)
  context_precision  × 0.30
  faithfulness       × 0.20
  answer_relevancy   × 0.15

출력: Dict[category_name, pd.DataFrame]
  컬럼: strategy, faithfulness, answer_relevancy, context_precision,
        context_recall, composite, pass_rate, rank
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

RAGAS_WEIGHTS = {
    "context_recall": 0.35,
    "context_precision": 0.30,
    "faithfulness": 0.20,
    "answer_relevancy": 0.15,
}

RAGAS_COLS = list(RAGAS_WEIGHTS.keys())


def load_results(run_dir: str | Path) -> Dict[str, dict]:
    """
    run_dir 하위 각 카테고리의 result.json 을 로드한다.

    Returns:
        Dict[category_name, raw_result_dict]
    """
    run_dir = Path(run_dir)
    results: Dict[str, dict] = {}

    for cat_dir in sorted(run_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        result_file = cat_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            category = data.get("category", cat_dir.name)
            if data.get("ragas"):
                results[category] = data
        except Exception as e:
            print(f"[ranker] {result_file} 로드 실패: {e}")

    return results


def _compute_composite(row: pd.Series) -> float:
    """RAGAS 가중 평균 점수 계산."""
    score = 0.0
    for col, w in RAGAS_WEIGHTS.items():
        score += row.get(col, 0.0) * w
    return round(score, 4)


def _compute_pass_rate(row: pd.Series) -> float:
    """context_recall > 0.5 기반 pass_rate (이미 집계된 평균값으로 계산)."""
    # 집계된 평균값 기준 (개별 샘플 없이)
    recall = row.get("context_recall", 0.0)
    # 평균 recall이 0.5 이상이면 pass_rate=100, 아니면 스케일 추정
    return round(min(recall * 100, 100.0), 1)


def rank_by_doc_type(
    raw_results: Dict[str, dict],
    latency_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    카테고리별 RAGAS 가중 복합 점수로 조합 순위를 반환한다.

    Args:
        raw_results: load_results() 반환값
        latency_dir: run_dir (latency.csv 로드용, 선택)

    Returns:
        Dict[category_name, DataFrame]
        컬럼: strategy, faithfulness, answer_relevancy, context_precision,
              context_recall, composite, pass_rate, avg_latency_ms, rank
    """
    ranked: Dict[str, pd.DataFrame] = {}

    for category, data in raw_results.items():
        ragas_records: List[dict] = data.get("ragas", [])
        if not ragas_records:
            continue

        df = pd.DataFrame(ragas_records)

        # RAGAS 컬럼 보정 (없으면 0 처리)
        for col in RAGAS_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0).clip(0.0, 1.0)

        # 복합 점수 + pass_rate
        df["composite"] = df.apply(_compute_composite, axis=1)
        df["pass_rate"] = df.apply(_compute_pass_rate, axis=1)

        # 레이턴시 로드 (있으면)
        if latency_dir is not None:
            lat_file = latency_dir / category / "latency.csv"
            if lat_file.exists():
                try:
                    lat_df = pd.read_csv(lat_file, encoding="utf-8-sig")
                    if "strategy" in lat_df.columns and "latency_ms" in lat_df.columns:
                        avg_lat = lat_df.groupby("strategy")["latency_ms"].mean().rename("avg_latency_ms")
                        df = df.merge(avg_lat, on="strategy", how="left")
                except Exception:
                    pass

        if "avg_latency_ms" not in df.columns:
            df["avg_latency_ms"] = float("nan")

        # 순위 (composite 내림차순, 동점이면 latency 오름차순)
        df = df.sort_values(
            ["composite", "avg_latency_ms"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        ranked[category] = df

    return ranked
