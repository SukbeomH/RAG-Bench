"""
colab_visualizer — 벤치마크 결과 시각화 유틸리티.

matplotlib, plotly, seaborn으로 차트를 생성한다.
Colab 노트북에서 inline 렌더링을 기본으로 한다.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 전략 그룹 색상 맵 (모듈 레벨 — H-2, H-4, M-4에서 공유)
# ---------------------------------------------------------------------------

_COLOR_MAP = {
    "Baseline":              "#3498db",
    "+ Reranker":            "#e74c3c",
    "+ Contextual":          "#e67e22",
    "Reranker + Contextual": "#9b59b6",
}


def _classify_group(strategy_name: str) -> str:
    """전략명에서 레이어 그룹을 분류한다."""
    has_ctx = "Contextual" in strategy_name
    has_rerank = "FlashRank" in strategy_name or "ColBERT" in strategy_name
    if has_ctx and has_rerank:
        return "Reranker + Contextual"
    if has_ctx:
        return "+ Contextual"
    if has_rerank:
        return "+ Reranker"
    return "Baseline"


# ---------------------------------------------------------------------------
# 1. 레이턴시 비교 (수평 막대)
# ---------------------------------------------------------------------------


def plot_latency_comparison(df: pd.DataFrame, top_n: int = 20) -> None:
    """전략별 평균 레이턴시 수평 막대 차트.

    Args:
        df: 'strategy', 'avg_latency' (또는 'avg_latency_ms') 컬럼 필요.
        top_n: 상위 N개만 표시.
    """
    import matplotlib.pyplot as plt

    df = df.copy()
    if "avg_latency" not in df.columns and "avg_latency_ms" in df.columns:
        df["avg_latency"] = df["avg_latency_ms"] / 1000

    df = df.nsmallest(top_n, "avg_latency")
    df = df.sort_values("avg_latency", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
    ax.barh(df["strategy"], df["avg_latency"], color=colors)
    ax.set_xlabel("Average Latency (s)")
    ax.set_title(f"Pass 1: Strategy Latency (Top {top_n})")

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["avg_latency"] + 0.002, i, f"{row['avg_latency']:.3f}s",
                va="center", fontsize=8)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. RAGAS 레이더 차트
# ---------------------------------------------------------------------------


def plot_ragas_radar(df: pd.DataFrame, top_n: int = 5) -> None:
    """RAGAS 메트릭 레이더/스파이더 차트 (plotly).

    Args:
        df: 'strategy' + RAGAS 메트릭 컬럼 필요.
        top_n: 상위 N개 전략 표시.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly가 필요합니다: pip install plotly")
        return

    metric_cols = [c for c in df.columns if c != "strategy" and df[c].dtype in ("float64", "float32")]
    if not metric_cols:
        print("RAGAS 메트릭 컬럼이 없습니다.")
        return

    df_top = df.head(top_n)
    fig = go.Figure()

    for _, row in df_top.iterrows():
        values = [row[c] for c in metric_cols]
        values.append(values[0])  # 닫기
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_cols + [metric_cols[0]],
            fill="toself",
            name=row["strategy"],
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f"RAGAS Metrics Radar (Top {top_n})",
        height=500,
        font=dict(family="NanumGothic, sans-serif"),
    )
    fig.show()


# ---------------------------------------------------------------------------
# 3. RAGAS 히트맵
# ---------------------------------------------------------------------------


def plot_ragas_heatmap(df: pd.DataFrame) -> None:
    """전략 x 메트릭 히트맵 (seaborn).

    Args:
        df: 'strategy' + RAGAS 메트릭 컬럼 필요.
    """
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn이 필요합니다: pip install seaborn")
        return

    metric_cols = [c for c in df.columns if c != "strategy" and df[c].dtype in ("float64", "float32")]
    if not metric_cols:
        print("RAGAS 메트릭 컬럼이 없습니다.")
        return

    heatmap_data = df.set_index("strategy")[metric_cols]

    fig, ax = plt.subplots(figsize=(10, max(4, len(heatmap_data) * 0.5)))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("RAGAS Metrics Heatmap")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. 레이턴시 vs 품질 산점도 + 파레토 프론티어
# ---------------------------------------------------------------------------


def plot_latency_vs_quality(
    lat_df: pd.DataFrame,
    ragas_df: pd.DataFrame,
    quality_metric: str = "faithfulness",
) -> None:
    """레이턴시 vs 품질 산점도 + 파레토 프론티어.

    Args:
        lat_df: 레이턴시 DataFrame ('strategy', 'avg_latency').
        ragas_df: RAGAS DataFrame ('strategy', quality_metric).
        quality_metric: Y축 메트릭.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly가 필요합니다: pip install plotly")
        return

    # quality_metric이 없으면 사용 가능한 수치 컬럼 자동 선택
    available_cols = [c for c in ragas_df.columns if c != "strategy" and pd.api.types.is_numeric_dtype(ragas_df[c])]
    if quality_metric not in ragas_df.columns:
        if not available_cols:
            print("RAGAS DataFrame에 수치 메트릭 컬럼이 없습니다.")
            return
        quality_metric = available_cols[0]
        print(f"[Info] quality_metric → '{quality_metric}' (자동 선택)")

    merged = lat_df.merge(ragas_df[["strategy", quality_metric]], on="strategy", how="inner")

    if merged.empty:
        print("레이턴시-품질 매칭 데이터가 없습니다.")
        return

    if "avg_latency" not in merged.columns and "avg_latency_ms" in merged.columns:
        merged["avg_latency"] = merged["avg_latency_ms"] / 1000

    fig = go.Figure()

    # 산점도
    fig.add_trace(go.Scatter(
        x=merged["avg_latency"],
        y=merged[quality_metric],
        mode="markers+text",
        text=merged["strategy"],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(size=12, color="steelblue"),
        name="Strategies",
    ))

    # 파레토 프론티어 계산
    pareto = _compute_pareto_front(
        merged[["avg_latency", quality_metric]].values,
        minimize_x=True,
        maximize_y=True,
    )
    if len(pareto) > 1:
        pareto_sorted = pareto[pareto[:, 0].argsort()]
        fig.add_trace(go.Scatter(
            x=pareto_sorted[:, 0],
            y=pareto_sorted[:, 1],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Pareto Frontier",
        ))

    fig.update_layout(
        xaxis_title="Average Latency (s)",
        yaxis_title=quality_metric,
        title=f"Latency vs {quality_metric} (Pareto Frontier)",
        height=500,
        font=dict(family="NanumGothic, sans-serif"),
    )
    fig.show()


def _compute_pareto_front(
    points: np.ndarray,
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> np.ndarray:
    """2D 파레토 프론티어 계산."""
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_better = (points[j, 0] <= points[i, 0]) if minimize_x else (points[j, 0] >= points[i, 0])
            y_better = (points[j, 1] >= points[i, 1]) if maximize_y else (points[j, 1] <= points[i, 1])
            x_strict = (points[j, 0] < points[i, 0]) if minimize_x else (points[j, 0] > points[i, 0])
            y_strict = (points[j, 1] > points[i, 1]) if maximize_y else (points[j, 1] < points[i, 1])
            if x_better and y_better and (x_strict or y_strict):
                is_pareto[i] = False
                break
    return points[is_pareto]


# ---------------------------------------------------------------------------
# 5. 레이어별 기여도 박스플롯
# ---------------------------------------------------------------------------


def plot_layer_contribution(
    combos: list,
    metric_df: pd.DataFrame,
    metric: str = "avg_latency",
) -> None:
    """레이어별 기여도 박스플롯.

    Args:
        combos: ComboSpec 목록 (spec, strategy) 튜플 또는 ComboSpec 목록.
        metric_df: 전략별 메트릭 DataFrame.
        metric: 분석할 메트릭 컬럼명.
    """
    import matplotlib.pyplot as plt

    if not combos:
        print("조합 목록이 비어 있습니다.")
        return

    # ComboSpec 추출
    specs = []
    for item in combos:
        if isinstance(item, tuple):
            specs.append(item[0])
        else:
            specs.append(item)

    layers = {
        "Dense Model": lambda s: s.dense,
        "Sparse Model": lambda s: s.sparse,
        "Reranker": lambda s: s.reranker or "none",
        "LLM Support": lambda s: s.llm_support or "none",
    }

    fig, axes = plt.subplots(1, len(layers), figsize=(16, 5))

    for ax, (layer_name, get_val) in zip(axes, layers.items()):
        groups: Dict[str, List[float]] = {}
        for spec in specs:
            val = get_val(spec)
            strategy_name = spec.label
            mask = metric_df["strategy"].str.contains(strategy_name, regex=False)
            if mask.any():
                metric_val = metric_df.loc[mask, metric].values[0]
                groups.setdefault(val, []).append(metric_val)

        if groups:
            labels = sorted(groups.keys())
            data = [groups[lbl] for lbl in labels]
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(labels)))):
                patch.set_facecolor(color)

        ax.set_title(layer_name)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle(f"Layer Contribution Analysis ({metric})", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5-H1. Ablation Waterfall — 레이어별 품질 기여도
# ---------------------------------------------------------------------------


def plot_ablation_waterfall(
    ragas_df: pd.DataFrame,
    metric: Optional[str] = None,
) -> None:
    """레이어 추가 시 품질 기여도 폭포 차트 (Ablation Waterfall).

    Baseline(Dense+Sparse) 대비 Reranker / Contextual 추가가
    품질 점수에 미치는 평균 기여도를 시각화한다.

    Args:
        ragas_df: 'strategy' + RAGAS 메트릭 컬럼 필요.
        metric: 분석할 메트릭. None이면 weighted_ 컬럼 또는 첫 수치 컬럼 자동 선택.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly가 필요합니다: pip install plotly")
        return

    if ragas_df is None or ragas_df.empty:
        print("RAGAS 데이터가 없습니다.")
        return

    numeric_cols = [
        c for c in ragas_df.columns
        if c != "strategy" and pd.api.types.is_numeric_dtype(ragas_df[c])
    ]
    if not numeric_cols:
        print("수치 메트릭 컬럼이 없습니다.")
        return

    if metric is None:
        weighted = [c for c in numeric_cols if c.startswith("weighted_")]
        metric = weighted[0] if weighted else numeric_cols[0]

    if metric not in ragas_df.columns:
        print(f"메트릭 '{metric}'이 없습니다. 사용 가능: {numeric_cols}")
        return

    # 전략명 패턴으로 레이어 분류
    def _classify(name: str):
        has_ctx = "Contextual" in name
        has_rerank = "FlashRank" in name or "ColBERT" in name
        return has_ctx, has_rerank

    groups: Dict[str, List[float]] = {
        "baseline": [], "reranker": [], "contextual": [], "both": []
    }
    for _, row in ragas_df.iterrows():
        score = row[metric]
        if pd.isna(score):
            continue
        has_ctx, has_rerank = _classify(str(row["strategy"]))
        if has_ctx and has_rerank:
            groups["both"].append(float(score))
        elif has_ctx:
            groups["contextual"].append(float(score))
        elif has_rerank:
            groups["reranker"].append(float(score))
        else:
            groups["baseline"].append(float(score))

    means: Dict[str, Optional[float]] = {
        k: float(np.mean(v)) if v else None for k, v in groups.items()
    }

    if means["baseline"] is None:
        print("Baseline 전략(Reranker/Contextual 없음)이 평가되지 않았습니다.")
        return

    base = means["baseline"]
    x_labels: List[str] = ["Baseline"]
    y_values: List[float] = [base]
    measures: List[str] = ["absolute"]
    text_vals: List[str] = [f"{base:.3f}"]

    # 누적값 추적
    cumulative = base
    if means["reranker"] is not None:
        delta = means["reranker"] - base
        x_labels.append("+ Reranker")
        y_values.append(delta)
        measures.append("relative")
        text_vals.append(f"{delta:+.3f}")
        cumulative += delta

    if means["contextual"] is not None:
        delta = means["contextual"] - base
        x_labels.append("+ Contextual")
        y_values.append(delta)
        measures.append("relative")
        text_vals.append(f"{delta:+.3f}")
        cumulative = base + delta  # contextual은 독립 경로

    if means["both"] is not None:
        # reranker + contextual 조합 → 누적 대비 추가 효과
        ref = base + (means["reranker"] - base if means["reranker"] else 0) \
                   + (means["contextual"] - base if means["contextual"] else 0)
        delta = means["both"] - ref
        x_labels.append("Reranker + Contextual")
        y_values.append(delta)
        measures.append("relative")
        text_vals.append(f"{delta:+.3f}")

    # 총합 bar
    final = sum(
        v for v, m in zip(y_values, measures) if m == "relative"
    ) + base
    x_labels.append("최종 (합산)")
    y_values.append(0)
    measures.append("total")
    text_vals.append(f"{final:.3f}")

    fig = go.Figure(go.Waterfall(
        name="품질 기여도",
        orientation="v",
        measure=measures,
        x=x_labels,
        y=y_values,
        text=text_vals,
        textposition="outside",
        connector={"line": {"color": "rgb(63,63,63)", "width": 1}},
        increasing={"marker": {"color": "#2ecc71"}},
        decreasing={"marker": {"color": "#e74c3c"}},
        totals={"marker": {"color": "#3498db"}},
    ))
    fig.update_layout(
        title=f"레이어별 품질 기여도 — Ablation Waterfall ({metric})",
        yaxis_title=f"{metric} 점수",
        yaxis=dict(range=[0, 1.15]),
        height=450,
        font=dict(family="NanumGothic, sans-serif"),
        showlegend=False,
    )
    fig.show()


# ---------------------------------------------------------------------------
# 5-H3. Layer Interaction Heatmap — Dense × Sparse 조합 상호작용
# ---------------------------------------------------------------------------


def plot_layer_interaction_heatmap(
    ragas_df: pd.DataFrame,
    metric: Optional[str] = None,
) -> None:
    """Dense × Sparse 조합별 품질 상호작용 히트맵.

    Reranker / Contextual 유무에 따라 서브플롯을 분리하여
    어떤 Dense+Sparse 조합이 각 조건에서 최적인지 시각화한다.

    Args:
        ragas_df: 'strategy' + RAGAS 메트릭 컬럼 필요.
        metric: 분석할 메트릭. None이면 자동 선택.
    """
    import re

    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ImportError:
        print("seaborn이 필요합니다: pip install seaborn")
        return

    if ragas_df is None or ragas_df.empty:
        print("RAGAS 데이터가 없습니다.")
        return

    numeric_cols = [
        c for c in ragas_df.columns
        if c != "strategy" and pd.api.types.is_numeric_dtype(ragas_df[c])
    ]
    if not numeric_cols:
        print("수치 메트릭 컬럼이 없습니다.")
        return

    if metric is None:
        weighted = [c for c in numeric_cols if c.startswith("weighted_")]
        metric = weighted[0] if weighted else numeric_cols[0]

    if metric not in ragas_df.columns:
        print(f"메트릭 '{metric}'이 없습니다.")
        return

    # 전략명에서 Dense / Sparse 추출
    # 패턴: DS({dense}+{sparse}) — 래퍼에 중첩될 수 있음
    def _parse(name: str):
        m = re.search(r"DS\(([^+]+)\+([^)]+)\)", name)
        if not m:
            return None
        return {
            "dense": m.group(1).strip(),
            "sparse": m.group(2).strip(),
            "has_rerank": "FlashRank" in name or "ColBERT" in name,
            "has_ctx": "Contextual" in name,
            metric: None,
        }

    parsed_rows = []
    for _, row in ragas_df.iterrows():
        info = _parse(str(row["strategy"]))
        if info is None:
            continue
        val = row[metric]
        if pd.isna(val):
            continue
        info[metric] = float(val)
        parsed_rows.append(info)

    if not parsed_rows:
        print("전략명에서 Dense/Sparse 정보를 추출할 수 없습니다.")
        return

    df_p = pd.DataFrame(parsed_rows)

    group_defs = [
        ("Baseline\n(No Reranker, No Contextual)",
         (~df_p["has_rerank"]) & (~df_p["has_ctx"])),
        ("+ Reranker",
         df_p["has_rerank"] & (~df_p["has_ctx"])),
        ("+ Contextual",
         (~df_p["has_rerank"]) & df_p["has_ctx"]),
        ("Reranker + Contextual",
         df_p["has_rerank"] & df_p["has_ctx"]),
    ]
    groups = [(label, df_p[mask]) for label, mask in group_defs if mask.any()]

    if not groups:
        print("히트맵을 그릴 데이터가 없습니다.")
        return

    n = len(groups)
    n_dense = df_p["dense"].nunique()
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, max(3, n_dense * 0.7 + 2.5)))
    if n == 1:
        axes = [axes]

    for ax, (label, gdf) in zip(axes, groups):
        pivot = gdf.pivot_table(
            index="dense", columns="sparse", values=metric, aggfunc="mean"
        )
        if pivot.empty:
            ax.set_visible(False)
            continue
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Sparse Model")
        ax.set_ylabel("Dense Model")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(f"Dense × Sparse 조합별 상호작용 ({metric})", fontsize=13, y=1.03)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5-H4. Tradeoff Bubble Chart — 레이턴시 × 품질 × 비용
# ---------------------------------------------------------------------------


def plot_tradeoff_bubble(
    latency_df: pd.DataFrame,
    ragas_df: pd.DataFrame,
    run_record: Optional[dict] = None,
    metric: Optional[str] = None,
) -> None:
    """레이턴시 × 품질 × 비용 3차원 버블 차트.

    x=레이턴시, y=품질, 버블 크기=인덱싱 비용(토큰),
    색상=레이어 구성(Baseline/Reranker/Contextual/Both).
    파레토 프론티어(점선)도 함께 표시한다.

    Args:
        latency_df: 'strategy', 'avg_latency' 컬럼 필요.
        ragas_df: 'strategy' + RAGAS 메트릭 컬럼 필요.
        run_record: run_history dict. 토큰 비용 추출에 사용 (없으면 균일 크기).
        metric: 품질 메트릭. None이면 자동 선택.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly가 필요합니다: pip install plotly")
        return

    if latency_df is None or ragas_df is None:
        print("latency_df와 ragas_df가 모두 필요합니다.")
        return

    lat_df = latency_df.copy()
    if "avg_latency" not in lat_df.columns and "avg_latency_ms" in lat_df.columns:
        lat_df["avg_latency"] = lat_df["avg_latency_ms"] / 1000

    numeric_cols = [
        c for c in ragas_df.columns
        if c != "strategy" and pd.api.types.is_numeric_dtype(ragas_df[c])
    ]
    if not numeric_cols:
        print("수치 메트릭 컬럼이 없습니다.")
        return

    if metric is None:
        weighted = [c for c in numeric_cols if c.startswith("weighted_")]
        metric = weighted[0] if weighted else numeric_cols[0]

    if metric not in ragas_df.columns:
        print(f"메트릭 '{metric}'이 없습니다.")
        return

    merged = lat_df.merge(ragas_df[["strategy", metric]], on="strategy", how="inner")
    if merged.empty:
        print("레이턴시-품질 매칭 데이터가 없습니다.")
        return

    # run_record에서 전략별 인덱싱 비용 추출
    cost_map: Dict[str, float] = {}
    if run_record:
        for st in run_record.get("strategy_timings", []):
            lbl = st.get("label", "")
            idx_tok = st.get("indexing_tokens") or {}
            cost_map[lbl] = float(idx_tok.get("total_cost_usd", 0.0) or 0.0)

    def _get_cost(name: str) -> float:
        # label 부분 매칭
        for lbl, cost in cost_map.items():
            if lbl and (lbl in name or name in lbl):
                return cost
        return 0.0

    merged["cost"] = merged["strategy"].apply(_get_cost)
    max_cost = merged["cost"].max()
    merged["bubble_size"] = (
        (merged["cost"] / max_cost * 28 + 10) if max_cost > 0 else 18
    ).clip(10, 38)

    # 레이어 구성으로 색상 분류
    merged["group"] = merged["strategy"].apply(_classify_group)

    fig = go.Figure()

    for group_name, color in _COLOR_MAP.items():
        gdf = merged[merged["group"] == group_name]
        if gdf.empty:
            continue
        hover = [
            f"<b>{r['strategy']}</b><br>"
            f"레이턴시: {r['avg_latency']:.3f}s<br>"
            f"{metric}: {r[metric]:.3f}<br>"
            f"인덱싱 비용: ${r['cost']:.4f}"
            for _, r in gdf.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=gdf["avg_latency"],
            y=gdf[metric],
            mode="markers",
            name=group_name,
            hovertext=hover,
            hoverinfo="text",
            marker=dict(
                size=gdf["bubble_size"],
                color=color,
                opacity=0.78,
                line=dict(color="white", width=1),
            ),
        ))

    # 파레토 프론티어
    if len(merged) > 1:
        pareto = _compute_pareto_front(
            merged[["avg_latency", metric]].values,
            minimize_x=True, maximize_y=True,
        )
        if len(pareto) > 1:
            ps = pareto[pareto[:, 0].argsort()]
            fig.add_trace(go.Scatter(
                x=ps[:, 0], y=ps[:, 1],
                mode="lines",
                line=dict(color="gray", dash="dot", width=1.5),
                name="파레토 프론티어",
            ))

    cost_note = " (버블 크기 = 인덱싱 비용)" if max_cost > 0 else ""
    fig.update_layout(
        title=f"레이턴시 × 품질 × 비용 비교{cost_note}",
        xaxis_title="평균 레이턴시 (s)",
        yaxis_title=metric,
        height=520,
        font=dict(family="NanumGothic, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()


# ---------------------------------------------------------------------------
# 6. 비용 파이 차트
# ---------------------------------------------------------------------------


def plot_cost_breakdown(cost_data: Dict[str, float]) -> None:
    """API 비용 파이 차트.

    Args:
        cost_data: {'category': cost_usd} dict.
    """
    import matplotlib.pyplot as plt

    if not cost_data:
        print("비용 데이터가 없습니다.")
        return

    labels = list(cost_data.keys())
    values = list(cost_data.values())
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90,
    )
    ax.set_title(f"API Cost Breakdown (Total: ${total:.2f})")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 7. 리더보드 테이블
# ---------------------------------------------------------------------------


def create_summary_table(
    lat_df: Optional[pd.DataFrame] = None,
    ragas_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """레이턴시 + RAGAS를 결합한 리더보드 테이블.

    Returns:
        스타일링된 DataFrame.
    """
    if lat_df is None and ragas_df is None:
        return pd.DataFrame()

    if lat_df is not None and ragas_df is not None:
        if "avg_latency" not in lat_df.columns and "avg_latency_ms" in lat_df.columns:
            lat_df = lat_df.copy()
            lat_df["avg_latency"] = lat_df["avg_latency_ms"] / 1000
        merged = lat_df[["strategy", "avg_latency"]].merge(
            ragas_df, on="strategy", how="outer"
        )
    elif ragas_df is not None:
        merged = ragas_df.copy()
    else:
        assert lat_df is not None
        merged = lat_df.copy()

    return merged


def display_styled_table(df: pd.DataFrame) -> None:
    """DataFrame을 Colab에서 스타일링하여 표시."""
    try:
        from IPython.display import display

        metric_cols = [c for c in df.columns if c != "strategy" and df[c].dtype in ("float64", "float32")]

        def _highlight_best(s):
            if s.name in metric_cols:
                if s.name == "avg_latency":
                    is_best = s == s.min()
                else:
                    is_best = s == s.max()
                return ["background-color: #d4edda" if v else "" for v in is_best]
            return [""] * len(s)

        styled = df.style.apply(_highlight_best).format(
            {c: "{:.4f}" for c in metric_cols}
        )
        display(styled)

    except ImportError:
        print(df.to_string(index=False))


def display_weighted_scores(
    reports: dict,
    scoring_profile: str = "balanced",
) -> Optional[pd.DataFrame]:
    """EvaluationReport의 가중 점수를 프로파일별로 표시.

    Args:
        reports: {strategy_name: EvaluationReport} dict.
        scoring_profile: 하이라이트할 프로파일.

    Returns:
        가중 점수 DataFrame.
    """
    if not reports:
        print("EvaluationReport가 없습니다.")
        return None

    try:
        from rag_bench.evaluation.evaluator import SCORING_PROFILES
    except ImportError:
        print("rag_bench.evaluation.evaluator를 import할 수 없습니다.")
        return None

    profile_names = list(SCORING_PROFILES.keys())
    rows = []
    for name, report in reports.items():
        ws = report.weighted_score
        row = {"strategy": name}
        for p in profile_names:
            row[p] = ws.get(p, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # scoring_profile 기준 내림차순 정렬
    if scoring_profile in df.columns:
        df = df.sort_values(scoring_profile, ascending=False).reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "rank"

    try:
        from IPython.display import display

        def _highlight_best(s):
            if s.name in profile_names:
                is_best = s == s.max()
                return ["background-color: #d4edda; font-weight: bold" if v else "" for v in is_best]
            return [""] * len(s)

        styled = df.style.apply(_highlight_best).format(
            {c: "{:.4f}" for c in profile_names}
        )
        display(styled)
    except ImportError:
        print(df.to_string())

    return df


# ---------------------------------------------------------------------------
# 8. 수행 이력 — 플랫폼 정보 카드
# ---------------------------------------------------------------------------


def plot_run_info(run_record: dict) -> None:
    """수행 이력의 플랫폼 정보와 설정을 요약 카드로 표시.

    Args:
        run_record: run_history JSON 로드 결과.
    """
    import matplotlib.pyplot as plt

    pf = run_record.get("platform_info", {})
    chip = pf.get("apple_chip", pf.get("processor", "N/A"))
    gpu = pf.get("gpu") or "None"
    ram = f"{pf.get('ram_total_gb', '?')} GB"
    cpu_cores = pf.get("cpu_count_logical", "?")

    rows = [
        ("Run ID", run_record.get("run_id", "")),
        ("Preset", run_record.get("preset", "")),
        ("Duration", f"{run_record.get('duration_s', 0):.1f}s"),
        ("Combos", str(run_record.get("num_combos", 0))),
        ("Queries", str(run_record.get("num_queries", 0))),
        ("Docs (chunks)", str(run_record.get("num_docs", 0))),
        ("Platform", f"{pf.get('os', '')} {pf.get('os_release', '')}"),
        ("Chip / CPU", f"{chip} ({cpu_cores} cores)"),
        ("RAM", ram),
        ("GPU", gpu),
        ("Python", pf.get("python_version", "")),
        ("Git Commit", pf.get("git_commit", "")),
    ]

    # 단계별 소요 시간 비중
    phases = run_record.get("phase_times", [])
    total_s = run_record.get("duration_s", 0) or 1
    if phases:
        rows.append(("", ""))  # 구분선
        rows.append(("Phase Breakdown", "시간 / 비중"))
        for p in phases:
            dur = p.get("duration_s", 0)
            if dur <= 0:
                continue
            pct = dur / total_s * 100
            label = _PHASE_LABELS.get(p["phase"], p["phase"])
            tok_str = ""
            t = p.get("tokens")
            if t and t.get("total_tokens", 0) > 0:
                tok_str = f"  [{t['total_tokens']:,} tok]"
            rows.append((f"  {label}", f"{dur:.1f}s ({pct:.1f}%){tok_str}"))

    token = run_record.get("token_usage_total", {})
    if token.get("total_tokens", 0) > 0:
        rows.append(("", ""))  # 구분선
        rows.append(("Total Tokens", f"{token['total_tokens']:,}"))
        rows.append(("  prompt / completion",
                      f"{token.get('prompt_tokens', 0):,} / {token.get('completion_tokens', 0):,}"))
        rows.append(("  API Cost", f"${token.get('total_cost_usd', 0):.4f}"))
        rows.append(("  LLM Calls", str(token.get("num_calls", 0))))

    fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.3)))
    ax.axis("off")

    table = ax.table(
        cellText=[[k, v] for k, v in rows],
        colLabels=["항목", "값"],
        cellLoc="left",
        loc="center",
        colWidths=[0.35, 0.65],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    # 헤더 스타일
    for j in range(2):
        table[0, j].set_facecolor("#4C78A8")
        table[0, j].set_text_props(color="white", weight="bold")

    # 행 스타일링
    for i, (k, _) in enumerate(rows, 1):
        if "Token" in k or "Cost" in k or "LLM Calls" in k:
            for j in range(2):
                table[i, j].set_facecolor("#FFF3CD")
        elif k == "Phase Breakdown":
            for j in range(2):
                table[i, j].set_facecolor("#D6EAF8")
                table[i, j].set_text_props(weight="bold")
        elif k.startswith("  ") and "%" in str(rows[i - 1][1] if i > 0 else ""):
            # phase 하위 행 — 연한 파랑
            for j in range(2):
                table[i, j].set_facecolor("#EBF5FB")

    ax.set_title("Benchmark Run Summary", fontsize=14, weight="bold", pad=20)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 9. 수행 이력 — 단계별 소요 시간 (가로 막대)
# ---------------------------------------------------------------------------

_PHASE_LABELS = {
    "qa_dataset_load": "QA 로드",
    "chunking": "문서 청킹",
    "qa_chunking": "QA 청킹",
    "qa_generation": "QA 생성 (LLM)",
    "strategy_build_and_indexing": "전략 빌드 & 인덱싱",
    "pass1_latency": "Pass 1: 레이턴시",
    "pass2_ragas": "Pass 2: RAGAS 평가",
    "ragas_evaluation_tokens": "RAGAS 토큰",
}


def plot_phase_timeline(run_record: dict) -> None:
    """단계별 소요 시간 가로 누적 막대.

    Args:
        run_record: run_history JSON 로드 결과.
    """
    import matplotlib.pyplot as plt

    phases = run_record.get("phase_times", [])
    if not phases:
        print("단계별 시간 데이터가 없습니다.")
        return

    # 0초인 메타 단계 제외
    phases = [p for p in phases if p.get("duration_s", 0) > 0]
    if not phases:
        return

    labels = [_PHASE_LABELS.get(p["phase"], p["phase"]) for p in phases]
    durations = [p["duration_s"] for p in phases]
    total = sum(durations)

    # 토큰 정보 수집
    token_labels = []
    for p in phases:
        t = p.get("tokens")
        if t and t.get("total_tokens", 0) > 0:
            token_labels.append(f"{t['total_tokens']:,} tok")
        else:
            token_labels.append("")

    fig, ax = plt.subplots(figsize=(12, max(2.5, len(phases) * 0.5)))
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(phases)))
    bars = ax.barh(labels, durations, color=colors, edgecolor="white", linewidth=0.5)

    for bar, dur, tok in zip(bars, durations, token_labels):
        pct = dur / total * 100 if total > 0 else 0
        text = f"{dur:.1f}s ({pct:.0f}%)"
        if tok:
            text += f"  [{tok}]"
        ax.text(bar.get_width() + total * 0.01, bar.get_y() + bar.get_height() / 2,
                text, va="center", fontsize=9)

    ax.set_xlabel("소요 시간 (s)")
    ax.set_title(f"Phase Timeline (total: {total:.1f}s)", fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 10. 수행 이력 — 전략별 빌드(인덱싱) 시간 + 토큰
# ---------------------------------------------------------------------------


def plot_build_times(run_record: dict, top_n: int = 30) -> None:
    """전략별 빌드(인덱싱) 시간 수평 막대. LLM 토큰 사용 전략을 강조.

    Args:
        run_record: run_history JSON 로드 결과.
        top_n: 표시할 전략 수.
    """
    import matplotlib.pyplot as plt

    timings = run_record.get("strategy_timings", [])
    if not timings:
        print("전략 타이밍 데이터가 없습니다.")
        return

    # 성공한 전략만, 빌드 시간 내림차순
    success = [t for t in timings if t.get("build_success", True)]
    success.sort(key=lambda t: t.get("build_time_s", 0), reverse=True)
    success = success[:top_n]

    labels = [t["label"] for t in success]
    build_times = [t.get("build_time_s", 0) for t in success]

    # LLM 토큰 사용 여부로 색상 분류
    colors = []
    for t in success:
        idx_tok = t.get("indexing_tokens")
        if idx_tok and idx_tok.get("total_tokens", 0) > 0:
            colors.append("#E45756")  # 빨강: LLM 사용
        else:
            colors.append("#4C78A8")  # 파랑: LLM 미사용

    total_duration = run_record.get("duration_s", 0) or 1
    total_build = sum(build_times) or 1

    fig, ax = plt.subplots(figsize=(10, max(4, len(success) * 0.35)))
    bars = ax.barh(labels, build_times, color=colors, edgecolor="white", linewidth=0.5)

    for bar, t in zip(bars, success):
        bt = t.get("build_time_s", 0)
        pct_build = bt / total_build * 100
        pct_total = bt / total_duration * 100
        text = f"{bt:.1f}s ({pct_build:.0f}% build, {pct_total:.1f}% total)"
        idx_tok = t.get("indexing_tokens")
        if idx_tok and idx_tok.get("total_tokens", 0) > 0:
            text += f"  [{idx_tok['total_tokens']:,} tok, ${idx_tok.get('total_cost_usd', 0):.4f}]"
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                text, va="center", fontsize=8)

    ax.set_xlabel("빌드 시간 (s)")
    ax.set_title(f"Strategy Build Time (Top {top_n})", fontsize=13)
    ax.invert_yaxis()

    # 범례
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color="#4C78A8", label="Embedding Only"),
        Patch(color="#E45756", label="+ LLM (Contextual)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 11. 수행 이력 — 토큰 사용량 파이 차트 (단계별)
# ---------------------------------------------------------------------------


def plot_token_usage(run_record: dict) -> None:
    """단계별 LLM 토큰 사용량 파이 + prompt/completion 비율 막대.

    Args:
        run_record: run_history JSON 로드 결과.
    """
    import matplotlib.pyplot as plt

    total_tok = run_record.get("token_usage_total", {})
    if total_tok.get("total_tokens", 0) == 0:
        print("토큰 사용량 데이터가 없습니다.")
        return

    # 단계별 토큰 집계
    phase_tokens = {}
    for p in run_record.get("phase_times", []):
        t = p.get("tokens")
        if t and t.get("total_tokens", 0) > 0:
            phase_tokens[_PHASE_LABELS.get(p["phase"], p["phase"])] = t["total_tokens"]

    # 전략별 인덱싱 토큰 합산
    indexing_total = 0
    for st in run_record.get("strategy_timings", []):
        idx_tok = st.get("indexing_tokens")
        if idx_tok and idx_tok.get("total_tokens", 0) > 0:
            indexing_total += idx_tok["total_tokens"]
    if indexing_total > 0:
        phase_tokens["인덱싱 (Contextual)"] = indexing_total

    if not phase_tokens:
        # 총량만 표시
        phase_tokens = {"전체": total_tok["total_tokens"]}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 좌: 단계별 토큰 파이
    ax1 = axes[0]
    labels = list(phase_tokens.keys())
    values = list(phase_tokens.values())
    colors = plt.cm.Set3(np.linspace(0, 0.8, len(labels)))
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colors, startangle=90, textprops={"fontsize": 9},
    )
    ax1.set_title(f"Token Usage by Phase\n(Total: {total_tok['total_tokens']:,} tokens)")

    # 우: prompt vs completion 막대
    ax2 = axes[1]
    prompt = total_tok.get("prompt_tokens", 0)
    completion = total_tok.get("completion_tokens", 0)
    bars = ax2.bar(
        ["Prompt", "Completion"],
        [prompt, completion],
        color=["#4C78A8", "#E45756"],
        width=0.5,
    )
    for bar, val in zip(bars, [prompt, completion]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(prompt, completion) * 0.02,
                 f"{val:,}", ha="center", fontsize=11)
    ax2.set_ylabel("Tokens")
    ax2.set_title(f"Prompt vs Completion\n(Cost: ${total_tok.get('total_cost_usd', 0):.4f})")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# H-2. Metric Violin — 전략 그룹별 Per-Sample 메트릭 분포
# ---------------------------------------------------------------------------


def plot_metric_violin(
    reports: Dict[str, Any],
    metrics: Optional[List[str]] = None,
) -> None:
    """전략 그룹별 per-sample RAGAS 점수 분포를 Violin 차트로 시각화.

    Args:
        reports: {strategy_name: EvaluationReport} dict.
                 EvaluationReport.per_sample_df에 per-sample 점수가 있어야 함.
        metrics: 시각화할 메트릭 목록. None이면 자동 선택 (최대 4개).
    """
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ImportError:
        print("seaborn이 필요합니다: pip install seaborn")
        return

    if not reports:
        print("EvaluationReport가 없습니다.")
        return

    # 1. per_sample_df 수집
    rows = []
    for name, report in reports.items():
        df_s = getattr(report, "per_sample_df", None)
        if df_s is None or (hasattr(df_s, "empty") and df_s.empty):
            continue
        df_s = df_s.copy()
        df_s["strategy"] = name
        df_s["group"] = _classify_group(name)
        rows.append(df_s)

    if not rows:
        print("[Info] per_sample_df가 없습니다. 집계 점수로 대체합니다.")
        _plot_metric_violin_fallback(reports, metrics)
        return

    combined = pd.concat(rows, ignore_index=True)

    # 2. 메트릭 컬럼 자동 선택
    exclude = {"strategy", "group", "user_input", "response", "retrieved_contexts", "reference"}
    numeric_cols = [
        c for c in combined.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(combined[c])
    ]
    if not numeric_cols:
        print("수치 메트릭 컬럼이 없습니다.")
        return
    if metrics is None:
        metrics = numeric_cols[:4]
    else:
        metrics = [m for m in metrics if m in combined.columns]

    if not metrics:
        print("지정한 메트릭이 데이터에 없습니다.")
        return

    # 3. Violin 서브플롯
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    group_order = [g for g in ["Baseline", "+ Reranker", "+ Contextual", "Reranker + Contextual"]
                   if g in combined["group"].values]
    palette = {k: v for k, v in _COLOR_MAP.items() if k in group_order}

    for ax, metric in zip(axes, metrics):
        sns.violinplot(
            data=combined,
            x="group", y=metric,
            order=group_order,
            palette=palette,
            inner="box",
            ax=ax,
            cut=0,
        )
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("Score" if ax is axes[0] else "")
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("전략 그룹별 메트릭 분포 — H-2 Metric Violin", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def _plot_metric_violin_fallback(reports: Dict[str, Any], metrics: Optional[List[str]]) -> None:
    """per_sample_df 없을 때 집계 점수 막대로 대체."""
    import matplotlib.pyplot as plt

    rows = []
    for name, report in reports.items():
        agg = getattr(report, "aggregate_dict", {})
        rows.append({"strategy": name, "group": _classify_group(name), **agg})
    if not rows:
        return

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c not in {"strategy", "group"}
                    and pd.api.types.is_numeric_dtype(df[c])]
    if metrics:
        numeric_cols = [m for m in metrics if m in numeric_cols]
    if not numeric_cols:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    for i, col in enumerate(numeric_cols[:4]):
        ax.bar([xi + i * 0.2 for xi in x], df[col], width=0.2, label=col, alpha=0.8)
    ax.set_xticks([xi + 0.3 for xi in x])
    ax.set_xticklabels(df["strategy"], rotation=30, ha="right", fontsize=8)
    ax.set_title("전략별 집계 점수 (per_sample_df 없음 — 집계 대체)")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# M-1. Pipeline Diagram — RAG 파이프라인 구조 다이어그램
# ---------------------------------------------------------------------------


def plot_pipeline_diagram(
    spec: Optional[Any] = None,
    strategy_name: Optional[str] = None,
) -> None:
    """RAG 파이프라인 레이어 구조를 박스-화살표 다이어그램으로 시각화.

    Args:
        spec: ComboSpec 인스턴스. None이면 strategy_name으로 파싱.
        strategy_name: 전략명 문자열 (spec이 없을 때 사용).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    # spec에서 레이어 정보 추출
    if spec is not None:
        dense = getattr(spec, "dense", "Dense")
        sparse = getattr(spec, "sparse", "Sparse")
        reranker = getattr(spec, "reranker", None)
        llm_support = getattr(spec, "llm_support", None)
    else:
        # strategy_name으로 파싱
        name = strategy_name or "DenseSparse"
        dense = "Dense"
        sparse = "Sparse"
        reranker = "flashrank" if "FlashRank" in name else ("colbert" if "ColBERT" in name else None)
        llm_support = "contextual" if "Contextual" in name else None

    has_rerank = reranker is not None
    has_ctx = llm_support is not None

    # 노드 정의: (label, x_center, y_center, color)
    nodes = [
        ("Documents\n(Markdown)", 0.5, 0.5, "#95a5a6"),
        ("Parent-Child\nChunking", 2.0, 0.5, "#7f8c8d"),
        (f"Dense\n({dense})", 3.5, 0.75, "#3498db"),
        (f"Sparse\n({sparse})", 3.5, 0.25, "#2ecc71"),
        ("Qdrant\nHybrid Index", 5.0, 0.5, "#1abc9c"),
        ("Hybrid\nRetrieval", 6.5, 0.5, "#16a085"),
    ]

    x_cursor = 8.0
    if has_rerank:
        nodes.append((f"Reranker\n({reranker})", x_cursor, 0.5, "#e74c3c"))
        x_cursor += 1.5
    if has_ctx:
        nodes.append((f"Contextual\nLLM", x_cursor, 0.5, "#e67e22"))
        x_cursor += 1.5

    nodes.append(("Answer\nLLM", x_cursor, 0.5, "#9b59b6"))
    x_cursor += 1.5
    nodes.append(("Response", x_cursor, 0.5, "#2c3e50"))

    fig, ax = plt.subplots(figsize=(max(14, x_cursor + 1), 3))
    ax.set_xlim(-0.2, x_cursor + 0.7)
    ax.set_ylim(0, 1)
    ax.axis("off")

    BOX_W, BOX_H = 1.2, 0.28

    def draw_box(label, cx, cy, color):
        x0, y0 = cx - BOX_W / 2, cy - BOX_H / 2
        box = FancyBboxPatch(
            (x0, y0), BOX_W, BOX_H,
            boxstyle="round,pad=0.02",
            linewidth=1.5,
            edgecolor="white",
            facecolor=color,
            alpha=0.88,
            zorder=3,
        )
        ax.add_patch(box)
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=7.5, color="white", weight="bold", zorder=4,
                multialignment="center")

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2 - BOX_W / 2, y2),
            xytext=(x1 + BOX_W / 2, y1),
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
            zorder=2,
        )

    # 박스 그리기
    for label, cx, cy, color in nodes:
        draw_box(label, cx, cy, color)

    # 화살표 (순서대로)
    prev = None
    for label, cx, cy, color in nodes:
        if prev:
            # Dense/Sparse → Qdrant 는 두 경로
            if prev[0].startswith("Parent") and label.startswith("Dense"):
                draw_arrow(prev[1], prev[2], cx, cy + 0.13)
                prev = (label, cx, cy, color)
                continue
            if prev[0].startswith("Parent") and label.startswith("Sparse"):
                draw_arrow(prev[1], prev[2], cx, cy - 0.13)
                prev = (label, cx, cy, color)
                continue
            if label.startswith("Qdrant") and (prev[0].startswith("Dense") or prev[0].startswith("Sparse")):
                draw_arrow(prev[1], prev[2], cx, cy)
                prev = (label, cx, cy, color)
                continue
        if prev and not prev[0].startswith("Dense") and not prev[0].startswith("Sparse"):
            draw_arrow(prev[1], prev[2], cx, cy)
        prev = (label, cx, cy, color)

    # 범례 레이블
    legend_patches = [
        mpatches.Patch(color="#3498db", label="Dense Embedding"),
        mpatches.Patch(color="#2ecc71", label="Sparse Embedding"),
        mpatches.Patch(color="#1abc9c", label="Vector Store"),
    ]
    if has_rerank:
        legend_patches.append(mpatches.Patch(color="#e74c3c", label=f"Reranker ({reranker})"))
    if has_ctx:
        legend_patches.append(mpatches.Patch(color="#e67e22", label="Contextual LLM"))
    legend_patches.append(mpatches.Patch(color="#9b59b6", label="Answer LLM"))

    ax.legend(handles=legend_patches, loc="upper left", fontsize=8,
              bbox_to_anchor=(0, 1.15), ncol=len(legend_patches))

    title_parts = [dense, sparse]
    if has_rerank:
        title_parts.append(reranker)
    if has_ctx:
        title_parts.append("contextual")
    ax.set_title(f"RAG Pipeline — {' + '.join(title_parts)}", fontsize=12, pad=30)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# M-3. Strategy Gantt — 전략 빌드 타임라인
# ---------------------------------------------------------------------------


def plot_strategy_gantt(
    run_record: dict,
    top_n: int = 40,
) -> None:
    """전략 빌드 시간을 누적 합산하여 Gantt 바 차트로 시각화.

    strategy_timings의 build_time_s를 순서대로 누적하여 start 시간을 계산.
    절대 타임스탬프가 없으므로 기록 순서 = 실행 순서로 가정.

    Args:
        run_record: run_history JSON 로드 결과.
        top_n: 표시할 최대 전략 수.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    timings = run_record.get("strategy_timings", [])
    if not timings:
        print("전략 타이밍 데이터가 없습니다.")
        return

    # 빌드 성공 여부와 관계없이 전부 표시 (실패는 회색)
    timings = timings[:top_n]

    # 누적 start 계산
    cursor = 0.0
    bars = []
    for t in timings:
        bt = t.get("build_time_s", 0)
        has_llm = bool((t.get("indexing_tokens") or {}).get("total_tokens", 0))
        success = t.get("build_success", True)
        bars.append({
            "label": t.get("label", "unknown"),
            "start": cursor,
            "duration": bt,
            "has_llm": has_llm,
            "success": success,
        })
        cursor += bt

    def _bar_color(b: dict) -> str:
        if not b["success"]:
            return "#bdc3c7"   # 실패: 회색
        if b["has_llm"]:
            return "#E45756"   # LLM 사용: 빨강
        return "#4C78A8"       # 일반: 파랑

    fig, ax = plt.subplots(figsize=(14, max(4, len(bars) * 0.35)))

    for b in bars:
        color = _bar_color(b)
        ax.barh(
            b["label"], b["duration"], left=b["start"],
            color=color, alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        if b["duration"] > 0:
            ax.text(
                b["start"] + b["duration"] / 2,
                b["label"],
                f"{b['duration']:.1f}s",
                va="center", ha="center", fontsize=7, color="white",
            )

    ax.set_xlabel("누적 경과 시간 (s)")
    ax.set_title(
        f"M-3: Strategy Build Gantt  (총 {cursor:.1f}s, {len(bars)}개 전략)",
        fontsize=13,
    )
    ax.invert_yaxis()

    # Phase 구분선 (strategy_build 단계 끝)
    phase_cursor = 0.0
    for p in run_record.get("phase_times", []):
        dur = p.get("duration_s", 0)
        if "build" in p.get("phase", "") or "strategy" in p.get("phase", ""):
            ax.axvline(
                phase_cursor + dur, color="orange",
                linestyle="--", linewidth=1.2, alpha=0.7,
                label="Phase 경계",
            )
        phase_cursor += dur

    ax.legend(handles=[
        Patch(color="#4C78A8", label="Embedding Only"),
        Patch(color="#E45756", label="+ LLM (Contextual)"),
        Patch(color="#bdc3c7", label="Failed"),
    ], loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# M-4. Cost-Efficiency — 비용 대비 품질 효율
# ---------------------------------------------------------------------------


def plot_cost_efficiency(
    run_record: dict,
    ragas_df: pd.DataFrame,
    latency_df: Optional[pd.DataFrame] = None,
    metric: Optional[str] = None,
    cost_per_query_usd: float = 1.5e-6,
) -> None:
    """비용 대비 품질 효율 산점도 (M-4 Cost-Efficiency).

    X축: 전략당 총 추정 비용(USD) = 인덱싱(LLM) 비용 + 쿼리 비용 추정
    Y축: RAGAS 품질 점수
    파레토 프론티어(저비용 고품질) + 효율 TOP 3 라벨 표시.

    Args:
        run_record: run_history JSON 로드 결과.
        ragas_df: 전략별 RAGAS 점수 DataFrame.
        latency_df: 전략별 레이턴시 DataFrame (쿼리 비용 추정용, 없으면 생략).
        metric: 품질 메트릭. None이면 자동 선택.
        cost_per_query_usd: 쿼리 1회당 추정 토큰 비용 단가.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly가 필요합니다: pip install plotly")
        return

    if ragas_df is None or ragas_df.empty:
        print("ragas_df가 없습니다.")
        return

    # 1. 전략별 인덱싱 비용 + 레이턴시 수집
    cost_map: Dict[str, float] = {}
    latency_map: Dict[str, float] = {}
    for st in run_record.get("strategy_timings", []):
        lbl = st.get("label", "")
        idx_tok = st.get("indexing_tokens") or {}
        cost_map[lbl] = float(idx_tok.get("total_cost_usd", 0.0))
        latency_map[lbl] = st.get("avg_latency_ms", 0) / 1000.0

    # latency_df 우선 사용
    if latency_df is not None and not latency_df.empty:
        lat_col = "avg_latency" if "avg_latency" in latency_df.columns else "avg_latency_ms"
        for _, row in latency_df.iterrows():
            name = str(row["strategy"])
            val = row[lat_col]
            if lat_col == "avg_latency_ms":
                val = val / 1000.0
            latency_map[name] = float(val)

    # 2. 품질 메트릭 선택
    non_metric_cols = {"strategy", "group"}
    numeric_cols = [
        c for c in ragas_df.columns
        if c not in non_metric_cols and pd.api.types.is_numeric_dtype(ragas_df[c])
    ]
    if not numeric_cols:
        print("수치 메트릭 컬럼이 없습니다.")
        return
    if metric is None:
        metric = numeric_cols[0]
    if metric not in ragas_df.columns:
        print(f"메트릭 '{metric}'이 없습니다.")
        return

    # 3. 병합 및 비용 계산
    n_queries = run_record.get("num_queries", 1) or 1
    rows = []
    for _, row in ragas_df.iterrows():
        name = str(row["strategy"])
        # 부분 문자열 매칭
        idx_cost = 0.0
        lat = 0.0
        matched = False
        for lbl, cost in cost_map.items():
            if lbl and (lbl in name or name in lbl):
                idx_cost = cost
                lat = latency_map.get(lbl, latency_map.get(name, 0.0))
                matched = True
                break
        if not matched:
            lat = latency_map.get(name, 0.0)

        query_cost = lat * n_queries * cost_per_query_usd
        total_cost = idx_cost + query_cost
        quality = row[metric]
        if pd.isna(quality):
            continue
        rows.append({
            "strategy": name,
            "group": _classify_group(name),
            "total_cost": total_cost,
            "quality": float(quality),
            "idx_cost": idx_cost,
            "lat": lat,
        })

    if not rows:
        print("매칭된 데이터가 없습니다.")
        return

    merged = pd.DataFrame(rows)

    # 4. 비용 효율 지수 + TOP 3
    merged["efficiency"] = merged["quality"] / (merged["total_cost"] + 1e-8)
    top3 = merged.nlargest(3, "efficiency")

    # log scale 여부 (전체 비용이 0에 매우 가까운 전략이 많으면 log)
    use_log = (merged["total_cost"] > 0).sum() > len(merged) * 0.3 and merged["total_cost"].max() > 1e-3

    # 5. Plotly scatter
    fig = go.Figure()
    for group_name, color in _COLOR_MAP.items():
        gdf = merged[merged["group"] == group_name]
        if gdf.empty:
            continue
        hover = [
            f"<b>{r['strategy']}</b><br>"
            f"총 비용: ${r['total_cost']:.6f}<br>"
            f"인덱싱: ${r['idx_cost']:.6f}<br>"
            f"레이턴시: {r['lat']:.3f}s<br>"
            f"{metric}: {r['quality']:.3f}<br>"
            f"효율 지수: {r['efficiency']:.1f}"
            for _, r in gdf.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=gdf["total_cost"],
            y=gdf["quality"],
            mode="markers",
            name=group_name,
            hovertext=hover,
            hoverinfo="text",
            marker=dict(size=12, color=color, opacity=0.82,
                        line=dict(color="white", width=1)),
        ))

    # 파레토 프론티어 (저비용 고품질)
    pts = merged[["total_cost", "quality"]].values
    if len(pts) > 1:
        pareto = _compute_pareto_front(pts, minimize_x=True, maximize_y=True)
        if len(pareto) > 1:
            ps = pareto[pareto[:, 0].argsort()]
            fig.add_trace(go.Scatter(
                x=ps[:, 0], y=ps[:, 1],
                mode="lines",
                line=dict(color="gray", dash="dot", width=1.5),
                name="파레토 프론티어",
            ))

    # TOP 3 효율 라벨
    for _, r in top3.iterrows():
        fig.add_annotation(
            x=r["total_cost"], y=r["quality"],
            text=f"★ {r['strategy'][:28]}",
            showarrow=True, arrowhead=2, arrowsize=1,
            font=dict(size=8, color="#2c3e50"),
            bgcolor="rgba(255,255,255,0.7)",
        )

    xaxis_opts: Dict[str, Any] = dict(title="총 추정 비용 (USD)")
    if use_log:
        xaxis_opts["type"] = "log"
        xaxis_opts["title"] = "총 추정 비용 (USD, log scale)"

    fig.update_layout(
        title=f"M-4: 비용 대비 품질 효율 — {metric}",
        xaxis=xaxis_opts,
        yaxis=dict(title=metric, range=[-0.05, 1.05]),
        height=520,
        font=dict(family="NanumGothic, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()


# ---------------------------------------------------------------------------
# 12. 통합 대시보드
# ---------------------------------------------------------------------------


def display_dashboard(
    lat_df: Optional[pd.DataFrame] = None,
    ragas_df: Optional[pd.DataFrame] = None,
    combos: Optional[list] = None,
    cost_data: Optional[Dict[str, float]] = None,
    run_record: Optional[dict] = None,
    reports: Optional[dict] = None,
    scoring_profile: str = "balanced",
) -> None:
    """전체 시각화 대시보드 (위 함수들 통합 호출).

    Args:
        lat_df: 레이턴시 DataFrame.
        ragas_df: RAGAS DataFrame.
        combos: ComboSpec 목록.
        cost_data: 비용 데이터.
        run_record: 수행 이력 JSON dict (run_history/*.json).
        reports: {strategy_name: EvaluationReport} dict (가중 점수용).
        scoring_profile: 하이라이트할 스코어링 프로파일.
    """
    print("=" * 60)
    print(" RAG Benchmark Dashboard")
    print("=" * 60)

    # 0. 수행 이력 요약
    if run_record:
        print("\n--- Run Summary ---")
        plot_run_info(run_record)

        print("\n--- Phase Timeline ---")
        plot_phase_timeline(run_record)

        print("\n--- Build Times ---")
        plot_build_times(run_record)

        print("\n--- Token Usage ---")
        plot_token_usage(run_record)

    # 1. 레이턴시 차트
    if lat_df is not None and not lat_df.empty:
        print("\n--- Latency Comparison ---")
        plot_latency_comparison(lat_df)

    # 2. RAGAS 차트
    if ragas_df is not None and not ragas_df.empty:
        print("\n--- RAGAS Radar ---")
        plot_ragas_radar(ragas_df)

        print("\n--- RAGAS Heatmap ---")
        plot_ragas_heatmap(ragas_df)

    # 2-H1. Ablation Waterfall
    if ragas_df is not None and not ragas_df.empty:
        print("\n--- Ablation Waterfall ---")
        plot_ablation_waterfall(ragas_df)

        print("\n--- Layer Interaction Heatmap ---")
        plot_layer_interaction_heatmap(ragas_df)

    # 2-H4. Tradeoff Bubble Chart
    if lat_df is not None and ragas_df is not None and not ragas_df.empty:
        print("\n--- Tradeoff Bubble Chart ---")
        plot_tradeoff_bubble(lat_df, ragas_df, run_record=run_record)

    # 3. 파레토 프론티어
    if lat_df is not None and ragas_df is not None:
        print("\n--- Latency vs Quality ---")
        # faithfulness가 있으면 사용, 없으면 첫 번째 float 컬럼
        metric_cols = [c for c in ragas_df.columns if c != "strategy" and ragas_df[c].dtype in ("float64", "float32")]
        quality_metric = "faithfulness" if "faithfulness" in metric_cols else metric_cols[0] if metric_cols else None
        if quality_metric:
            plot_latency_vs_quality(lat_df, ragas_df, quality_metric)

    # 4. 레이어 기여도
    if combos and lat_df is not None:
        print("\n--- Layer Contribution ---")
        plot_layer_contribution(combos, lat_df)

    # 5. 비용
    if cost_data:
        print("\n--- Cost Breakdown ---")
        plot_cost_breakdown(cost_data)

    # 2-H2. Metric Violin
    if reports:
        print("\n--- H-2: Metric Violin (Per-Sample Distribution) ---")
        plot_metric_violin(reports)

    # M-1. Pipeline Diagram
    if combos:
        print("\n--- M-1: Pipeline Diagram ---")
        sample_spec = next(
            (s for s in combos if getattr(s, "reranker", None) and getattr(s, "llm_support", None)),
            combos[0],
        )
        plot_pipeline_diagram(spec=sample_spec)

    # M-3. Gantt
    if run_record:
        print("\n--- M-3: Strategy Build Gantt ---")
        plot_strategy_gantt(run_record)

    # M-4. Cost-Efficiency
    if run_record and ragas_df is not None and not ragas_df.empty:
        print("\n--- M-4: Cost-Efficiency ---")
        plot_cost_efficiency(run_record, ragas_df, lat_df)

    # 6. Weighted Scores
    if reports:
        print("\n--- Weighted Scores ---")
        display_weighted_scores(reports, scoring_profile=scoring_profile)

    # 7. 리더보드
    print("\n--- Leaderboard ---")
    summary = create_summary_table(lat_df, ragas_df)
    if not summary.empty:
        display_styled_table(summary)
