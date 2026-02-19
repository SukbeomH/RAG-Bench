"""
colab_visualizer — 벤치마크 결과 시각화 유틸리티.

matplotlib, plotly, seaborn으로 차트를 생성한다.
Colab 노트북에서 inline 렌더링을 기본으로 한다.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


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
        groups = {}
        for spec in specs:
            val = get_val(spec)
            strategy_name = spec.label
            mask = metric_df["strategy"].str.contains(strategy_name, regex=False)
            if mask.any():
                metric_val = metric_df.loc[mask, metric].values[0]
                groups.setdefault(val, []).append(metric_val)

        if groups:
            labels = sorted(groups.keys())
            data = [groups[l] for l in labels]
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


# ---------------------------------------------------------------------------
# 8. 통합 대시보드
# ---------------------------------------------------------------------------


def display_dashboard(
    lat_df: Optional[pd.DataFrame] = None,
    ragas_df: Optional[pd.DataFrame] = None,
    combos: Optional[list] = None,
    cost_data: Optional[Dict[str, float]] = None,
) -> None:
    """전체 시각화 대시보드 (위 함수들 통합 호출).

    Args:
        lat_df: 레이턴시 DataFrame.
        ragas_df: RAGAS DataFrame.
        combos: ComboSpec 목록.
        cost_data: 비용 데이터.
    """
    print("=" * 60)
    print(" RAG Benchmark Dashboard")
    print("=" * 60)

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

    # 6. 리더보드
    print("\n--- Leaderboard ---")
    summary = create_summary_table(lat_df, ragas_df)
    if not summary.empty:
        display_styled_table(summary)
