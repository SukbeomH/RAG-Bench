"""
colab_visualizer — 벤치마크 결과 시각화 유틸리티.

matplotlib, plotly, seaborn으로 차트를 생성한다.
Colab 노트북에서 inline 렌더링을 기본으로 한다.
"""

from typing import Dict, Optional

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

    # 6. Weighted Scores
    if reports:
        print("\n--- Weighted Scores ---")
        display_weighted_scores(reports, scoring_profile=scoring_profile)

    # 7. 리더보드
    print("\n--- Leaderboard ---")
    summary = create_summary_table(lat_df, ragas_df)
    if not summary.empty:
        display_styled_table(summary)
