"""
HTML 벤치마크 보고서 자동 생성.

벤치마크 결과(레이턴시 + RAGAS)를 시각화 차트와 함께 HTML 보고서로 출력한다.
외부 의존성 없이 순수 Python f-string + Bootstrap CDN + matplotlib(base64 인라인) 사용.
"""

import base64
import io
import time
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# 한글 폰트 설정 (macOS AppleGothic / Linux NanumGothic 자동 선택)
# ---------------------------------------------------------------------------


def _set_korean_font():
    """matplotlib 한글 폰트를 플랫폼에 맞게 설정한다."""
    try:
        import platform
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        system = platform.system()
        candidates = []
        if system == "Darwin":
            candidates = ["AppleGothic", "Apple SD Gothic Neo", "Noto Sans KR"]
        elif system == "Linux":
            candidates = ["NanumGothic", "Noto Sans KR", "UnDotum"]
        else:
            candidates = ["Malgun Gothic", "NanumGothic", "Noto Sans KR"]

        available = {f.name for f in font_manager.fontManager.ttflist}
        chosen = next((c for c in candidates if c in available), None)
        if chosen:
            plt.rcParams["font.family"] = chosen
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


_set_korean_font()


# ---------------------------------------------------------------------------
# 차트 생성 유틸리티
# ---------------------------------------------------------------------------


def _fig_to_base64(fig) -> str:
    """matplotlib figure → base64 PNG 문자열."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _agg_latency(latency_df: pd.DataFrame) -> pd.DataFrame:
    """per-query 행을 전략별로 집계하여 avg_latency_ms 컬럼을 생성한다."""
    if latency_df is None or latency_df.empty:
        return latency_df
    if "avg_latency_ms" in latency_df.columns or "avg_latency" in latency_df.columns:
        return latency_df  # 이미 집계됨
    if "latency_ms" in latency_df.columns and "strategy" in latency_df.columns:
        agg = (
            latency_df.groupby("strategy")["latency_ms"]
            .agg(avg_latency_ms="mean", min_latency_ms="min", max_latency_ms="max",
                 p50_latency_ms=lambda x: x.quantile(0.5))
            .reset_index()
        )
        agg["avg_latency_ms"] = agg["avg_latency_ms"].round(1)
        return agg
    return latency_df


def _build_latency_chart(latency_df: pd.DataFrame) -> str:
    """레이턴시 수평 막대 차트 → base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        latency_df = _agg_latency(latency_df)
        sort_col = "avg_latency" if "avg_latency" in latency_df.columns else "avg_latency_ms"
        if sort_col not in latency_df.columns:
            return ""

        df = latency_df[["strategy", sort_col]].copy()
        df = df.sort_values(sort_col, ascending=True).head(20)

        n = len(df)
        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.4)))
        colors = ["#2196F3" if i < 3 else "#90CAF9" for i in range(n)]
        bars = ax.barh(df["strategy"], df[sort_col], color=colors)

        unit = "s" if sort_col == "avg_latency" else "ms"
        ax.set_xlabel(f"평균 레이턴시 ({unit})")
        ax.set_title("전략별 평균 레이턴시 (낮을수록 우수)")
        ax.invert_yaxis()

        for bar, val in zip(bars, df[sort_col]):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}{unit}", va="center", fontsize=8)

        plt.tight_layout()
        img = _fig_to_base64(fig)
        plt.close(fig)
        return img
    except Exception:
        return ""


def _build_ragas_heatmap(ragas_df: pd.DataFrame) -> str:
    """RAGAS 메트릭 히트맵 → base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
        if not metric_cols:
            return ""

        data = ragas_df.set_index("strategy")[metric_cols].astype(float)

        fig, ax = plt.subplots(figsize=(max(6, len(metric_cols) * 1.5), max(4, len(data) * 0.5)))
        im = ax.imshow(data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(metric_cols)))
        ax.set_xticklabels(metric_cols, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data.index, fontsize=8)
        ax.set_title("RAGAS 메트릭 히트맵 (높을수록 녹색)")

        # 셀 값 표시
        for i in range(len(data)):
            for j in range(len(metric_cols)):
                val = data.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black")

        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        img = _fig_to_base64(fig)
        plt.close(fig)
        return img
    except Exception:
        return ""


def _build_scatter_chart(latency_df: pd.DataFrame, ragas_df: pd.DataFrame) -> str:
    """레이턴시 vs 품질 산점도 → base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        latency_df = _agg_latency(latency_df)
        sort_col = "avg_latency" if "avg_latency" in latency_df.columns else "avg_latency_ms"
        if sort_col not in latency_df.columns or ragas_df.empty:
            return ""

        metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
        if not metric_cols:
            return ""

        # 가중 평균 품질 점수 (메트릭 단순 평균)
        ragas_copy = ragas_df.copy()
        ragas_copy["quality"] = ragas_copy[metric_cols].mean(axis=1)

        merged = latency_df[["strategy", sort_col]].merge(
            ragas_copy[["strategy", "quality"]], on="strategy", how="inner"
        )
        if merged.empty:
            return ""

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(merged[sort_col], merged["quality"], alpha=0.7, s=80, color="#4CAF50")

        for _, row in merged.iterrows():
            ax.annotate(
                row["strategy"][:20],
                (row[sort_col], row["quality"]),
                fontsize=7, alpha=0.8,
                xytext=(3, 3), textcoords="offset points",
            )

        unit = "s" if sort_col == "avg_latency" else "ms"
        ax.set_xlabel(f"평균 레이턴시 ({unit})")
        ax.set_ylabel("평균 품질 점수 (RAGAS 메트릭 평균)")
        ax.set_title("레이턴시 vs 품질 분포")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        img = _fig_to_base64(fig)
        plt.close(fig)
        return img
    except Exception:
        return ""


def _build_radar_chart(ragas_df: pd.DataFrame, top_n: int = 5) -> str:
    """상위 N개 전략 레이더 차트 → base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
        if len(metric_cols) < 3:
            return ""

        top_df = ragas_df.head(min(top_n, len(ragas_df)))
        N = len(metric_cols)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]

        for i, (_, row) in enumerate(top_df.iterrows()):
            vals = [float(row.get(m, 0)) for m in metric_cols]
            vals += vals[:1]
            ax.plot(angles, vals, color=colors[i % len(colors)], linewidth=2, label=row["strategy"][:20])
            ax.fill(angles, vals, color=colors[i % len(colors)], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_cols, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"상위 {len(top_df)}개 전략 레이더 차트", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

        plt.tight_layout()
        img = _fig_to_base64(fig)
        plt.close(fig)
        return img
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# HTML 빌더
# ---------------------------------------------------------------------------


def _compute_weighted_scores(ragas_df: pd.DataFrame) -> pd.Series:
    """단순 메트릭 평균으로 가중 점수 계산."""
    metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
    if not metric_cols:
        return pd.Series(dtype=float)
    return ragas_df[metric_cols].mean(axis=1)


def _ragas_table_html(ragas_df: pd.DataFrame) -> str:
    """RAGAS 메트릭 테이블 HTML (색상 코딩)."""
    metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
    if not metric_cols:
        return "<p>RAGAS 데이터 없음</p>"

    rows_html = ""
    for _, row in ragas_df.iterrows():
        cells = f"<td>{row['strategy']}</td>"
        for col in metric_cols:
            val = row.get(col, 0.0)
            if isinstance(val, float):
                pct = int(val * 100)
                color = f"hsl({int(val * 120)}, 70%, 85%)"
                cells += f'<td style="background:{color}; text-align:center">{val:.4f}</td>'
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    headers = "".join(f"<th>{c}</th>" for c in ["전략"] + metric_cols)
    return f"""
    <table class="table table-sm table-bordered">
      <thead class="table-dark"><tr>{headers}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


def _latency_table_html(latency_df: pd.DataFrame) -> str:
    """레이턴시 요약 테이블 HTML."""
    latency_df = _agg_latency(latency_df)
    sort_col = "avg_latency" if "avg_latency" in latency_df.columns else "avg_latency_ms"
    if sort_col not in latency_df.columns:
        return "<p>레이턴시 데이터 없음</p>"

    df = latency_df.sort_values(sort_col).head(20)
    unit = "s" if sort_col == "avg_latency" else "ms"

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows(), 1):
        val = row[sort_col]
        badge = ' <span class="badge bg-warning">TOP3</span>' if i <= 3 else ""
        rows_html += f"<tr><td>{i}</td><td>{row['strategy']}{badge}</td><td>{val:.3f}{unit}</td></tr>"

    return f"""
    <table class="table table-sm table-hover">
      <thead class="table-secondary"><tr><th>#</th><th>전략</th><th>평균 레이턴시</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


def _recommendations_html(ragas_df: pd.DataFrame, latency_df: pd.DataFrame) -> str:
    """가중 점수 기준 Top 3 전략 추천."""
    if ragas_df is None or ragas_df.empty:
        return "<p class='text-muted'>RAGAS 데이터가 없어 추천을 생성할 수 없습니다.</p>"

    weighted = _compute_weighted_scores(ragas_df)
    ragas_copy = ragas_df.copy()
    ragas_copy["_weighted"] = weighted
    top3 = ragas_copy.nlargest(3, "_weighted")

    items = ""
    medals = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(top3.iterrows()):
        name = row["strategy"]
        score = row["_weighted"]

        # 레이턴시 가져오기
        agg_lat = _agg_latency(latency_df) if latency_df is not None else None
        sort_col = "avg_latency_ms" if agg_lat is not None and "avg_latency_ms" in agg_lat.columns else (
            "avg_latency" if agg_lat is not None and "avg_latency" in agg_lat.columns else None
        )
        lat_str = ""
        if sort_col and agg_lat is not None:
            lat_row = agg_lat[agg_lat["strategy"] == name]
            if not lat_row.empty:
                unit = "s" if sort_col == "avg_latency" else "ms"
                lat_str = f" | 레이턴시: {lat_row[sort_col].values[0]:.1f}{unit}"

        items += f"""
        <li class="list-group-item">
          <strong>{medals[i]} {name}</strong><br>
          <small class="text-muted">가중 점수: {score:.4f}{lat_str}</small>
        </li>"""

    return f'<ol class="list-group list-group-numbered">{items}</ol>'


def _env_table_html(run_record: Optional[dict]) -> str:
    """실행 환경 정보 테이블."""
    if run_record is None:
        return "<p class='text-muted'>실행 환경 정보 없음</p>"

    pi = run_record.get("platform_info", {})
    rows = [
        ("OS", f"{pi.get('os', 'N/A')} {pi.get('os_release', '')}"),
        ("CPU", f"{pi.get('apple_chip', pi.get('processor', 'N/A'))} ({pi.get('cpu_count_logical', '?')} cores)"),
        ("RAM", f"{pi.get('ram_total_gb', '?')} GB"),
        ("GPU", pi.get("gpu") or "None"),
        ("Python", pi.get("python_version", "N/A")),
        ("Git Commit", pi.get("git_commit", "N/A")),
    ]
    rows_html = "".join(f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>" for k, v in rows)
    return f"""
    <table class="table table-sm">
      <tbody>{rows_html}</tbody>
    </table>"""


def _summary_cards_html(
    latency_df: Optional[pd.DataFrame],
    ragas_df: Optional[pd.DataFrame],
    run_record: Optional[dict],
) -> str:
    """요약 통계 카드."""
    _lat_agg = _agg_latency(latency_df) if latency_df is not None else None
    n_strategies = len(_lat_agg["strategy"].unique()) if _lat_agg is not None and "strategy" in _lat_agg.columns else 0
    n_evaluated = len(ragas_df) if ragas_df is not None else 0

    total_s = "N/A"
    api_cost = "N/A"
    if run_record:
        dur = run_record.get("duration_s")
        if dur:
            total_s = f"{dur:.0f}s"
        tu = run_record.get("token_usage_total", {})
        if tu and tu.get("total_cost_usd", 0) > 0:
            api_cost = f"${tu['total_cost_usd']:.4f}"

    return f"""
    <div class="row g-3 mb-4">
      <div class="col-md-3">
        <div class="card text-center border-primary">
          <div class="card-body">
            <h2 class="card-title text-primary">{n_strategies}</h2>
            <p class="card-text text-muted">총 전략 수</p>
          </div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card text-center border-success">
          <div class="card-body">
            <h2 class="card-title text-success">{n_evaluated}</h2>
            <p class="card-text text-muted">RAGAS 평가 완료</p>
          </div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card text-center border-warning">
          <div class="card-body">
            <h2 class="card-title text-warning">{total_s}</h2>
            <p class="card-text text-muted">총 소요 시간</p>
          </div>
        </div>
      </div>
      <div class="col-md-3">
        <div class="card text-center border-danger">
          <div class="card-body">
            <h2 class="card-title text-danger">{api_cost}</h2>
            <p class="card-text text-muted">API 비용</p>
          </div>
        </div>
      </div>
    </div>"""


def _token_breakdown_html(run_record: Optional[dict]) -> str:
    """토큰 사용량 세분화 테이블 HTML (카테고리 × 프로바이더)."""
    if run_record is None:
        return "<p class='text-muted'>토큰 데이터 없음</p>"

    breakdown = run_record.get("token_breakdown", {})
    total = run_record.get("token_usage_total", {})

    # breakdown이 없으면 total만 표시
    if not breakdown:
        tt = total.get("total_tokens", 0)
        cost = total.get("total_cost_usd", 0)
        calls = total.get("num_calls", 0)
        if tt == 0:
            return "<p class='text-muted'>토큰 사용 내역 없음</p>"
        return f"""
        <table class='table table-sm table-bordered'>
          <thead class='table-dark'><tr><th>카테고리</th><th>프로바이더</th><th>총 토큰</th><th>프롬프트</th><th>컴플리션</th><th>비용($)</th><th>호출수</th></tr></thead>
          <tbody><tr><td colspan='2'><em>합계</em></td>
            <td>{total.get('total_tokens',0):,}</td>
            <td>{total.get('prompt_tokens',0):,}</td>
            <td>{total.get('completion_tokens',0):,}</td>
            <td>{total.get('total_cost_usd',0):.4f}</td>
            <td>{total.get('num_calls',0)}</td></tr></tbody>
        </table>"""

    CATEGORY_LABELS = {
        "qa_generation": "QA 생성",
        "contextual_indexing": "Contextual 인덱싱",
        "ragas_evaluation": "RAGAS 평가",
        "embedding_indexing": "임베딩 인덱싱",
        "embedding_query": "임베딩 쿼리",
    }
    PROVIDER_BADGES = {
        "openai": '<span class="badge bg-primary">OpenAI</span>',
        "upstage": '<span class="badge bg-success">Upstage</span>',
        "local": '<span class="badge bg-secondary">Local</span>',
    }

    rows_html = ""
    grand_total = grand_prompt = grand_comp = grand_cost = grand_calls = 0
    for key in sorted(breakdown.keys()):
        cat, _, prov = key.partition(".")
        u = breakdown[key]
        tt = u.get("total_tokens", 0)
        pt = u.get("prompt_tokens", 0)
        ct = u.get("completion_tokens", 0)
        cost = u.get("total_cost_usd", 0.0)
        nc = u.get("num_calls", 0)
        grand_total += tt; grand_prompt += pt; grand_comp += ct
        grand_cost += cost; grand_calls += nc
        cat_label = CATEGORY_LABELS.get(cat, cat)
        prov_badge = PROVIDER_BADGES.get(prov, prov)
        rows_html += (
            f"<tr><td>{cat_label}</td><td>{prov_badge}</td>"
            f"<td>{tt:,}</td><td>{pt:,}</td><td>{ct:,}</td>"
            f"<td>{cost:.4f}</td><td>{nc}</td></tr>"
        )

    rows_html += (
        f"<tr class='table-warning fw-bold'><td colspan='2'>합계</td>"
        f"<td>{grand_total:,}</td><td>{grand_prompt:,}</td><td>{grand_comp:,}</td>"
        f"<td>{grand_cost:.4f}</td><td>{grand_calls}</td></tr>"
    )

    return f"""
    <table class='table table-sm table-bordered table-hover'>
      <thead class='table-dark'>
        <tr><th>카테고리</th><th>프로바이더</th><th>총 토큰</th><th>프롬프트</th><th>컴플리션</th><th>비용($)</th><th>호출수</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------


def generate_html_report(
    latency_df: Optional[pd.DataFrame],
    ragas_df: Optional[pd.DataFrame],
    output_path: str,
    session_id: str = "",
    run_record: Optional[dict] = None,
) -> str:
    """벤치마크 결과를 HTML 보고서로 생성.

    Args:
        latency_df: 레이턴시 결과 DataFrame.
        ragas_df: RAGAS 점수 DataFrame.
        output_path: 출력 HTML 파일 경로.
        session_id: 세션 식별자.
        run_record: RunTracker 기록 dict (선택).

    Returns:
        출력 파일 경로.
    """
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S")
    title = f"RAG Benchmark Report — {session_id}" if session_id else "RAG Benchmark Report"

    # 차트 생성
    latency_chart = _build_latency_chart(latency_df) if latency_df is not None else ""
    ragas_heatmap = _build_ragas_heatmap(ragas_df) if ragas_df is not None else ""
    scatter_chart = _build_scatter_chart(latency_df, ragas_df) if (latency_df is not None and ragas_df is not None) else ""
    radar_chart = _build_radar_chart(ragas_df) if ragas_df is not None else ""

    # 테이블 HTML
    lat_table = _latency_table_html(latency_df) if latency_df is not None else "<p>데이터 없음</p>"
    ragas_table = _ragas_table_html(ragas_df) if ragas_df is not None else "<p>데이터 없음</p>"
    summary_cards = _summary_cards_html(latency_df, ragas_df, run_record)
    env_table = _env_table_html(run_record)
    recommendations = _recommendations_html(ragas_df, latency_df)
    token_breakdown_table = _token_breakdown_html(run_record)

    def _img_html(b64: str, alt: str) -> str:
        if not b64:
            return f'<p class="text-muted">차트 생성 실패 (matplotlib 필요)</p>'
        return f'<img src="data:image/png;base64,{b64}" class="img-fluid" alt="{alt}" />'

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body {{ font-family: 'Noto Sans KR', sans-serif; }}
    .section-title {{ border-left: 4px solid #2196F3; padding-left: 12px; margin: 24px 0 12px 0; }}
    .chart-box {{ background: #f8f9fa; border-radius: 8px; padding: 16px; margin-bottom: 24px; text-align: center; }}
    .strategy-desc {{ background: #fff3e0; border-radius: 6px; padding: 12px; margin-bottom: 8px; }}
    footer {{ background: #263238; color: #cfd8dc; padding: 20px; text-align: center; margin-top: 40px; }}
    .metric-badge {{ font-size: 0.75rem; }}
  </style>
</head>
<body>
<div class="container-fluid px-4 py-4">

  <!-- 헤더 -->
  <div class="p-4 mb-4 bg-dark text-white rounded">
    <h1 class="display-6">{title}</h1>
    <p class="mb-0 text-secondary">생성 시각: {generated_at} | 세션: {session_id or "N/A"}</p>
  </div>

  <!-- 요약 카드 -->
  <h4 class="section-title">요약 통계</h4>
  {summary_cards}

  <!-- 실행 환경 -->
  <h4 class="section-title">실행 환경</h4>
  {env_table}

  <!-- API 토큰 사용량 -->
  <h4 class="section-title">API 토큰 사용량 (카테고리 × 프로바이더)</h4>
  <div class="table-responsive mb-4">
    {token_breakdown_table}
  </div>

  <!-- ══════════════════════════════════════════════════════════ -->
  <!-- 차트 Row 1: 레이턴시 막대 + 레이더 차트                        -->
  <!-- ══════════════════════════════════════════════════════════ -->
  <h4 class="section-title">성능 시각화</h4>

  <div class="row g-4 mb-4">
    <!-- 레이턴시 막대 차트 -->
    <div class="col-md-6">
      <div class="chart-card">
        <h5 class="chart-title">① 전략별 평균 레이턴시 (낮을수록 좋음)</h5>
        <div class="chart-explain">
          <strong>무엇을 보여주나요?</strong><br>
          각 RAG 전략이 쿼리 1건을 처리하는 데 걸린 평균 시간(ms)을 수평 막대로 표시합니다.
          막대가 짧을수록 응답이 빠르고, 실시간 서비스에 유리합니다.<br><br>
          <strong>어떻게 읽나요?</strong><br>
          • <span class="badge bg-primary">파란색</span> 상위 3개 전략 — 속도 최강군<br>
          • 로컬 임베딩(KoSimCSE, BGE-M3)은 일반적으로 API 호출 없이 빠름<br>
          • ColBERT/FlashRank 리랭킹이 붙으면 레이턴시가 크게 증가<br>
          • Contextual은 청크 요약을 미리 캐시하므로 쿼리 시점에는 속도 영향 없음
        </div>
        <div class="chart-box mt-2">
          {_img_html(latency_chart, "레이턴시 막대 차트")}
        </div>
      </div>
    </div>

    <!-- 레이더 차트 -->
    <div class="col-md-6">
      <div class="chart-card">
        <h5 class="chart-title">② 상위 전략 RAGAS 레이더 (넓을수록 좋음)</h5>
        <div class="chart-explain">
          <strong>무엇을 보여주나요?</strong><br>
          RAGAS 평가를 통과한 상위 5개 전략의 4가지 품질 지표를
          다각형으로 겹쳐 그립니다. 다각형 면적이 클수록 종합 품질이 높습니다.<br><br>
          <strong>어떻게 읽나요?</strong><br>
          • <strong>faithfulness</strong> — 답변이 검색된 문서에만 근거하는가 (할루시네이션 억제)<br>
          • <strong>answer_relevancy</strong> — 답변이 질문에 얼마나 직접적으로 대응하는가<br>
          • <strong>context_precision</strong> — 검색된 청크 중 실제 유용한 비율<br>
          • <strong>context_recall</strong> — 정답에 필요한 정보를 빠짐없이 가져왔는가<br>
          • 이상적인 전략은 모든 축에서 외곽선 근처까지 채워진 형태
        </div>
        <div class="chart-box mt-2">
          {_img_html(radar_chart, "레이더 차트")}
        </div>
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════ -->
  <!-- 차트 Row 2: RAGAS 히트맵 + 품질-속도 산점도                    -->
  <!-- ══════════════════════════════════════════════════════════ -->
  <div class="row g-4 mb-4">
    <!-- RAGAS 히트맵 -->
    <div class="col-md-6">
      <div class="chart-card">
        <h5 class="chart-title">③ RAGAS 메트릭 히트맵 (진할수록 높은 점수)</h5>
        <div class="chart-explain">
          <strong>무엇을 보여주나요?</strong><br>
          평가된 모든 전략의 RAGAS 4개 지표를 색상 강도로 한눈에 비교합니다.
          셀 색이 진초록일수록 점수가 높고, 옅을수록 낮습니다.<br><br>
          <strong>어떻게 읽나요?</strong><br>
          • 행(가로) = 개별 전략, 열(세로) = RAGAS 지표<br>
          • 전체 행이 고르게 진한 전략 → 균형 잡힌 우수 전략<br>
          • 특정 열만 옅은 전략 → 해당 지표에 약점 존재<br>
          • context_recall이 옅으면 필요한 정보를 놓친 것 → 청크 크기·k 값 재검토 필요
        </div>
        <div class="chart-box mt-2">
          {_img_html(ragas_heatmap, "RAGAS 히트맵")}
        </div>
      </div>
    </div>

    <!-- 산점도 -->
    <div class="col-md-6">
      <div class="chart-card">
        <h5 class="chart-title">④ 레이턴시 vs 품질 산점도 (왼쪽 위가 최적)</h5>
        <div class="chart-explain">
          <strong>무엇을 보여주나요?</strong><br>
          X축 = 평균 레이턴시, Y축 = RAGAS 메트릭 평균으로 각 전략을 점으로 찍습니다.
          "빠르면서 품질도 높은" 최적 전략을 시각적으로 식별할 수 있습니다.<br><br>
          <strong>어떻게 읽나요?</strong><br>
          • <strong>왼쪽 위</strong> = 이상적 전략 (빠름 + 고품질)<br>
          • <strong>오른쪽 위</strong> = 고품질이지만 느림 (오프라인 처리용)<br>
          • <strong>왼쪽 아래</strong> = 빠르지만 품질 미흡 (추가 튜닝 필요)<br>
          • 파레토 프론티어(좌상단 경계선) 근처 전략들이 실용적 후보군
        </div>
        <div class="chart-box mt-2">
          {_img_html(scatter_chart, "산점도")}
        </div>
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════ -->
  <!-- 데이터 테이블                                               -->
  <!-- ══════════════════════════════════════════════════════════ -->
  <div class="row g-4 mb-4">
    <!-- 레이턴시 순위 테이블 -->
    <div class="col-md-6">
      <div class="chart-card">
        <h5 class="chart-title">레이턴시 순위 Top 20</h5>
        <div class="chart-explain">
          전략별 평균 레이턴시(ms) 순위표입니다.
          <span class="badge bg-warning text-dark">TOP3</span> 표시 전략은 속도 최강군으로,
          실시간 챗봇·검색 서비스에 우선 검토를 권장합니다.
        </div>
        <div class="table-responsive mt-2">
          {lat_table}
        </div>
      </div>
    </div>

    <!-- RAGAS 메트릭 테이블 -->
    <div class="col-md-6">
      <div class="chart-card">
        <h5 class="chart-title">RAGAS 메트릭 상세 테이블</h5>
        <div class="chart-explain">
          Pass 2에서 평가된 전략별 RAGAS 4개 지표입니다.
          셀 배경색이 진할수록(초록) 점수가 높습니다.
          모든 지표가 균형 잡힌 전략이 장기적으로 안정적입니다.
        </div>
        <div class="table-responsive mt-2">
          {ragas_table}
        </div>
      </div>
    </div>
  </div>

  <!-- 전략 설명 -->
  <h4 class="section-title">전략 유형 가이드</h4>
  <div class="row g-3 mb-4">
    <div class="col-md-4">
      <div class="strategy-desc">
        <strong>Dense + Sparse (Hybrid)</strong><br>
        <small>Dense 임베딩(의미적 유사도)과 Sparse BM25/SPLADE(키워드 매칭)를 병합한 하이브리드 검색.
        단독 방식 대비 recall과 precision 모두 향상되는 경우가 많음.</small>
      </div>
    </div>
    <div class="col-md-4">
      <div class="strategy-desc">
        <strong>ColBERT Reranker</strong><br>
        <small>Late-Interaction 방식의 ColBERT 모델로 초기 검색 후 재순위화.
        정밀도는 높지만 레이턴시가 크게 증가. 배치/비실시간 처리에 적합.</small>
      </div>
    </div>
    <div class="col-md-4">
      <div class="strategy-desc">
        <strong>FlashRank Reranker</strong><br>
        <small>경량 Cross-Encoder 기반 빠른 재순위화.
        ColBERT보다 빠르고 단순 Hybrid보다 정밀. 속도·품질 균형이 필요한 상황에 적합.</small>
      </div>
    </div>
    <div class="col-md-4">
      <div class="strategy-desc">
        <strong>Contextual Retrieval</strong><br>
        <small>각 청크에 LLM 요약 문맥을 미리 부착(인덱싱 시점).
        쿼리 시점 레이턴시 증가 없이 검색 품질 향상. API 비용은 인덱싱 시 일회성 발생.</small>
      </div>
    </div>
    <div class="col-md-4">
      <div class="strategy-desc">
        <strong>OpenAI Embedding</strong><br>
        <small>text-embedding-3-large API 기반 Dense 검색.
        고차원(3072d) 벡터로 높은 의미 표현력. API 비용과 네트워크 레이턴시가 트레이드오프.</small>
      </div>
    </div>
    <div class="col-md-4">
      <div class="strategy-desc">
        <strong>Upstage Solar Embedding</strong><br>
        <small>문서용(passage)과 쿼리용(query) 모델을 분리 운용하는 비대칭 임베딩.
        한국어 최적화 모델로 국내 도메인 문서에 강점.</small>
      </div>
    </div>
  </div>

  <!-- 결론 & 권장사항 -->
  <h4 class="section-title">결론 &amp; 권장사항 (Top 3)</h4>
  {recommendations}

</div>

<footer>
  <small>AutoRAG Benchmark Report | Generated by generate_html_report.py | {generated_at}</small>
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""

    from pathlib import Path
    Path(output_path).write_text(html, encoding="utf-8")
    print(f"  HTML 보고서: {output_path}")
    return output_path
