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
# 차트 생성 유틸리티
# ---------------------------------------------------------------------------


def _fig_to_base64(fig) -> str:
    """matplotlib figure → base64 PNG 문자열."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _build_latency_chart(latency_df: pd.DataFrame) -> str:
    """레이턴시 수평 막대 차트 → base64 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
        sort_col = "avg_latency" if latency_df is not None and "avg_latency" in latency_df.columns else None
        lat_str = ""
        if sort_col and latency_df is not None:
            lat_row = latency_df[latency_df["strategy"] == name]
            if not lat_row.empty:
                lat_str = f" | 레이턴시: {lat_row[sort_col].values[0]:.3f}s"

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
    n_strategies = len(latency_df) if latency_df is not None else 0
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

  <!-- 레이턴시 순위 -->
  <h4 class="section-title">레이턴시 순위 (Top 20)</h4>
  <div class="row">
    <div class="col-md-6">
      {lat_table}
    </div>
    <div class="col-md-6">
      <div class="chart-box">
        {_img_html(latency_chart, "레이턴시 막대 차트")}
      </div>
    </div>
  </div>

  <!-- RAGAS 메트릭 -->
  <h4 class="section-title">RAGAS 메트릭 테이블</h4>
  <div class="table-responsive mb-4">
    {ragas_table}
  </div>

  <!-- 히트맵 -->
  <h4 class="section-title">RAGAS 히트맵</h4>
  <div class="chart-box">
    {_img_html(ragas_heatmap, "RAGAS 히트맵")}
  </div>

  <!-- 레이더 차트 -->
  <h4 class="section-title">상위 전략 레이더 차트</h4>
  <div class="chart-box">
    {_img_html(radar_chart, "레이더 차트")}
  </div>

  <!-- 산점도 -->
  <h4 class="section-title">레이턴시 vs 품질 산점도</h4>
  <div class="chart-box">
    {_img_html(scatter_chart, "산점도")}
  </div>

  <!-- 전략 설명 -->
  <h4 class="section-title">전략 유형 설명</h4>
  <div class="row g-3 mb-4">
    <div class="col-md-6">
      <div class="strategy-desc">
        <strong>Dense + Sparse (Hybrid)</strong><br>
        <small>Dense 임베딩(코사인 유사도)과 Sparse BM25/SPLADE를 결합한 하이브리드 검색.</small>
      </div>
      <div class="strategy-desc">
        <strong>ColBERT Reranker</strong><br>
        <small>ColBERT 후기 상호작용 모델로 초기 검색 결과를 재순위화.</small>
      </div>
      <div class="strategy-desc">
        <strong>FlashRank Reranker</strong><br>
        <small>경량 Cross-Encoder로 빠른 재순위화.</small>
      </div>
    </div>
    <div class="col-md-6">
      <div class="strategy-desc">
        <strong>Contextual Retrieval</strong><br>
        <small>각 청크에 LLM 기반 문맥 요약을 부착하여 검색 품질 향상.</small>
      </div>
      <div class="strategy-desc">
        <strong>OpenAI Embedding</strong><br>
        <small>OpenAI text-embedding-3-small/large API 기반 Dense 검색.</small>
      </div>
      <div class="strategy-desc">
        <strong>Upstage Solar Embedding</strong><br>
        <small>Upstage solar-embedding-1-large: passage/query 모델 분리 운용.</small>
      </div>
    </div>
  </div>

  <!-- 결론 & 권장사항 -->
  <h4 class="section-title">결론 & 권장사항 (Top 3)</h4>
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
