"""
HTML 벤치마크 보고서 자동 생성.

벤치마크 결과(레이턴시 + RAGAS)를 시각화 차트와 함께 HTML 보고서로 출력한다.
외부 의존성 없이 순수 Python f-string + Bootstrap CDN + matplotlib(base64 인라인) 사용.
"""

import base64
import io
import json
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
    """per-query 행을 전략별로 집계하여 avg_latency_ms 컬럼을 생성한다.

    ★ 이상치 강건성: 주 정렬·표시 지표로 중앙값(median)을 사용한다.
      avg_latency_ms = median (정렬·차트·순위의 기준값)
      mean_latency_ms = 산술 평균 (참고용 — 이상치에 민감)

    추가 통계:
    - has_outlier: IQR×1.5 초과 쿼리가 1개 이상 존재하면 True
    - cv_pct: 변동계수(std/mean×100). 높을수록 샘플 간 편차가 큼.
    """
    if latency_df is None or latency_df.empty:
        return latency_df
    if "avg_latency_ms" in latency_df.columns or "avg_latency" in latency_df.columns:
        return latency_df  # 이미 집계됨
    if "latency_ms" in latency_df.columns and "strategy" in latency_df.columns:
        def _has_outlier(x):
            """IQR×1.5 기준으로 이상치 존재 여부 반환."""
            if len(x) < 3:
                return False
            q1, q3 = x.quantile(0.25), x.quantile(0.75)
            iqr = q3 - q1
            return bool(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).any())

        agg = (
            latency_df.groupby("strategy")["latency_ms"]
            .agg(
                # avg_latency_ms = median (이상치 강건 지표, 정렬 기준)
                avg_latency_ms=lambda x: x.quantile(0.5),
                mean_latency_ms="mean",
                min_latency_ms="min",
                max_latency_ms="max",
                std_latency_ms="std",
                n_queries="count",
                has_outlier=_has_outlier,
            )
            .reset_index()
        )
        agg["avg_latency_ms"] = agg["avg_latency_ms"].round(1)
        agg["mean_latency_ms"] = agg["mean_latency_ms"].round(1)
        # CV(변동계수): std/mean×100 — 샘플 간 편차 상대 지표
        agg["cv_pct"] = (
            agg["std_latency_ms"].fillna(0) / agg["mean_latency_ms"].replace(0, float("nan")) * 100
        ).round(1)
        return agg
    return latency_df


def _shorten_name(name: str) -> str:
    """차트 표시용 전략명 축약.

    긴 이름 형태를 식별 가능한 짧은 형태로 변환합니다:
      'Contextual Retrieval (DS(model+sparse))' → 'Ctx·DS(model+sparse)'
      'ColBERT Rerank (DS(model+sparse))'       → 'CB·DS(model+sparse)'
      'FlashRank Rerank (DS(model+sparse))'     → 'FR·DS(model+sparse)'
    """
    prefixes = [
        ("Contextual Retrieval (", "Ctx·"),
        ("ColBERT Rerank (", "CB·"),
        ("FlashRank Rerank (", "FR·"),
    ]
    for long_prefix, short_prefix in prefixes:
        if name.startswith(long_prefix):
            inner = name[len(long_prefix):]
            if inner.endswith(")"):
                inner = inner[:-1]
            return short_prefix + inner
    return name


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
        df["strategy"] = df["strategy"].apply(_shorten_name)

        n = len(df)
        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.4)))
        colors = ["#2196F3" if i < 3 else "#90CAF9" for i in range(n)]
        bars = ax.barh(df["strategy"], df[sort_col], color=colors)

        unit = "s" if sort_col == "avg_latency" else "ms"
        ax.set_xlabel(f"중앙값 레이턴시 ({unit})")
        ax.set_title("전략별 중앙값 레이턴시 (낮을수록 우수)")
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


def _build_scatter_chart(latency_df: pd.DataFrame, ragas_df: pd.DataFrame, top_n: int = 10) -> str:
    """레이턴시 vs 품질 산점도 → base64 PNG. 상위 top_n개 전략만 표시.

    개선:
    - ColBERT 포함 전략: 파란색 (#2196F3), non-ColBERT: 녹색 (#4CAF50)
    - x축에 실용 레이턴시 기준선 3000ms 수직 점선 추가 (빨간색)
    - 범례에 ColBERT / non-ColBERT 구분 추가
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

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

        # 품질 기준 상위 top_n개만 추출
        merged = merged.nlargest(top_n, "quality").reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(10, 6))

        # ColBERT 포함 여부로 색상 분리
        colbert_mask = merged["strategy"].str.contains("ColBERT", case=False, na=False)
        colbert_df = merged[colbert_mask]
        non_colbert_df = merged[~colbert_mask]

        scatter_colbert = ax.scatter(
            colbert_df[sort_col], colbert_df["quality"],
            alpha=0.8, s=90, color="#2196F3", zorder=5
        )
        scatter_non = ax.scatter(
            non_colbert_df[sort_col], non_colbert_df["quality"],
            alpha=0.8, s=90, color="#4CAF50", zorder=5
        )

        for _, row in merged.iterrows():
            ax.annotate(
                _shorten_name(row["strategy"]),
                (row[sort_col], row["quality"]),
                fontsize=7, alpha=0.8,
                xytext=(3, 3), textcoords="offset points",
            )

        unit = "s" if sort_col == "avg_latency" else "ms"

        # 실용 레이턴시 기준선: ms 단위면 3000ms, s 단위면 3s
        threshold = 3.0 if sort_col == "avg_latency" else 3000.0
        ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5, alpha=0.8, zorder=4)
        y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
        ax.text(
            threshold + (threshold * 0.01), y_max * 0.97,
            "실용 한계 3s",
            color="red", fontsize=8, va="top", ha="left"
        )

        ax.set_xlabel(f"중앙값 레이턴시 ({unit})")
        ax.set_ylabel("평균 품질 점수 (RAGAS 메트릭 평균)")
        ax.set_title("레이턴시 vs 품질 분포")
        ax.grid(True, alpha=0.3)

        # 범례: ColBERT / non-ColBERT 구분
        patch_colbert = mpatches.Patch(color="#2196F3", label="ColBERT 포함 전략")
        patch_non = mpatches.Patch(color="#4CAF50", label="non-ColBERT 전략")
        ax.legend(handles=[patch_colbert, patch_non], loc="lower right", fontsize=8)

        plt.tight_layout()
        img = _fig_to_base64(fig)
        plt.close(fig)
        return img
    except Exception:
        return ""


def _build_radar_chart(ragas_df: pd.DataFrame, top_n: int = 5) -> str:
    """Best 1 vs Worst 1 전략 레이더 차트 → base64 PNG.

    개선:
    - top_n 인자는 유지하되 Best 1 vs Worst 1 비교로 변경
    - Best: weighted score 1위, Worst: weighted score 최하위
    - 범례에 "최고 전략" / "최저 전략" 표시
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
        if len(metric_cols) < 3:
            return ""

        # weighted score 계산 후 Best/Worst 선택
        scores = _compute_weighted_scores(ragas_df)
        ragas_copy = ragas_df.copy()
        ragas_copy["_score"] = scores

        best_row = ragas_copy.loc[ragas_copy["_score"].idxmax()]
        worst_row = ragas_copy.loc[ragas_copy["_score"].idxmin()]

        N = len(metric_cols)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        # Best: 파란색
        best_vals = [float(best_row.get(m, 0)) for m in metric_cols]
        best_vals += best_vals[:1]
        ax.plot(angles, best_vals, color="#2196F3", linewidth=2.5,
                label=f"최고 전략: {_shorten_name(best_row['strategy'])}")
        ax.fill(angles, best_vals, color="#2196F3", alpha=0.2)

        # Worst: 빨간색
        worst_vals = [float(worst_row.get(m, 0)) for m in metric_cols]
        worst_vals += worst_vals[:1]
        ax.plot(angles, worst_vals, color="#F44336", linewidth=2.5,
                label=f"최저 전략: {_shorten_name(worst_row['strategy'])}")
        ax.fill(angles, worst_vals, color="#F44336", alpha=0.2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_cols, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title("최고 vs 최저 전략 레이더 비교", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.15), fontsize=8)

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
    """단순 메트릭 평균으로 가중 점수 계산 (NaN skipna=True).

    주의: pandas mean(axis=1, skipna=True) 기본 동작으로 NaN 메트릭은 제외 후 평균.
    예) faithfulness=NaN, 나머지 3개=0.7 → score=0.7 (4개 기준이면 0.525)
    과대평가 가능성이 있으므로 NaN 포함 전략은 별도 주석으로 표기 권장.
    """
    metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
    if not metric_cols:
        return pd.Series(dtype=float)
    return ragas_df[metric_cols].mean(axis=1)


def _compute_weighted_scores_nan_penalized(ragas_df: pd.DataFrame) -> pd.Series:
    """NaN을 0으로 대체한 후 전체 메트릭 수 기준으로 평균 계산 (보수적 점수).

    NaN skipna=True 방식 대비 과대평가를 방지한다.
    """
    metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
    if not metric_cols:
        return pd.Series(dtype=float)
    filled = ragas_df[metric_cols].copy()
    for col in metric_cols:
        filled[col] = pd.to_numeric(filled[col], errors="coerce").fillna(0.0)
    return filled.mean(axis=1)


def _compute_radar_areas(ragas_df: pd.DataFrame) -> pd.DataFrame:
    """각 전략의 레이더 폴리곤 면적을 Shoelace 공식으로 계산한다.

    등간격 N개 축(각도 2π/N씩 간격)에서 반지름 rᵢ인 폴리곤의 면적:
        A = (1/2) * sin(2π/N) * Σᵢ rᵢ * r(i+1 mod N)

    반환 DataFrame: strategy, area (0~1 사이 정규화), rank
    """
    import math
    metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
    if len(metric_cols) < 2:
        return pd.DataFrame(columns=["strategy", "area", "rank"])

    N = len(metric_cols)
    factor = 0.5 * math.sin(2 * math.pi / N)
    # 최대 면적 (모든 값=1일 때): factor * N
    max_area = factor * N

    rows = []
    for _, row in ragas_df.iterrows():
        vals = [float(row.get(m, 0)) for m in metric_cols]
        area = factor * sum(vals[i] * vals[(i + 1) % N] for i in range(N))
        rows.append({"strategy": row["strategy"], "area": area})

    df = pd.DataFrame(rows)
    df["area_norm"] = (df["area"] / max_area * 100).round(1)  # 최대 대비 %
    df = df.sort_values("area", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def _radar_area_rank_html(ragas_df: pd.DataFrame, top_n: int = 5) -> str:
    """레이더 차트 면적 순위 테이블 HTML."""
    if ragas_df is None or ragas_df.empty:
        return ""
    area_df = _compute_radar_areas(ragas_df).head(top_n)
    if area_df.empty:
        return ""

    medals = ["🥇", "🥈", "🥉", "4위", "5위"]
    rows_html = ""
    max_area_norm = area_df["area_norm"].max() or 1
    for i, (_, row) in enumerate(area_df.iterrows()):
        medal = medals[i] if i < len(medals) else f"{i+1}위"
        bar_pct = int(row["area_norm"] / max_area_norm * 100)
        rows_html += f"""
        <tr>
          <td class="text-center fw-bold">{medal}</td>
          <td style="font-size:0.8rem">{row['strategy']}</td>
          <td class="text-center">{row['area_norm']}%</td>
          <td>
            <div class="progress" style="height:10px">
              <div class="progress-bar bg-primary" style="width:{bar_pct}%"></div>
            </div>
          </td>
        </tr>"""

    return f"""
    <table class="table table-sm table-hover mt-2">
      <thead class="table-light">
        <tr><th>#</th><th>전략</th><th>면적(%)</th><th>상대 크기</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div class="text-muted" style="font-size:0.75rem">
      * 면적(%) = 레이더 폴리곤이 최대값(모든 지표=1.0) 대비 차지하는 비율.
      Shoelace 공식 적용. 메트릭 수={len([c for c in ragas_df.columns if c != 'strategy'])}개 기준.
    </div>"""


def _ragas_table_html(ragas_df: pd.DataFrame) -> str:
    """RAGAS 메트릭 테이블 HTML (색상 코딩)."""
    import math
    metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
    if not metric_cols:
        return "<p>RAGAS 데이터 없음</p>"

    rows_html = ""
    for _, row in ragas_df.iterrows():
        cells = f"<td style='font-size:0.78rem'>{row['strategy']}</td>"
        for col in metric_cols:
            val = row.get(col, 0.0)
            if isinstance(val, float):
                if math.isnan(val):
                    cells += '<td style="text-align:center;color:#aaa">N/A</td>'
                else:
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


def _ragas_split_tables_html(ragas_df: pd.DataFrame) -> str:
    """Contextual 포함/제외 기준으로 RAGAS 테이블을 Bootstrap 탭으로 분리."""
    if ragas_df is None or ragas_df.empty:
        return "<p>데이터 없음</p>"

    no_ctx = ragas_df[~ragas_df["strategy"].str.contains("Contextual", case=False, na=False)]
    ctx    = ragas_df[ ragas_df["strategy"].str.contains("Contextual", case=False, na=False)]

    tbl_no_ctx = _ragas_table_html(no_ctx) if not no_ctx.empty else "<p class='text-muted'>데이터 없음</p>"
    tbl_ctx    = _ragas_table_html(ctx)    if not ctx.empty    else "<p class='text-muted'>데이터 없음</p>"

    n_off = len(no_ctx)
    n_on  = len(ctx)

    return f"""
    <ul class="nav nav-tabs mb-0" id="ragasTab" role="tablist">
      <li class="nav-item" role="presentation">
        <button class="nav-link active" id="ragas-tab-off" data-bs-toggle="tab"
                data-bs-target="#ragas-off" type="button" role="tab">
          <span class="badge bg-secondary me-1">OFF</span>Contextual OFF
          <span class="badge bg-light text-dark ms-1">{n_off}</span>
        </button>
      </li>
      <li class="nav-item" role="presentation">
        <button class="nav-link" id="ragas-tab-on" data-bs-toggle="tab"
                data-bs-target="#ragas-on" type="button" role="tab">
          <span class="badge bg-success me-1">ON</span>Contextual ON
          <span class="badge bg-light text-dark ms-1">{n_on}</span>
        </button>
      </li>
    </ul>
    <div class="tab-content border border-top-0 p-3 bg-white rounded-bottom">
      <div class="tab-pane fade show active" id="ragas-off" role="tabpanel">
        <div class="table-responsive">{tbl_no_ctx}</div>
      </div>
      <div class="tab-pane fade" id="ragas-on" role="tabpanel">
        <div class="table-responsive">{tbl_ctx}</div>
      </div>
    </div>"""


def _latency_table_html(latency_df: pd.DataFrame) -> str:
    """레이턴시 요약 테이블 HTML.

    이상치(IQR×1.5) 감지 시 ⚠️ 마킹 + 중앙값(p50) 부표시.
    소규모 샘플(5개 이하)에서는 단일 이상치가 평균을 크게 왜곡할 수 있음을 주석으로 명시.
    """
    latency_df = _agg_latency(latency_df)
    sort_col = "avg_latency" if "avg_latency" in latency_df.columns else "avg_latency_ms"
    if sort_col not in latency_df.columns:
        return "<p>레이턴시 데이터 없음</p>"

    df = latency_df.sort_values(sort_col).head(20)
    unit = "s" if sort_col == "avg_latency" else "ms"

    has_outlier_col = "has_outlier" in df.columns
    has_mean = "mean_latency_ms" in df.columns  # 참고용 평균(이상치 민감)
    has_cv = "cv_pct" in df.columns
    has_n = "n_queries" in df.columns

    rows_html = ""
    any_outlier = False
    for i, (_, row) in enumerate(df.iterrows(), 1):
        val = row[sort_col]
        badge = ' <span class="badge bg-warning text-dark">TOP3</span>' if i <= 3 else ""

        # 이상치 경고 배지
        outlier_warn = ""
        if has_outlier_col and row.get("has_outlier", False):
            any_outlier = True
            cv = row.get("cv_pct", "") if has_cv else ""
            cv_str = f"CV={cv:.0f}%" if isinstance(cv, float) and not pd.isna(cv) else ""
            outlier_warn = (
                f' <span style="color:#e65100;font-size:0.72rem;cursor:help" '
                f'title="이상치 쿼리 포함 가능성 ({cv_str}) — 중앙값 참조 권장">⚠️이상치</span>'
            )

        # 평균 부표시 (중앙값이 주 지표이므로 평균을 참고용으로 표시)
        median_str = ""
        if has_mean and sort_col == "avg_latency_ms":
            mean_val = row.get("mean_latency_ms", None)
            if mean_val is not None and not (isinstance(mean_val, float) and pd.isna(mean_val)):
                median_str = f'<br><span style="font-size:0.71rem;color:#888">평균: {mean_val:.1f}{unit}</span>'

        rows_html += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td style='font-size:0.82rem'>{row['strategy']}{badge}{outlier_warn}{median_str}</td>"
            f"<td style='white-space:nowrap'>{val:.1f}{unit}</td>"
            f"</tr>"
        )

    # 이상치 주석 (이상치 전략이 하나라도 있을 때)
    n_val = int(df["n_queries"].iloc[0]) if has_n and not df.empty else 0
    outlier_note = ""
    if any_outlier:
        sample_warn = (
            f" 특히 이 벤치마크는 <strong>{n_val}개 쿼리</strong> 기준이므로 단일 이상치가 평균에 미치는 영향이 큽니다."
            if n_val > 0 else ""
        )
        outlier_note = f"""
    <div class="mt-2 p-2 rounded" style="background:#fff3cd;border-left:3px solid #ffc107;font-size:0.75rem;color:#664d03;">
      ⚠️ <strong>이상치 경고</strong>: IQR×1.5 기준 이상치 쿼리가 포함된 전략입니다.{sample_warn}
      표시된 값은 <strong>중앙값(Median)</strong>이므로 이상치 영향을 최소화했습니다. 참고용 평균은 각 전략 아래에 표시됩니다.
    </div>"""

    return f"""
    <table class="table table-sm table-hover">
      <thead class="table-secondary">
        <tr><th>#</th><th>전략</th><th>중앙값 레이턴시</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>{outlier_note}"""


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


def _executive_summary_html(
    ragas_df: pd.DataFrame,
    latency_df: Optional[pd.DataFrame],
) -> str:
    """최적 전략 추천 Executive Summary 카드.

    _compute_weighted_scores로 1위 전략을 산출하고,
    전략 이름에서 특성(ColBERT/Contextual/BM25/SPLADE 등)을 파싱해
    한줄 이유를 자동 생성합니다.
    """
    if ragas_df is None or ragas_df.empty:
        return ""

    try:
        scores = _compute_weighted_scores(ragas_df)
        ragas_copy = ragas_df.copy()
        ragas_copy["_score"] = scores
        best_idx = ragas_copy["_score"].idxmax()
        best_row = ragas_copy.loc[best_idx]
        best_name = best_row["strategy"]
        best_score = best_row["_score"]

        # 레이턴시 정보 가져오기
        lat_str = "N/A"
        lat_badge_color = "#6c757d"
        if latency_df is not None:
            agg_lat = _agg_latency(latency_df)
            sort_col = (
                "avg_latency_ms" if "avg_latency_ms" in agg_lat.columns
                else ("avg_latency" if "avg_latency" in agg_lat.columns else None)
            )
            if sort_col:
                lat_row = agg_lat[agg_lat["strategy"] == best_name]
                if not lat_row.empty:
                    lat_val = lat_row[sort_col].values[0]
                    unit = "s" if sort_col == "avg_latency" else "ms"
                    lat_str = f"{lat_val:.1f}{unit}"
                    # 레이턴시 색상: 3000ms 기준
                    ms_val = lat_val * 1000 if sort_col == "avg_latency" else lat_val
                    lat_badge_color = "#4CAF50" if ms_val <= 3000 else "#FF9800"

        # 전략 이름에서 특성 파싱 → 한줄 이유 생성
        reasons = []
        name_lower = best_name.lower()
        if "contextual" in name_lower:
            reasons.append("Contextual Retrieval로 청크 문맥 품질 강화")
        if "colbert" in name_lower:
            reasons.append("ColBERT Late-Interaction 리랭킹으로 정밀도 향상")
        elif "flashrank" in name_lower:
            reasons.append("FlashRank 경량 리랭킹으로 속도·품질 균형")
        if "splade" in name_lower:
            reasons.append("SPLADE Sparse 검색으로 유의어 확장 강점")
        elif "bm25" in name_lower:
            reasons.append("Korean BM25로 정확한 키워드 매칭")
        if not reasons:
            reasons.append("4개 RAGAS 지표 균형 최우수")

        reason_text = " · ".join(reasons)

        # 메트릭별 점수 배지
        metric_cols = [c for c in ragas_df.columns if c not in ("strategy",)]
        metric_badges = ""
        for m in metric_cols:
            val = best_row.get(m, None)
            if val is not None:
                try:
                    fval = float(val)
                    color = "#4CAF50" if fval >= 0.7 else ("#FF9800" if fval >= 0.5 else "#F44336")
                    metric_badges += (
                        f'<span style="background:{color};color:#fff;border-radius:12px;'
                        f'padding:3px 10px;font-size:0.78rem;margin:2px;display:inline-block;">'
                        f'{m}: {fval:.4f}</span>'
                    )
                except (ValueError, TypeError):
                    pass

        return f"""
    <div style="
      background: linear-gradient(135deg, #0d1b2a 0%, #1a3a5c 50%, #0d1b2a 100%);
      border-radius: 16px;
      padding: 28px 32px;
      margin-bottom: 28px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.35);
      color: #ffffff;
      border: 1px solid rgba(33,150,243,0.3);
    ">
      <div class="d-flex align-items-center mb-3">
        <span style="font-size:2rem;margin-right:12px;">🏆</span>
        <h4 style="margin:0;font-weight:800;letter-spacing:0.5px;color:#e3f2fd;">
          최적 전략 추천
        </h4>
        <span style="margin-left:auto;font-size:0.8rem;color:#90caf9;opacity:0.8;">
          RAGAS 가중 점수 기준
        </span>
      </div>

      <div style="
        background: rgba(33,150,243,0.15);
        border: 1px solid rgba(33,150,243,0.4);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
      ">
        <div style="font-size:0.82rem;color:#90caf9;margin-bottom:4px;">1위 전략</div>
        <div style="font-size:1.1rem;font-weight:700;color:#ffffff;word-break:break-all;">
          {best_name}
        </div>
      </div>

      <div class="row g-3 mb-3">
        <div class="col-md-4">
          <div style="
            background:rgba(255,255,255,0.08);
            border-radius:8px;
            padding:12px 16px;
            text-align:center;
          ">
            <div style="font-size:0.78rem;color:#90caf9;margin-bottom:4px;">종합 RAGAS 점수</div>
            <div style="font-size:1.6rem;font-weight:800;color:#4fc3f7;">{best_score:.4f}</div>
          </div>
        </div>
        <div class="col-md-4">
          <div style="
            background:rgba(255,255,255,0.08);
            border-radius:8px;
            padding:12px 16px;
            text-align:center;
          ">
            <div style="font-size:0.78rem;color:#90caf9;margin-bottom:4px;">중앙값 레이턴시</div>
            <div style="font-size:1.6rem;font-weight:800;color:{lat_badge_color};">{lat_str}</div>
          </div>
        </div>
        <div class="col-md-4">
          <div style="
            background:rgba(255,255,255,0.08);
            border-radius:8px;
            padding:12px 16px;
            text-align:center;
          ">
            <div style="font-size:0.78rem;color:#90caf9;margin-bottom:4px;">평가 전략 수</div>
            <div style="font-size:1.6rem;font-weight:800;color:#aed6f1;">{len(ragas_df)}</div>
          </div>
        </div>
      </div>

      <div style="margin-bottom:14px;">
        <div style="font-size:0.8rem;color:#90caf9;margin-bottom:6px;">선정 이유</div>
        <div style="
          background:rgba(255,255,255,0.07);
          border-left:3px solid #2196F3;
          border-radius:0 8px 8px 0;
          padding:10px 14px;
          font-size:0.88rem;
          color:#e3f2fd;
          line-height:1.6;
        ">{reason_text}</div>
      </div>

      <div>
        <div style="font-size:0.8rem;color:#90caf9;margin-bottom:6px;">메트릭별 점수</div>
        <div>{metric_badges}</div>
      </div>
    </div>"""
    except Exception:
        return ""


def _layer_contribution_html(ragas_df: pd.DataFrame) -> str:
    """레이어별 기여도 요약표 HTML.

    ragas_df를 입력받아 레이어 조합별 평균 RAGAS 점수 차이를 분석합니다.
    - Reranker 효과: None vs ColBERT vs FlashRank
    - Contextual 효과: OFF vs ON
    각 그룹의 평균 weighted score + 향상폭(Delta) 표시.
    """
    if ragas_df is None or ragas_df.empty:
        return ""

    try:
        scores = _compute_weighted_scores(ragas_df)
        ragas_copy = ragas_df.copy()
        ragas_copy["_score"] = scores

        # --- Reranker 효과 분석 ---
        def _reranker_group(name: str) -> str:
            if "ColBERT Rerank" in name or "ColBERT Reranker" in name or "colbert" in name.lower():
                return "ColBERT"
            elif "FlashRank" in name:
                return "FlashRank"
            else:
                return "없음"

        ragas_copy["_reranker"] = ragas_copy["strategy"].apply(_reranker_group)
        reranker_agg = (
            ragas_copy.groupby("_reranker")["_score"]
            .agg(mean_score="mean", count="count")
            .reset_index()
            .rename(columns={"_reranker": "group"})
        )
        # 기준: Reranker 없음
        base_reranker = reranker_agg.loc[reranker_agg["group"] == "없음", "mean_score"]
        base_reranker_val = base_reranker.values[0] if not base_reranker.empty else None

        reranker_order = ["없음", "ColBERT", "FlashRank"]
        reranker_rows_html = ""
        for grp in reranker_order:
            row = reranker_agg[reranker_agg["group"] == grp]
            if row.empty:
                continue
            score = row["mean_score"].values[0]
            cnt = int(row["count"].values[0])
            if base_reranker_val is not None and grp != "없음":
                delta = score - base_reranker_val
                delta_str = f'+{delta:.4f}' if delta >= 0 else f'{delta:.4f}'
                delta_color = "#4CAF50" if delta >= 0 else "#F44336"
            else:
                delta_str = "기준"
                delta_color = "#6c757d"
            bar_pct = int(score * 100)
            reranker_rows_html += f"""
            <tr>
              <td><strong>{grp}</strong></td>
              <td class="text-center">{cnt}</td>
              <td class="text-center">{score:.4f}</td>
              <td>
                <div class="progress" style="height:10px">
                  <div class="progress-bar bg-primary" style="width:{bar_pct}%"></div>
                </div>
              </td>
              <td class="text-center fw-bold" style="color:{delta_color};">{delta_str}</td>
            </tr>"""

        # --- Contextual 효과 분석 ---
        ragas_copy["_contextual"] = ragas_copy["strategy"].apply(
            lambda n: "ON" if "Contextual" in n else "OFF"
        )
        ctx_agg = (
            ragas_copy.groupby("_contextual")["_score"]
            .agg(mean_score="mean", count="count")
            .reset_index()
            .rename(columns={"_contextual": "group"})
        )
        base_ctx = ctx_agg.loc[ctx_agg["group"] == "OFF", "mean_score"]
        base_ctx_val = base_ctx.values[0] if not base_ctx.empty else None

        ctx_rows_html = ""
        for grp in ["OFF", "ON"]:
            row = ctx_agg[ctx_agg["group"] == grp]
            if row.empty:
                continue
            score = row["mean_score"].values[0]
            cnt = int(row["count"].values[0])
            if base_ctx_val is not None and grp == "ON":
                delta = score - base_ctx_val
                delta_str = f'+{delta:.4f}' if delta >= 0 else f'{delta:.4f}'
                delta_color = "#4CAF50" if delta >= 0 else "#F44336"
            else:
                delta_str = "기준"
                delta_color = "#6c757d"
            bar_pct = int(score * 100)
            ctx_rows_html += f"""
            <tr>
              <td><strong>Contextual {grp}</strong></td>
              <td class="text-center">{cnt}</td>
              <td class="text-center">{score:.4f}</td>
              <td>
                <div class="progress" style="height:10px">
                  <div class="progress-bar bg-success" style="width:{bar_pct}%"></div>
                </div>
              </td>
              <td class="text-center fw-bold" style="color:{delta_color};">{delta_str}</td>
            </tr>"""

        # --- 순수 Contextual 효과: 동일 base 전략 1:1 쌍 비교 ---
        # OFF 전략 이름에서 DS(...) 핵심 부분 추출 → ON 짝 찾기
        # OFF 패턴: "DS(X)", "ColBERT Rerank (DS(X))", "FlashRank Rerank (DS(X))"
        # ON  패턴: "Contextual Retrieval (DS(X))", "ColBERT Rerank (Contextual Retrieval (DS(X)))", ...
        import re as _re

        def _extract_base_key(name: str):
            """전략명에서 (Reranker종류, DS(X)) 형태의 base key 추출."""
            # ColBERT Rerank (Contextual Retrieval (DS(X))) → ("ColBERT", "X")
            m = _re.match(r"ColBERT Rerank \(Contextual Retrieval \(DS\((.+)\)\)\)", name)
            if m:
                return ("ColBERT", m.group(1))
            # FlashRank Rerank (Contextual Retrieval (DS(X))) → ("FlashRank", "X")
            m = _re.match(r"FlashRank Rerank \(Contextual Retrieval \(DS\((.+)\)\)\)", name)
            if m:
                return ("FlashRank", m.group(1))
            # Contextual Retrieval (DS(X)) → ("None", "X")
            m = _re.match(r"Contextual Retrieval \(DS\((.+)\)\)", name)
            if m:
                return ("None", m.group(1))
            # ColBERT Rerank (DS(X)) → ("ColBERT", "X")
            m = _re.match(r"ColBERT Rerank \(DS\((.+)\)\)", name)
            if m:
                return ("ColBERT", m.group(1))
            # FlashRank Rerank (DS(X)) → ("FlashRank", "X")
            m = _re.match(r"FlashRank Rerank \(DS\((.+)\)\)", name)
            if m:
                return ("FlashRank", m.group(1))
            # DS(X) → ("None", "X")
            m = _re.match(r"DS\((.+)\)", name)
            if m:
                return ("None", m.group(1))
            return None

        ragas_copy["_base_key"] = ragas_copy["strategy"].apply(_extract_base_key)
        off_df = ragas_copy[ragas_copy["_contextual"] == "OFF"][["_base_key", "_score"]].dropna(subset=["_base_key"])
        on_df  = ragas_copy[ragas_copy["_contextual"] == "ON"][["_base_key", "_score"]].dropna(subset=["_base_key"])
        # key를 문자열로 변환하여 merge
        off_df = off_df.copy(); off_df["_key_str"] = off_df["_base_key"].astype(str)
        on_df  = on_df.copy();  on_df["_key_str"]  = on_df["_base_key"].astype(str)
        paired = off_df.merge(on_df, on="_key_str", suffixes=("_off", "_on"))
        paired["_pure_delta"] = paired["_score_on"] - paired["_score_off"]

        pure_delta_html = ""
        if not paired.empty:
            pure_mean  = paired["_pure_delta"].mean()
            n_pairs    = len(paired)
            pos_pairs  = int((paired["_pure_delta"] > 0).sum())
            neg_pairs  = int((paired["_pure_delta"] < 0).sum())
            zero_pairs = n_pairs - pos_pairs - neg_pairs
            pure_color = "#4CAF50" if pure_mean >= 0 else "#F44336"
            pure_str   = f'+{pure_mean:.4f}' if pure_mean >= 0 else f'{pure_mean:.4f}'
            pure_delta_html = f"""
        <div class="mt-2 p-2 rounded" style="background:#e8f5e9;border-left:3px solid #4CAF50;font-size:0.78rem;color:#1b5e20;">
          🔬 <strong>순수 Contextual 효과 (동일 base 전략 {n_pairs}쌍 1:1 비교)</strong>:
          평균 Delta = <strong style="color:{pure_color};">{pure_str}</strong>
          &nbsp;|&nbsp; 향상: {pos_pairs}쌍 / 동일: {zero_pairs}쌍 / 하락: {neg_pairs}쌍<br>
          <span style="font-size:0.72rem;color:#388e3c;">
            Contextual ON/OFF 전체 그룹 평균 Delta와 부호가 다를 수 있습니다 — 그룹 평균은 Reranker 종류 등 다른 레이어 효과를 포함하기 때문입니다.
          </span>
        </div>"""

        # --- NaN 감지 + 과대평가 폭 계산 ---
        metric_cols_local = [c for c in ragas_df.columns if c not in ("strategy",)]
        n_metrics = len(metric_cols_local)
        nan_notes = []
        for grp_name, grp_key, grp_col in [
            ("FlashRank", "FlashRank", "_reranker"),
            ("ColBERT", "ColBERT", "_reranker"),
            ("없음", "없음", "_reranker"),
        ]:
            grp = ragas_copy[ragas_copy[grp_col] == grp_key]
            for m in metric_cols_local:
                nan_rows = grp[grp[m].isna()]
                nan_cnt = len(nan_rows)
                if nan_cnt > 0:
                    # skipna=True 점수 vs NaN=0 점수 비교
                    overest_parts = []
                    for _, nan_row in nan_rows.iterrows():
                        # metric/strategy 컬럼만 남겨 추가된 _reranker/_score 등 제거
                        orig_cols = ["strategy"] + metric_cols_local
                        row_df = nan_row[orig_cols].to_frame().T.reset_index(drop=True)
                        skipna_score_raw = _compute_weighted_scores(row_df).iloc[0]
                        penalized_score_raw = _compute_weighted_scores_nan_penalized(row_df).iloc[0]
                        # 전체 NaN(skipna_score=NaN)이면 과대평가 계산 불가
                        if pd.isna(skipna_score_raw):
                            overest_parts.append(
                                f"<em>{str(nan_row['strategy'])[:60]}…</em>: "
                                f"모든 메트릭 NaN — 가중 점수 계산 불가 "
                                f"(<strong style='color:#c62828'>순위 집계에서 제외 권장</strong>)"
                            )
                            continue
                        skipna_score = float(skipna_score_raw)
                        penalized_score = float(penalized_score_raw)
                        gap = skipna_score - penalized_score
                        overest_parts.append(
                            f"<em>{nan_row['strategy'][:60]}…</em>: "
                            f"현재 {skipna_score:.4f} vs NaN=0 기준 {penalized_score:.4f} "
                            f"(<strong style='color:#c62828'>+{gap:.4f} 과대평가</strong>)"
                        )
                    nan_notes.append(
                        f"<strong>{grp_name}</strong> 그룹 <code>{m}</code>: NaN {nan_cnt}개<br>"
                        + "<br>".join(overest_parts)
                    )

        nan_note_html = ""
        if nan_notes:
            nan_note_html = f"""
        <div class="mt-2 p-2 rounded" style="background:#fce4ec;border-left:3px solid #e91e63;font-size:0.75rem;color:#880e4f;">
          🔴 <strong>NaN 처리 주의</strong>: NaN 메트릭은 평균 계산에서 제외(skipna=True)되므로
          해당 전략의 가중 점수가 실제보다 높게 산출될 수 있습니다.<br>
          {'<br>'.join(nan_notes)}
        </div>"""

        return f"""
    <div class="p-3 rounded mb-3" style="background:#fff;border:1px solid #dee2e6;box-shadow:0 2px 6px rgba(0,0,0,0.06);">
      <h6 style="font-size:0.9rem;font-weight:700;color:#1a237e;margin-bottom:14px;">
        📊 레이어별 기여도 분석 — 평균 RAGAS 가중 점수 비교
      </h6>

      <!-- 혼합 그룹 비교 한계 경고 -->
      <div class="mb-3 p-2 rounded" style="background:#fff8e1;border-left:3px solid #ffc107;font-size:0.78rem;color:#5d4037;">
        ⚠️ <strong>해석 주의</strong>: 이 비교는 <strong>그룹 전체 평균</strong>으로, 각 그룹 내에 다른 레이어(Dense 모델, Sparse 모델 등) 효과가 혼입됩니다.
        예를 들어 Contextual ON/OFF 비교 시 ON 그룹에는 FlashRank+Contextual처럼 Reranker 효과도 포함됩니다.
        <strong>순수 레이어 효과</strong>를 측정하려면 동일한 base 전략(OFF vs ON, 또는 Reranker 없음 vs 있음)을 1:1 대비해야 합니다.
      </div>

      <!-- Reranker 효과 -->
      <div class="mb-3">
        <div style="font-size:0.82rem;font-weight:700;color:#37474f;margin-bottom:6px;">
          Layer 3: Reranker 효과
        </div>
        <div class="table-responsive">
          <table class="table table-sm table-hover table-bordered mb-1" style="font-size:0.82rem;">
            <thead class="table-light">
              <tr><th>Reranker</th><th class="text-center">전략 수</th><th class="text-center">평균 점수</th><th>점수 분포</th><th class="text-center">향상폭 (Δ)</th></tr>
            </thead>
            <tbody>{reranker_rows_html}</tbody>
          </table>
        </div>
        <div class="text-muted" style="font-size:0.75rem;">* 기준: Reranker 없음 그룹 평균 대비 향상폭 (Layer 1·2·4 효과 혼입)</div>
      </div>

      <!-- Contextual 효과 -->
      <div>
        <div style="font-size:0.82rem;font-weight:700;color:#37474f;margin-bottom:6px;">
          Layer 4: Contextual Retrieval 효과
        </div>
        <div class="table-responsive">
          <table class="table table-sm table-hover table-bordered mb-1" style="font-size:0.82rem;">
            <thead class="table-light">
              <tr><th>Contextual</th><th class="text-center">전략 수</th><th class="text-center">평균 점수</th><th>점수 분포</th><th class="text-center">향상폭 (Δ)</th></tr>
            </thead>
            <tbody>{ctx_rows_html}</tbody>
          </table>
        </div>
        <div class="text-muted" style="font-size:0.75rem;">* 기준: Contextual OFF 그룹 평균 대비 향상폭 (Layer 1·2·3 효과 혼입)</div>
        {pure_delta_html}
      </div>

      {nan_note_html}
    </div>"""
    except Exception:
        return ""


def _total_timing_table_html(timing_df: Optional[pd.DataFrame], top_n: int = 20, exclude_contextual: bool = False, evaluated_only: bool = True) -> str:
    """조합별 총 소요시간 테이블 (인덱싱+검색+평가 합계 순위).

    combo_timing.csv의 컬럼: label, dense, sparse, reranker, llm_support,
                              build_s, pass1_s, pass1_s_per_qa, pass2_s, pass2_s_per_qa, total_s
    """
    if timing_df is None or timing_df.empty:
        return "<p class='text-muted'>타이밍 데이터 없음 (combo_timing.csv 필요)</p>"
    if "total_s" not in timing_df.columns:
        return "<p class='text-muted'>total_s 컬럼 없음</p>"

    df = timing_df.copy()
    # Pass 2 평가가 실제로 수행된 전략만 포함 (pass2_s > 0)
    if evaluated_only and "pass2_s" in df.columns:
        df = df[df["pass2_s"] > 0]
    if exclude_contextual:
        if "llm_support" in df.columns:
            df = df[df["llm_support"].fillna("none") != "contextual"]
        elif "label" in df.columns:
            df = df[~df["label"].str.contains("contextual", case=False, na=False)]
    df = df.sort_values("total_s").head(top_n)

    # 컬럼 안전 접근
    def _fmt(row, col, unit="s"):
        v = row.get(col, None)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        return f"{v:.1f}{unit}"

    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows(), 1):
        label = str(row.get("label", ""))
        build_s = float(row.get("build_s", 0) or 0)
        pass1_s = float(row.get("pass1_s", 0) or 0)
        pass2_s = float(row.get("pass2_s", 0) or 0)
        total_s = float(row.get("total_s", 0) or 0)
        badge = ' <span class="badge bg-warning text-dark">TOP3</span>' if i <= 3 else ""
        # 비율 계산 (build/total)
        build_pct = int(build_s / total_s * 100) if total_s > 0 else 0
        rows_html += f"""
        <tr>
          <td class="text-center">{i}</td>
          <td style="font-size:0.82rem">{label}{badge}</td>
          <td class="text-center">{build_s:.1f}s</td>
          <td class="text-center">{pass1_s:.1f}s</td>
          <td class="text-center">{pass2_s:.1f}s</td>
          <td class="text-center fw-bold">{total_s:.1f}s</td>
          <td>
            <div class="progress" style="height:8px" title="인덱싱 비중 {build_pct}%">
              <div class="progress-bar bg-info" style="width:{build_pct}%" title="인덱싱"></div>
            </div>
            <div style="font-size:0.7rem;color:#888">인덱싱 {build_pct}%</div>
          </td>
        </tr>"""

    return f"""
    <table class="table table-sm table-hover table-bordered">
      <thead class="table-dark">
        <tr>
          <th>#</th>
          <th>전략 조합 (Label)</th>
          <th>인덱싱 (build)</th>
          <th>Pass 1 (검색)</th>
          <th>Pass 2 (평가)</th>
          <th>합계</th>
          <th>인덱싱 비중</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div class="text-muted" style="font-size:0.76rem">
      * <strong>인덱싱(build)</strong> = 임베딩+벡터DB 구축 소요 시간 (캐시 재사용 시 단축).
      <strong>Pass 1</strong> = 전체 전략 레이턴시 측정. <strong>Pass 2</strong> = RAGAS 평가용 추론 시간.
    </div>"""


def _load_cold_on_times(history_dir: str) -> Optional[pd.DataFrame]:
    """run_history JSON에서 Contextual ON 비캐시 최초 빌드 시간을 추출.

    Returns:
        dense, sparse, on_s 컬럼을 가진 DataFrame, 또는 None.
    """
    import glob as _glob
    import os as _os

    files = sorted(_glob.glob(_os.path.join(history_dir, "run_*.json")))
    if not files:
        return None

    records = []
    for f in files:
        try:
            with open(f) as fp:
                d = json.load(fp)
        except Exception:
            continue
        for t in d.get("strategy_timings", []):
            if (t.get("llm_support") or "") == "contextual":
                records.append({
                    "dense": t.get("dense_model", ""),
                    "sparse": t.get("sparse_model", ""),
                    "build_time_s": float(t.get("build_time_s") or 0),
                })

    if not records:
        return None

    hist = pd.DataFrame(records)
    # 실제 LLM 호출이 있었던 최초 인덱싱 시간 = build_time_s > 0 중 최대값
    on_cold = (
        hist[hist["build_time_s"] > 0]
        .groupby(["dense", "sparse"], as_index=False)["build_time_s"]
        .max()
        .rename(columns={"build_time_s": "on_s"})
    )
    return on_cold if not on_cold.empty else None


def _layer4_timing_html(
    timing_df: Optional[pd.DataFrame],
    history_dir: Optional[str] = None,
) -> str:
    """Layer 4 — Contextual Retrieval 인덱싱 시간 비교표.

    OFF는 현재 timing_df 기준, ON은 history_dir가 있으면 비캐시 최초 시간으로 표기.
    """
    if timing_df is None or timing_df.empty:
        return ""
    required = {"dense", "sparse", "llm_support", "build_s"}
    if not required.issubset(timing_df.columns):
        return ""

    df = timing_df.copy()
    df["llm_support"] = df["llm_support"].fillna("none")
    df["build_s"] = pd.to_numeric(df["build_s"], errors="coerce").fillna(0.0)

    # OFF: (dense, sparse)별 평균
    off_df = (
        df[df["llm_support"] == "none"][["dense", "sparse", "build_s"]]
        .groupby(["dense", "sparse"], as_index=False)["build_s"].mean()
        .rename(columns={"build_s": "off_s"})
    )

    # ON: history에서 비캐시 시간 우선, 없으면 현재 timing_df max 사용
    on_cold = None
    if history_dir:
        try:
            on_cold = _load_cold_on_times(history_dir)
        except Exception:
            on_cold = None

    if on_cold is not None:
        on_df = on_cold
        note = "* Contextual ON 시간은 이전 실행 기록 기준 <strong>캐시 없는 최초 LLM 인덱싱 시간</strong>입니다."
    else:
        on_df = (
            df[df["llm_support"] == "contextual"][["dense", "sparse", "build_s"]]
            .groupby(["dense", "sparse"], as_index=False)["build_s"].max()
            .rename(columns={"build_s": "on_s"})
        )
        note = "* Contextual ON 시간은 현재 실행 기준입니다 (캐시 효과 포함)."

    merged = off_df.merge(on_df, on=["dense", "sparse"], how="inner")
    if merged.empty:
        return ""

    merged["off_s"] = pd.to_numeric(merged["off_s"], errors="coerce").fillna(0.0)
    merged["on_s"] = pd.to_numeric(merged["on_s"], errors="coerce").fillna(0.0)
    merged["overhead_s"] = merged["on_s"]
    merged = merged.sort_values("on_s", ascending=False)

    rows_html = ""
    for _, row in merged.iterrows():
        rows_html += f"""
        <tr>
          <td><span class="badge bg-primary" style="font-size:0.72rem">{row['dense']}</span></td>
          <td><span class="badge bg-warning text-dark" style="font-size:0.72rem">{row['sparse']}</span></td>
          <td class="text-center">{row['off_s']:.1f}s</td>
          <td class="text-center">{row['on_s']:.1f}s</td>
          <td class="text-center">{row['overhead_s']:.1f}s</td>
        </tr>"""

    avg_on_s = merged["on_s"].mean()
    return f"""
    <div class="chart-explain mb-2">
      Contextual ON 시 LLM이 각 청크에 문맥 요약을 부착하므로, 평균 <strong>+{avg_on_s:.0f}s</strong> 의 인덱싱 시간이 추가됩니다.
      이 비용은 <strong>일회성</strong>으로, 캐시 이후 재인덱싱 시 단축됩니다.
    </div>
    <div class="table-responsive">
      <table class="table table-sm table-hover table-bordered">
        <thead class="table-dark">
          <tr><th>Dense 모델</th><th>Sparse</th><th>Off (s)</th><th>On — LLM 추가 (s)</th><th>추가 시간</th></tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    <div class="text-muted" style="font-size:0.76rem">
      {note}
      Contextual 캐시는 <strong>청크 내용 해시 기준</strong>으로 공유됩니다.
      동일 문서의 첫 번째 Contextual 전략이 LLM 호출 비용을 부담하며, 이후 전략은 캐시를 재사용합니다.
    </div>"""


def _benchmark_methodology_html(n_combos: int = 60, top_n_eval: int = 0) -> str:
    """벤치마크 방법론 설명 카드."""
    top_n_text = f"상위 {top_n_eval}개 전략" if top_n_eval > 0 else "상위 N개 전략"
    return f"""
    <div class="row g-3 mb-4">
      <div class="col-md-6">
        <div class="chart-card h-100">
          <h5 class="chart-title">⚙️ 2-Pass 벤치마크 방식</h5>
          <div class="chart-explain mb-3">
            효율적인 비교를 위해 두 단계로 분리 실행합니다.
          </div>
          <div class="d-flex align-items-start mb-2">
            <span class="badge bg-secondary me-2 mt-1" style="min-width:60px">Pass 1</span>
            <div style="font-size:0.84rem">
              <strong>레이턴시 측정</strong> — 전체 {n_combos}개 전략을 대상으로 쿼리 응답 속도를 측정합니다.
              각 전략이 동일한 질문에 걸리는 평균 시간을 기록하여 속도 순위를 결정합니다.
            </div>
          </div>
          <div class="d-flex align-items-start">
            <span class="badge bg-primary me-2 mt-1" style="min-width:60px">Pass 2</span>
            <div style="font-size:0.84rem">
              <strong>RAGAS 평가</strong> — 속도 상위 {top_n_text}에 대해서만 RAGAS 4개 지표를 평가합니다.
              LLM 기반 평가라 시간·비용이 많이 들어 선별 전략만 평가합니다.
            </div>
          </div>
          <div class="mt-3 p-2 rounded" style="background:#f1f8e9;font-size:0.8rem">
            💡 이 방식으로 전략 수에 비례해 폭발적으로 늘어날 수 있는 LLM 평가 비용을 관리합니다.
          </div>
        </div>
      </div>
      <div class="col-md-6">
        <div class="chart-card h-100">
          <h5 class="chart-title">🔢 4-Layer × {n_combos}개 전략 구성</h5>
          <div class="chart-explain mb-3">
            4개 레이어의 독립적 선택지를 카테시안 곱으로 조합합니다.
          </div>
          <table class="table table-sm mb-2" style="font-size:0.83rem">
            <thead class="table-light"><tr><th>레이어</th><th>역할</th><th>선택지</th><th>개수</th></tr></thead>
            <tbody>
              <tr>
                <td><span class="badge bg-primary">Layer 1</span></td>
                <td>Dense 임베딩</td>
                <td>KoSimCSE · E5 · BGE-M3 · OpenAI · Upstage</td>
                <td class="text-center fw-bold">5</td>
              </tr>
              <tr>
                <td><span class="badge bg-warning text-dark">Layer 2</span></td>
                <td>Sparse 검색</td>
                <td>Korean BM25 · SPLADE</td>
                <td class="text-center fw-bold">2</td>
              </tr>
              <tr>
                <td><span class="badge bg-danger">Layer 3</span></td>
                <td>Reranker</td>
                <td>없음 · ColBERT · FlashRank</td>
                <td class="text-center fw-bold">3</td>
              </tr>
              <tr>
                <td><span class="badge bg-success">Layer 4</span></td>
                <td>Contextual</td>
                <td>Off · On</td>
                <td class="text-center fw-bold">2</td>
              </tr>
              <tr class="table-dark">
                <td colspan="3"><strong>총 조합 = 5 × 2 × 3 × 2</strong></td>
                <td class="text-center fw-bold">{n_combos}</td>
              </tr>
            </tbody>
          </table>
          <div style="font-size:0.8rem;color:#666">
            각 레이어는 독립적으로 교체 가능 — 기여도를 분리 분석할 수 있습니다.
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
    timing_df: Optional[pd.DataFrame] = None,
    history_dir: Optional[str] = None,
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

    # 차트 생성 (히트맵은 RAGAS 테이블로 대체됐으므로 생략)
    latency_chart = _build_latency_chart(latency_df) if latency_df is not None else ""
    scatter_chart = _build_scatter_chart(latency_df, ragas_df) if (latency_df is not None and ragas_df is not None) else ""
    radar_chart = _build_radar_chart(ragas_df) if ragas_df is not None else ""

    # 테이블 HTML
    lat_table = _latency_table_html(latency_df) if latency_df is not None else "<p>데이터 없음</p>"
    ragas_table = _ragas_split_tables_html(ragas_df) if ragas_df is not None else "<p>데이터 없음</p>"
    summary_cards = _summary_cards_html(latency_df, ragas_df, run_record)
    env_table = _env_table_html(run_record)
    recommendations = _recommendations_html(ragas_df, latency_df)
    token_breakdown_table = _token_breakdown_html(run_record)
    radar_area_rank = _radar_area_rank_html(ragas_df) if ragas_df is not None else ""
    layer4_timing = _layer4_timing_html(timing_df, history_dir=history_dir)

    # 신규: Executive Summary + 레이어 기여도
    executive_summary = _executive_summary_html(ragas_df, latency_df) if ragas_df is not None else ""
    layer_contribution = _layer_contribution_html(ragas_df) if ragas_df is not None else ""

    # 전략 수 / top_n 계산 (벤치마크 방법론 섹션용)
    _agg_lat = _agg_latency(latency_df) if latency_df is not None else None
    _n_all_strategies = int(_agg_lat["strategy"].nunique()) if _agg_lat is not None and "strategy" in _agg_lat.columns else 60
    _n_evaluated_ragas = len(ragas_df) if ragas_df is not None else 0
    methodology_html = _benchmark_methodology_html(n_combos=_n_all_strategies, top_n_eval=_n_evaluated_ragas)

    # 평가 완료된 전략 수 계산 (pass2_s > 0)
    _n_evaluated_timing = 0
    if timing_df is not None and "pass2_s" in timing_df.columns:
        _n_evaluated_timing = int((timing_df["pass2_s"] > 0).sum())
    total_timing_all = _total_timing_table_html(timing_df) if timing_df is not None else ""
    total_timing_no_ctx = _total_timing_table_html(timing_df, exclude_contextual=True) if timing_df is not None else ""
    has_timing = timing_df is not None and not timing_df.empty and _n_evaluated_timing > 0

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
    body {{ font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif; background: #f5f7fa; }}
    .section-title {{ border-left: 4px solid #2196F3; padding-left: 12px; margin: 32px 0 16px 0; font-weight: 700; }}
    .chart-box {{ background: #fff; border-radius: 8px; padding: 12px; text-align: center; }}
    .chart-card {{
      background: #fff;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      height: 100%;
    }}
    .chart-title {{
      font-size: 0.95rem;
      font-weight: 700;
      color: #1a237e;
      border-bottom: 2px solid #e3f2fd;
      padding-bottom: 8px;
      margin-bottom: 10px;
    }}
    .chart-explain {{
      background: #e8f4fd;
      border-left: 3px solid #2196F3;
      border-radius: 0 6px 6px 0;
      padding: 10px 14px;
      font-size: 0.82rem;
      color: #37474f;
      line-height: 1.6;
    }}
    .strategy-desc {{
      background: #fff8e1;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 10px;
      border-left: 3px solid #ffc107;
      font-size: 0.85rem;
    }}
    .ragas-metric-card {{
      background: #fff;
      border-radius: 10px;
      padding: 16px 20px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.07);
      margin-bottom: 12px;
      border-top: 4px solid;
    }}
    .ragas-metric-card.faithfulness {{ border-color: #4CAF50; }}
    .ragas-metric-card.answer_relevancy {{ border-color: #2196F3; }}
    .ragas-metric-card.context_precision {{ border-color: #FF9800; }}
    .ragas-metric-card.context_recall {{ border-color: #9C27B0; }}
    .ragas-metric-card .metric-name {{ font-weight: 700; font-size: 0.95rem; }}
    .ragas-metric-card .metric-range {{ font-size: 0.78rem; color: #888; }}
    .ragas-metric-card .metric-desc {{ font-size: 0.83rem; color: #37474f; margin-top: 6px; line-height: 1.5; }}
    footer {{ background: #263238; color: #cfd8dc; padding: 20px; text-align: center; margin-top: 40px; border-radius: 8px 8px 0 0; }}
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

  <!-- Executive Summary — 최적 전략 추천 -->
  {executive_summary}

  <!-- 실행 환경 -->
  <h4 class="section-title">실행 환경</h4>
  {env_table}

  <!-- API 토큰 사용량 -->
  <h4 class="section-title">API 토큰 사용량 (카테고리 × 프로바이더)</h4>
  <div class="table-responsive mb-4">
    {token_breakdown_table}
  </div>

  <!-- ══════════════════════════════════════════════════════════ -->
  <!-- 벤치마크 방법론                                               -->
  <!-- ══════════════════════════════════════════════════════════ -->
  <h4 class="section-title">벤치마크 방법론</h4>
  {methodology_html}

  <!-- ══════════════════════════════════════════════════════════ -->
  <!-- 전략 레이어 설명 (차트 위)                                     -->
  <!-- ══════════════════════════════════════════════════════════ -->
  <h4 class="section-title">전략 레이어 구조</h4>
  <p class="text-muted mb-3" style="font-size:0.88rem;">
    4개 레이어의 독립적 선택지를 조합해 총 60개 전략을 비교합니다.
    각 레이어는 서로 독립적으로 교체 가능하며, 레이어별 기여도를 분리해 분석할 수 있습니다.
    Layer 4(Contextual Retrieval)는 Layer 1~3의 어느 조합에도 독립적으로 on/off 적용이 가능한
    인덱싱 품질 강화 기법입니다.
  </p>

  <!-- Layer 1: Dense Model -->
  <div class="mb-3">
    <div class="d-flex align-items-center mb-2">
      <span class="badge bg-primary me-2" style="font-size:0.85rem;">Layer 1</span>
      <strong>Dense 임베딩 모델</strong>
      <span class="text-muted ms-2" style="font-size:0.82rem;">— 의미적 유사도 기반 벡터 검색의 핵심</span>
    </div>
    <div class="row g-2">
      <div class="col-md-4">
        <div class="strategy-desc">
          <strong>KoSimCSE-roberta</strong> <span class="badge bg-secondary">로컬</span><br>
          <small>한국어 SimCSE 방식으로 학습된 BERT 계열 모델(768d). 로컬 추론으로 레이턴시가 가장 낮음.
          한국어 문장 간 의미 유사도에 특화되어 있으며, 도메인 특화 코퍼스에서 강점을 보임.</small>
        </div>
      </div>
      <div class="col-md-4">
        <div class="strategy-desc">
          <strong>multilingual-E5-large</strong> <span class="badge bg-secondary">로컬</span><br>
          <small>100개 언어를 지원하는 Microsoft의 다국어 임베딩(1024d). "query: " / "passage: " 접두사를 붙여 비대칭 검색을 수행.
          한국어와 영어가 혼합된 문서에서 균형 잡힌 성능을 발휘.</small>
        </div>
      </div>
      <div class="col-md-4">
        <div class="strategy-desc">
          <strong>BGE-M3</strong> <span class="badge bg-secondary">로컬</span><br>
          <small>BAAI의 Multi-Lingual·Multi-Granularity·Multi-Functionality 임베딩(1024d).
          Dense + Sparse + ColBERT 세 가지 검색 방식을 단일 모델로 지원.
          한국어 포함 다국어에서 최고 수준의 성능을 보임.</small>
        </div>
      </div>
      <div class="col-md-4">
        <div class="strategy-desc">
          <strong>text-embedding-3-large</strong> <span class="badge bg-info text-dark">OpenAI API</span><br>
          <small>OpenAI의 최신 임베딩 모델(3072d). 높은 의미 표현력으로 품질이 우수하나,
          API 호출 비용과 네트워크 레이턴시가 트레이드오프. 대규모 문서 인덱싱 시 비용 검토 필요.</small>
        </div>
      </div>
      <div class="col-md-4">
        <div class="strategy-desc">
          <strong>Solar Embedding (Upstage)</strong> <span class="badge bg-success">Upstage API</span><br>
          <small>문서(passage)용과 쿼리(query)용 모델을 분리 운용하는 비대칭 임베딩.
          한국어에 최적화되어 있으며, 국내 도메인 문서에서 강점을 보임.
          API 응답 시간에 따라 레이턴시 변동 가능.</small>
        </div>
      </div>
    </div>
  </div>

  <!-- Layer 2: Sparse Model -->
  <div class="mb-3">
    <div class="d-flex align-items-center mb-2">
      <span class="badge bg-warning text-dark me-2" style="font-size:0.85rem;">Layer 2</span>
      <strong>Sparse 검색 모델</strong>
      <span class="text-muted ms-2" style="font-size:0.82rem;">— 키워드 매칭으로 Dense가 놓치는 정확한 용어를 보완</span>
    </div>
    <div class="row g-2">
      <div class="col-md-6">
        <div class="strategy-desc">
          <strong>Korean BM25</strong> <span class="badge bg-secondary">로컬</span><br>
          <small>Okapi BM25 알고리즘에 한국어 형태소 분석기(Okt)를 결합한 Sparse 검색.
          정확한 키워드·고유명사 매칭에 강하고, 추가 인프라 없이 빠르게 동작.
          Dense 임베딩이 의미를 잘 잡지 못하는 전문 용어나 약어에 특히 유용.</small>
        </div>
      </div>
      <div class="col-md-6">
        <div class="strategy-desc">
          <strong>SPLADE</strong> <span class="badge bg-secondary">로컬</span><br>
          <small>Sparse Lexical And Expansion 방식으로 BERT 기반 모델이 단어 가중치를 학습.
          BM25와 달리 유의어 확장(query expansion)을 자동으로 수행하여 재현율이 높음.
          메모리와 추론 시간이 BM25보다 더 필요하지만, 검색 품질이 일반적으로 우수.</small>
        </div>
      </div>
    </div>
  </div>

  <!-- Layer 3: Reranker -->
  <div class="mb-3">
    <div class="d-flex align-items-center mb-2">
      <span class="badge bg-danger me-2" style="font-size:0.85rem;">Layer 3</span>
      <strong>리랭커 (Reranker)</strong>
      <span class="text-muted ms-2" style="font-size:0.82rem;">— Dense+Sparse 결합 결과를 정밀하게 재순위화하는 후처리 단계</span>
    </div>
    <div class="row g-2">
      <div class="col-md-4">
        <div class="strategy-desc">
          <strong>없음 (Hybrid 기본)</strong> <span class="badge bg-secondary">추가 비용 없음</span><br>
          <small>Dense와 Sparse 점수를 RRF(Reciprocal Rank Fusion)로 합산.
          별도 리랭커 없이 속도가 가장 빠름. 베이스라인 성능 기준으로 활용.</small>
        </div>
      </div>
      <div class="col-md-4">
        <div class="strategy-desc">
          <strong>ColBERT Reranker</strong> <span class="badge bg-danger">고레이턴시</span><br>
          <small>Late-Interaction 방식으로 쿼리와 문서의 토큰 수준 상호작용을 계산.
          정밀도가 높지만 쿼리당 ~20초 이상 소요. 배치 처리나 비실시간 파이프라인에 적합.</small>
        </div>
      </div>
      <div class="col-md-4">
        <div class="strategy-desc">
          <strong>FlashRank Reranker</strong> <span class="badge bg-warning text-dark">중간 레이턴시</span><br>
          <small>경량 Cross-Encoder로 빠른 재순위화. ColBERT보다 속도가 빠르고 단순 Hybrid보다 정밀.
          속도·품질 균형이 필요한 대화형 서비스에 적합.</small>
        </div>
      </div>
    </div>
  </div>

  <!-- Layer 4: Contextual Retrieval -->
  <div class="mb-4">
    <div class="d-flex align-items-center mb-2">
      <span class="badge bg-success me-2" style="font-size:0.85rem;">Layer 4</span>
      <strong>Contextual Retrieval (선택적 인덱스 강화)</strong>
      <span class="text-muted ms-2" style="font-size:0.82rem;">— Layer 1~3과 독립적으로 적용 가능한 인덱싱 품질 향상 기법</span>
    </div>
    <div class="row g-2 mb-2">
      <div class="col-md-6">
        <div class="strategy-desc" style="border-left-color: #198754;">
          <strong>Off (기본)</strong> <span class="badge bg-secondary">추가 비용 없음</span><br>
          <small>청크 원문 그대로 인덱싱. 빠르고 저렴하지만 청크가 문서 전체 맥락을 잃을 수 있음.
          쿼리가 "지난 분기"처럼 문서 내 위치에 의존하는 표현을 포함할 때 검색 실패 가능.</small>
        </div>
      </div>
      <div class="col-md-6">
        <div class="strategy-desc" style="border-left-color: #198754;">
          <strong>On (Contextual Retrieval)</strong> <span class="badge bg-success">인덱싱 시 1회 API 비용</span><br>
          <small>인덱싱 시 LLM이 각 청크에 문서 전체 맥락 요약(Contextual Prefix)을 자동으로 부착.
          쿼리 시점 레이턴시 추가 <em>없이</em> 검색 품질을 향상.
          Anthropic 실험 결과 검색 실패율 49% 감소, 리랭킹 결합 시 67% 감소.
          API 비용은 인덱싱 시 일회성으로만 발생하며, 이후 캐시로 재사용.</small>
        </div>
      </div>
    </div>

    <!-- 레이어별 기여도 요약표 (Layer 4 인덱싱 시간 비교 바로 위) -->
    {layer_contribution}

    <!-- Layer 4 인덱싱 시간 비교표 -->
    <div class="p-3 rounded border border-success" style="background:#f0fdf4;">
      <h6 style="font-size:0.88rem;font-weight:700;color:#166534;margin-bottom:8px;">
        ⏱️ Layer 4 인덱싱 시간 비교 — Contextual OFF vs ON (캐시 없는 최초 인덱싱 기준)
      </h6>
      {layer4_timing}
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════ -->
  <!-- 섹션 A: 속도 분석 — 레이턴시 막대 + 산점도 + 순위 테이블       -->
  <!-- ══════════════════════════════════════════════════════════ -->
  <h4 class="section-title">① 속도 분석</h4>

  <div class="row g-4 mb-3">
    <!-- 레이턴시 막대 차트 -->
    <div class="col-md-7">
      <div class="chart-card">
        <h5 class="chart-title">전략별 중앙값 레이턴시 (낮을수록 좋음)</h5>
        <div class="chart-explain">
          쿼리 1건을 처리하는 <strong>중앙값</strong> 시간(ms). 이상치에 강건한 지표입니다. 막대가 짧을수록 실시간 서비스에 유리합니다.<br>
          • <span class="badge bg-primary">파란색</span> 상위 3개 전략 — 속도 최강군 &nbsp;
          • 로컬 임베딩(KoSimCSE, BGE-M3)은 네트워크 없이 빠름<br>
          • ColBERT/FlashRank 리랭킹 시 크게 증가 &nbsp;
          • Contextual은 미리 캐시하므로 쿼리 시점 속도 영향 없음<br>
          <span style="font-size:0.75rem;color:#1565c0;">★ 정렬·차트 기준: <strong>중앙값(Median)</strong> — 이상치 쿼리가 있어도 순위가 왜곡되지 않습니다.
          이상치가 있는 전략은 우측 테이블에서 ⚠️이상치 표시와 참고용 평균을 함께 확인하세요.</span>
        </div>
        <div class="chart-box mt-2">
          {_img_html(latency_chart, "레이턴시 막대 차트")}
        </div>
      </div>
    </div>

    <!-- 레이턴시 순위 테이블 -->
    <div class="col-md-5">
      <div class="chart-card">
        <h5 class="chart-title">레이턴시 순위 Top 20</h5>
        <div class="chart-explain">
          <span class="badge bg-warning text-dark">TOP3</span> 전략은 실시간 서비스에 우선 검토를 권장합니다.<br>
          ★ 표시값은 <strong>중앙값(Median)</strong> 기준 — 이상치 영향 최소화.<br>
          <span style="color:#e65100;font-weight:600;">⚠️이상치</span> 전략은 특정 쿼리에서 극단값 발생. 참고용 평균(mean)을 함께 표시합니다.<br>
          <span style="font-size:0.75rem;color:#666;">* 이 벤치마크는 5개 쿼리 기준 — 소규모 샘플에서는 이상치 1건도 순위에 영향을 줄 수 있습니다.</span>
        </div>
        <div class="table-responsive mt-2">
          {lat_table}
        </div>
      </div>
    </div>
  </div>

  <!-- 산점도 (속도 vs 품질) — 전체 너비 -->
  <div class="row g-4 mb-4">
    <div class="col-12">
      <div class="chart-card">
        <h5 class="chart-title">레이턴시 vs 품질 산점도 (왼쪽 위가 최적)</h5>
        <div class="chart-explain">
          X축 = 중앙값 레이턴시, Y축 = RAGAS 메트릭 평균. "빠르면서 품질도 높은" 최적 전략을 한눈에 파악합니다.<br>
          • <strong style="color:#2196F3;">파란 점</strong> = ColBERT 포함 전략 (고정밀, 고레이턴시) &nbsp;
          • <strong style="color:#4CAF50;">녹색 점</strong> = non-ColBERT 전략 (속도 우세) &nbsp;
          • <strong style="color:red;">빨간 점선</strong> = 실용 한계 3s 기준선<br>
          • <strong>왼쪽 위</strong> = 이상적 전략 (빠름 + 고품질) &nbsp;
          • 파레토 프론티어(좌상단 경계선) 근처 전략들이 실용적 후보군
        </div>
        <div class="chart-box mt-2">
          {_img_html(scatter_chart, "산점도")}
        </div>
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════ -->
  <!-- 섹션 B: RAGAS 종합 분석 — 레이더 + 상세 테이블 + 지표 해설   -->
  <!-- ══════════════════════════════════════════════════════════ -->
  <h4 class="section-title">② RAGAS 종합 분석</h4>

  <div class="row g-4 mb-4">
    <!-- 레이더 차트 + 면적 순위 -->
    <div class="col-md-5">
      <div class="chart-card">
        <h5 class="chart-title">최고 vs 최저 전략 레이더 비교</h5>
        <div class="chart-explain">
          RAGAS 가중 점수 1위(최고)와 최하위(최저) 전략의 4가지 품질 지표를 비교합니다.<br>
          • <strong style="color:#2196F3;">파란 영역</strong> = 최고 전략 &nbsp;
          • <strong style="color:#F44336;">빨간 영역</strong> = 최저 전략<br>
          • 두 전략 간 면적 차이가 클수록 전략 선택의 중요성이 높습니다
        </div>
        <div class="chart-box mt-2">
          {_img_html(radar_chart, "레이더 차트")}
        </div>
        <h6 class="mt-3 mb-1" style="font-size:0.85rem;font-weight:700;color:#1a237e;">📐 면적 순위 (Shoelace 공식)</h6>
        {radar_area_rank}
      </div>
    </div>

    <!-- RAGAS 상세 테이블 + 지표 해설 -->
    <div class="col-md-7">
      <div class="chart-card">
        <h5 class="chart-title">RAGAS 메트릭 상세 테이블</h5>
        <div class="chart-explain">
          Pass 2에서 평가된 전략별 4개 지표입니다. 셀 배경색이 진할수록(초록) 점수가 높습니다.
          모든 지표가 균형 잡힌 전략이 장기적으로 안정적입니다.
          <div class="mt-2 p-2 rounded" style="background:#fff3cd;border-left:3px solid #ffc107;font-size:0.8rem;color:#664d03;">
            ⚠️ FlashRank 전략의 faithfulness가 낮게 측정된 것은 짧은 재랭킹 윈도우(top-k 한정 스코어링)로 인해 문맥 다양성이 떨어지기 때문일 수 있습니다.
          </div>
          <div class="mt-2 p-2 rounded" style="background:#e8f4fd;border-left:3px solid #2196F3;font-size:0.8rem;color:#0d47a1;">
            📊 본 평가는 5개 쿼리 기준으로 통계적 변동성이 있을 수 있습니다. 운영 도입 전 더 많은 쿼리로 재검증을 권장합니다.
          </div>
        </div>
        <div class="table-responsive mt-2">
          {ragas_table}
        </div>

        <!-- RAGAS 지표 해설 -->
        <h6 class="mt-3 mb-2" style="font-size:0.85rem;font-weight:700;color:#1a237e;">📖 RAGAS 지표 해설</h6>
        <div class="row g-2">
          <div class="col-6">
            <div class="ragas-metric-card faithfulness py-2 px-3 mb-0">
              <div class="metric-name" style="font-size:0.82rem;">🟢 Faithfulness (충실도)</div>
              <div class="metric-range">0~1 | 높을수록 좋음</div>
              <div class="metric-desc">
                답변 문장이 검색된 컨텍스트에 근거하는 비율.<br>낮으면 <strong>할루시네이션</strong> 신호.
              </div>
            </div>
          </div>
          <div class="col-6">
            <div class="ragas-metric-card answer_relevancy py-2 px-3 mb-0">
              <div class="metric-name" style="font-size:0.82rem;">🔵 Answer Relevancy (답변 관련성)</div>
              <div class="metric-range">0~1 | 높을수록 좋음</div>
              <div class="metric-desc">
                답변이 질문에 얼마나 직접 대응하는가.<br>낮으면 답변이 <strong>주제를 벗어남</strong>.
              </div>
            </div>
          </div>
          <div class="col-6">
            <div class="ragas-metric-card context_precision py-2 px-3 mb-0">
              <div class="metric-name" style="font-size:0.82rem;">🟠 Context Precision (정밀도)</div>
              <div class="metric-range">0~1 | 높을수록 좋음</div>
              <div class="metric-desc">
                검색된 청크 중 실제 유용한 비율.<br>낮으면 <strong>노이즈 청크 과다</strong>.
              </div>
            </div>
          </div>
          <div class="col-6">
            <div class="ragas-metric-card context_recall py-2 px-3 mb-0">
              <div class="metric-name" style="font-size:0.82rem;">🟣 Context Recall (재현율)</div>
              <div class="metric-range">0~1 | 높을수록 좋음</div>
              <div class="metric-desc">
                정답에 필요한 정보를 빠짐없이 커버하는가.<br>낮으면 <strong>핵심 청크 미검색</strong>.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ══════════════════════════════════════════════════════════ -->
  <!-- 총 소요시간 순위 (인덱싱+검색+평가)                          -->
  <!-- ══════════════════════════════════════════════════════════ -->
  {"" if not has_timing else f"""
  <h4 class="section-title">상위 {_n_evaluated_timing}개 전략 총 소요시간 순위</h4>
  <p class="text-muted mb-3" style="font-size:0.88rem;">
    Pass 2(RAGAS 평가)까지 완료된 <strong>{_n_evaluated_timing}개 전략</strong>의 인덱싱(build) + Pass 1(검색) + Pass 2(RAGAS 평가) 합산 시간입니다.
    RAGAS 평가를 거치지 않은 전략은 pass2_s=0이므로 공정 비교를 위해 제외했습니다.
    인덱싱 시간은 캐시 재사용 시 단축되며, <strong>빠른 순으로 정렬</strong>됩니다.
  </p>

  <!-- Bootstrap 탭으로 포함/제외 분리 -->
  <ul class="nav nav-tabs mb-0" id="timingTab" role="tablist">
    <li class="nav-item" role="presentation">
      <button class="nav-link active" id="tab-all" data-bs-toggle="tab" data-bs-target="#timing-all"
              type="button" role="tab">전체 전략 (Contextual 포함)</button>
    </li>
    <li class="nav-item" role="presentation">
      <button class="nav-link" id="tab-noctx" data-bs-toggle="tab" data-bs-target="#timing-noctx"
              type="button" role="tab">Contextual 제외</button>
    </li>
  </ul>
  <div class="tab-content border border-top-0 p-3 mb-4 bg-white rounded-bottom">
    <div class="tab-pane fade show active" id="timing-all" role="tabpanel">
      <div class="table-responsive mt-2">{total_timing_all}</div>
    </div>
    <div class="tab-pane fade" id="timing-noctx" role="tabpanel">
      <div class="chart-explain mb-2">
        Contextual Retrieval은 인덱싱 시 LLM 호출로 build_s가 크게 증가합니다.
        API 비용 없는 전략만 비교할 때 이 탭을 사용하세요.
      </div>
      <div class="table-responsive mt-2">{total_timing_no_ctx}</div>
    </div>
  </div>
  """}

  <!-- 결론 & 권장사항 -->
  <h4 class="section-title">결론 &amp; 권장사항 (Top 3)</h4>
  <p class="text-muted mb-3" style="font-size:0.88rem;">
    RAGAS 4개 지표(Faithfulness · Answer Relevancy · Context Precision · Context Recall)를
    동일 가중치로 평균하여 종합 품질 점수를 계산합니다.
    레이턴시와 RAGAS 점수를 함께 고려하여 서비스 목적에 맞는 전략을 선택하세요.
  </p>
  {recommendations}
  <div class="mt-3 p-3 rounded" style="background:#e8f4fd;border-left:4px solid #2196F3;font-size:0.83rem">
    <strong>📌 전략 선택 가이드</strong><br>
    • <strong>실시간 대화형 서비스</strong>: 레이턴시 순위 Top 3 + RAGAS 점수 0.8 이상 전략 우선 검토<br>
    • <strong>오프라인 배치 처리</strong>: RAGAS 종합 점수가 가장 높은 전략 선택 (레이턴시 무관)<br>
    • <strong>비용 최소화</strong>: 로컬 임베딩(KoSimCSE·BGE-M3·E5) + Contextual Off 조합 권장<br>
    • <strong>품질 최우선</strong>: Contextual On + FlashRank Reranker 조합이 속도·품질 균형에 유리<br>
    • <strong>한국어 특화 도메인</strong>: KoSimCSE or Upstage + Korean BM25 조합이 전문 용어 매칭에 강점
  </div>

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
