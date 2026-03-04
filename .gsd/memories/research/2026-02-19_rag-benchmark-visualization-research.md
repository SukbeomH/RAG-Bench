---
title: "RAG 벤치마크 시각화 리서치 — 단계별 영향도 + 구성 다이어그램 강화 방안"
tags:
  - research
  - visualization
  - rag
  - benchmark
  - matplotlib
  - plotly
type: research
created: "2026-02-19T00:00:00+09:00"
contextual_description: "현재 colab_visualizer.py 14개 함수의 한계 분석 + RAG 벤치마크 시각화 업계 패턴 리서치 + 신규 차트 8종 추가 방안"
keywords:
  - visualization
  - ablation study
  - layer contribution
  - sankey
  - waterfall
  - violin plot
  - component diagram
  - RAG pipeline
  - colab_visualizer
related:
  - 2026-02-19_colab-graphrag-lint-mypy-cleanup
  - 2026-02-19_colab-matplotlib-korean-font-broken
---

## RAG 벤치마크 시각화 리서치

---

## 1. 현재 시각화 현황 및 한계

### 현재 14개 함수 요약

| 함수 | 차트 | 데이터 | 한계 |
|------|------|--------|------|
| `plot_latency_comparison` | 수평 막대 | avg_latency | 분포 없음, 단일 값 |
| `plot_ragas_radar` | 레이더 (plotly) | RAGAS 메트릭 | 상위 N개만 비교 |
| `plot_ragas_heatmap` | 히트맵 (seaborn) | 전략×메트릭 | 값 범위 고정 |
| `plot_latency_vs_quality` | 산점도+파레토 (plotly) | latency×quality | 단일 품질 지표 |
| `plot_layer_contribution` | 박스플롯 4개 | Layer×metric | 레이어 간 상호작용 없음 |
| `plot_cost_breakdown` | 파이 차트 | 비용 카테고리 | 세부 전략별 비용 없음 |
| `plot_run_info` | 정보 카드 테이블 | run_record | 정적, 텍스트 위주 |
| `plot_phase_timeline` | 누적 막대 | phase_times | 단계 간 의존성 없음 |
| `plot_build_times` | 수평 막대 | 전략별 빌드 | 레이어 구성 구분 없음 |
| `plot_token_usage` | 파이+막대 | token_usage | 전략별 토큰 비교 없음 |
| `display_styled_table` | HTML 테이블 | ragas_df | 정렬/필터 없음 |
| `display_weighted_scores` | HTML 테이블 | reports | 프로파일별 1개씩 |
| `create_summary_table` | DataFrame | latency+ragas | 시각화 없음 |
| `display_dashboard` | 통합 호출 | 전체 | 레이아웃 없음 |

### 현재 시각화에서 빠진 핵심 정보

1. **단계별 구성 흐름도** — 인덱싱 → 쿼리 → 평가 단계 간 의존성과 데이터 흐름
2. **레이어별 독립 기여도** — Dense/Sparse/Reranker/Contextual 각각이 품질에 미치는 영향
3. **레이어 조합 상호작용** — 특정 Dense × Reranker 조합의 시너지/충돌 효과
4. **레이턴시 분포** — avg만 있고 p50/p95/분포 없음
5. **비용-성능 비율** — 토큰 비용 대비 품질 점수 효율
6. **전략별 에러율** — 실패한 전략의 원인 분포

---

## 2. 업계 RAG 시각화 패턴 리서치

### 2.1 아키텍처 다이어그램 표준 (2024~2025)

연구 논문(RAGBench 2024, EMNLP 2024 Best Practices, RankRAG NeurIPS 2024)과
Arize Phoenix, Qdrant 가이드에서 공통적으로 사용하는 시각화 패턴:

#### 패턴 A: 3-레이어 파이프라인 플로우

```
[문서] → [청킹] → [임베딩] → [벡터 DB]
                                    ↓
[쿼리] → [쿼리 임베딩] → [검색(Dense+Sparse)] → [리랭킹] → [컨텍스트화] → [LLM] → [답변]
```
- **표현 방법**: matplotlib `FancyArrow` + `Rectangle` 패치 조합 또는 plotly `Sankey`
- **표시 정보**: 각 단계의 소요 시간, 토큰 수, 실패율을 박스 내 표기

#### 패턴 B: Sankey 다이어그램 (데이터 흐름)

품질 점수가 각 레이어를 통해 어떻게 변화하는지 흐름으로 표현.
- Dense 모델 선택 → 품질 분기 → Sparse 추가 효과 → 리랭커 효과 → Contextual 효과
- plotly `go.Sankey` 로 구현

#### 패턴 C: Ablation Waterfall Chart

베이스라인 대비 각 컴포넌트 추가 시 품질 변화를 폭포 형태로 표현.
```
Baseline (bge-m3+bm25)     0.45  ████
+ Sparse hybrid            +0.08  ████
+ FlashRank reranker       +0.06  ████
+ Contextual Retrieval     +0.12  ████
= Full pipeline            0.71  ███████
```
- matplotlib `bar` with bottom offset 또는 plotly `waterfall`
- 연구 논문(RAGBench, RankRAG)에서 ablation 결과 표현에 가장 많이 사용

#### 패턴 D: Violin / Box + Strip Plot (레이턴시 분포)

단순 평균이 아닌 쿼리별 레이턴시 분포를 전략 간 비교.
```
bge-m3 │ ░░░▓████▓░ │ p50=55ms  p95=120ms
minilm │ ░▓████▓░   │ p50=18ms  p95=45ms
```
- seaborn `violinplot` + `stripplot` 오버레이
- 이상치(outlier) 탐지에 효과적

#### 패턴 E: Bubble Chart (3차원 비교)

x=레이턴시, y=품질, size=비용(토큰) 으로 3가지 지표 동시 표현.
- 현재 `plot_latency_vs_quality` 산점도를 버블 차트로 확장
- plotly `go.Scatter(marker=dict(size=cost_normalized))`

#### 패턴 F: Correlation Heatmap (레이어 상호작용)

Dense × Sparse × Reranker 조합별 품질 점수를 피벗 테이블 히트맵으로 표현.
```
         bm25   splade  fastembed
bge-m3   0.71    0.68    0.70
minilm   0.63    0.60    0.62
e5       0.65    0.63    0.66
```
- pandas pivot_table + seaborn `heatmap(annot=True)`

#### 패턴 G: Gantt-style Phase Timeline

단계별 시간을 Gantt 차트로 표현 — 병렬 가능 구간 시각화.
현재 `plot_phase_timeline`의 누적 막대를 Gantt 스타일로 개선.
- matplotlib `broken_barh`
- 표시 정보: 단계명, 소요시간, 토큰, 병렬 가능 여부

#### 패턴 H: Radar + Score Breakdown 병렬 표시

레이더 차트(RAGAS 다차원)와 가중 점수 막대를 subplot으로 나란히 표시.
- 상위 전략 N개를 한 화면에서 비교

---

## 3. 추가 구현 권장 차트 목록

### 우선순위 High (즉시 효과 큰 것)

#### [H-1] Ablation Waterfall Chart
**목적**: 각 레이어 추가 시 품질 점수 변화를 직관적으로 표현
**데이터**: `ragas_df` + `combos` (ComboSpec 분해)
**라이브러리**: plotly `go.Waterfall`
**구현 방법**:
```python
def plot_ablation_waterfall(ragas_df, combos, base_metric="weighted_balanced"):
    # 1. baseline = reranker 없음, llm_support 없음 전략
    # 2. +sparse hybrid, +reranker, +contextual 순으로 delta 계산
    # 3. plotly Waterfall으로 시각화
```

#### [H-2] Latency Distribution Violin Plot
**목적**: 평균이 아닌 실제 분포로 전략 신뢰도 비교
**데이터**: `run_record["strategy_timings"]` — per_query_latencies
**라이브러리**: seaborn `violinplot` + `stripplot`
**구현 방법**:
```python
def plot_latency_distribution(run_record):
    # strategy_timings에서 per-query latency 리스트 추출
    # seaborn violinplot으로 전략 간 분포 비교
    # p50/p95 수평선 오버레이
```
> **주의**: `run_tracker.py`의 `StrategyTiming`에 `per_query_latencies` 필드 추가 필요 확인

#### [H-3] Layer Interaction Heatmap
**목적**: Dense × Sparse 조합 품질 피벗 테이블
**데이터**: `ragas_df` + `combos`
**라이브러리**: seaborn `heatmap(annot=True, fmt=".2f")`
**구현 방법**:
```python
def plot_layer_interaction_heatmap(ragas_df, combos, metric="weighted_balanced"):
    # ComboSpec에서 dense, sparse 추출
    # pivot_table(index=dense, columns=sparse, values=metric)
    # reranker별 서브플롯으로 분리 (없음 / flashrank / colbert)
```

#### [H-4] Bubble Chart (Latency × Quality × Cost)
**목적**: 3차원 트레이드오프 한 눈에 파악
**데이터**: `latency_df` + `ragas_df` + token 정보
**라이브러리**: plotly `go.Scatter`
**구현 방법**:
```python
def plot_tradeoff_bubble(latency_df, ragas_df, run_record):
    # 버블 크기 = 토큰 사용량(비용)
    # x = avg_latency, y = weighted_score, size = total_tokens
    # 호버 시 전략명 + 모든 지표 표시
```

### 우선순위 Medium

#### [M-1] RAG Pipeline Architecture Diagram (정적)
**목적**: 3-Layer 구성(Dense/Sparse/Reranker/Contextual)을 다이어그램으로 표현
**라이브러리**: matplotlib patches + FancyArrowPatch
**표시 내용**: 각 레이어 박스 + 연결 화살표 + 각 레이어 옵션 목록

#### [M-2] Sankey Diagram (데이터 흐름 + 품질 분기)
**목적**: 각 레이어 선택이 최종 품질에 미치는 흐름 시각화
**라이브러리**: plotly `go.Sankey`
**구현 방법**:
```
Dense(4종) → Sparse(3종) → Reranker(3종) → Quality Score 구간
```
노드 색상 = 품질 점수 구간 (낮음→높음: 빨강→초록)

#### [M-3] Gantt-style Phase Timeline 개선
**목적**: 현재 누적 막대 → 병렬 가능 단계 시각적 구분
**라이브러리**: matplotlib `broken_barh`
**개선점**: 청킹/임베딩 등 병렬 가능 단계를 다른 행에 배치

#### [M-4] Cost-Efficiency Scatter
**목적**: 토큰 비용 대비 품질 효율 비교
**데이터**: `strategy_timings[].indexing_tokens` + ragas scores
**라이브러리**: matplotlib scatter + 파레토 라인

### 우선순위 Low

#### [L-1] Error Rate Bar (전략별 실패율)
- `strategy_timings[].error_count` / `total_queries` 시각화

#### [L-2] Multi-run Trend Line
- 여러 `run_history/*.json` 파일 로드 → 시간 경과에 따른 성능 추세

#### [L-3] RAGAS per-sample Distribution
- 샘플별 점수 분포 (hist/KDE) — 평균의 신뢰도 검증용

---

## 4. 구현 우선순위 로드맵

```
Phase 1 (즉시, 현재 데이터로 구현 가능)
├── H-1: Ablation Waterfall    → ragas_df + combos 있음
├── H-3: Layer Interaction Heatmap → ragas_df + combos 있음
└── H-4: Bubble Chart          → latency_df + ragas_df + run_record 있음

Phase 2 (run_tracker 데이터 추가 필요)
├── H-2: Violin Plot           → per_query_latencies 필드 필요
├── M-1: Pipeline Diagram      → 정적, 코드만 작성
└── M-3: Gantt Timeline 개선   → phase_times 있음

Phase 3 (설계 필요)
├── M-2: Sankey Diagram        → 조합-품질 매핑 설계 필요
└── L-2: Multi-run Trend       → run_history 로더 필요
```

---

## 5. 함수별 코드 스니펫 (구현 참고)

### H-1: plotly Waterfall
```python
import plotly.graph_objects as go

fig = go.Figure(go.Waterfall(
    name="품질 기여도", orientation="v",
    measure=["absolute", "relative", "relative", "relative", "total"],
    x=["Baseline", "+Sparse", "+Reranker", "+Contextual", "최종"],
    y=[0.45, 0.08, 0.06, 0.12, 0],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    increasing={"marker": {"color": "#2ecc71"}},
    decreasing={"marker": {"color": "#e74c3c"}},
    totals={"marker": {"color": "#3498db"}},
))
fig.update_layout(
    title="레이어별 품질 기여도 (Ablation)",
    font=dict(family="NanumGothic, sans-serif"),
)
```

### H-3: seaborn Layer Interaction Heatmap
```python
import seaborn as sns
import matplotlib.pyplot as plt

pivot = merged_df.pivot_table(
    index="dense", columns="sparse", values="weighted_balanced", aggfunc="mean"
)
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=0, vmax=1)
ax.set_title("Dense × Sparse 조합별 가중 점수")
```

### H-4: Bubble Chart
```python
fig = go.Figure(go.Scatter(
    x=merged["avg_latency"],
    y=merged["weighted_balanced"],
    mode="markers+text",
    text=merged["strategy"],
    marker=dict(
        size=merged["total_tokens"].fillna(1000) / 500,
        color=merged["weighted_balanced"],
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="품질 점수"),
    ),
))
fig.update_layout(
    title="레이턴시 × 품질 × 비용 (버블 크기 = 토큰)",
    xaxis_title="평균 레이턴시 (s)",
    yaxis_title="가중 품질 점수",
    font=dict(family="NanumGothic, sans-serif"),
)
```

---

## 6. 참고 도구 및 레퍼런스

| 도구/논문 | 관련 시각화 | 링크 |
|---------|-----------|------|
| Arize Phoenix | 단계별 trace 시각화, 실행 경로 | [arize.com](https://arize.com/phoenix) |
| RAGBench 2024 | TRACe 프레임워크, ablation 테이블 | [arxiv 2407.11005](https://arxiv.org/abs/2407.11005) |
| RankRAG NeurIPS 2024 | 컴포넌트 ablation 막대 차트 | [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/db93ccb6cf392f352570dd5af0a223d3-Paper-Conference.pdf) |
| Best Practices RAG EMNLP 2024 | 모듈별 성능 비교 표 | [ACL Anthology](https://aclanthology.org/2024.emnlp-main.981.pdf) |
| Qdrant RAG Eval Guide | 레이턴시-품질 트레이드오프 | [qdrant.tech](https://qdrant.tech/blog/rag-evaluation-guide/) |
| plotly Waterfall | 폭포 차트 공식 문서 | [plotly.com/python/waterfall-charts](https://plotly.com/python/waterfall-charts/) |
| seaborn Violin Plot | 분포 시각화 | [seaborn.pydata.org](https://seaborn.pydata.org/generated/seaborn.violinplot.html) |
