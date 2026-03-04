# Session Handoff: 시각화 Phase 2 완료

## Date: 2026-02-20
## Branch: master

---

## What Was Done

### 1. 시각화 Phase 2 — 4개 차트 구현

파일: `rag_bench_colab/colab_visualizer.py` (+613줄 → 총 ~1,900줄)

#### H-2: `plot_metric_violin(reports)`
- 전략 그룹(Baseline / +Reranker / +Contextual / Both)별 per-sample RAGAS 메트릭 분포
- seaborn `violinplot(inner="box")` — 최대 4개 메트릭 서브플롯
- `per_sample_df` 없으면 집계 막대 fallback (`_plot_metric_violin_fallback`)

#### M-1: `plot_pipeline_diagram(spec, strategy_name)`
- ComboSpec → Documents→Chunking→Dense+Sparse→Qdrant→Retrieval→[Reranker?]→[Contextual LLM?]→Answer LLM 흐름도
- matplotlib `FancyBboxPatch` + `annotate` 화살표
- `spec` 없으면 `strategy_name` 문자열 파싱으로 fallback

#### M-3: `plot_strategy_gantt(run_record, top_n=40)`
- `strategy_timings`의 `build_time_s` 누적 합산으로 Gantt 바 생성 (절대 타임스탬프 없음)
- LLM 사용(Contextual)=빨강, 일반=파랑, 실패=회색
- Phase 경계선 표시

#### M-4: `plot_cost_efficiency(run_record, ragas_df, latency_df, metric)`
- X=총 추정 비용(인덱싱+쿼리), Y=RAGAS 품질
- plotly scatter + 파레토 프론티어(점선) + 효율 TOP3 라벨
- 비용 분산 시 log scale 자동 적용

### 2. 공통 리팩토링

- `_COLOR_MAP` dict + `_classify_group()` 함수 → 모듈 레벨 상수로 추출
  (기존에는 `plot_tradeoff_bubble()` 내 지역 변수였음)
- `display_dashboard()` — H-2/M-1/M-3/M-4 4개 호출 블록 추가

### 3. 노트북 업데이트

파일: `rag_bench_colab/rag_benchmark.ipynb`
- Section 8 끝에 Cell 8-5 ~ 8-8 추가 (H-2, M-1, M-3, M-4)

### 4. Colab 노트북 이전 업데이트 (동일 세션)

- Cell 1.3 Smoke Test: `from rag_bench.scripts.run_all_combos import ...` → `from rag_bench.combo import ...`
- Cell 9 비용 요약: `"GPT-3.5-turbo"` 레이블 → `"GPT-4o-mini"`

---

## 커밋 목록 (이번 세션)

```
0a92ff4 feat(viz): H-2 Violin / M-1 Pipeline / M-3 Gantt / M-4 Cost-Efficiency 구현
93dc492 fix(notebook): combo 패키지 import 경로 수정 + GPT-4o-mini 레이블 통일
854da96 fix(utils): detect_device __init__.py re-export 누락 수정
d5258b4 refactor(colab): patch_dense_device/patch_qdrant_memory_mode deprecated 처리
97b9d38 refactor(dense): DenseSparseStrategy device/:memory: 파라미터 내장
68467d6 refactor(utils): load_qa_dataset / print_ragas_table 공유 모듈 추출
8dbb0ab refactor(combo): ComboSpec/CacheConfig/IndexCacheManager → rag_bench/combo/ 패키지 추출
```

---

## Current Architecture State

### 완료된 모든 리팩토링

| Phase | 내용 | 상태 |
|-------|------|------|
| 3 | StrategyRetriever 5개 → 1개 통합 | ✅ |
| 4 | detect_device() 중앙화 | ✅ |
| 6 | LLM 상수 중앙화 | ✅ |
| 1 | rag_bench/combo/ 패키지 추출 | ✅ |
| 5 | qa_loader / report 공유 모듈 추출 | ✅ |
| 2 | monkey-patch → DI 패턴 | ✅ |

### 완료된 시각화

| 코드 | 함수 | 상태 |
|------|------|------|
| H-1 | `plot_ablation_waterfall` | ✅ |
| H-2 | `plot_metric_violin` | ✅ (이번 세션) |
| H-3 | `plot_layer_interaction_heatmap` | ✅ |
| H-4 | `plot_tradeoff_bubble` | ✅ |
| M-1 | `plot_pipeline_diagram` | ✅ (이번 세션) |
| M-3 | `plot_strategy_gantt` | ✅ (이번 세션) |
| M-4 | `plot_cost_efficiency` | ✅ (이번 세션) |

---

## Critical Notes

- `ARCHITECTURE.md`, `STACK.md` — GSD 에이전트 자동 생성, 미커밋 상태 (루트에 존재)
- `patch_dense_device()`, `patch_qdrant_memory_mode()` — no-op + DeprecationWarning 상태
  기존 Colab 노트북 셀에서 호출해도 에러 없음
- `_COLOR_MAP` — 이제 모듈 레벨 상수 (line ~14). H-2/H-4/M-4 공유

---

## What Needs To Be Done Next

1. **126개 조합 Colab 벤치마크 실행** — 전체 파이프라인 end-to-end 검증
2. **M-2 차트** — 아직 미구현 (플랜에 없음, 필요 시 추가)
3. `ARCHITECTURE.md`, `STACK.md` 커밋 여부 결정
