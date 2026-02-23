---
title: "Session Handoff: 60개 조합 재편 + Colab 설치 최적화"
tags:
  - handoff
  - session
  - benchmark
  - colab
type: session-handoff
created: 2026-02-23T00:00:00+09:00
contextual_description: "Dense 5종(HF 3 + openai-large + upstage) × Sparse 2종 = 60개 조합 확정. Colab Cell 1.2 uv + flash-attn wheel 캐시 적용. README 전면 업데이트 완료."
keywords:
  - 60-combos
  - openai-large
  - upstage
  - korean-bm25
  - splade
  - uv
  - flash-attn
  - colab
related:
  - 2026-02-23_combo-reorg-colab-install-optimization
  - 2026-02-20_viz-phase2-complete
---

# Session Handoff: 60개 조합 재편 + Colab 설치 최적화

## Date: 2026-02-23
## Branch: master

---

## Current Architecture State

### 벤치마크 조합 구조 (확정)

```
Layer 1 Dense (5종):
  HF:   kosimcse(768d) | e5(1024d) | bge-m3(1024d)
  유료: openai-large(3072d, text-embedding-3-large)
        upstage(4096d, solar-embedding-1-query)

Layer 2 Sparse (2종):
  korean_bm25 (BM25 + KoNLPy OKt)
  splade      (SPLADE learned sparse)

Layer 3 Mode (6종):
  hybrid | +contextual | +colbert | +colbert+ctx | +flashrank | +flashrank+ctx

총합: 5 × 2 × 6 = 60개
```

### 프리셋

| 프리셋 | 조합 수 | Dense | Sparse |
|--------|---------|-------|--------|
| `quick` | 2 | bge-m3 | korean_bm25 |
| `standard` | 20 | 5종 | 2종 |
| `full` | 60 | 5종 | 2종 |

### 핵심 파일 현황

| 파일 | 상태 |
|------|------|
| `rag_bench/strategies/dense_sparse.py` | DENSE_MODELS 5종, SPARSE_TYPES 2종 |
| `rag_bench/combo/spec.py` | `_HF_DENSE_MODELS`/`_ALL_DENSE_MODELS` 분리 |
| `rag_bench_colab/rag_benchmark.ipynb` | Cell 1.2 uv+flash-attn, 38셀 |
| `rag_bench_colab/colab_runner.py` | parallel_queries 지원, 체크포인트 |
| `README.md` / `rag_bench/README.md` / `rag_bench_colab/README.md` | 60개 기준 최신화 완료 |

### 완료된 리팩토링 (이전 세션들)

| Phase | 내용 | 상태 |
|-------|------|------|
| 3 | StrategyRetriever 5개 → 1개 통합 | ✅ |
| 4 | detect_device() 중앙화 | ✅ |
| 6 | LLM 상수 중앙화 | ✅ |
| 1 | rag_bench/combo/ 패키지 추출 | ✅ |
| 5 | qa_loader / report 공유 모듈 추출 | ✅ |
| 2 | monkey-patch → DI 패턴 | ✅ |

### 완료된 시각화 (2026-02-20)

| 코드 | 함수 | 상태 |
|------|------|------|
| H-1 | `plot_ablation_waterfall` | ✅ |
| H-2 | `plot_metric_violin` | ✅ |
| H-3 | `plot_layer_interaction_heatmap` | ✅ |
| H-4 | `plot_tradeoff_bubble` | ✅ |
| M-1 | `plot_pipeline_diagram` | ✅ |
| M-3 | `plot_strategy_gantt` | ✅ |
| M-4 | `plot_cost_efficiency` | ✅ |

---

## What Was Done (이번 세션)

1. **벤치마크 조합 재구성**
   - `minilm`, `fastembed_bm25` 제거
   - `openai-large`, `upstage` 복원 → 60개 조합

2. **Colab Cell 1.2 설치 최적화**
   - uv 패키지 매니저 (pip 대비 ~8배)
   - flash-attn 사전 빌드 wheel + Drive 캐시 (30~120분 → ~30초, 2회차 ~5초)

3. **README 전면 업데이트** (3개 파일)
   - 72개 → 60개, 모델 표, 다이어그램, 프리셋, 레거시 모드 제거

---

## What Needs To Be Done Next

1. **미커밋 파일**: `ARCHITECTURE.md`, `STACK.md` 루트에 존재 — 커밋 여부 결정
2. **M-2 차트**: `plot_strategy_comparison_matrix` 아직 미구현 (필요 시)
3. **60개 조합 Colab 벤치마크 실행**: 전체 파이프라인 end-to-end 검증
   - openai-large, upstage API 키 필요
4. **유료 API 모델 조건부 처리**: `quick` 프리셋에 유료 모델 제외 확인 (현재 bge-m3만 사용 중 → OK)

---

## Critical Notes

- `patch_dense_device()`, `patch_qdrant_memory_mode()` — deprecated (no-op + DeprecationWarning). 기존 호출 코드도 에러 없이 동작
- `_ALL_DENSE_MODELS` = HF 3종 + 유료 2종 → `standard`/`full` 프리셋에 사용
- `_HF_DENSE_MODELS` = HF 3종만 → `quick` 프리셋에 사용 (유료 API 키 없어도 실행 가능)
- flash-attn wheel 이름 규칙: `flash_attn-{ver}+cu{cu}torch{torch_xy}cxx11abi{bool}-cp{py}-cp{py}-linux_x86_64.whl`
- `UV_CACHE_DIR=Drive/colab_cache/uv` 세션 간 재사용 (Drive 마운트 필요)
