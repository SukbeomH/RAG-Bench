---
title: "RAG Bench 실 서비스 모듈화 — Cookbook 방식 권장"
tags:
  - pattern-discovery
  - modularization
  - service
  - cookbook
  - architecture
  - reuse
type: pattern-discovery
created: 2026-02-20T00:00:00+09:00
contextual_description: "rag_bench/ 코드의 70-80%는 실 서비스에서 재사용 가능. 별도 SDK 레이어나 in-place 리팩토링보다 cookbook/ 디렉토리 + 표준 예시 코드 방식이 최소 비용·최대 재사용성으로 권장됨."
keywords:
  - 모듈화
  - cookbook
  - 실 서비스
  - BaseRAGStrategy
  - 재사용
  - rag_sdk
  - 코드 분리
related:
  - 2026-02-20_benchmark-efficiency-api-extension-html-report
---

## RAG Bench 실 서비스 모듈화 — Cookbook 방식 권장

### 분석 배경
사용자 요청: rag_bench/ 로직을 실 서비스 개발에 재사용할 수 있도록 블록화.
세 가지 방안 비교 분석 후 권장안 도출.

---

### 코드베이스 재사용성 분석

#### 재사용 가능 (70-80%)
| 컴포넌트 | 파일 | 비고 |
|---------|------|------|
| BaseRAGStrategy (ABC) | `rag_bench/base.py` | 순수 인터페이스, 0 수정 |
| DenseSparseStrategy | `rag_bench/strategies/dense_sparse.py` | 설정값만 조정 |
| ColBERTStrategy | `rag_bench/strategies/colbert.py` | 그대로 사용 가능 |
| OpenAIEmbedStrategy | `rag_bench/strategies/openai_embed.py` | 그대로 사용 가능 |
| UpstageEmbedStrategy | `rag_bench/strategies/upstage_embed.py` | 그대로 사용 가능 |
| pdf_converter.py | `rag_bench/indexing/pdf_converter.py` | 파라미터 튜닝만 |
| chunker.py | `rag_bench/indexing/chunker.py` | 그대로 사용 가능 |

#### 벤치마크 전용 (20-30%, 실 서비스에서 미사용)
| 컴포넌트 | 이유 |
|---------|------|
| `run_all_combos.py` | 모든 조합 순회 — 서비스에선 단일 전략 선택 |
| `generate_qa.py` | 평가용 QA 자동 생성 — 서비스 불필요 |
| `colab_runner.py` | Colab 환경 전용 오케스트레이터 |
| `generate_html_report.py` | 벤치마크 결과 시각화 |
| RAGAS 평가 파이프라인 | 오프라인 품질 평가 전용 |

---

### 3가지 방안 비교

| 항목 | A: 별도 rag_sdk/ | B: In-place 리팩토링 | C: Cookbook (권장) |
|------|----------------|---------------------|-------------------|
| 비용 | 高 (패키지화, CI 추가) | 中 (전체 리팩토링) | 低 (디렉토리 추가만) |
| 리스크 | 低 (완전 분리) | 高 (기존 벤치마크 영향) | 低 (기존 코드 불변) |
| 재사용성 | 高 | 高 | 中-高 |
| 서비스 맞춤화 | 高 | 中 | 高 |
| 권장 시점 | 팀 협업·배포 시 | 대규모 프로젝트 | **즉시 시작 가능** |

---

### 권장안: Option C — Hybrid Cookbook

#### 디렉토리 구조
```
autorag/
├── rag_bench/          # 기존 벤치마크 (수정 없음)
│   └── strategies/     # 재사용 가능한 전략들
├── cookbook/           # 실 서비스용 예시 코드 (신규)
│   ├── README.md
│   ├── 01_basic_rag/          # DenseSparse + 기본 Q&A
│   │   ├── build_index.py
│   │   ├── query.py
│   │   └── config.yaml
│   ├── 02_colbert_rag/        # ColBERT Late Interaction
│   ├── 03_openai_rag/         # OpenAI Embedding
│   ├── 04_upstage_rag/        # Upstage Solar Embedding
│   ├── 05_hybrid_rerank/      # Dense+Sparse + FlashRank
│   └── shared/
│       ├── loader.py          # PDF 로딩 래퍼
│       └── retriever_factory.py  # 전략 팩토리
└── docs/
    └── service_modularization_proposal.md  # 상세 제안서
```

#### 핵심 원칙
1. **`rag_bench/strategies/`는 수정 없이 직접 import** — 패키지 경계 없음
2. **cookbook/ 각 예시는 독립 실행 가능** — 단일 진입점 스크립트
3. **설정은 YAML/env 분리** — 코드 수정 없이 모델·경로 변경 가능
4. **장기적으로 팀 협업 시 Option A(rag_sdk/) 전환 고려**

#### 즉시 사용 가능한 import 패턴
```python
# cookbook/에서 rag_bench 전략 직접 활용
from rag_bench.strategies.openai_embed import OpenAIEmbedStrategy
from rag_bench.strategies.upstage_embed import UpstageEmbedStrategy
from rag_bench.strategies.dense_sparse import DenseSparseStrategy
from rag_bench.indexing.pdf_converter import pdfs_to_markdowns
from rag_bench.indexing.chunker import build_parent_child_chunks

# 단 3줄로 RAG 파이프라인 구성
strategy = OpenAIEmbedStrategy(model="text-embedding-3-small")
strategy.index(child_chunks)
results = strategy.retrieve("사용자 질문", k=5)
```

---

### 저장 위치
상세 제안서: `/Users/sukbeom/Desktop/autorag/docs/service_modularization_proposal.md`
