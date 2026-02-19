# RAG Bench Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<user>/autorag/blob/main/rag_bench_colab/rag_benchmark.ipynb)

Google Colab T4 GPU에서 72개 RAG 전략 조합을 벤치마크합니다.

## Quick Start

1. 위 Colab 뱃지 클릭 (또는 `rag_benchmark.ipynb`를 Colab에 업로드)
2. **런타임 → 런타임 유형 변경 → GPU (T4)** 선택
3. Colab Secrets에 `OPENAI_API_KEY` 등록
4. 노트북 셀 순서대로 실행

## 프리셋

| 프리셋 | 조합 수 | 예상 시간 | API 비용 |
|--------|---------|----------|---------|
| `quick` | 4 | ~15분 | ~$0.5 |
| `standard` | 24 | ~50분 | ~$2 |
| `full` | 72 | ~3시간 | ~$5 |

## 디렉토리 구조

```
rag_bench_colab/
├── README.md                  # 이 파일
├── rag_benchmark.ipynb        # 메인 Colab 노트북
├── colab_config.py            # Colab 환경 설정 + rag_bench 패치
├── colab_runner.py            # 체크포인트 지원 벤치마크 러너
├── colab_visualizer.py        # 시각화 유틸리티
├── requirements_colab.txt     # Colab 전용 의존성
└── data/
    ├── qa_dataset.json        # QA 데이터셋 (20쌍)
    └── docs/                  # 벤치마크 대상 마크다운 문서
```

## 체크포인트

세션이 끊기더라도 Google Drive에 체크포인트가 저장됩니다.
커널 재시작 후 동일 설정으로 다시 실행하면 완료된 전략은 건너뜁니다.

저장 위치: `Google Drive > MyDrive > rag_bench_colab > checkpoints/`

## Qdrant 모드

| 모드 | 설명 | 영속성 |
|------|------|--------|
| `ephemeral` | `/content/qdrant_workspace` (로컬) | 세션 종료 시 삭제 |
| `drive` | Google Drive 저장 | 영속 |
| `memory` | 인메모리 (`:memory:`) | 세션 종료 시 삭제 |
