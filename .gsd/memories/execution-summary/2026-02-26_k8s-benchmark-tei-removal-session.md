---
title: "K8s 벤치마크 세션 — TEI 제거 + 전략 실행 시작"
tags:
  - execution
  - summary
  - k8s
  - benchmark
  - tei-removal
type: execution-summary
created: 2026-02-26T09:55:00+09:00
contextual_description: "TEI 코드 전면 제거(7개 파일), K8s 리소스/toleration 조정, 문서 최신화 완료. general 카테고리 6조합 벤치마크 오케스트레이터 실행 시작"
keywords:
  - TEI removal
  - toleration
  - ai-platform
  - benchmark
  - orchestrator
  - resource defaults
related:
  - 2026-02-26_tei-removal-direct-embedding
  - 2026-02-26_tei-oom-sigsegv-cluster-incompatibility
---

## K8s 벤치마크 세션 — TEI 제거 + 전략 실행 시작

### 완료 작업

1. **TEI 코드 전면 제거** (7개 파일)
   - orchestrator.py, worker_entrypoint.py, dense_sparse.py, cache.py
   - bench-job-template.yaml, tei-deployment.yaml(삭제)
   - ARCHITECTURE.md (메인 + k8s)

2. **K8s 리소스/스케줄링 조정**
   - ai-platform toleration 추가 (prep, bench 매니페스트)
   - 리소스 기본값: Prep CPU 1/1 Mem 4Gi/8Gi, Bench CPU 1/2 Mem 4Gi/8Gi
   - 5노드(~13 vCPU) 활용 가능

3. **문서 최신화**
   - k8s/DEPLOY_GUIDE.md: CLI 명령어 간소화, 5노드 스케줄링
   - k8s/ARCHITECTURE.md: TEI 섹션 제거, 리소스 기본값
   - docs/k8s_benchmark_findings.md: 신규 생성 (테스트 결과 + 구성 제안)

4. **커밋 & 푸시**
   - `d3b6ea6` refactor(k8s): TEI 서빙 코드 제거 + toleration/리소스 조정
   - `48e9252` docs(k8s): 벤치마크 테스트 결과 문서화 + 아키텍처 최신화

### 진행 중
- `general` 카테고리 service 프리셋 (6 조합) 오케스트레이터 실행 중
- Prep Job (`prep-general-20260226-0948`) → 완료 후 6개 Bench Job 자동 생성 예정

### 주의사항
- 현재 Docker 이미지에는 TEI 제거 변경 미반영 (EMBEDDING_API_URL=""이면 무시되므로 안전)
- 다음 이미지 빌드 시 반영됨
