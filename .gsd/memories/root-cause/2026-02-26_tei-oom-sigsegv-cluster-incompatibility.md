---
title: "TEI OOM/SIGSEGV — 클러스터 CPU/메모리 비호환"
tags:
  - debug
  - root-cause
  - k8s
  - tei
  - oom
  - sigsegv
type: root-cause
created: 2026-02-26T09:30:00+09:00
contextual_description: "TEI CPU 이미지가 AMD EPYC에서 Intel MKL SIGSEGV 크래시, bge-m3는 float32 ONNX warm-up에서 12Gi OOM — 근본 원인 2건 확인"
keywords:
  - TEI
  - OOMKilled
  - SIGSEGV
  - Intel MKL
  - AMD EPYC
  - r5a.xlarge
  - ONNX
  - float32
related:
  - 2026-02-26_tei-removal-direct-embedding
---

## TEI OOM/SIGSEGV — 클러스터 CPU/메모리 비호환

### 증상
1. `tei-bge-m3` Pod: OOMKilled (8Gi, 12Gi 모두 실패)
2. `tei-kosimcse` Pod: SIGSEGV exit 139 (8Gi 충분하나 크래시)

### 근본 원인

**원인 1 — 메모리 (bge-m3)**
- TEI는 ONNX 모델을 float32로 전체 메모리 로드 + warm-up 배치 처리
- bge-m3: 568M param × 4bytes = ~2.3GB 가중치 + ONNX 런타임 오버헤드 + warm-up 텐서
- 총 필요 메모리 > 12Gi

**원인 2 — CPU 호환 (kosimcse)**
- TEI CPU 이미지 `ghcr.io/huggingface/text-embeddings-inference:cpu-1.7`는 Intel AVX 명령어 최적화
- r5a.xlarge 노드는 AMD EPYC CPU → Intel MKL SGEMM 파라미터 오류 발생
- 로그: `Intel MKL SGEMM parameter error` → SIGSEGV

### 재현 경로
```bash
# bge-m3 OOM 재현
kubectl apply -f tei-deployment.yaml  # MODEL_KEY=bge-m3, memory limit 12Gi
kubectl logs deployment/tei-bge-m3 -n rag-bench-test  # OOMKilled

# kosimcse SIGSEGV 재현
kubectl apply -f tei-deployment.yaml  # MODEL_KEY=kosimcse, memory limit 8Gi
kubectl logs deployment/tei-kosimcse -n rag-bench-test  # exit 139
```

### 해결
TEI 코드 전면 제거, 워커 Pod 내 직접 HF 임베딩 방식으로 전환.
