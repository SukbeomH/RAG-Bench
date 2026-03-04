---
title: "HF 데이터셋 streaming 우회 패턴"
tags:
  - pattern
  - streaming
  - huggingface
  - hf-dataset
  - workaround
type: pattern-discovery
created: 2026-02-24T06:53:03Z
contextual_description: "HF CAS 다운로드 오류 시 streaming=True로 우회하는 재사용 패턴"
keywords:
  - HuggingFace streaming
  - CAS 우회
  - load_dataset streaming
  - max_queries 루프
related:
  - 2026-02-24_hf-3
---

## HF 데이터셋 streaming 우회 패턴

## HF 데이터셋 streaming 우회 패턴

### 패턴 설명
HuggingFace 데이터셋 로드 시 CAS(Content Addressable Storage) 다운로드 오류 발생하면
load_dataset(..., streaming=True) 사용으로 우회 가능.

### 적용 규칙
1. datasets.load_dataset() 기본 호출 → CAS 다운로드 오류 발생 가능
2. streaming=True 추가 시 parquet 파일 다운로드 없이 스트리밍으로 직접 읽기
3. streaming 모드에서는 split 이름을 직접 지정해야 함 (ds['corpus'] 등)
4. max_queries 제한: 루프 내 len(queries) >= max_queries 조건으로 break

### 코드 패턴
```python
ds = load_dataset('dataset_name', 'config_name', streaming=True)
for row in ds['split_name']:
    if max_queries > 0 and len(results) >= max_queries:
        break
    # 처리
```

### 주의사항
- streaming 모드에서는 len(ds['split']) 불가 (이터레이터)
- 2-pass 불가: corpus 먼저 로드 후 qrels 필터링 시 set으로 캐시 필요
- trust_remote_code=True 제거 필요 (더 이상 지원 안 함)
