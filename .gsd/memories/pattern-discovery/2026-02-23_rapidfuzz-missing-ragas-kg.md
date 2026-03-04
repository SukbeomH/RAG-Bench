---
title: "RAGAS KG 생성 실패 — rapidfuzz 미설치"
tags:
  - pattern-discovery
  - ragas
  - rapidfuzz
  - dependency
  - kg
type: pattern-discovery
created: 2026-02-23T12:00:00+09:00
contextual_description: "RAGAS KG(Knowledge Graph) 구축 중 rapidfuzz 모듈 없음 오류로 실패. ragas 패키지가 rapidfuzz를 선택적 의존성으로 처리하여 자동 설치 안 됨. uv add rapidfuzz로 해결."
keywords:
  - rapidfuzz
  - ragas
  - ModuleNotFoundError
  - KG
  - generate_qa
  - _generate_qa_ragas
related:
  - 2026-02-23_pdf-page-sampling-run-all-combos
---

## RAGAS KG 생성 실패 — rapidfuzz 미설치 (2026-02-23)

### 증상
```
ModuleNotFoundError: No module named 'rapidfuzz'
ImportError: rapidfuzz is required for string distance.
             Please install it using `pip install rapidfuzz`
```
`_generate_qa_ragas()` → RAGAS KG 구축 단계에서 발생.

### 원인
`ragas` 패키지가 `rapidfuzz`를 선택적 의존성으로 처리 — `pip install ragas`만으로는 설치되지 않음.

### 해결
```bash
uv add rapidfuzz
# 또는
pip install 'rapidfuzz>=3.0'
```

### 부작용
- `rapidfuzz` 없이 실행 시 `_generate_qa_ragas()`가 예외를 던지고 None 반환
- 기존 코드가 `if qa_pairs_raw:` 로 폴백하여 **구 qa_dataset.json을 그대로 사용**
- 에러 메시지만 출력하고 벤치마크는 정상 진행되므로 **자동으로 눈치채기 어려움**

### 예방
`pyproject.toml` 또는 설치 스크립트에 `rapidfuzz>=3.0` 명시 권장.
