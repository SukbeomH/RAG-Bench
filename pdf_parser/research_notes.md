# PDF 파싱 기술 리서치 노트

> 조사일: 2026-02-25

---

## 1. 벤치마크 기준: OmniDocBench (CVPR 2025)

업계 표준 벤치마크. 1,355 PDF 페이지, 9가지 문서 유형, 4가지 레이아웃, 3개 언어 포함.

평가 모듈: 텍스트 / 표 / 수식 / 읽기 순서 (4개 독립 지표)

### 주요 파서 처리 속도 비교 (Nvidia L4 GPU)

| 파서     | 속도 (sec/page) | 강점                              |
|----------|-----------------|-----------------------------------|
| MinerU   | 0.21            | 가장 빠름, 한국어/중국어/과학논문 |
| Docling  | 0.49            | 기업 환경, air-gapped, IBM 개발   |
| Marker   | 0.86            | 이미지/표 구조 충실도, 멀티언어   |

---

## 2. 신규 프로젝트 (2025~2026)

### VLM 기반 파서

| 프로젝트               | 개발사       | 출시       | 특징                                                        |
|------------------------|--------------|------------|-------------------------------------------------------------|
| NVIDIA Nemotron-Parse 1.1 | NVIDIA    | 2025.11    | 885M 파라미터, Markdown+LaTeX+bounding box 동시 출력, TC 변형(+20% 속도) |
| dots.ocr-1.5           | 小紅書       | 2026.02    | 100+ 언어, 레이아웃+OCR 단일 VLM, 1.7B, 모든 인류 문자 지원 |
| Dolphin / Dolphin-v2   | ByteDance    | ACL 2025   | Heterogeneous Anchor Prompting, 학술 문서 특화              |
| GLM-OCR                | 智谱AI       | 2025       | 0.9B 경량, 다국어 문서 이해                                 |
| olmOCR                 | Allen AI     | 2025       | 7B, 학술 문서 특화                                          |
| SmolDocling            | HuggingFace  | 2025       | 소형 모델, 경량 배포 특화                                   |

### 로컬 VLM 추천 순위 (2026 기준)

| 순위 | 모델              | 파라미터        | 용도                        |
|------|-------------------|-----------------|-----------------------------|
| 1    | Qwen2.5-VL-72B   | 72B             | 기업급, 구조화 데이터 추출  |
| 2    | DeepSeek-VL2     | MoE (sparse)    | 비용효율, 고볼륨            |
| 3    | GLM-4.5V         | 106B (12B active) | MoE 아키텍처              |

---

## 3. 아키텍처 진화 방향

### 기존 방식 (문서 단위 라우팅)
```
문서 분석 → Simple / Medium / Complex 분류 → 단일 도구 적용
```

### Hybrid Backend 방식 (페이지 단위 라우팅, MinerU 2.0+)
```
페이지별 분석
  ├─ 텍스트 직접 추출 가능 → Rule-based (빠름)
  └─ 스캔/이미지 중심    → VLM backend (정확)
```
문서 단위보다 세밀하며, 혼합 문서에서 정확도와 속도를 동시에 확보.

### Single-pass VLM (Docling VLM pipeline)
- 기존: 레이아웃 감지 → OCR → 구조화 (3단계, 오류 누적)
- 신규: 단일 VLM이 전체 처리 → 오류 전파 없음
- 대표 모델: Granite-Docling-258M

---

## 4. 현재 코드 개선 포인트

| 현재 구조           | 개선 방향                                          |
|---------------------|----------------------------------------------------|
| 문서 단위 라우팅    | 페이지 단위 라우팅 (Hybrid backend)                |
| Gemini만 지원       | olmOCR / dots.ocr 로컬 VLM 옵션 추가              |
| 품질 검사 단순      | OmniDocBench 지표 방식 (표/수식/읽기순서 분리 평가) |
| 동기 처리           | 페이지 병렬 처리 (async)                           |

---

## 5. 참고 링크

- [OmniDocBench GitHub (CVPR 2025)](https://github.com/opendatalab/OmniDocBench)
- [MinerU GitHub](https://github.com/opendatalab/MinerU)
- [dots.ocr GitHub](https://github.com/rednote-hilab/dots.ocr)
- [Dolphin (ByteDance, ACL 2025)](https://github.com/bytedance/Dolphin)
- [NVIDIA Nemotron-Parse 1.1](https://developer.nvidia.com/blog/turn-complex-documents-into-usable-data-with-vlm-nvidia-nemotron-parse-1-1/)
- [Marker vs MinerU vs MarkItDown 비교 (2026)](https://jimmysong.io/blog/pdf-to-markdown-open-source-deep-dive/)
- [12개 오픈소스 파서 비교](https://liduos.com/en/ai-develope-tools-series-2-open-source-doucment-parsing.html)
- [Docling 논문 (arXiv)](https://arxiv.org/pdf/2501.17887)
