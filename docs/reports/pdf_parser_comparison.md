# PDF Parser 백엔드 전체 비교 보고서
> 생성일: 2026-03-03  |  총 6 백엔드  |  11 PDF 유형

---

## 1. 핵심 결론

- **텍스트 정확도(NED) 1위**: PaddleOCR-VL (0.7594)
- **표 정확도(TEDS) 1위**: Upstage (0.6185)
- **텍스트형 최적**: Upstage (0.8493)
- **표형 최적 (NED)**: Upstage (0.8068)
- **표형 최적 (TEDS)**: Upstage (0.6316)
- **그래프형 최적**: PaddleOCR-VL (0.7259)

- **Upstage vs Enhanced**: Upstage 우세 (NED 차이 0.2082)

## 2. 백엔드 종합 순위표

> NED: 텍스트 일치도 (0~1, 높을수록 좋음) | TEDS: 표 구조 정확도 (0~1, 높을수록 좋음)
> †K8s Phase1 결과 (run-id: 20260227-1118), 로컬 재현 불가 (docling: transformers 호환 문제)

| 순위 | 백엔드 | 유형 | 전체 평균 NED | 전체 평균 TEDS | 비고 |
|:---:|---|:---:|:---:|:---:|---|
| 1 | **PaddleOCR-VL** | 💻 로컬 | 0.7594 | 0.5586 |  |
| 2 | **Upstage** | 🌐 API | 0.6937 | 0.6185 |  |
| 3 | **OpenAI GPT-4o** | 🌐 API | 0.5890 | 0.2859 |  |
| 4 | **PyMuPDF** | 💻 로컬 | 0.5463 | 0.4500 | K8s 결과† |
| 5 | **Docling** | 💻 로컬 | 0.4954 | 0.4400 | K8s 결과† |
| 6 | **Upstage Enhanced** | 🌐 API | 0.4855 | 0.5157 |  |

## 3. 문서 유형별 비교

### 3-1. 📄 텍스트형

| 백엔드 | NED | TEDS | 비고 |
|---|:---:|:---:|---|
| Upstage ★ | 0.8493 | 0.5530 |  |
| Upstage Enhanced | 0.8430 | 0.5501 |  |
| PaddleOCR-VL | 0.8253 | 0.5510 |  |
| Docling | 0.7431 | 0.5348 | K8s 결과† |
| OpenAI GPT-4o | 0.7192 | 0.4288 |  |
| PyMuPDF | 0.6577 | 0.4928 | K8s 결과† |

### 3-2. 📊 표형

| 백엔드 | NED | TEDS | 비고 |
|---|:---:|:---:|---|
| Upstage ★ | 0.8068 | 0.6316 |  |
| Upstage Enhanced | 0.7877 | 0.5088 |  |
| PaddleOCR-VL | 0.7798 | 0.5601 |  |
| PyMuPDF | 0.6269 | 0.4072 | K8s 결과† |
| OpenAI GPT-4o | 0.5927 | 0.2573 |  |
| Docling | 0.4848 | 0.3926 | K8s 결과† |

### 3-3. 📈 그래프형

| 백엔드 | NED | TEDS | 비고 |
|---|:---:|:---:|---|
| PaddleOCR-VL ★ | 0.7259 | N/A |  |
| OpenAI GPT-4o | 0.5592 | N/A |  |
| Upstage | 0.5494 | N/A |  |
| Docling | 0.3821 | N/A | K8s 결과† |
| PyMuPDF | 0.3543 | N/A | K8s 결과† |
| Upstage Enhanced | 0.1118 | N/A |  |

## 4. 세부 결과표 (백엔드 × PDF)

> NED 기준 정렬 | TEDS N/A = 해당 PDF에 표 없음 (graph_rich 계열)

### 📄 텍스트형

| PDF 유형 | PyMuPDF NED | PyMuPDF TEDS | Docling NED | Docling TEDS | OpenAI GPT-4o NED | OpenAI GPT-4o TEDS | Upstage NED | Upstage TEDS | Upstage Enhanced NED | Upstage Enhanced TEDS | PaddleOCR-VL NED | PaddleOCR-VL TEDS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 텍스트 전용 | 0.6577 | 0.4928 | 0.7431 | 0.5348 | 0.7192 | 0.4288 | 0.8493 | 0.5530 | 0.8430 | 0.5501 | 0.8253 | 0.5510 |

### 📊 표형

| PDF 유형 | PyMuPDF NED | PyMuPDF TEDS | Docling NED | Docling TEDS | OpenAI GPT-4o NED | OpenAI GPT-4o TEDS | Upstage NED | Upstage TEDS | Upstage Enhanced NED | Upstage Enhanced TEDS | PaddleOCR-VL NED | PaddleOCR-VL TEDS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 표 (네이티브) | 0.6269 | 0.4072 | 0.6995 | 0.5253 | 0.6941 | 0.2874 | 0.8098 | 0.6423 | 0.7825 | 0.5181 | 0.7735 | 0.5010 |
| 표 (이미지) | — | — | 0.2700 | 0.2600 | 0.7010 | 0.2435 | 0.8080 | 0.6437 | 0.7946 | 0.5095 | 0.7766 | 0.5787 |
| 표 (이미지, 72dpi) | — | — | — | — | 0.6352 | 0.1781 | 0.7990 | 0.5849 | 0.7833 | 0.4911 | 0.7745 | 0.5585 |
| 표 (이미지, 150dpi) | — | — | — | — | 0.2569 | 0.3071 | 0.8087 | 0.6437 | 0.7820 | 0.5151 | 0.7980 | 0.5838 |
| 표 (이미지, 200dpi) | — | — | — | — | 0.6764 | 0.2704 | 0.8087 | 0.6435 | 0.7962 | 0.5101 | 0.7762 | 0.5785 |

### 📈 그래프형

| PDF 유형 | PyMuPDF NED | PyMuPDF TEDS | Docling NED | Docling TEDS | OpenAI GPT-4o NED | OpenAI GPT-4o TEDS | Upstage NED | Upstage TEDS | Upstage Enhanced NED | Upstage Enhanced TEDS | PaddleOCR-VL NED | PaddleOCR-VL TEDS |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 그래프 (텍스트 혼재) | 0.3543 | N/A | 0.6242 | N/A | 0.5647 | N/A | 0.6478 | N/A | 0.1141 | N/A | 0.7280 | N/A |
| 그래프+이미지 | — | — | 0.1400 | N/A | 0.5886 | N/A | 0.5329 | N/A | 0.1120 | N/A | 0.7281 | N/A |
| 그래프+이미지 (72dpi) | — | — | — | — | 0.4867 | N/A | 0.5294 | N/A | 0.1011 | N/A | 0.7096 | N/A |
| 그래프+이미지 (150dpi) | — | — | — | — | 0.5696 | N/A | 0.4955 | N/A | 0.1147 | N/A | 0.7285 | N/A |
| 그래프+이미지 (200dpi) | — | — | — | — | 0.5863 | N/A | 0.5415 | N/A | 0.1169 | N/A | 0.7355 | N/A |

## 5. 속도 참고 (평균 처리 시간/페이지)

> ⚠️ 속도는 환경(네트워크, 서버 부하)에 따라 편차가 큼. 순위 판단에 미반영.

| 백엔드 | 텍스트형 | 표형 (avg) | 그래프형 (avg) | 비고 |
|---|:---:|:---:|:---:|---|
| PyMuPDF | 2s | 2s | 2s | K8s 결과† |
| Docling | 45s | 1.1min | 1.0min | K8s 결과† |
| OpenAI GPT-4o | 3.2min | 4.5min | 1.6min |  |
| Upstage | 0s | 17s | 17s |  |
| Upstage Enhanced | 18s | 24s | 23s |  |
| PaddleOCR-VL | 1.8min | 1.7min | 58s |  |

## 6. 백엔드 선택 가이드

| 문서 유형 | 1순위 추천 | 2순위 추천 | 근거 |
|---|---|---|---|
| 📄 텍스트형 | **Upstage** | Upstage Enhanced | NED 0.8493 |
| 📊 표형 | **Upstage** | Upstage Enhanced | NED 0.8068, TEDS 0.6316 |
| 📈 그래프형 | **PaddleOCR-VL** | OpenAI GPT-4o | NED 0.7259 |

> 💡 **비용·보안 고려 시**: API 백엔드(OpenAI, Upstage) 대신 PaddleOCR-VL 활용 가능
> 💡 **Phase 5 예정**: MinerU Pipeline 추가 예정 — 현재 순위 변동 가능

## 7. 데이터 출처

| 백엔드 | Run ID | PDF 수 | 상태 |
|---|---|:---:|---|
| PyMuPDF | `20260227-1118 (K8s)` | 5† | K8s 결과 (부분) |
| Docling | `20260227-1118 (K8s)` | 5† | K8s 결과 (부분), 로컬 실행 불가 |
| OpenAI GPT-4o | `vlm-20260303-1550` | 11 | 완료 |
| Upstage | `upstage-20260303-1635` | 11 | 완료 |
| Upstage Enhanced | `upstage-20260303-1635` | 11 | 완료 |
| PaddleOCR-VL | `paddleocr-20260303-1541` | 11 | 완료 |

† pymupdf/docling: K8s Phase1 실행 결과. 로컬 docling은 `transformers>=4.49.0` 호환 문제로 실행 불가 (Task 2 예정).
