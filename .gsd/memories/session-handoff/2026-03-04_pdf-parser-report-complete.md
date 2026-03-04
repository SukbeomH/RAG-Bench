# Session Handoff: PDF Parser 종합 보고서 작성 완료

## 날짜: 2026-03-04
## 브랜치: master (666fca3)

## 완료 작업

### RAG Parser 종합 벤치마크 보고서
- **로컬**: `docs/reports/pdf_parser_comparison.md` (요약) + `docs/reports/rag_parser_full_report.md` (상세, 커밋 15fd042)
- **Notion**: https://www.notion.so/319b7135d33b813090f5ebbea039a325
  - RAG Parser 허브(313b7135d33b8049a0c9dc542b06f3b4) 하위에 생성
- 보고서 구조: 용어사전 → 프로젝트 개요 → 백엔드 상세(5종) → 종합 순위 → 효율 분석 → 유형별 순위 → 세부 결과(11 PDF × 5 백엔드) → DPI 분석 → 모델 선택 → 속도 → 데이터 출처
- **Upstage Enhanced / MinerU Pipeline 제거**: 사용자 요청에 따라 보고서에서 완전 제거

### 보고서 핵심 결론
- NED 종합 1위: PaddleOCR-VL (0.7594)
- TEDS 종합 1위: Upstage (0.6185)
- 텍스트/표형 추천: Upstage, 그래프형 추천: PaddleOCR-VL
- 효율(NED/파라미터): PaddleOCR-VL 0.844 NED/B — 0.9B 모델로 가성비 최고

### 기타 정리
- `rag_bench/uv.lock` 삭제 (모노레포 전환으로 불필요)
- `pdf_parser/reports/` 중복 디렉토리 제거 (docs/reports/로 이동 완료)
- GSD 세션 메모리 반영

## 미결 / 후속 작업
- **Phase 5**: MinerU Pipeline 백엔드 추가 벤치마크 (spec.py에 "mineru" 예약됨)
- **Docling 로컬 실행 불가**: transformers>=4.49.0 호환 문제 미해결 (K8s 결과만 존재)
- **Notion 부분 업데이트 이슈**: `replace_content_range`로 Notion 페이지 부분 수정 시 매칭 실패 빈번 → `replace_content` 전체 교체 사용 권장

## 참고 데이터 소스
- Notion RAG Parser 허브: `313b7135d33b8049a0c9dc542b06f3b4`
- Notion Phase 4 결과: `318b7135d33b81188128c8a869022be0`
- Notion Parser 기술 리서치: `313b7135d33b8017ab63fdd406de210e`
