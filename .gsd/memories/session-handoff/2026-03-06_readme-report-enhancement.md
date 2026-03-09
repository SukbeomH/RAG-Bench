# Session Handoff: README 최신화 + 보고서 자동 인사이트

## 날짜
2026-03-06

## 브랜치
master

## 완료 작업

### 1. PDF 보고서 자동 인사이트 도출 (report.py)
- `_derive_exec_insights()`: Executive Summary용 핵심 발견 자동 생성
  - 1위/2위 격차, 로컬 vs API 비교, 문서유형별 특이사항
- `_derive_weakness()`: 미선정 백엔드 주요 감점 기준 자동 도출
- Executive Summary 강화: 문서유형별 1위 테이블, 핵심 발견, 즉시 실행 항목
- 추천 섹션 강화: 용도별 추천 테이블 (보안/정확도/표형/텍스트)
- 미선정 사유 테이블에 "주요 약점" 컬럼 추가
- 표형 분석: NED vs TEDS 역전 인사이트 자동 삽입
- Next Steps: 유형별 1위 기반 Hybrid 라우팅 후보 자동 제안

### 2. README.md 최신화
- 리서치 현황: Phase 5 최종 결과 반영 (PaddleOCR-VL 종합 1위)
- PDF 파서 백엔드: 6→8종 (openai-4.1, upstage-enhanced 추가)
- 벤치마크 프리셋: 11개 전체 테이블 추가
- CLI 옵션: 전체 옵션 테이블 정리
- 정규화 규칙: `<|SEP|>` 토큰 반영
- 테스트 섹션: 유동적 개수 제거

## 커밋 이력
```
ef7a6ce feat: PDF 보고서 자동 인사이트 도출 — Executive Summary 강화, 용도별 추천, 미선정 약점 분석
8e8cd11 docs: README 최신화 — Phase 5 결과 반영, 백엔드 8종, 프리셋 11개, CLI 옵션 전체 정리
9275288 chore: GSD 세션 메모리 + CURRENT.md 업데이트
```

## 다음 세션 참고
- report.py의 자동 인사이트 기능은 `generate_report()` 호출 시 자동 적용됨
- `--report-only` 로 기존 결과에 새 보고서 형식 재생성 가능
- Smart Router 리서치 미착수 상태 유지
