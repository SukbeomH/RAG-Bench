# Current Session Context

## Session Narrative
> On 2026-03-10 16:00, PaddleOCRVL 파이프라인의 _IMAGE_LABELS 실측 검증 및 수정 완료. VL 파이프라인이 layout_parsing과 다른 이미지 할당 로직을 사용함을 확인하고, 5종(image, header_image, footer_image, seal, chart)으로 정정.

## Context Snapshot
- **Active Task**: PaddleOCRVL _IMAGE_LABELS 실측 정정 (완료)
- **Branch**: master
- **Last Commit**: bfa723b fix: _IMAGE_LABELS를 PaddleOCRVL 실측 기준 5종으로 정정
- **Last Updated**: 2026-03-10 16:00

## Key Decisions
- PaddleOCRVL 파이프라인(paddleocr_vl/pipeline.py)의 실제 이미지 할당 대상: 5종
- table/formula/figure는 VL에서 이미지 미할당 확인
- 상세 레퍼런스: docs/paddleocr-vl-image-labels-reference.md

## Recent Commits
```
bfa723b fix: _IMAGE_LABELS를 PaddleOCRVL 실측 기준 5종으로 정정
d6194eb chore: GSD 세션 메모리 + 핸드오프 문서 업데이트
4db8845 fix: _html_to_markdown fallback에서 table HTML→Markdown 변환 구현
```
