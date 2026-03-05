# Current Session Context

## Session Narrative
> On 2026-03-05 13:44:26, the developer was working on the **master** branch, modifying 30 files across `.,.gsd,packages/pdf-eval`. The recent work involved: fix: openai_compat mlx-vlm 호환성 수정 + GSD 세션 핸드오프.

## Context Snapshot
- **Active Task**: fix: openai_compat mlx-vlm 호환성 수정 + GSD 세션 핸드오프
- **Branch**: master
- **Files Changed**: 30
- **Last Updated**: 2026-03-05 13:44:26

## Working Files
```
 D .gsd/.modified-this-session
 M .gsd/CURRENT.md
 M packages/pdf-eval/pyproject.toml
 M packages/pdf-eval/src/autorag_pdf_eval/omnidoc_metrics.py
 M packages/pdf-eval/src/autorag_pdf_eval/report.py
 M packages/pdf-eval/tests/test_omnidoc_metrics.py
 D packages/pdf-eval/tests/test_report.py
 M packages/pdf-eval/tests/test_unit_eval.py
 M servers/mlx_vlm/run.sh
 M uv.lock
?? .gsd/memories/session-summary/2026-03-04_session-2026-03-04-16-20-27-master.md
?? .gsd/memories/session-summary/2026-03-04_session-2026-03-04-16-28-32-master.md
?? .gsd/memories/session-summary/2026-03-04_session-2026-03-04-16-31-52-master.md
?? .gsd/memories/session-summary/2026-03-04_session-2026-03-04-16-36-48-master.md
?? .gsd/memories/session-summary/2026-03-04_session-2026-03-04-17-30-45-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-08-36-13-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-13-37-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-15-44-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-18-58-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-32-08-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-36-02-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-36-49-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-41-27-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-53-38-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-55-24-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-56-31-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-09-59-16-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-10-00-05-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-10-07-37-master.md
?? .gsd/memories/session-summary/2026-03-05_session-2026-03-05-10-09-42-master.md
```

## Recent Commits
```
2d202b2 fix: openai_compat mlx-vlm 호환성 수정 + GSD 세션 핸드오프
7563d50 feat: OCR 서버 독립 uv 프로젝트화 + mlx-vlm 래퍼 신규 추가
97c9e33 feat: LangGraph 기반 rag-pipeline 패키지 신규 생성
```

## Diff Stats
```
 packages/pdf-eval/tests/test_report.py             | 330 -----------------
 packages/pdf-eval/tests/test_unit_eval.py          |  27 +-
 servers/mlx_vlm/run.sh                             |   1 -
 uv.lock                                            |   2 +
 10 files changed, 475 insertions(+), 514 deletions(-)
```
