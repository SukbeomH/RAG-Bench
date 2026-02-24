"""
pipeline — 분석 파이프라인 공통 진입점.

reporter.py(기술용)와 reporter_exec.py(경영진용) 모두 이 파이프라인을 공유한다.
렌더링 로직은 각 reporter에 유지한다.

사용 예시:
    from rag_bench.analysis.pipeline import run_analysis_pipeline

    result = run_analysis_pipeline(run_dir="_benchdata/service_run")
    print(result.selection.default_recommendation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from rag_bench.analysis.ranker import load_results, rank_by_doc_type
from rag_bench.analysis.insight import analyze_strengths_weaknesses
from rag_bench.analysis.deduplication import compress_similar_results, format_tie_groups_summary
from rag_bench.analysis.selector import generate_selection_report, SelectionReport


@dataclass
class AnalysisResult:
    """분석 파이프라인 전체 결과를 담는 컨테이너."""
    raw_results: Dict[str, dict] = field(default_factory=dict)
    ranked: Dict[str, pd.DataFrame] = field(default_factory=dict)
    insights: Dict[str, dict] = field(default_factory=dict)
    compressed: Dict[str, pd.DataFrame] = field(default_factory=dict)
    tie_summary: Dict[str, List[dict]] = field(default_factory=dict)
    selection: SelectionReport = field(default_factory=SelectionReport)

    @property
    def categories(self) -> List[str]:
        return list(self.ranked.keys())

    @property
    def n_combos(self) -> int:
        if not self.ranked:
            return 0
        return len(next(iter(self.ranked.values())))


def run_analysis_pipeline(
    run_dir: str | Path,
    similarity_threshold: float = 0.05,
    verbose: bool = True,
) -> AnalysisResult:
    """
    벤치마크 결과 디렉토리를 분석하여 AnalysisResult를 반환한다.

    파이프라인 단계:
      1. 결과 로드 (load_results)
      2. 순위 계산 (rank_by_doc_type)
      3. 강점/약점 분석 (analyze_strengths_weaknesses)
      4. 동점 그룹 압축 (compress_similar_results)
      5. 최종 선정 보고서 생성 (generate_selection_report)

    Args:
        run_dir: run_service_bench.py 의 출력 디렉토리
        similarity_threshold: 동점 판정 임계값 (기본 0.05 = 5%)
        verbose: 진행 상태 출력 여부

    Returns:
        AnalysisResult — 렌더링 전 중간 결과 전체를 포함
    """
    run_dir = Path(run_dir)

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    _log(f"\n[1/5] 결과 로드 중... ({run_dir})")
    raw_results = load_results(run_dir)
    if not raw_results:
        print(f"오류: {run_dir} 에서 result.json 파일을 찾을 수 없습니다.")
        return AnalysisResult()

    _log(f"  로드된 카테고리: {list(raw_results.keys())}")

    _log("\n[2/5] 순위 계산 중...")
    ranked = rank_by_doc_type(raw_results, latency_dir=run_dir)
    _log(f"  순위 완료: {list(ranked.keys())}")

    _log("\n[3/5] 강점/약점 분석 중...")
    insights = analyze_strengths_weaknesses(ranked)
    _log(f"  분석 완료: {len(insights)}개 조합")

    _log("\n[4/5] 동점 그룹 압축 중...")
    compressed = compress_similar_results(ranked, similarity_threshold)
    tie_summary = format_tie_groups_summary(compressed)

    _log("\n[5/5] 최종 선정 보고서 생성 중...")
    selection = generate_selection_report(ranked, insights, compressed)

    return AnalysisResult(
        raw_results=raw_results,
        ranked=ranked,
        insights=insights,
        compressed=compressed,
        tie_summary=tie_summary,
        selection=selection,
    )
