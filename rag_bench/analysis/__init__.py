"""
analysis — 서비스 벤치마크 결과 분석 모듈.

벤치마크 결과 데이터에서 모델 선정 근거를 도출하는 파이프라인:
  load_results() → rank_by_doc_type() → analyze_strengths_weaknesses()
  → compress_similar_results() → generate_selection_report()
  → generate_report()

빠른 사용법:
    from rag_bench.analysis import generate_report
    generate_report(run_dir="_benchdata/service_run")
"""

from rag_bench.analysis.ranker import rank_by_doc_type, load_results
from rag_bench.analysis.insight import analyze_strengths_weaknesses
from rag_bench.analysis.deduplication import compress_similar_results
from rag_bench.analysis.selector import generate_selection_report
from rag_bench.analysis.reporter import generate_report
from rag_bench.analysis.pipeline import run_analysis_pipeline, AnalysisResult
from rag_bench.analysis.reporter_exec import generate_exec_report

__all__ = [
    "load_results",
    "rank_by_doc_type",
    "analyze_strengths_weaknesses",
    "compress_similar_results",
    "generate_selection_report",
    "generate_report",
    "run_analysis_pipeline",
    "AnalysisResult",
    "generate_exec_report",
]
