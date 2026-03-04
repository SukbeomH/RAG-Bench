"""CLI entrypoint for the RAG pipeline."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="autorag-pipeline",
        description="Run AutoRAG LangGraph pipelines from the command line.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run the basic RAG pipeline")
    run_parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    run_parser.add_argument("--query", required=True, help="Query to ask")
    run_parser.add_argument("--backend", default="pymupdf", help="Parser backend")
    run_parser.add_argument("--k", type=int, default=5, help="Number of results")
    run_parser.add_argument("--chunk-size", type=int, default=512)
    run_parser.add_argument("--chunk-overlap", type=int, default=64)
    run_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _run_pipeline(args)


def _run_pipeline(args: argparse.Namespace) -> None:
    from autorag_pipeline.graphs.rag_pipeline import build_rag_pipeline

    graph = build_rag_pipeline()

    initial_state = {
        "pdf_path": args.pdf,
        "query": args.query,
        "backend": args.backend,
        "k": args.k,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
    }

    result = graph.invoke(initial_state)

    if args.output == "json":
        output = {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "doc_id": result.get("doc_id", ""),
            "total_parse_time_s": result.get("total_parse_time_s", 0.0),
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(f"\n답변: {result.get('answer', '')}\n")
        citations = result.get("citations", [])
        if citations:
            print("출처:")
            for i, c in enumerate(citations, 1):
                print(
                    f"  [{i}] {c.get('source_path', '')} "
                    f"p.{c.get('page_number', '?')} "
                    f"({c.get('chunk_id', '')})"
                )


if __name__ == "__main__":
    main()
