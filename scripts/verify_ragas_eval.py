"""
Verify RAGAS Integration
"""

import sys
import os
from typing import List

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_bench.base import BaseRAGStrategy
from rag_bench.runner import BenchmarkRunner
from rag_bench.evaluation import ExtendedRAGEvaluator


class MockStrategy(BaseRAGStrategy):
    """Mock strategy for testing."""

    @property
    def name(self) -> str:
        return "Mock Strategy"

    @property
    def description(self) -> str:
        return "Returns fixed documents for testing."

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        return [
            Document(
                page_content=f"This is a relevant document for query: {query}. Content info.",
                metadata={"id": i},
            )
            for i in range(k)
        ]

    def index(self, documents):
        pass

    def get_retriever(self, k=5):
        pass


def main():
    print("Verifying RAGAS Integration...")

    # 1. Initialize Evaluator
    try:
        evaluator = ExtendedRAGEvaluator()
        print("✅ ExtendedRAGEvaluator initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize ExtendedRAGEvaluator: {e}")
        return

    # 2. Setup Benchmark
    strategy = MockStrategy()
    queries = ["What is RAG?", "Explain Dense Retrieval."]

    runner = BenchmarkRunner(
        strategies=[strategy], queries=queries, k=2, evaluator=evaluator
    )

    # 3. Run Benchmark (Retrieval)
    print("\nRunning Retrieval...")
    runner.run()

    # 4. Run Evaluation
    # Note: This requires OPENAI_API_KEY for RAGAS default metrics
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "⚠️ OPENAI_API_KEY not found. Skipping actual RAGAS execution to avoid failure."
        )
        print("✅ integration logic seems correct otherwise.")
        return

    print("\nRunning Evaluation...")
    try:
        # Provide ground truths for Context Recall/Precision
        ground_truths = [
            "RAG stands for Retrieval-Augmented Generation.",
            "Dense retrieval uses vector embeddings to find relevant documents.",
        ]
        df = runner.evaluate(ground_truths=ground_truths)
        if df is not None:
            print("\nEvaluation Results:")
            print(df)
            print("✅ Evaluation complete.")
        else:
            print("❌ Evaluation returned None.")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")


if __name__ == "__main__":
    main()
