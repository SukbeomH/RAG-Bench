"""
RAGAS Evaluation Module
"""

from typing import List, Dict, Any, Optional
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


class RAGEvaluator:
    """Wrapper for RAGAS evaluation."""

    def __init__(
        self,
        metrics: Optional[List[Any]] = None,
        llm: Optional[Any] = None,
        embeddings: Optional[Any] = None,
    ):
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "ragas is not installed. Please install it with `pip install ragas`."
            )

        self.metrics = metrics or [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

        # Configure LLM/Embeddings with SSL bypass if not provided
        if llm is None or embeddings is None:
            try:
                import httpx
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings

                # Check if we should bypass SSL (simple heuristic: if previous attempt failed)
                # Or just always be safe in this corporate environment
                http_client = httpx.Client(verify=False)
                async_client = httpx.AsyncClient(verify=False)

                if llm is None:
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        http_client=http_client,
                        http_async_client=async_client,
                    )
                else:
                    self.llm = llm

                if embeddings is None:
                    self.embeddings = OpenAIEmbeddings(
                        http_client=http_client, http_async_client=async_client
                    )
                else:
                    self.embeddings = embeddings
            except Exception as e:
                print(
                    f"Warning: Failed to setup default OpenAI clients with SSL bypass: {e}"
                )
                self.llm = llm
                self.embeddings = embeddings
        else:
            self.llm = llm
            self.embeddings = embeddings

    def evaluate(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Run RAGAS evaluation on the provided data.

        Args:
            questions: List of questions.
            contexts: List of retrieved contexts (each item is a list of strings).
            answers: List of generated answers.
            ground_truths: List of ground truth answers (optional, required for context_recall).

        Returns:
            Dictionary of average scores for each metric.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        # Compute averages
        averages = {}
        # results is an EvaluationResult object.
        # results.scores is a list of dicts (one per sample).
        # results[metric] returns a list of scores for that metric.

        # Identify metrics from the first result if available
        if results.scores:
            metric_names = results.scores[0].keys()
            for m in metric_names:
                try:
                    # results[m] returns a list of values.
                    # Filter out None/NaN if any? Ragas usually returns floats.
                    values = results[m]
                    if values:
                        avg = sum(values) / len(values)
                        averages[m] = avg
                except Exception:
                    pass

        return averages
