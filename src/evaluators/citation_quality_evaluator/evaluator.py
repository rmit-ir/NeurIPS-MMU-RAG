"""
Citation Quality Evaluator for assessing citation usage in RAG responses.

This evaluator uses LLM-as-a-Judge to assess citation quality including:
- Citation Relevance: How relevant citations are to the claims made
- Citation Accuracy: Whether citations accurately support the claims
- Citation Completeness: Whether all claims are properly cited
- Citation Attribution: Whether citations are properly attributed
"""

import time
import re
import json
from typing import List, Dict, Any, Optional
from statistics import mean
import concurrent.futures
from threading import Lock

from src.evaluators.evaluator_interface import EvaluatorInterface, EvaluationResult
import openai
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CITATION_QUALITY_PROMPT_TEMPLATE = """
You are evaluating the quality of citations in a RAG (Retrieval-Augmented Generation) system response.

**Task:** Evaluate how well citations are used in the response on a scale from 0-5 for each dimension:

1. **Citation Relevance (0-5)**: How relevant are the citations to the claims made in the response?
   - 5: All citations are highly relevant and directly support the claims
   - 3: Citations are mostly relevant but some are tangential
   - 1: Citations are largely irrelevant to the claims made
   - 0: No citations provided or completely irrelevant

2. **Citation Accuracy (0-5)**: Do the citations accurately support the claims made?
   - 5: All citations accurately and precisely support the claims
   - 3: Citations generally support claims but with some inaccuracies
   - 1: Citations do not accurately support the claims
   - 0: Citations contradict the claims made

3. **Citation Completeness (0-5)**: Are all factual claims properly cited?
   - 5: Every factual claim is supported by appropriate citations
   - 3: Most claims are cited, but some minor claims lack citations
   - 1: Major claims lack citations or citations are missing for key facts
   - 0: No citations provided for any claims

4. **Citation Attribution (0-5)**: Are citations properly attributed and formatted?
   - 5: Citations are clearly attributed and properly formatted throughout
   - 3: Citations are mostly well-attributed but with minor formatting issues
   - 1: Citations are poorly attributed or formatted incorrectly
   - 0: No citations or citations are not attributable

**Response Format:** Return a JSON object with the following structure:
{
    "citation_relevance": <score>,
    "citation_accuracy": <score>,
    "citation_completeness": <score>,
    "citation_attribution": <score>,
    "overall_citation_quality": <average_score>,
    "reasoning": "<brief explanation of scores>"
}

**Question:** {question}

**Response:** {response}

**Available Citations:** {citations}

**Evaluation:**
"""


class CitationQualityEvaluator(EvaluatorInterface):
    """
    LLM-based evaluator for citation quality assessment.

    This evaluator assesses how well citations are used in RAG responses,
    measuring relevance, accuracy, completeness, and proper attribution.
    """

    def __init__(
        self,
        model_id: str = "qwen.qwen3-32b-v1:0",
        temperature: float = 0.0,
        max_tokens: int = 6000,
        silent_errors: bool = True,
        num_threads: int = 1,
        api_base: str = "https://mmu-proxy-server-llm-proxy.rankun.org",
        api_key: Optional[str] = None
    ):
        """
        Initialize the citation quality evaluator.

        Args:
            model_id: Model ID for evaluation
            temperature: LLM temperature setting
            max_tokens: Maximum tokens to generate
            silent_errors: Whether to log errors and continue
            num_threads: Number of threads for concurrent evaluations
            api_base: Base URL for the MMU PROXY Router server
            api_key: API key for authentication
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.silent_errors = silent_errors
        self.num_threads = num_threads
        self.api_base = api_base
        self.api_key = api_key or "dummy-key"  # Will be overridden by env var

        # Initialize OpenAI client
        self.client = openai.OpenAI(
            base_url=self.api_base,
            api_key=self.api_key
        )

    @property
    def name(self) -> str:
        """Return evaluator name."""
        return "CitationQualityEvaluator"

    @property
    def description(self) -> str:
        """Return evaluator description."""
        return "LLM-based evaluation of citation quality in RAG responses"

    def evaluate(
        self,
        system_outputs: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate citation quality in system outputs.

        Args:
            system_outputs: List of system outputs with citations
            references: List of reference data (may include ground truth citations)

        Returns:
            EvaluationResult with citation quality metrics
        """
        start_time = time.time()

        # Validate inputs
        self.validate_inputs(system_outputs, references)

        # Merge data
        merged_data = self._merge_data(system_outputs, references)

        if not merged_data:
            raise ValueError("No matching data found between outputs and references")

        # Evaluate citation quality
        try:
            if self.num_threads > 1:
                metrics, row_results = self._evaluate_concurrent(merged_data)
            else:
                metrics, row_results = self._evaluate_sequential(merged_data)

            # Calculate execution time
            total_time_ms = (time.time() - start_time) * 1000

            return EvaluationResult(
                metrics=metrics,
                evaluator_name=self.name,
                sample_count=len(merged_data),
                rows=row_results,
                total_time_ms=total_time_ms
            )

        except Exception as e:
            if self.silent_errors:
                logger.error(f"Citation quality evaluation failed: {str(e)}")
                # Return empty results
                return EvaluationResult(
                    metrics={},
                    evaluator_name=self.name,
                    sample_count=len(merged_data),
                    rows=[],
                    total_time_ms=(time.time() - start_time) * 1000
                )
            else:
                raise RuntimeError(f"Citation quality evaluation failed: {str(e)}")

    def _merge_data(
        self,
        system_outputs: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge system outputs with references by ID."""
        # Create lookup dictionaries
        outputs_by_id = {}
        for item in system_outputs:
            key = item.get('query_id') or item.get('iid')
            if key:
                outputs_by_id[key] = item

        references_by_id = {}
        for item in references:
            key = item.get('query_id') or item.get('iid')
            if key:
                references_by_id[key] = item

        merged_data = []
        for query_id in outputs_by_id:
            if query_id in references_by_id:
                output = outputs_by_id[query_id]
                reference = references_by_id[query_id]

                merged_data.append({
                    'query_id': query_id,
                    'question': output.get('question') or reference.get('question', ''),
                    'generated_response': output.get('generated_response', ''),
                    'citations': output.get('citations', []),
                    'reference': reference.get('generated_response') or reference.get('reference', ''),
                    'reference_citations': reference.get('citations', [])
                })

        return merged_data

    def _evaluate_sequential(
        self,
        merged_data: List[Dict[str, Any]]
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Evaluate citation quality sequentially."""
        all_scores = {
            'citation_relevance': [],
            'citation_accuracy': [],
            'citation_completeness': [],
            'citation_attribution': [],
            'overall_citation_quality': []
        }
        row_results = []

        for item in merged_data:
            scores = self._evaluate_single_item(item)
            if scores:
                for key in all_scores:
                    if key in scores:
                        all_scores[key].append(scores[key])

                row_results.append({
                    'query_id': item['query_id'],
                    **scores
                })

        # Calculate aggregate metrics
        metrics = {}
        for key, scores in all_scores.items():
            if scores:
                metrics[f'mean_{key}'] = mean(scores)
                metrics[f'std_{key}'] = (sum((x - metrics[f'mean_{key}']) ** 2 for x in scores) / len(scores)) ** 0.5
                metrics[f'min_{key}'] = min(scores)
                metrics[f'max_{key}'] = max(scores)

        return metrics, row_results

    def _evaluate_concurrent(
        self,
        merged_data: List[Dict[str, Any]]
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Evaluate citation quality concurrently."""
        all_scores = {
            'citation_relevance': [],
            'citation_accuracy': [],
            'citation_completeness': [],
            'citation_attribution': [],
            'overall_citation_quality': []
        }
        row_results = []

        lock = Lock()

        def evaluate_item(item):
            scores = self._evaluate_single_item(item)
            with lock:
                if scores:
                    for key in all_scores:
                        if key in scores:
                            all_scores[key].append(scores[key])
                    row_results.append({
                        'query_id': item['query_id'],
                        **scores
                    })

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(evaluate_item, item) for item in merged_data]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Wait for completion

        # Calculate aggregate metrics
        metrics = {}
        for key, scores in all_scores.items():
            if scores:
                metrics[f'mean_{key}'] = mean(scores)
                metrics[f'std_{key}'] = (sum((x - metrics[f'mean_{key}']) ** 2 for x in scores) / len(scores)) ** 0.5
                metrics[f'min_{key}'] = min(scores)
                metrics[f'max_{key}'] = max(scores)

        return metrics, row_results

    def _evaluate_single_item(self, item: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Evaluate citation quality for a single item."""
        try:
            question = item['question']
            response = item['generated_response']
            citations = item['citations']

            # Format citations for the prompt
            citations_text = self._format_citations(citations)

            # Create evaluation prompt
            prompt = CITATION_QUALITY_PROMPT_TEMPLATE.format(
                question=question,
                response=response,
                citations=citations_text
            )

            # Call LLM for evaluation
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            evaluation_text = response.choices[0].message.content.strip()

            # Parse JSON response
            scores = self._parse_evaluation_response(evaluation_text)

            return scores

        except Exception as e:
            if self.silent_errors:
                logger.error(f"Error evaluating item {item.get('query_id', 'unknown')}: {str(e)}")
                return None
            else:
                raise e

    def _format_citations(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations for the evaluation prompt."""
        if not citations:
            return "No citations provided"

        formatted = []
        for i, citation in enumerate(citations, 1):
            citation_text = f"[{i}] "
            if isinstance(citation, dict):
                if 'url' in citation:
                    citation_text += f"URL: {citation['url']}"
                if 'title' in citation:
                    citation_text += f" | Title: {citation['title']}"
                if 'text_preview' in citation:
                    citation_text += f" | Preview: {citation['text_preview'][:100]}..."
            else:
                citation_text += str(citation)
            formatted.append(citation_text)

        return "\n".join(formatted)

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, float]:
        """Parse the JSON evaluation response from the LLM."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                # Ensure all expected keys are present
                scores = {}
                expected_keys = [
                    'citation_relevance', 'citation_accuracy',
                    'citation_completeness', 'citation_attribution',
                    'overall_citation_quality'
                ]

                for key in expected_keys:
                    if key in parsed:
                        scores[key] = float(parsed[key])
                    else:
                        scores[key] = 0.0  # Default to 0 if missing

                return scores
            else:
                # Fallback: try to parse the entire response as JSON
                parsed = json.loads(response_text)
                return {
                    'citation_relevance': float(parsed.get('citation_relevance', 0)),
                    'citation_accuracy': float(parsed.get('citation_accuracy', 0)),
                    'citation_completeness': float(parsed.get('citation_completeness', 0)),
                    'citation_attribution': float(parsed.get('citation_attribution', 0)),
                    'overall_citation_quality': float(parsed.get('overall_citation_quality', 0))
                }

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            if self.silent_errors:
                logger.error(f"Failed to parse evaluation response: {response_text[:200]}... Error: {str(e)}")
                # Return default scores
                return {
                    'citation_relevance': 0.0,
                    'citation_accuracy': 0.0,
                    'citation_completeness': 0.0,
                    'citation_attribution': 0.0,
                    'overall_citation_quality': 0.0
                }
            else:
                raise ValueError(f"Invalid evaluation response format: {response_text}") from e


if __name__ == "__main__":
    # Test the evaluator
    print("Testing Citation Quality Evaluator...")

    # Sample data
    system_outputs = [
        {
            'query_id': '1',
            'question': 'What is the capital of France?',
            'generated_response': 'Paris is the capital city of France. [1]',
            'citations': [
                {
                    'url': 'https://en.wikipedia.org/wiki/Paris',
                    'title': 'Paris - Wikipedia',
                    'text_preview': 'Paris is the capital and most populous city of France.'
                }
            ]
        }
    ]

    references = [
        {
            'iid': '1',
            'question': 'What is the capital of France?',
            'reference': 'The capital of France is Paris.'
        }
    ]

    evaluator = CitationQualityEvaluator()

    print(f"Evaluator: {evaluator.name}")
    print(f"Description: {evaluator.description}")

    try:
        result = evaluator.evaluate(system_outputs, references)
        print(f"Metrics: {result.metrics}")
        print(f"Sample count: {result.sample_count}")
        if result.rows:
            print(f"Sample row: {result.rows[0]}")
    except Exception as e:
        print(f"Test error: {e}")

    print("Test complete.")