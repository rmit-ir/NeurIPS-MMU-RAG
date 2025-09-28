"""VLLM-based reranker implementation using Qwen3-Reranker."""

from typing import List, Optional, TypedDict
from vllm import LLM

from tools.web_search import SearchResult
from tools.logging_utils import get_logger

logger = get_logger('reranker_vllm')

# Global singleton instance
_reranker_instance: Optional['VLLMReranker'] = None


# TODO: update web_search.py to use TypedDict instead
class SearchResultRanked(TypedDict):
    """SearchResult with added score field for ranking."""
    text: str
    id: str
    sid: str
    dump: str
    url: str
    date: str
    file_path: str
    language: str
    language_score: float
    token_count: int
    score: float


class VLLMReranker:
    """VLLM-based reranker using Qwen3-Reranker model."""

    def __init__(self,
                 model_name="Qwen/Qwen3-Reranker-0.6B",
                 drop_irrelevant_threshold: Optional[float] = None):
        """Initialize the VLLM reranker."""
        self.model_name = model_name
        self.gpu_memory_utilization = 0.15
        self.max_model_len = 16000  # the max input that can fit in 24Gx0.15
        # e.g., 0.5 to drop irrelevant results
        self.drop_irrelevant_threshold = drop_irrelevant_threshold

        # Templates that works for Qwen3-Reranker only
        self.prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
        self.document_template = "<Document>: {doc}{suffix}"

        self.llm = self._get_llm()
        logger.info("VLLMReranker initialized", model_name=self.model_name)

    def _get_llm(self) -> LLM:
        """Initialize and return the LLM model for Qwen3-Reranker."""
        return LLM(
            model=self.model_name,
            runner="pooling",
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            hf_overrides={
                "architectures": ["Qwen3ForSequenceClassification"],
                "classifier_from_token": ["no", "yes"],
                "is_original_qwen3_reranker": True,
            },
        )

    def _cut_to_words(self, text: str, max_words: int) -> str:
        """Cut the text to the first max_words words."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words])

    def _search_result_to_text(self, result: SearchResult) -> str:
        """Convert SearchResult to formatted text."""
        return f"Web URL: {result.url.strip()}\n\nContent: {result.text.strip()}\n\n"

    def rerank(self, query: str, search_results: List[SearchResult], max_words: int = 4000) -> List[SearchResultRanked]:
        """
        Rerank search results based on query relevance.

        Args:
            query: The search query
            search_results: List of SearchResult objects to rerank
            max_words: Maximum words per document (default: 4000)

        Returns:
            List of SearchResultRanked objects sorted by score (descending)
        """
        if not search_results:
            logger.warning("No search results to rerank")
            return []

        instruction = (
            "Given the web search query, is the retrieved document "
            "(1) from a high quality and relevant website based on the URL, "
            "(2) published recently, and "
            "(3) contains key information that helps answering the query?"
        )

        # Format query and docs
        query_fmt = self.query_template.format(
            prefix=self.prefix,
            instruction=instruction,
            query=query
        )
        docs_fmt = [
            self.document_template.format(
                doc=self._cut_to_words(
                    self._search_result_to_text(result), max_words),
                suffix=self.suffix
            )
            for result in search_results
        ]

        # Get scores from vLLM
        try:
            outputs = self.llm.score(query_fmt, docs_fmt)
            scores = [output.outputs.score for output in outputs]

            logger.info("Reranking completed",
                        num_results=len(search_results),
                        query_length=len(query))

        except Exception as e:
            logger.error("Error during reranking", error=str(e))
            # Fallback: return original results with score 0.0
            scores = [0.0] * len(search_results)

        # Create ranked results
        ranked_results = [
            SearchResultRanked(**(result._asdict()), score=score)
            for result, score in zip(search_results, scores)
        ]

        # Sort by score descending
        ranked_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        if self.drop_irrelevant_threshold is not None:
            # Filter out results with non-positive scores
            ranked_results = [
                res for res in ranked_results if res['score'] > self.drop_irrelevant_threshold]
            logger.info("Filtered irrelevant results",
                        num_remaining=len(ranked_results))

        return ranked_results


def get_reranker(model_name="Qwen/Qwen3-Reranker-0.6B",
                 drop_irrelevant_threshold: Optional[float] = None) -> VLLMReranker:
    """
    Get the global singleton VLLMReranker instance.

    Returns:
        VLLMReranker instance
    """
    global _reranker_instance

    if _reranker_instance is None:
        logger.info("Creating new VLLMReranker instance")
        _reranker_instance = VLLMReranker(
            model_name=model_name,
            drop_irrelevant_threshold=drop_irrelevant_threshold)
    else:
        logger.debug("Returning existing VLLMReranker instance")

    return _reranker_instance
