"""
Citation Quality Evaluator for assessing citation usage in RAG responses.

This evaluator measures how well citations are used in generated responses,
including relevance, accuracy, and proper attribution.
"""

from .evaluator import CitationQualityEvaluator

__all__ = ["CitationQualityEvaluator"]