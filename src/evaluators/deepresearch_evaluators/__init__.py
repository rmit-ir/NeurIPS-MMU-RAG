"""
DeepResearch evaluators based on the benchmarking framework from cxcscmu/deepresearch_benchmarking.

This module contains evaluators for:
- Citation Precision: Checks if claims are supported by their citations
- Citation Recall: Measures percentage of claims with sources
- Holistic Quality: Multi-criteria quality assessment
- Key Point Recall: Coverage of ground-truth key points
- Combined DeepResearch: Runs all evaluators together
"""

from .citation_precision_evaluator import CitationPrecisionEvaluator
from .citation_recall_evaluator import CitationRecallEvaluator
from .holistic_quality_evaluator import HolisticQualityEvaluator
from .key_point_recall_evaluator import KeyPointRecallEvaluator
from .combined_deepresearch_evaluator import CombinedDeepResearchEvaluator

__all__ = [
    "CitationPrecisionEvaluator",
    "CitationRecallEvaluator", 
    "HolisticQualityEvaluator",
    "KeyPointRecallEvaluator",
    "CombinedDeepResearchEvaluator"
]