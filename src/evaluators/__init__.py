"""
Evaluators package for MMU-RAG system.

This package contains various evaluators for assessing RAG system performance,
following a modular design pattern inspired by the G-RAG-LiveRAG project.
"""

from .evaluator_interface import EvaluatorInterface, EvaluationResult

# Import evaluators (with graceful handling of missing dependencies)
try:
    from .deepeval_evaluator import DeepEvalEvaluator
except ImportError:
    DeepEvalEvaluator = None

try:
    from .citation_quality_evaluator import CitationQualityEvaluator
except ImportError:
    CitationQualityEvaluator = None

try:
    from .nlp_metrics.evaluator import NLPMetricsEvaluator
except ImportError:
    NLPMetricsEvaluator = None

# DeepResearch evaluators
try:
    from .deepresearch_evaluators import (
        CitationPrecisionEvaluator,
        CitationRecallEvaluator,
        HolisticQualityEvaluator,
        KeyPointRecallEvaluator
    )
except ImportError:
    CitationPrecisionEvaluator = None
    CitationRecallEvaluator = None
    HolisticQualityEvaluator = None
    KeyPointRecallEvaluator = None

__all__ = [
    'EvaluatorInterface',
    'EvaluationResult',
    'DeepEvalEvaluator',
    'CitationQualityEvaluator',
    'NLPMetricsEvaluator',
    'CitationPrecisionEvaluator',
    'CitationRecallEvaluator',
    'HolisticQualityEvaluator',
    'KeyPointRecallEvaluator'
]