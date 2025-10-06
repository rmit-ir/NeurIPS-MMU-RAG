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

__all__ = [
    'EvaluatorInterface',
    'EvaluationResult',
    'DeepEvalEvaluator'
]