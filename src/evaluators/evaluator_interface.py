"""
Abstract interface for RAG evaluators.

This module defines the abstract base class that all RAG evaluators should
implement to ensure consistency and interoperability.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime


class EvaluationResult:
    """
    Container for evaluation results.
    
    Holds both aggregated metrics and optional row-level results.
    """
    
    def __init__(
        self,
        metrics: Dict[str, float],
        evaluator_name: str,
        sample_count: int,
        system_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        rows: Optional[List[Dict[str, Any]]] = None,
        total_time_ms: Optional[float] = None,
        total_cost: Optional[float] = None
    ):
        """
        Initialize evaluation result.
        
        Args:
            metrics: Dictionary of metric names to values
            evaluator_name: Name of the evaluator used
            sample_count: Number of samples evaluated
            system_name: Optional name of the system being evaluated
            timestamp: When the evaluation was performed
            rows: Optional list of per-row evaluation results
            total_time_ms: Total evaluation time in milliseconds
            total_cost: Total cost of evaluation (e.g., API costs)
        """
        self.metrics = metrics
        self.evaluator_name = evaluator_name
        self.sample_count = sample_count
        self.system_name = system_name
        self.timestamp = timestamp or datetime.now()
        self.rows = rows or []
        self.total_time_ms = total_time_ms
        self.total_cost = total_cost
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            'metrics': self.metrics,
            'evaluator_name': self.evaluator_name,
            'sample_count': self.sample_count,
            'timestamp': self.timestamp.isoformat()
        }
        
        if self.system_name:
            result['system_name'] = self.system_name
        if self.total_time_ms is not None:
            result['total_time_ms'] = self.total_time_ms
        if self.total_cost is not None:
            result['total_cost'] = self.total_cost
        if self.rows:
            result['rows'] = self.rows
            
        return result


class EvaluatorInterface(ABC):
    """
    Abstract base class for RAG evaluators.
    
    All RAG evaluators should implement this interface to ensure consistency
    and interoperability across different evaluation metrics and approaches.
    """
    
    @abstractmethod
    def evaluate(
        self, 
        system_outputs: List[Dict[str, Any]], 
        references: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate system outputs against reference data.
        
        Args:
            system_outputs: List of system outputs to evaluate
            references: List of reference data to compare against
            
        Returns:
            EvaluationResult containing metrics and optionally row-level results
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the evaluator name."""
        pass
    
    @property
    def description(self) -> str:
        """Return a description of what this evaluator measures."""
        return self.__doc__ or f"{self.name} evaluator"
    
    def validate_inputs(
        self, 
        system_outputs: List[Dict[str, Any]], 
        references: List[Dict[str, Any]]
    ) -> bool:
        """
        Validate input data format.
        
        Args:
            system_outputs: System outputs to validate
            references: Reference data to validate
            
        Returns:
            True if inputs are valid, raises ValueError otherwise
        """
        if not system_outputs:
            raise ValueError("System outputs cannot be empty")
        if not references:
            raise ValueError("References cannot be empty")
        
        # Check that we have matching IDs
        output_ids = {item.get('iid', item.get('query_id')) for item in system_outputs}
        reference_ids = {item.get('iid', item.get('query_id')) for item in references}
        
        if not output_ids.intersection(reference_ids):
            raise ValueError("No matching IDs found between system outputs and references")
        
        return True