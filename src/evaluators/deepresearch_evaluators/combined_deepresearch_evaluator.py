"""
Combined DeepResearch Evaluator

This evaluator runs all DeepResearch benchmarking evaluators together:
- Citation Recall
- Citation Precision (requires URL crawling)
- Key Point Recall
- Holistic Quality

Usage:
    evaluator = CombinedDeepResearchEvaluator(
        model="openai.gpt-oss-120b-1:0",
        num_threads=4
    )
    result = evaluator.evaluate(system_outputs, references)
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..evaluator_interface import EvaluatorInterface, EvaluationResult
from .citation_recall_evaluator import CitationRecallEvaluator
from .citation_precision_evaluator import CitationPrecisionEvaluator
from .key_point_recall_evaluator import KeyPointRecallEvaluator
from .holistic_quality_evaluator import HolisticQualityEvaluator


class CombinedDeepResearchEvaluator(EvaluatorInterface):
    """
    Combined evaluator that runs all DeepResearch benchmarking metrics.
    
    This evaluator provides a comprehensive assessment by running:
    1. Citation Recall - measures if claims have supporting citations
    2. Citation Precision - verifies if citations actually support claims (requires crawling)
    3. Key Point Recall - checks if key points are supported/omitted/contradicted
    4. Holistic Quality - evaluates 6 quality criteria (Clarity, Depth, Balance, etc.)
    """
    
    @property
    def name(self) -> str:
        """Return the evaluator name."""
        return "combined_deepresearch"
    
    def __init__(
        self,
        model: str = "openai.gpt-oss-120b-1:0",
        num_threads: int = 1,
        silent_errors: bool = True,
        include_citation_precision: bool = False,
        **kwargs
    ):
        """
        Initialize the combined DeepResearch evaluator.
        
        Args:
            model: LLM model to use for evaluation
            num_threads: Number of threads for parallel processing
            silent_errors: Whether to silence errors and continue evaluation
            include_citation_precision: Whether to include citation precision (slow, requires crawling)
            **kwargs: Additional arguments passed to individual evaluators
        """
        super().__init__()
        self.model = model
        self.num_threads = num_threads
        self.silent_errors = silent_errors
        self.include_citation_precision = include_citation_precision
        
        # Initialize all evaluators
        self.citation_recall = CitationRecallEvaluator(
            model=model,
            num_threads=num_threads,
            silent_errors=silent_errors
        )
        
        self.key_point_recall = KeyPointRecallEvaluator(
            model=model,
            num_threads=num_threads,
            silent_errors=silent_errors
        )
        
        self.holistic_quality = HolisticQualityEvaluator(
            model=model,
            num_threads=num_threads,
            silent_errors=silent_errors
        )
        
        # Citation precision is optional (slower, requires URL crawling)
        if include_citation_precision:
            self.citation_precision = CitationPrecisionEvaluator(
                model=model,
                num_threads=num_threads,
                silent_errors=silent_errors
            )
        else:
            self.citation_precision = None
    
    def evaluate(
        self,
        system_outputs: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Run all DeepResearch evaluators on the provided data.
        
        Args:
            system_outputs: List of system-generated outputs with keys:
                - iid or query_id: unique identifier
                - answer or generated_response: the system's response
                - citations: list of citations (optional)
                - contexts: list of retrieved contexts (optional)
            references: List of reference data with keys:
                - iid or query_id: unique identifier matching system outputs
                - query: the question/query
                - reference: reference answer text
                - key_points: list of key points to check (optional)
        
        Returns:
            EvaluationResult with combined metrics from all evaluators
        """
        start_time = time.time()
        
        # Validate inputs
        self.validate_inputs(system_outputs, references)
        
        print(f"\n{'='*80}")
        print(f"Running Combined DeepResearch Evaluation")
        print(f"{'='*80}")
        print(f"Samples: {len(system_outputs)}")
        print(f"Model: {self.model}")
        print(f"Threads: {self.num_threads}")
        print(f"Include Citation Precision: {self.include_citation_precision}")
        print(f"{'='*80}\n")
        
        # Run all evaluators
        results = {}
        rows_by_id = {}
        
        # 1. Citation Recall
        print("Running Citation Recall Evaluator...")
        citation_recall_result = self.citation_recall.evaluate(system_outputs, references)
        results['citation_recall'] = citation_recall_result
        
        # Merge rows
        for row in citation_recall_result.rows:
            query_id = row['query_id']
            if query_id not in rows_by_id:
                rows_by_id[query_id] = {'query_id': query_id}
            rows_by_id[query_id].update({
                'citation_recall': row['citation_recall'],
                'total_claims': row['total_claims'],
                'supported_claims': row['supported_claims']
            })
        
        print(f"✓ Citation Recall: {citation_recall_result.metrics['citation_recall']:.2%}\n")
        
        # 2. Key Point Recall
        print("Running Key Point Recall Evaluator...")
        key_point_result = self.key_point_recall.evaluate(system_outputs, references)
        results['key_point_recall'] = key_point_result
        
        # Merge rows
        for row in key_point_result.rows:
            query_id = row['query_id']
            if query_id not in rows_by_id:
                rows_by_id[query_id] = {'query_id': query_id}
            rows_by_id[query_id].update({
                'key_point_recall': row['key_point_recall'],
                'supported_points': row['supported_count'],
                'omitted_points': row['omitted_count'],
                'contradicted_points': row['contradicted_count']
            })
        
        print(f"✓ Key Point Recall: {key_point_result.metrics['key_point_recall']:.2%}\n")
        
        # 3. Holistic Quality
        print("Running Holistic Quality Evaluator...")
        holistic_result = self.holistic_quality.evaluate(system_outputs, references)
        results['holistic_quality'] = holistic_result
        
        # Merge rows
        for row in holistic_result.rows:
            query_id = row['query_id']
            if query_id not in rows_by_id:
                rows_by_id[query_id] = {'query_id': query_id}
            rows_by_id[query_id].update({
                'holistic_quality': row['holistic_quality'],
                'quality_scores': row['scores']
            })
        
        print(f"✓ Holistic Quality: {holistic_result.metrics['holistic_quality']:.2f}%\n")
        
        # 4. Citation Precision (optional, slower)
        if self.citation_precision:
            print("Running Citation Precision Evaluator (this may take longer)...")
            citation_precision_result = self.citation_precision.evaluate(system_outputs, references)
            results['citation_precision'] = citation_precision_result
            
            # Merge rows
            for row in citation_precision_result.rows:
                query_id = row['query_id']
                if query_id not in rows_by_id:
                    rows_by_id[query_id] = {'query_id': query_id}
                rows_by_id[query_id].update({
                    'citation_precision': row['citation_precision'],
                    'verified_citations': row['verified_citations'],
                    'total_citations': row['total_citations']
                })
            
            print(f"✓ Citation Precision: {citation_precision_result.metrics['citation_precision']:.2%}\n")
        
        # Combine metrics
        combined_metrics = {
            'citation_recall': citation_recall_result.metrics['citation_recall'],
            'citation_recall_total_claims': citation_recall_result.metrics['total_claims'],
            'citation_recall_supported': citation_recall_result.metrics['total_supported_claims'],
            
            'key_point_recall': key_point_result.metrics['key_point_recall'],
            'key_point_support_rate': key_point_result.metrics['avg_support_rate'],
            'key_point_omitted_rate': key_point_result.metrics['avg_omitted_rate'],
            'key_point_contradicted_rate': key_point_result.metrics['avg_contradicted_rate'],
            
            'holistic_quality': holistic_result.metrics['holistic_quality'],
            'clarity_avg': holistic_result.metrics.get('clarity_avg', 0),
            'depth_avg': holistic_result.metrics.get('depth_avg', 0),
            'balance_avg': holistic_result.metrics.get('balance_avg', 0),
            'breadth_avg': holistic_result.metrics.get('breadth_avg', 0),
            'support_avg': holistic_result.metrics.get('support_avg', 0),
            'insightfulness_avg': holistic_result.metrics.get('insightfulness_avg', 0),
        }
        
        if self.citation_precision:
            combined_metrics.update({
                'citation_precision': citation_precision_result.metrics['citation_precision'],
                'citation_precision_verified': citation_precision_result.metrics['total_verified_citations'],
                'citation_precision_total': citation_precision_result.metrics['total_citations']
            })
        
        # Calculate overall score (average of normalized metrics)
        # Normalize all metrics to 0-100 scale
        normalized_scores = [
            citation_recall_result.metrics['citation_recall'] * 100,  # Already 0-1
            key_point_result.metrics['key_point_recall'] * 100,  # Already 0-1
            holistic_result.metrics['holistic_quality']  # Already 0-100
        ]
        
        if self.citation_precision:
            normalized_scores.append(citation_precision_result.metrics['citation_precision'] * 100)
        
        overall_score = sum(normalized_scores) / len(normalized_scores)
        combined_metrics['overall_deepresearch_score'] = overall_score
        
        # Convert rows dict to list
        combined_rows = list(rows_by_id.values())
        
        # Calculate total time and cost
        total_time_ms = (time.time() - start_time) * 1000
        total_cost = sum(
            r.total_cost for r in results.values() 
            if r.total_cost is not None
        ) or None
        
        print(f"{'='*80}")
        print(f"Combined DeepResearch Evaluation Complete")
        print(f"{'='*80}")
        print(f"Overall Score: {overall_score:.2f}%")
        print(f"  - Citation Recall: {combined_metrics['citation_recall']:.2%}")
        print(f"  - Key Point Recall: {combined_metrics['key_point_recall']:.2%}")
        print(f"  - Holistic Quality: {combined_metrics['holistic_quality']:.2f}%")
        if self.citation_precision:
            print(f"  - Citation Precision: {combined_metrics['citation_precision']:.2%}")
        print(f"\nTotal Time: {total_time_ms/1000:.2f}s")
        if total_cost:
            print(f"Total Cost: ${total_cost:.4f}")
        print(f"{'='*80}\n")
        
        return EvaluationResult(
            metrics=combined_metrics,
            evaluator_name=self.name,
            sample_count=len(system_outputs),
            rows=combined_rows,
            total_time_ms=total_time_ms,
            total_cost=total_cost,
            timestamp=datetime.now()
        )
    
    def get_sub_results(
        self,
        system_outputs: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> Dict[str, EvaluationResult]:
        """
        Run evaluation and return individual results from each sub-evaluator.
        
        This is useful if you need access to detailed results from each evaluator.
        
        Returns:
            Dictionary with keys: 'citation_recall', 'key_point_recall', 
            'holistic_quality', and optionally 'citation_precision'
        """
        results = {}
        
        results['citation_recall'] = self.citation_recall.evaluate(system_outputs, references)
        results['key_point_recall'] = self.key_point_recall.evaluate(system_outputs, references)
        results['holistic_quality'] = self.holistic_quality.evaluate(system_outputs, references)
        
        if self.citation_precision:
            results['citation_precision'] = self.citation_precision.evaluate(system_outputs, references)
        
        return results
