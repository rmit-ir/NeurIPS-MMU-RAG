"""
DeepEval evaluator implementation.

This module provides a DeepEval-based evaluator that uses custom LLMs
to evaluate RAG system outputs using multiple metrics:
- Faithfulness: Whether the output is factually aligned with retrieval context
- Answer Relevancy: Whether the output is relevant to the input
- Contextual Relevancy: Whether the retrieval context is relevant to the input
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

try:
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRelevancyMetric
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

from .custom_llm import MMUCustomLLM

from src.evaluators.evaluator_interface import EvaluatorInterface, EvaluationResult


class DeepEvalEvaluator(EvaluatorInterface):
    """
    DeepEval-based evaluator using custom MMU LLM.
    
    This evaluator measures:
    - Faithfulness: Factual alignment with retrieval context
    - Answer Relevancy: Relevance of output to input
    - Contextual Relevancy: Quality of retrieved context
    """
    
    def __init__(
        self,
        model: str = "openai.gpt-oss-120b-1:0",
        base_url: str = "https://mmu-proxy-server-llm-proxy.rankun.org",
        api_key: Optional[str] = None,
        faithfulness_threshold: float = 0.7,
        answer_relevancy_threshold: float = 0.7,
        contextual_relevancy_threshold: float = 0.7,
        include_reason: bool = True,
        verbose: bool = False,
        num_workers: Optional[int] = None
    ):
        """
        Initialize DeepEval evaluator.
        
        Args:
            model: Model name to use
            base_url: Base URL for MMU proxy server
            api_key: API key (defaults to MMU_OPENAI_API_KEY env var)
            faithfulness_threshold: Minimum passing threshold for faithfulness
            answer_relevancy_threshold: Minimum passing threshold for answer relevancy
            contextual_relevancy_threshold: Minimum passing threshold for contextual relevancy
            include_reason: Whether to include reasoning in results
            verbose: Whether to print detailed logs
            num_workers: Number of worker processes for parallel evaluation (default: None for sequential)
        """
        if not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "DeepEval is not installed. Please install it with: pip install deepeval"
            )
        
        self.verbose = verbose
        self.include_reason = include_reason
        self.num_workers = num_workers or multiprocessing.cpu_count()
        self.api_key = api_key
        
        # Initialize custom LLM
        self.custom_llm = MMUCustomLLM(
            model=model,
            base_url=base_url,
            api_key=api_key
        )
        
        # Initialize metrics
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=faithfulness_threshold,
            model=self.custom_llm,
            include_reason=include_reason,
            verbose_mode=verbose
        )
        
        self.answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=answer_relevancy_threshold,
            model=self.custom_llm,
            include_reason=include_reason,
            verbose_mode=verbose
        )
        
        self.contextual_relevancy_metric = ContextualRelevancyMetric(
            threshold=contextual_relevancy_threshold,
            model=self.custom_llm,
            include_reason=include_reason,
            verbose_mode=verbose
        )
        
        if self.verbose:
            print(f"Initialized DeepEval evaluator with model: {model}")
    
    def _evaluate_single_sample(
        self, 
        output: Dict[str, Any], 
        reference: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Evaluate a single sample with all metrics.
        
        Args:
            output: System output dictionary
            reference: Reference dictionary
            
        Returns:
            Tuple of (output_id, result_dict)
        """
        output_id = output.get('iid', output.get('query_id'))
        
        try:
            # Create test case
            test_case = self._create_test_case(output, reference)
            
            # Measure all metrics
            self.faithfulness_metric.measure(test_case)
            faithfulness_score = self.faithfulness_metric.score
            
            self.answer_relevancy_metric.measure(test_case)
            answer_relevancy_score = self.answer_relevancy_metric.score
            
            self.contextual_relevancy_metric.measure(test_case)
            contextual_relevancy_score = self.contextual_relevancy_metric.score
            
            # Create result
            result = {
                'id': output_id,
                'faithfulness': faithfulness_score,
                'answer_relevancy': answer_relevancy_score,
                'contextual_relevancy': contextual_relevancy_score
            }
            
            if self.include_reason:
                result['faithfulness_reason'] = self.faithfulness_metric.reason
                result['answer_relevancy_reason'] = self.answer_relevancy_metric.reason
                result['contextual_relevancy_reason'] = self.contextual_relevancy_metric.reason
            
            return output_id, result
            
        except Exception as e:
            return output_id, {
                'id': output_id,
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'contextual_relevancy': 0.0,
                'error': str(e)
            }
    @property
    def name(self) -> str:
        """Return evaluator name."""
        return "DeepEval"
    
    @property
    def description(self) -> str:
        """Return evaluator description."""
        return (
            "DeepEval evaluator using custom MMU LLM for faithfulness, "
            "answer relevancy, and contextual relevancy metrics"
        )
    
    def _create_test_case(self, system_output: Dict[str, Any], reference: Dict[str, Any]) -> LLMTestCase:
        """
        Create a DeepEval test case from system output and reference.
        
        Args:
            system_output: System output dictionary
            reference: Reference dictionary
            
        Returns:
            LLMTestCase instance
        """
        # Extract fields from system output and reference
        # Try to get query from multiple sources: system_output, then reference
        input_text = (
            system_output.get('query') or
            system_output.get('input') or
            reference.get('query') or
            reference.get('input') or
            ''
        )
        
        actual_output = (
            system_output.get('response') or
            system_output.get('generated_response') or
            system_output.get('output') or
            system_output.get('actual_output') or
            ''
        )
        
        # Extract retrieval context
        retrieval_context = []
        if 'contexts' in system_output:
            retrieval_context = system_output['contexts']
        elif 'retrieved_contexts' in system_output:
            retrieval_context = system_output['retrieved_contexts']
        elif 'context' in system_output:
            ctx = system_output['context']
            retrieval_context = [ctx] if isinstance(ctx, str) else ctx
        
        # Create test case
        return LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
    
    def evaluate(
        self,
        system_outputs: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate system outputs using DeepEval metrics.
        
        Args:
            system_outputs: List of system outputs to evaluate
            references: List of reference data
            
        Returns:
            EvaluationResult with faithfulness, answer relevancy, and contextual relevancy scores
        """
        # Validate inputs
        self.validate_inputs(system_outputs, references)
        
        start_time = time.time()
        
        # Create lookup for references by ID
        ref_lookup = {}
        for ref in references:
            ref_id = ref.get('iid', ref.get('query_id'))
            if ref_id:
                ref_lookup[ref_id] = ref
        
        # Prepare evaluation tasks
        evaluation_tasks = []
        for output in system_outputs:
            output_id = output.get('iid', output.get('query_id'))
            reference = ref_lookup.get(output_id, {})
            evaluation_tasks.append((output, reference))
        
        # Evaluate samples (parallel or sequential based on num_workers)
        row_results = []
        faithfulness_scores = []
        answer_relevancy_scores = []
        contextual_relevancy_scores = []
        
        if self.num_workers > 1 and len(evaluation_tasks) > 1:
            # Use parallel processing
            if self.verbose:
                print(f"Using {self.num_workers} workers for parallel evaluation")
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(
                        _evaluate_sample_worker,
                        output, reference,
                        self.custom_llm.model_name, self.custom_llm.base_url, self.api_key,
                        self.faithfulness_metric.threshold,
                        self.answer_relevancy_metric.threshold,
                        self.contextual_relevancy_metric.threshold,
                        self.include_reason, self.verbose
                    ): (output, reference) for output, reference in evaluation_tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    output_id, result = future.result()
                    row_results.append(result)
                    
                    # Collect scores
                    faithfulness_scores.append(result['faithfulness'])
                    answer_relevancy_scores.append(result['answer_relevancy'])
                    contextual_relevancy_scores.append(result['contextual_relevancy'])
                    
                    if self.verbose:
                        print(f"Completed evaluation for sample ID: {output_id}")
        else:
            # Use sequential processing
            if self.verbose:
                print("Using sequential evaluation")
            
            for i, (output, reference) in enumerate(evaluation_tasks):
                output_id, result = self._evaluate_single_sample(output, reference)
                row_results.append(result)
                
                # Collect scores
                faithfulness_scores.append(result['faithfulness'])
                answer_relevancy_scores.append(result['answer_relevancy'])
                contextual_relevancy_scores.append(result['contextual_relevancy'])
                
                if self.verbose:
                    print(f"Evaluating sample {i+1}/{len(evaluation_tasks)} (ID: {output_id})")
                    print(f"  Faithfulness: {result['faithfulness']:.3f}")
                    print(f"  Answer Relevancy: {result['answer_relevancy']:.3f}")
                    print(f"  Contextual Relevancy: {result['contextual_relevancy']:.3f}")
        
        # Calculate average metrics
        avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
        avg_answer_relevancy = sum(answer_relevancy_scores) / len(answer_relevancy_scores) if answer_relevancy_scores else 0.0
        avg_contextual_relevancy = sum(contextual_relevancy_scores) / len(contextual_relevancy_scores) if contextual_relevancy_scores else 0.0
        
        # Calculate overall average
        overall_avg = (avg_faithfulness + avg_answer_relevancy + avg_contextual_relevancy) / 3
        
        total_time = (time.time() - start_time) * 1000  # Convert to ms
        
        metrics = {
            'faithfulness': avg_faithfulness,
            'answer_relevancy': avg_answer_relevancy,
            'contextual_relevancy': avg_contextual_relevancy,
            'overall': overall_avg
        }
        
        if self.verbose:
            print(f"\n{'='*50}")
            print(f"Evaluation Complete")
            print(f"{'='*50}")
            print(f"Average Faithfulness: {avg_faithfulness:.3f}")
            print(f"Average Answer Relevancy: {avg_answer_relevancy:.3f}")
            print(f"Average Contextual Relevancy: {avg_contextual_relevancy:.3f}")
            print(f"Overall Average: {overall_avg:.3f}")
            print(f"Total Time: {total_time:.2f}ms")
        
        return EvaluationResult(
            metrics=metrics,
            evaluator_name=self.name,
            sample_count=len(system_outputs),
            timestamp=datetime.now(),
            rows=row_results,
            total_time_ms=total_time
        )


def _evaluate_sample_worker(
    output: Dict[str, Any],
    reference: Dict[str, Any],
    model: str,
    base_url: str,
    api_key: Optional[str],
    faithfulness_threshold: float,
    answer_relevancy_threshold: float,
    contextual_relevancy_threshold: float,
    include_reason: bool,
    verbose: bool
) -> Tuple[str, Dict[str, Any]]:
    """
    Worker function for evaluating a single sample with all metrics.
    This function is designed to be used with multiprocessing.
    """
    output_id = output.get('iid', output.get('query_id'))
    
    try:
        # Initialize custom LLM (each worker gets its own instance)
        custom_llm = MMUCustomLLM(
            model=model,
            base_url=base_url,
            api_key=api_key
        )
        
        # Initialize metrics
        faithfulness_metric = FaithfulnessMetric(
            threshold=faithfulness_threshold,
            model=custom_llm,
            include_reason=include_reason,
            verbose_mode=verbose
        )
        
        answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=answer_relevancy_threshold,
            model=custom_llm,
            include_reason=include_reason,
            verbose_mode=verbose
        )
        
        contextual_relevancy_metric = ContextualRelevancyMetric(
            threshold=contextual_relevancy_threshold,
            model=custom_llm,
            include_reason=include_reason,
            verbose_mode=verbose
        )
        
        # Create test case
        input_text = (
            output.get('query') or
            output.get('input') or
            reference.get('query') or
            reference.get('input') or
            ''
        )
        
        actual_output = (
            output.get('response') or
            output.get('generated_response') or
            output.get('output') or
            output.get('actual_output') or
            ''
        )
        
        # Extract retrieval context
        retrieval_context = []
        if 'contexts' in output:
            retrieval_context = output['contexts']
        elif 'retrieved_contexts' in output:
            retrieval_context = output['retrieved_contexts']
        elif 'context' in output:
            ctx = output['context']
            retrieval_context = [ctx] if isinstance(ctx, str) else ctx
        
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context
        )
        
        # Measure all metrics
        faithfulness_metric.measure(test_case)
        faithfulness_score = faithfulness_metric.score
        
        answer_relevancy_metric.measure(test_case)
        answer_relevancy_score = answer_relevancy_metric.score
        
        contextual_relevancy_metric.measure(test_case)
        contextual_relevancy_score = contextual_relevancy_metric.score
        
        # Create result
        result = {
            'id': output_id,
            'faithfulness': faithfulness_score,
            'answer_relevancy': answer_relevancy_score,
            'contextual_relevancy': contextual_relevancy_score
        }
        
        if include_reason:
            result['faithfulness_reason'] = faithfulness_metric.reason
            result['answer_relevancy_reason'] = answer_relevancy_metric.reason
            result['contextual_relevancy_reason'] = contextual_relevancy_metric.reason
        
        return output_id, result
        
    except Exception as e:
        return output_id, {
            'id': output_id,
            'faithfulness': 0.0,
            'answer_relevancy': 0.0,
            'contextual_relevancy': 0.0,
            'error': str(e)
        }


def load_system_outputs(filepath: str) -> List[Dict[str, Any]]:
    """Load system outputs from JSONL file."""
    outputs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            outputs.append({
                'iid': data.get('query_id', data.get('iid')),
                'response': data.get('generated_response', data.get('response', '')),
                'contexts': data.get('contexts', []),
                'query': data.get('query', '')
            })
    return outputs


def load_references(filepath: str) -> List[Dict[str, Any]]:
    """Load references from JSONL file."""
    references = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            references.append({
                'iid': data.get('id', data.get('iid', data.get('query_id'))),
                'query': data.get('narrative', data.get('query', '')),
                'reference': data.get('reference', '')
            })
    return references


def main():
    """Main entry point for running DeepEval evaluator from command line."""
    parser = argparse.ArgumentParser(
        description='Run DeepEval evaluator on system outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--system-outputs',
        required=True,
        help='Path to system outputs JSONL file'
    )
    parser.add_argument(
        '--references',
        required=True,
        help='Path to references JSONL file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model',
        default='qwen.qwen3-32b-v1:0',
        help='Model name (default: qwen.qwen3-32b-v1:0)'
    )
    parser.add_argument(
        '--base-url',
        default='https://mmu-proxy-server-llm-proxy.rankun.org',
        help='Base URL for MMU proxy server'
    )
    parser.add_argument(
        '--api-key',
        default=None,
        help='API key (defaults to MMU_OPENAI_API_KEY env var)'
    )
    parser.add_argument(
        '--faithfulness-threshold',
        type=float,
        default=0.7,
        help='Faithfulness threshold (default: 0.7)'
    )
    parser.add_argument(
        '--answer-relevancy-threshold',
        type=float,
        default=0.7,
        help='Answer relevancy threshold (default: 0.7)'
    )
    parser.add_argument(
        '--contextual-relevancy-threshold',
        type=float,
        default=0.7,
        help='Contextual relevancy threshold (default: 0.7)'
    )
    parser.add_argument(
        '--no-reason',
        action='store_true',
        help='Disable reasons in results (faster)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--output-file',
        default=None,
        help='Output JSON file (default: auto-generated in results/)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='Number of worker processes for parallel evaluation (default: CPU count)'
    )
    
    args = parser.parse_args()
    
    # Set API key
    if args.api_key:
        os.environ["MMU_OPENAI_API_KEY"] = args.api_key
    
    print("="*70)
    print("DeepEval Evaluation")
    print("="*70)
    
    try:
        # Load data
        print("\n📁 Loading data...")
        system_outputs = load_system_outputs(args.system_outputs)
        print(f"   ✓ Loaded {len(system_outputs)} system outputs")
        
        references = load_references(args.references)
        print(f"   ✓ Loaded {len(references)} references")
        
        # Match outputs with references
        ref_lookup = {ref['iid']: ref for ref in references}
        matched_outputs = []
        
        for output in system_outputs:
            iid = output['iid']
            if iid in ref_lookup:
                if not output['query']:
                    output['query'] = ref_lookup[iid]['query']
                matched_outputs.append(output)
        
        print(f"   ✓ Matched {len(matched_outputs)} outputs")
        
        # Initialize evaluator
        print("\n🔧 Initializing evaluator...")
        evaluator = DeepEvalEvaluator(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            faithfulness_threshold=args.faithfulness_threshold,
            answer_relevancy_threshold=args.answer_relevancy_threshold,
            contextual_relevancy_threshold=args.contextual_relevancy_threshold,
            include_reason=not args.no_reason,
            verbose=args.verbose,
            num_workers=args.num_workers
        )
        
        # Evaluate
        print("\n🚀 Starting evaluation...")
        result = evaluator.evaluate(matched_outputs, references)
        
        # Print results
        print("\n" + "="*70)
        print("📊 RESULTS")
        print("="*70)
        print(f"Faithfulness:         {result.metrics['faithfulness']:.3f}")
        print(f"Answer Relevancy:     {result.metrics['answer_relevancy']:.3f}")
        print(f"Contextual Relevancy: {result.metrics['contextual_relevancy']:.3f}")
        print(f"Overall:              {result.metrics['overall']:.3f}")
        print(f"Time:                 {result.total_time_ms/1000:.2f}s")
        
        # Save results
        if args.output_file:
            output_file = args.output_file
        else:
            output_name = Path(args.system_outputs).stem
            output_file = f'results/deepeval_{output_name}_results.json'
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        print(f"\n✅ Saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
