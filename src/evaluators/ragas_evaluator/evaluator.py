"""
RAGAS Evaluator for semantic evaluation of RAG systems.

This evaluator uses RAGAS (RAG Assessment) framework to measure:
- Faithfulness: How factually consistent the answer is with the context
- Context Precision: How relevant the retrieved context is to the question
"""

import os
import time
from typing import List, Dict, Any, Optional
import pandas as pd
from datasets import Dataset

from src.evaluators.evaluator_interface import EvaluatorInterface, EvaluationResult


class RAGASEvaluator(EvaluatorInterface):
    """
    Evaluator using RAGAS framework for semantic evaluation of RAG systems.
    
    Measures faithfulness and context precision using LLM-based evaluation.
    """
    
    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        include_faithfulness: bool = True,
        include_context_precision: bool = True
    ):
        """
        Initialize RAGAS evaluator.
        
        Args:
            model_name: Model name for evaluation (supports OpenRouter format)
            api_key: API key for the model provider
            include_faithfulness: Whether to include faithfulness metric
            include_context_precision: Whether to include context precision metric
        """
        self.model_name = model_name
        self.api_key = api_key
        self.include_faithfulness = include_faithfulness
        self.include_context_precision = include_context_precision
        
        if not (include_faithfulness or include_context_precision):
            raise ValueError("At least one metric must be enabled")
    
    @property
    def name(self) -> str:
        """Return evaluator name."""
        return "RAGASEvaluator"
    
    @property
    def description(self) -> str:
        """Return evaluator description."""
        metrics = []
        if self.include_faithfulness:
            metrics.append("Faithfulness")
        if self.include_context_precision:
            metrics.append("Context Precision")
        return f"RAGAS semantic evaluation: {', '.join(metrics)}"
    
    def evaluate(
        self,
        system_outputs: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate system outputs using RAGAS metrics.
        
        Args:
            system_outputs: List of system outputs with keys: query_id, generated_response, citations
            references: List of references with keys: iid/query_id, query, reference
            
        Returns:
            EvaluationResult with RAGAS metrics
        """
        start_time = time.time()
        
        # Validate inputs
        self.validate_inputs(system_outputs, references)
        
        # Merge data
        merged_data = self._merge_data(system_outputs, references)
        
        if not merged_data:
            raise ValueError("No matching data found between outputs and references")
        
        # Run RAGAS evaluation
        try:
            metrics, row_results = self._run_ragas_evaluation(merged_data)
            
            # Calculate execution time
            total_time_ms = (time.time() - start_time) * 1000
            
            return EvaluationResult(
                metrics=metrics,
                evaluator_name=self.name,
                sample_count=len(merged_data),
                timestamp=None,  # Will be set automatically
                rows=row_results,
                total_time_ms=total_time_ms
            )
            
        except Exception as e:
            raise RuntimeError(f"RAGAS evaluation failed: {str(e)}")
    
    def _merge_data(
        self,
        system_outputs: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge system outputs with references by ID."""
        # Create lookup dictionaries - handle both query_id and iid
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
                    'query': reference.get('query', ''),
                    'generated_response': output.get('generated_response', ''),
                    'reference': reference.get('generated_response') or reference.get('reference', ''),
                    'citations': output.get('citations', [])
                })
        
        return merged_data
    
    def _run_ragas_evaluation(
        self,
        merged_data: List[Dict[str, Any]]
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Run RAGAS evaluation on merged data."""
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, context_precision
            from ragas.llms import LangchainLLMWrapper
            
            # Configure model
            self._configure_model()
            
            # Create LangChain model
            langchain_llm = self._create_langchain_model()
            ragas_llm = LangchainLLMWrapper(langchain_llm)
            
            # Prepare dataset
            formatted_data = self._format_for_ragas(merged_data)
            dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
            
            # Configure metrics
            metrics_to_use = []
            if self.include_faithfulness:
                faithfulness.llm = ragas_llm
                metrics_to_use.append(faithfulness)
            if self.include_context_precision:
                context_precision.llm = ragas_llm
                metrics_to_use.append(context_precision)
            
            # Run evaluation
            result = evaluate(dataset=dataset, metrics=metrics_to_use)
            result_df = result.to_pandas()
            
            # Extract metrics
            aggregated_metrics = {}
            if self.include_faithfulness and 'faithfulness' in result_df.columns:
                faith_scores = result_df['faithfulness'].dropna()
                aggregated_metrics['mean_faithfulness'] = float(faith_scores.mean())
                aggregated_metrics['std_faithfulness'] = float(faith_scores.std())
                aggregated_metrics['min_faithfulness'] = float(faith_scores.min())
                aggregated_metrics['max_faithfulness'] = float(faith_scores.max())
            
            if self.include_context_precision and 'context_precision' in result_df.columns:
                cp_scores = result_df['context_precision'].dropna()
                aggregated_metrics['mean_context_precision'] = float(cp_scores.mean())
                aggregated_metrics['std_context_precision'] = float(cp_scores.std())
                aggregated_metrics['min_context_precision'] = float(cp_scores.min())
                aggregated_metrics['max_context_precision'] = float(cp_scores.max())
            
            # Extract row results
            row_results = []
            for idx, row in result_df.iterrows():
                row_result = {
                    'query_id': merged_data[idx]['query_id'],
                    'query': merged_data[idx]['query']
                }
                
                if self.include_faithfulness and 'faithfulness' in result_df.columns:
                    row_result['faithfulness'] = float(row.get('faithfulness', 0))
                
                if self.include_context_precision and 'context_precision' in result_df.columns:
                    row_result['context_precision'] = float(row.get('context_precision', 0))
                
                row_results.append(row_result)
            
            return aggregated_metrics, row_results
            
        except ImportError as e:
            raise ImportError(f"RAGAS dependencies not available: {e}")
        except Exception as e:
            raise RuntimeError(f"RAGAS evaluation error: {e}")
    
    def _configure_model(self):
        """Configure environment variables for model access."""
        if self.api_key and self.api_key.startswith("sk-or-"):
            # OpenRouter configuration
            os.environ["OPENROUTER_API_KEY"] = self.api_key
            os.environ["OPENAI_API_KEY"] = self.api_key
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        else:
            # Standard OpenAI configuration
            os.environ["OPENAI_API_KEY"] = self.api_key
    
    def _create_langchain_model(self):
        """Create LangChain model for RAGAS."""
        from langchain_openai import ChatOpenAI
        
        if self.api_key and self.api_key.startswith("sk-or-"):
            # OpenRouter model
            return ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0,
                max_retries=3,
                request_timeout=120
            )
        else:
            # Standard OpenAI model
            return ChatOpenAI(
                model_name=self.model_name,
                temperature=0
            )
    
    def _format_for_ragas(
        self,
        merged_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Format data for RAGAS evaluation."""
        formatted_data = []
        
        for item in merged_data:
            # Create contexts from citations or use response excerpt
            citations = item.get('citations', [])
            if citations:
                contexts = [f"Retrieved context from {cite}" for cite in citations[:5]]
            else:
                response = item['generated_response']
                contexts = [f"Context: {response[:300]}..." if len(response) > 300 else response]
            
            formatted_data.append({
                'question': item['query'],
                'answer': item['generated_response'],
                'ground_truth': item['reference'],
                'contexts': contexts
            })
        
        return formatted_data


if __name__ == "__main__":
    # Test the evaluator
    print("Testing RAGAS Evaluator...")
    
    # Sample data
    system_outputs = [
        {
            'query_id': '1',
            'generated_response': 'Paris is the capital of France.',
            'citations': ['France_Wikipedia']
        }
    ]
    
    references = [
        {
            'iid': '1',
            'query': 'What is the capital of France?',
            'reference': 'The capital of France is Paris.'
        }
    ]
    
    evaluator = RAGASEvaluator(
        model_name="openai/gpt-4o-mini",
        api_key=None  # Would need real API key for actual testing
    )
    
    print(f"Evaluator: {evaluator.name}")
    print(f"Description: {evaluator.description}")
    
    # Note: Actual evaluation would require API key
    print("Test setup complete. Actual evaluation requires API key.")