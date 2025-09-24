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
    
    Measures faithfulness, context precision, and answer relevancy using LLM-based evaluation.
    """
    
    def __init__(
        self,
        model_name: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        include_faithfulness: bool = True,
        include_context_precision: bool = False,
        include_answer_relevancy: bool = False
    ):
        """
        Initialize RAGAS evaluator.
        
        Args:
            model_name: Model name for evaluation (supports OpenRouter format: provider/model, 
                       Bedrock format: bedrock/model_id, or OpenAI format: openai/model)
            api_key: API key for the model provider (not needed for Bedrock)
            include_faithfulness: Whether to include faithfulness metric
            include_context_precision: Whether to include context precision metric (only evaluated 
                                     if RAG system provides actual document contexts)
            include_answer_relevancy: Whether to include answer relevancy metric
        """
        self.model_name = model_name
        self.api_key = api_key
        self.include_faithfulness = include_faithfulness
        self.include_context_precision = include_context_precision
        self.include_answer_relevancy = include_answer_relevancy
        
        if not (include_faithfulness or include_context_precision or include_answer_relevancy):
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
        if self.include_answer_relevancy:
            metrics.append("Answer Relevancy")
        return f"RAGAS semantic evaluation: {', '.join(metrics)}"
    
    def evaluate(
        self,
        system_outputs: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate system outputs using RAGAS metrics.
        
        Args:
            system_outputs: List of system outputs with keys: query_id, generated_response, citations, contexts
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
                    'citations': output.get('citations', []),
                    'contexts': output.get('contexts', [])
                })
        
        return merged_data
    
    def _run_ragas_evaluation(
        self,
        merged_data: List[Dict[str, Any]]
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Run RAGAS evaluation on merged data."""
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, context_precision, answer_relevancy
            from ragas.llms import LangchainLLMWrapper
            
            # Configure model
            self._configure_model()
            
            # Create LangChain model
            langchain_llm = self._create_langchain_model()
            ragas_llm = LangchainLLMWrapper(langchain_llm)
            
            # Prepare dataset
            formatted_data = self._format_for_ragas(merged_data)
            dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
            
            # Check if real contexts are available for context precision
            has_real_contexts = any(item.get('contexts', []) for item in merged_data)
            
            # Configure metrics
            metrics_to_use = []
            if self.include_faithfulness:
                faithfulness.llm = ragas_llm
                metrics_to_use.append(faithfulness)
            if self.include_context_precision and has_real_contexts:
                context_precision.llm = ragas_llm
                metrics_to_use.append(context_precision)
            if self.include_answer_relevancy:
                answer_relevancy.llm = ragas_llm
                metrics_to_use.append(answer_relevancy)
            
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
            
            if self.include_context_precision and has_real_contexts and 'context_precision' in result_df.columns:
                cp_scores = result_df['context_precision'].dropna()
                aggregated_metrics['mean_context_precision'] = float(cp_scores.mean())
                aggregated_metrics['std_context_precision'] = float(cp_scores.std())
                aggregated_metrics['min_context_precision'] = float(cp_scores.min())
                aggregated_metrics['max_context_precision'] = float(cp_scores.max())
            
            if self.include_answer_relevancy and 'answer_relevancy' in result_df.columns:
                ar_scores = result_df['answer_relevancy'].dropna()
                aggregated_metrics['mean_answer_relevancy'] = float(ar_scores.mean())
                aggregated_metrics['std_answer_relevancy'] = float(ar_scores.std())
                aggregated_metrics['min_answer_relevancy'] = float(ar_scores.min())
                aggregated_metrics['max_answer_relevancy'] = float(ar_scores.max())
            
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
                
                if self.include_answer_relevancy and 'answer_relevancy' in result_df.columns:
                    row_result['answer_relevancy'] = float(row.get('answer_relevancy', 0))
                
                row_results.append(row_result)
            
            return aggregated_metrics, row_results
            
        except ImportError as e:
            raise ImportError(f"RAGAS dependencies not available: {e}")
        except Exception as e:
            raise RuntimeError(f"RAGAS evaluation error: {e}")
    
    def _configure_model(self):
        """Configure environment variables for model access."""
        if self.model_name.startswith("bedrock/"):
            # AWS Bedrock configuration - no API key needed
            pass  # Bedrock uses AWS credentials from environment or IAM roles
        elif self.api_key and self.api_key.startswith("sk-or-"):
            # OpenRouter configuration
            os.environ["OPENROUTER_API_KEY"] = self.api_key
            os.environ["OPENAI_API_KEY"] = self.api_key
            os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        else:
            # Standard OpenAI configuration
            os.environ["OPENAI_API_KEY"] = self.api_key
    
    def _create_langchain_model(self):
        """Create LangChain model for RAGAS."""
        if self.model_name.startswith("bedrock/"):
            # AWS Bedrock model
            from langchain_aws import ChatBedrock
            
            # Extract model ID from model_name (remove "bedrock/" prefix)
            model_id = self.model_name.replace("bedrock/", "")
            
            return ChatBedrock(
                model_id=model_id,
                temperature=0,
                max_tokens=4096
            )
        elif self.api_key and self.api_key.startswith("sk-or-"):
            # OpenRouter model
            from langchain_openai import ChatOpenAI
            
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
            from langchain_openai import ChatOpenAI
            
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
            # Use actual contexts from the system output
            contexts = item.get('contexts', [])
            # Note: If no contexts are available, context_precision metric will be skipped
            
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
            'citations': ['France_Wikipedia'],
            'contexts': ['France is a country in Western Europe. Its capital and largest city is Paris, which is located in the north-central part of the country.']
        }
    ]
    
    references = [
        {
            'iid': '1',
            'query': 'What is the capital of France?',
            'reference': 'The capital of France is Paris.'
        }
    ]
    
    # Test with AWS Bedrock gpt-oss-120b model
    evaluator = RAGASEvaluator(
        model_name="bedrock/openai.gpt-oss-120b-1:0",
        api_key=None,  # Not needed for Bedrock - uses AWS credentials
        include_faithfulness=True,
        include_context_precision=False,
        include_answer_relevancy=False  
    )
    
    print(f"Evaluator: {evaluator.name}")
    print(f"Description: {evaluator.description}")
    print("Model configured for AWS Bedrock: openai.gpt-oss-120b-1:0")
    
    try:
        result = evaluator.evaluate(system_outputs, references)
        print("\nEvaluation Results:")
        print(f"Sample count: {result.sample_count}")
        print(f"Execution time: {result.total_time_ms:.2f} ms")
        print("\nMetrics:")
        for metric_name, value in result.metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        print("\nRow Results:")
        for row in result.rows:
            print(f"  Query ID: {row['query_id']}")
            print(f"  Query: {row['query']}")
            if 'faithfulness' in row:
                print(f"  Faithfulness: {row['faithfulness']:.4f}")
            if 'context_precision' in row:
                print(f"  Context Precision: {row['context_precision']:.4f}")
            if 'answer_relevancy' in row:
                print(f"  Answer Relevancy: {row['answer_relevancy']:.4f}")
            print()
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print("This may be due to missing RAGAS dependencies or API issues.")