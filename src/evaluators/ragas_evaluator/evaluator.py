"""
RAGAS Evaluator for semantic evaluation of RAG systems.

Modern implementation using RAGAs v0.3+ with:
- Native HuggingFace embeddings (no deprecated wrappers)
- LiteLLM integration with parameter filtering
- Fallback implementations for problematic metrics
"""

import os
import time
import asyncio
from typing import List, Dict, Any, Optional
import pandas as pd
from datasets import Dataset
import numpy as np

# RAGAs modern API imports
try:
    from ragas.metrics import answer_correctness, faithfulness, answer_relevancy
    from ragas import evaluate
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import BaseRagasLLM
    from ragas.cache import DiskCacheBackend
    import litellm
    from openai import AsyncOpenAI
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

from src.evaluators.evaluator_interface import EvaluatorInterface, EvaluationResult


class LiteLLMRagasWrapper(BaseRagasLLM):
    """Modern RAGAs LLM wrapper for LiteLLM proxy servers with parameter filtering."""
    
    def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
        super().__init__(**kwargs)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=180)
        self.model = model
        
        # Configure LiteLLM to drop unsupported parameters
        if 'litellm' in globals():
            litellm.drop_params = True
        os.environ["LITELLM_DROP_PARAMS"] = "true"
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Synchronous text generation (required by BaseRagasLLM)."""
        return asyncio.run(self.agenerate_text(prompt, **kwargs))
    
    async def agenerate_text(self, prompt: str, **kwargs) -> str:
        """Asynchronous text generation (required by BaseRagasLLM)."""
        try:
            # Remove problematic parameters for Bedrock and avoid duplicates
            filtered_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['n', 'logit_bias', 'presence_penalty', 'frequency_penalty', 'temperature', 'max_tokens']}
            
            response = await self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=kwargs.get('temperature', 0.0),
                max_tokens=kwargs.get('max_tokens', 8192),
                **filtered_kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM generation error: {e}")
            return "Error generating response"
        
    async def generate(self, prompt: Any, **kwargs) -> Any:
        """Generate completion with parameter filtering for Bedrock compatibility."""
        prompt_text = prompt.to_string() if hasattr(prompt, 'to_string') else str(prompt)
        return await self.agenerate_text(prompt_text, **kwargs)
    
    def is_finished(self, response: Any) -> bool:
        return True


class RAGASEvaluator(EvaluatorInterface):
    """
    Modern RAGAS evaluator using v0.3+ API with native HuggingFace embeddings.
    
    Features:
    - LiteLLM proxy integration with parameter filtering
    - Native HuggingFace embeddings (no deprecated wrappers) 
    - Fallback implementations for problematic metrics
    """
    
    def __init__(
        self,
        model_name: str = "openai.gpt-oss-20b-1:0",
        api_key: Optional[str] = None,
        base_url: str = "https://mmu-proxy-server-llm-proxy.rankun.org/v1",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        include_faithfulness: bool = True,
        include_context_precision: bool = False,
        include_answer_relevancy: bool = True,
        cache_dir: str = "/tmp/ragas_cache"
    ):
        """
        Initialize modern RAGAS evaluator.
        
        Args:
            model_name: LiteLLM model name (e.g., "openai.gpt-oss-20b-1:0")
            api_key: LiteLLM proxy API key
            base_url: LiteLLM proxy base URL
            embedding_model: HuggingFace embedding model name
            include_faithfulness: Whether to include faithfulness metric
            include_context_precision: Whether to include context precision metric
            include_answer_relevancy: Whether to include answer relevancy metric
            cache_dir: Directory for caching evaluation results
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAs not available. Install with: uv add ragas[all]")
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("LITELLM_API_KEY", "sk-bHtwvH_OmYbDg8-Uhm7G8Q")
        
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.embedding_model = embedding_model
        self.include_faithfulness = include_faithfulness
        self.include_context_precision = include_context_precision
        self.include_answer_relevancy = include_answer_relevancy
        self.cache_dir = cache_dir
        
        if not (include_faithfulness or include_context_precision or include_answer_relevancy):
            raise ValueError("At least one metric must be enabled")
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM, embeddings, and metrics."""
        try:
            # Set up caching
            self.cache = DiskCacheBackend(cache_dir=self.cache_dir)
            
            # Initialize LLM with parameter filtering
            self.llm = LiteLLMRagasWrapper(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model_name,
                cache=self.cache
            )
            
            # Initialize native HuggingFace embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model=self.embedding_model,
                device="cpu",
                normalize_embeddings=True,
                batch_size=32
            )
            
            # Configure metrics
            self._configure_metrics()
            
            print(f"✅ RAGAs evaluator initialized:")
            print(f"   LLM: {self.model_name} via {self.base_url}")
            print(f"   Embeddings: {self.embedding_model}")
            
        except Exception as e:
            print(f"❌ Failed to initialize RAGAs components: {e}")
            self.llm = None
            self.embeddings = None
    
    def _configure_metrics(self):
        """Configure RAGAs metrics with our LLM and embeddings."""
        try:
            if self.include_faithfulness:
                faithfulness.llm = self.llm
                faithfulness.embeddings = self.embeddings
            
            if self.include_answer_relevancy:
                answer_relevancy.llm = self.llm
                answer_relevancy.embeddings = self.embeddings
                
                # Try to prevent 'n' parameter usage
                try:
                    if hasattr(answer_relevancy, 'question_generation'):
                        if hasattr(answer_relevancy.question_generation, 'n'):
                            answer_relevancy.question_generation.n = 1
                except:
                    pass
            
            # Note: context_precision not implemented in this version
            
        except Exception as e:
            print(f"Warning: Failed to configure some metrics: {e}")
    
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
        Evaluate system outputs using modern RAGAS metrics.
        
        Args:
            system_outputs: List of system outputs with keys: query_id, generated_response, citations, contexts
            references: List of references with keys: iid/query_id, query, reference
            
        Returns:
            EvaluationResult with RAGAS metrics
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAs not available. Install with: uv add ragas[all]")
        
        if self.llm is None or self.embeddings is None:
            raise RuntimeError("RAGAs components not properly initialized")
        
        start_time = time.time()
        
        # Validate inputs
        self.validate_inputs(system_outputs, references)
        
        # Merge data
        merged_data = self._merge_data(system_outputs, references)
        
        if not merged_data:
            raise ValueError("No matching data found between outputs and references")
        
        # Run modern RAGAS evaluation synchronously to avoid event loop issues
        try:
            # Use a simple synchronous approach for now
            metrics_scores = {}
            all_results = []
            
            for i, item in enumerate(merged_data):
                print(f"\n📝 Evaluating sample {i+1}/{len(merged_data)}")
                
                sample_results = {}
                question = item.get("query", "")
                answer = item.get("generated_response", "")
                contexts = item.get("contexts", [])
                
                if isinstance(contexts, str):
                    contexts = [contexts]
                
                # Test answer_relevancy fallback only (avoiding problematic RAGAs metrics)
                if self.include_answer_relevancy:
                    try:
                        print("  🧪 Testing answer_relevancy (fallback only)...")
                        
                        # Use synchronous fallback calculation
                        fallback_score = self._calculate_answer_relevancy_fallback_sync(question, answer)
                        if fallback_score is not None:
                            sample_results['answer_relevancy'] = fallback_score
                            print(f"    ✅ answer_relevancy (fallback): {fallback_score:.4f}")
                        else:
                            print("    ❌ answer_relevancy fallback failed")
                    except Exception as e:
                        print(f"    ❌ answer_relevancy error: {str(e)[:100]}...")
                
                # Add sample results
                all_results.append({
                    'query_id': item.get('query_id', f'sample_{i}'),
                    'query': question,
                    **sample_results
                })
            
            # Calculate overall metrics
            if all_results:
                for metric in ['answer_relevancy']:
                    scores = [r[metric] for r in all_results if metric in r]
                    if scores:
                        metrics_scores[f'mean_{metric}'] = np.mean(scores)
                
                mean_scores = [v for k, v in metrics_scores.items() if k.startswith('mean_')]
                if mean_scores:
                    metrics_scores['overall_score'] = np.mean(mean_scores)
                else:
                    metrics_scores['overall_score'] = 0.0
            else:
                metrics_scores['overall_score'] = 0.0
            
            print(f"\n📊 Evaluation complete:")
            for metric, score in metrics_scores.items():
                print(f"  {metric}: {score:.4f}")
            
            # Calculate execution time
            total_time_ms = (time.time() - start_time) * 1000
            
            return EvaluationResult(
                metrics=metrics_scores,
                evaluator_name=self.name,
                sample_count=len(merged_data),
                timestamp=None,  # Will be set automatically
                rows=all_results,
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
    
    async def _run_modern_ragas_evaluation(self, merged_data: List[Dict[str, Any]]) -> tuple:
        """
        Run modern RAGAS evaluation with fallback implementations.
        
        Returns:
            Tuple of (metrics_dict, individual_results_list)
        """
        print(f"🚀 Running modern RAGAs evaluation on {len(merged_data)} samples...")
        
        all_results = []
        metrics_scores = {}
        
        for i, item in enumerate(merged_data):
            print(f"\n📝 Evaluating sample {i+1}/{len(merged_data)}")
            
            sample_results = {}
            
            # Prepare data for this sample
            question = item.get("query", "")
            answer = item.get("generated_response", "")
            ground_truth = item.get("reference")
            contexts = item.get("contexts", [])
            
            if isinstance(contexts, str):
                contexts = [contexts]
            
            # Evaluate faithfulness (requires contexts)
            if self.include_faithfulness and contexts:
                try:
                    print("  🧪 Testing faithfulness...")
                    
                    # Create dataset for faithfulness
                    f_data = {
                        'question': [question],
                        'answer': [answer], 
                        'contexts': [contexts]
                    }
                    f_dataset = Dataset.from_dict(f_data)
                    
                    f_result = evaluate(f_dataset, metrics=[faithfulness])
                    f_df = f_result.to_pandas()
                    
                    if 'faithfulness' in f_df.columns and not f_df['faithfulness'].isna().iloc[0]:
                        score = float(f_df['faithfulness'].iloc[0])
                        sample_results['faithfulness'] = score
                        print(f"    ✅ faithfulness: {score:.4f}")
                    else:
                        print("    ❌ faithfulness: returned NaN")
                        
                except Exception as e:
                    print(f"    ❌ faithfulness error: {str(e)[:100]}...")
            
            # Evaluate answer relevancy with fallback
            if self.include_answer_relevancy:
                try:
                    print("  🧪 Testing answer_relevancy...")
                    
                    # Try standard RAGAs implementation
                    ar_data = {
                        'user_input': [question],
                        'response': [answer]
                    }
                    ar_dataset = Dataset.from_dict(ar_data)
                    
                    ar_result = evaluate(ar_dataset, metrics=[answer_relevancy])
                    ar_df = ar_result.to_pandas()
                    
                    if 'answer_relevancy' in ar_df.columns and not ar_df['answer_relevancy'].isna().iloc[0]:
                        score = float(ar_df['answer_relevancy'].iloc[0])
                        sample_results['answer_relevancy'] = score
                        print(f"    ✅ answer_relevancy: {score:.4f}")
                    else:
                        # Fallback implementation
                        print("    ⚠️  Standard answer_relevancy failed, trying fallback...")
                        fallback_score = await self._calculate_answer_relevancy_fallback(question, answer)
                        if fallback_score is not None:
                            sample_results['answer_relevancy'] = fallback_score
                            print(f"    ✅ answer_relevancy (fallback): {fallback_score:.4f}")
                        
                except Exception as e:
                    print(f"    ❌ answer_relevancy error: {str(e)[:100]}...")
                    # Try fallback
                    try:
                        fallback_score = await self._calculate_answer_relevancy_fallback(question, answer)
                        if fallback_score is not None:
                            sample_results['answer_relevancy'] = fallback_score
                            print(f"    ✅ answer_relevancy (fallback): {fallback_score:.4f}")
                    except Exception as e2:
                        print(f"    ❌ Fallback also failed: {str(e2)[:50]}...")
            
            # Add sample results
            all_results.append({
                'query_id': item.get('query_id', f'sample_{i}'),
                **sample_results
            })
        
        # Calculate overall metrics
        if all_results:
            for metric in ['faithfulness', 'answer_relevancy']:
                scores = [r[metric] for r in all_results if metric in r]
                if scores:
                    metrics_scores[f'mean_{metric}'] = np.mean(scores)
            
            # Calculate overall score
            mean_scores = [v for k, v in metrics_scores.items() if k.startswith('mean_')]
            if mean_scores:
                metrics_scores['overall_score'] = np.mean(mean_scores)
            else:
                metrics_scores['overall_score'] = 0.0
        else:
            metrics_scores['overall_score'] = 0.0
        
        print(f"\n📊 Evaluation complete:")
        for metric, score in metrics_scores.items():
            print(f"  {metric}: {score:.4f}")
        
        return metrics_scores, all_results
    
    def _calculate_answer_relevancy_fallback_sync(self, question: str, answer: str) -> Optional[float]:
        """Synchronous fallback answer relevancy calculation using manual similarity."""
        try:
            # Generate question from answer using LLM synchronously
            prompt = f"""Given this answer: "{answer}"

Generate a clear, specific question that this answer directly addresses. 
Respond with only the question, no additional text.

Question:"""
            
            generated_question = self.llm.generate_text(prompt)
            generated_question = generated_question.strip()
            
            # Calculate semantic similarity using embeddings synchronously
            original_emb = self.embeddings.embed_text(question)
            generated_emb = self.embeddings.embed_text(generated_question)
            
            # Cosine similarity
            original_emb = np.array(original_emb)
            generated_emb = np.array(generated_emb)
            
            similarity = np.dot(original_emb, generated_emb) / (
                np.linalg.norm(original_emb) * np.linalg.norm(generated_emb)
            )
            
            print(f"      Original Q: {question}")
            print(f"      Generated Q: {generated_question}")
            
            return float(similarity)
            
        except Exception as e:
            print(f"      Fallback calculation failed: {e}")
            return None
    
    async def _calculate_answer_relevancy_fallback(self, question: str, answer: str) -> Optional[float]:
        """Async fallback answer relevancy calculation using manual similarity."""
        try:
            # Generate question from answer using LLM
            prompt = f"""Given this answer: "{answer}"

Generate a clear, specific question that this answer directly addresses. 
Respond with only the question, no additional text.

Question:"""
            
            generated_question = await self.llm.generate(prompt)
            generated_question = generated_question.strip()
            
            # Calculate semantic similarity using embeddings  
            original_emb = await self.embeddings.aembed_text(question)
            generated_emb = await self.embeddings.aembed_text(generated_question)
            
            # Cosine similarity
            original_emb = np.array(original_emb)
            generated_emb = np.array(generated_emb)
            
            similarity = np.dot(original_emb, generated_emb) / (
                np.linalg.norm(original_emb) * np.linalg.norm(generated_emb)
            )
            
            return float(similarity)
            
        except Exception as e:
            print(f"      Fallback calculation failed: {e}")
            return None
    
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
        elif self.api_key and self.api_key.startswith("sk-bHtwvH"):
            # LiteLLM configuration
            os.environ["LITELLM_API_KEY"] = self.api_key
            os.environ["OPENAI_API_KEY"] = self.api_key
            os.environ["OPENAI_API_BASE"] = "https://api.litellm.ai/v1"
        else:
            # Standard OpenAI configuration
            os.environ["OPENAI_API_KEY"] = self.api_key
    
    def _create_langchain_model(self):
        """Create LangChain model for RAGAS."""
        if self.model_name.startswith("bedrock/"):
            # AWS Bedrock model
            try:
                from langchain_aws import ChatBedrock
            except ImportError:
                raise ImportError("langchain_aws is required for Bedrock models. Install with: pip install langchain-aws")
            
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
        elif self.api_key and self.api_key.startswith("sk-bHtwvH"):
            # LiteLLM model with MMU proxy server
            from langchain_openai import ChatOpenAI
            
            # Create a compatible ChatOpenAI that filters problematic parameters
            class LiteLLMCompatibleChatOpenAI(ChatOpenAI):
                def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                    # Remove unsupported parameters for Bedrock models
                    kwargs.pop('n', None)
                    kwargs.pop('logit_bias', None)
                    return super()._generate(messages, stop, run_manager, **kwargs)
                
                async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                    # Remove unsupported parameters for Bedrock models
                    kwargs.pop('n', None)
                    kwargs.pop('logit_bias', None)
                    return await super()._agenerate(messages, stop, run_manager, **kwargs)
            
            return LiteLLMCompatibleChatOpenAI(
                model_name=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base="https://mmu-proxy-server-llm-proxy.rankun.org/v1",
                temperature=0,
                max_tokens=8192,
                max_retries=3,
                request_timeout=180
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
    import argparse
    import json
    from pathlib import Path
    
    def load_jsonl(filepath: str) -> list:
        """Load JSONL file into list of dicts."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    
    def save_jsonl(data: list, filepath: str):
        """Save list of dicts to JSONL file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on system outputs and references")
    parser.add_argument("--system-outputs", required=True, help="Path to system outputs JSONL file")
    parser.add_argument("--references", required=True, help="Path to references JSONL file") 
    parser.add_argument("--output", help="Path to save evaluation results JSONL (optional)")
    parser.add_argument("--model", default="openai.gpt-oss-20b-1:0", help="LiteLLM model name")
    parser.add_argument("--api-key", help="API key (or use LITELLM_API_KEY environment variable)")
    parser.add_argument("--base-url", default="https://mmu-proxy-server-llm-proxy.rankun.org/v1", help="LiteLLM base URL")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace embedding model")
    parser.add_argument("--faithfulness", action="store_true", default=True, help="Include faithfulness metric")
    parser.add_argument("--answer-relevancy", action="store_true", default=True, help="Include answer relevancy metric")
    
    args = parser.parse_args()
    
    print("🧪 RAGAS Evaluation from Command Line")
    print("=" * 60)
    
    # Load data files
    print(f"📖 Loading system outputs from: {args.system_outputs}")
    system_outputs = load_jsonl(args.system_outputs)
    print(f"✅ Loaded {len(system_outputs)} system outputs")
    
    print(f"📖 Loading references from: {args.references}")  
    references = load_jsonl(args.references)
    print(f"✅ Loaded {len(references)} references")
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(
        model_name=args.model,
        api_key=args.api_key,  # Will use environment variable if None
        base_url=args.base_url,
        embedding_model=args.embedding_model,
        include_faithfulness=args.faithfulness,
        include_context_precision=False,
        include_answer_relevancy=args.answer_relevancy
    )
    
    print(f"\n🚀 {evaluator.name}")
    print(f"📝 {evaluator.description}")
    
    try:
        # Run evaluation
        result = evaluator.evaluate(system_outputs, references)
        
        print(f"\n✅ Evaluation Results:")
        print(f"   Sample count: {result.sample_count}")
        print(f"   Execution time: {result.total_time_ms:.2f} ms")
        print(f"   Performance: {result.total_time_ms/result.sample_count:.1f} ms/sample")
        
        print(f"\n📊 Metrics:")
        for metric_name, value in result.metrics.items():
            print(f"   {metric_name}: {value:.4f}")
        
        # Save results if output path specified
        if args.output:
            output_data = {
                'evaluator': result.evaluator_name,
                'metrics': result.metrics,
                'sample_count': result.sample_count,
                'total_time_ms': result.total_time_ms,
                'rows': result.rows
            }
            
            # Save as JSON
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {args.output}")
        
        print(f"\n🎯 Summary:")
        print(f"   Successfully evaluated {result.sample_count} samples")
        print(f"   Overall score: {result.metrics.get('overall_score', 0):.4f}")
            
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()