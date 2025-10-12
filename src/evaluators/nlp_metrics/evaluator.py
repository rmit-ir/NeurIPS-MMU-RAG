"""
NLP Metrics Evaluator for traditional text similarity evaluation.

This evaluator computes surface-level text similarity metrics:
- ROUGE-L: Longest common subsequence overlap with reference
- BLEU: N-gram precision similarity with reference
- BERTScore: Semantic similarity using BERT embeddings
"""

import time
from typing import List, Dict, Any
import pandas as pd

from src.evaluators.evaluator_interface import EvaluatorInterface, EvaluationResult


class NLPMetricsEvaluator(EvaluatorInterface):
    """
    Evaluator for traditional NLP metrics (ROUGE-L, BLEU, and BERTScore).
    
    Measures surface-level and semantic text similarity between generated responses and references.
    """
    
    def __init__(
        self,
        include_rouge_l: bool = True,
        include_bleu: bool = True,
        include_bertscore: bool = True,
        use_stemmer: bool = True,
        bertscore_model: str = "microsoft/DialoGPT-medium"
    ):
        """
        Initialize NLP metrics evaluator.
        
        Args:
            include_rouge_l: Whether to compute ROUGE-L scores
            include_bleu: Whether to compute BLEU scores
            include_bertscore: Whether to compute BERTScore
            use_stemmer: Whether to use stemming for ROUGE-L calculation
            bertscore_model: BERT model to use for BERTScore (default: RoBERTa-large)
        """
        self.include_rouge_l = include_rouge_l
        self.include_bleu = include_bleu
        self.include_bertscore = include_bertscore
        self.use_stemmer = use_stemmer
        self.bertscore_model = bertscore_model
        
        if not (include_rouge_l or include_bleu or include_bertscore):
            raise ValueError("At least one metric must be enabled")
    
    @property
    def name(self) -> str:
        """Return evaluator name."""
        return "NLPMetricsEvaluator"
    
    @property
    def description(self) -> str:
        """Return evaluator description."""
        metrics = []
        if self.include_rouge_l:
            metrics.append("ROUGE-L")
        if self.include_bleu:
            metrics.append("BLEU")
        if self.include_bertscore:
            metrics.append("BERTScore")
        return f"Traditional NLP metrics: {', '.join(metrics)}"
    
    def evaluate(
        self,
        system_outputs: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """
        Evaluate system outputs using NLP metrics.
        
        Args:
            system_outputs: List of system outputs with keys: query_id, generated_response
            references: List of references with keys: iid/query_id, reference
            
        Returns:
            EvaluationResult with NLP metrics
        """
        start_time = time.time()
        
        # Validate inputs
        self.validate_inputs(system_outputs, references)
        
        # Merge data
        merged_data = self._merge_data(system_outputs, references)
        
        if not merged_data:
            raise ValueError("No matching data found between outputs and references")
        
        # Compute metrics
        try:
            metrics, row_results = self._compute_nlp_metrics(merged_data)
            
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
            raise RuntimeError(f"NLP metrics evaluation failed: {str(e)}")
    
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
                    'generated_response': output.get('generated_response', ''),
                    'reference': reference.get('generated_response') or reference.get('reference', '')
                })
        
        return merged_data
    
    def _compute_nlp_metrics(
        self,
        merged_data: List[Dict[str, Any]]
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Compute ROUGE-L, BLEU, and BERTScore."""
        try:
            if self.include_rouge_l:
                from rouge_score import rouge_scorer
            
            if self.include_bleu:
                import nltk
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                # Ensure punkt tokenizer is available
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
            
            if self.include_bertscore:
                from bert_score import score
            
            rouge_l_scores = []
            bleu_scores = []
            bertscore_f1_scores = []
            row_results = []
            
            # Initialize ROUGE scorer if needed
            if self.include_rouge_l:
                rouge_scorer_obj = rouge_scorer.RougeScorer(
                    ['rougeL'], use_stemmer=self.use_stemmer
                )
            
            # Initialize BLEU smoothing function if needed
            if self.include_bleu:
                smoothie = SmoothingFunction().method4
            
            for item in merged_data:
                generated = item['generated_response']
                reference = item['reference']
                
                row_result = {
                    'query_id': item['query_id']
                }
                
                # Calculate ROUGE-L
                if self.include_rouge_l:
                    rouge_scores = rouge_scorer_obj.score(reference, generated)
                    rouge_l_f1 = rouge_scores['rougeL'].fmeasure
                    rouge_l_scores.append(rouge_l_f1)
                    row_result['rouge_l'] = rouge_l_f1
                
                # Calculate BLEU
                if self.include_bleu:
                    # Tokenize for BLEU calculation
                    reference_tokens = nltk.word_tokenize(reference.lower())
                    generated_tokens = nltk.word_tokenize(generated.lower())
                    
                    # Calculate BLEU score
                    bleu_score = sentence_bleu(
                        [reference_tokens], generated_tokens, 
                        smoothing_function=smoothie
                    )
                    bleu_scores.append(bleu_score)
                    row_result['bleu'] = bleu_score
                
                # Calculate BERTScore
                if self.include_bertscore:
                    # BERTScore expects lists of strings
                    P, R, F1 = score([generated], [reference], model_type=self.bertscore_model, verbose=False)
                    bertscore_f1 = float(F1[0])
                    bertscore_f1_scores.append(bertscore_f1)
                    row_result['bertscore_f1'] = bertscore_f1
                
                row_results.append(row_result)
            
            # Aggregate metrics
            aggregated_metrics = {}
            
            if self.include_rouge_l and rouge_l_scores:
                rouge_series = pd.Series(rouge_l_scores)
                aggregated_metrics['mean_rouge_l'] = float(rouge_series.mean())
                aggregated_metrics['std_rouge_l'] = float(rouge_series.std())
                aggregated_metrics['min_rouge_l'] = float(rouge_series.min())
                aggregated_metrics['max_rouge_l'] = float(rouge_series.max())
            
            if self.include_bleu and bleu_scores:
                bleu_series = pd.Series(bleu_scores)
                aggregated_metrics['mean_bleu'] = float(bleu_series.mean())
                aggregated_metrics['std_bleu'] = float(bleu_series.std())
                aggregated_metrics['min_bleu'] = float(bleu_series.min())
                aggregated_metrics['max_bleu'] = float(bleu_series.max())
            
            if self.include_bertscore and bertscore_f1_scores:
                bertscore_series = pd.Series(bertscore_f1_scores)
                aggregated_metrics['mean_bertscore_f1'] = float(bertscore_series.mean())
                aggregated_metrics['std_bertscore_f1'] = float(bertscore_series.std())
                aggregated_metrics['min_bertscore_f1'] = float(bertscore_series.min())
                aggregated_metrics['max_bertscore_f1'] = float(bertscore_series.max())
            
            return aggregated_metrics, row_results
            
        except ImportError as e:
            raise ImportError(f"NLP metrics dependencies not available: {e}")
        except Exception as e:
            raise RuntimeError(f"NLP metrics computation error: {e}")


if __name__ == "__main__":
    # Test the evaluator
    print("Testing NLP Metrics Evaluator...")
    
    # Sample data
    system_outputs = [
        {
            'query_id': '1',
            'generated_response': 'Paris is the capital city of France.'
        }
    ]
    
    references = [
        {
            'iid': '1',
            'reference': 'The capital of France is Paris.'
        }
    ]
    
    evaluator = NLPMetricsEvaluator()
    
    print(f"Evaluator: {evaluator.name}")
    print(f"Description: {evaluator.description}")
    
    try:
        result = evaluator.evaluate(system_outputs, references)
        print(f"Metrics: {result.metrics}")
        print(f"Sample count: {result.sample_count}")
    except Exception as e:
        print(f"Test error (expected without dependencies): {e}")
    
    print("Test complete.")