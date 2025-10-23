"""
Citation Recall Evaluator for DeepResearch benchmarking.

Evaluates citation recall by measuring the percentage of gold reference claims
that are covered in the system's answer. This uses gold standard references
to measure how well the system covers expected claims.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
import time
import concurrent.futures
import logging

from openai import OpenAI
from pydantic import BaseModel

from src.evaluators.evaluator_interface import EvaluatorInterface, EvaluationResult

# Load environment variables
from dotenv import load_dotenv
load_dotenv()  # Load from .env file in project root

logger = logging.getLogger(__name__)

class ClaimEntry(BaseModel):
    claim_id: int
    claim: str
    sources: List[str]

class ClaimsModel(BaseModel):
    claims: List[ClaimEntry]

class CitationRecallEvaluator(EvaluatorInterface):
    """
    Evaluates citation recall for deep research reports using gold references.

    Measures the percentage of gold reference claims that are covered in the system output.
    Higher scores indicate better coverage of expected claims from gold standard.
    """

    def __init__(
        self,
        model: str = "openai.gpt-oss-120b-1:0",
        temperature: float = 0.0,
        max_tokens: int = 15000,
        silent_errors: bool = True,
        num_threads: int = 1,
        api_base: str = "https://mmu-proxy-server-llm-proxy.rankun.org",
        api_key: Optional[str] = None
    ):
        """
        Initialize the citation recall evaluator.

        Args:
            model: OpenAI model to use for LLM judgments
            temperature: Temperature for LLM calls
            max_tokens: Max tokens for LLM responses
            silent_errors: Whether to log errors and continue
            num_threads: Number of threads for concurrent evaluation
            api_base: Base URL for OpenAI API
            api_key: OpenAI API key
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.silent_errors = silent_errors
        self.num_threads = num_threads

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key or os.getenv("MMU_OPENAI_API_KEY")
        )
        if not self.client.api_key:
            raise ValueError("API key is not set. Please provide it via 'api_key' or 'MMU_OPENAI_API_KEY' environment variable.")

    @property
    def name(self) -> str:
        return "citation_recall"

    def create_prompt_extractor(self, answer: str) -> str:
        """Create prompt for extracting claims from answer (based on deepresearch_benchmarking repo)."""
        return f"""You are an information extraction expert.

Given a structured report containing claims and their supporting sources (usually in the form of inline hyperlinks or referenced URLs), extract all distinct factual or argumentative claims in the text.
If a claim is supported by one or more sources, return the supporting URLs as sources.
If a claim is not supported by any source, return an empty list of sources.

Return a JSON object like this:
{{
  "claims": [
    {{
      "claim_id": 1,
      "claim": "<claim_1>",
      "sources": ["<url_1>", "<url_2>", ...]
    }},
    {{
      "claim_id": 2,
      "claim": "<claim_2>",
      "sources": []
    }},
    ...
  ]
}}

Where:

- The root is "claims", which contains a list of claim objects.
- Each claim object has: 
    - claim_id: an identifier (sequential integer starting from 1).
    - claim: a concise but complete sentence restating the claim.
    - sources: a list of URLs that explicitly support the claim, or an empty list if no URLs support it. 

(**IMPORTANT**: Only include URLs that are **explicitly present in the report text**, typically as inline hyperlinks or reference-style citations. Do not infer or fabricate URLs. Do not include non-URL citations such as book titles, paper references, or other non-URL sources.)

(**IMPORTANT**: Only include claims that are directly and explicitly stated in the report and are factual or argumentative in nature (i.e., statements that can be verified or refuted). Do not include general summaries, personal opinions, or meta-commentary.)

Process the full report carefully to ensure all claims are included and accurately captured.

Now extract the claims from the report below:

{answer}

Return the JSON object, and nothing else.
"""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract structured data."""
        import re
        # Try to extract JSON using regex
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code block, try to find JSON directly
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                logger.error(f"Failed to extract JSON from LLM response: {response}")
                return {}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {}

    def _evaluate_single(self, system_output: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single system output for citation recall.
        """
        try:
            system_answer = system_output.get('answer', system_output.get('generated_response', ''))
            gold_answer = reference.get('generated_response', reference.get('answer', ''))
            
            if not gold_answer:
                return {
                    "citation_recall": 0.0,
                    "total_gold_claims": 0,
                    "covered_claims": 0,
                    "details": "No gold reference answer found."
                }
            
            if not system_answer:
                return {
                    "citation_recall": 0.0,
                    "total_gold_claims": 0,
                    "covered_claims": 0,
                    "details": "No system answer provided."
                }
            
            # Extract gold claims from reference answer
            logger.info("Extracting gold claims from reference answer...")
            gold_prompt = self.create_prompt_extractor(gold_answer)
            
            try:
                gold_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": gold_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                gold_llm_response = gold_response.choices[0].message.content
                gold_result = self._parse_llm_response(gold_llm_response)
                gold_claims_data = gold_result.get("claims", [])
                
            except Exception as e:
                logger.warning(f"Error extracting gold claims: {e}")
                gold_claims_data = []
            
            if not gold_claims_data:
                return {
                    "citation_recall": 0.0,
                    "total_gold_claims": 0,
                    "covered_claims": 0,
                    "details": "No claims extracted from gold reference answer."
                }
            
            # Extract claims from system answer
            logger.info("Extracting claims from system answer...")
            system_prompt = self.create_prompt_extractor(system_answer)
            
            try:
                system_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": system_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                system_llm_response = system_response.choices[0].message.content
                system_result = self._parse_llm_response(system_llm_response)
                system_claims_data = system_result.get("claims", [])
                
            except Exception as e:
                logger.warning(f"Error extracting claims from system answer: {e}")
                system_claims_data = []
            
            if not system_claims_data:
                return {
                    "citation_recall": 0.0,
                    "total_gold_claims": len(gold_claims_data),
                    "covered_claims": 0,
                    "details": "No claims extracted from system answer."
                }
            
            # Combine all system claims into one text for matching
            system_claims = [claim.get("claim", "") for claim in system_claims_data if claim.get("claim")]
            system_claims_combined = " ".join(system_claims)
            
            # Check each gold claim for coverage in system claims
            coverage_results = []
            covered_count = 0
            
            for gold_claim_obj in gold_claims_data:
                gold_claim = gold_claim_obj.get("claim", "")
                if not gold_claim or not gold_claim.strip():
                    continue
                
                # Use word overlap heuristic to check coverage
                gold_claim_lower = gold_claim.lower().strip()
                system_combined_lower = system_claims_combined.lower()
                
                # Extract alphanumeric sequences (potential key terms)
                gold_words = set(re.findall(r'\b\w{4,}\b', gold_claim_lower))
                system_words = set(re.findall(r'\b\w{4,}\b', system_combined_lower))
                
                # Calculate overlap
                if gold_words:
                    overlap = len(gold_words & system_words) / len(gold_words)
                    # If at least 60% of key words from gold claim appear in system output
                    is_covered = overlap >= 0.6
                else:
                    is_covered = False
                
                if is_covered:
                    covered_count += 1
                
                coverage_results.append({
                    "gold_claim_id": gold_claim_obj.get("claim_id", 0),
                    "gold_claim": gold_claim[:200] + "..." if len(gold_claim) > 200 else gold_claim,
                    "covered": is_covered,
                    "overlap_ratio": round(overlap, 2) if gold_words else 0.0
                })
            
            total_gold_claims = len(gold_claims_data)
            recall_score = covered_count / total_gold_claims if total_gold_claims > 0 else 0.0
            
            return {
                "citation_recall": recall_score,
                "total_gold_claims": total_gold_claims,
                "covered_claims": covered_count,
                "details": coverage_results
            }

        except Exception as e:
            if self.silent_errors:
                logger.error(f"Error during citation recall evaluation: {e}")
                return {
                    "citation_recall": 0.0,
                    "total_gold_claims": 0,
                    "covered_claims": 0,
                    "details": f"Evaluation error: {str(e)}"
                }
            else:
                raise e

    def evaluate(self, system_outputs: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate citation recall for the system outputs using gold references.
        
        For each system output, we:
        1. Find the matching gold reference by query_id
        2. Extract gold claims from the reference
        3. Check how many gold claims are covered in the system output
        4. Calculate recall = covered_claims / total_gold_claims
        """
        start_time = time.time()

        # Validate inputs
        self.validate_inputs(system_outputs, references)

        # Create lookup for references by query_id
        ref_lookup = {ref.get('query_id', ref.get('iid')): ref for ref in references}

        rows = []
        recall_scores = []
        total_gold_claims_list = []
        covered_claims_list = []

        def evaluate_sample(output, reference):
            result = self._evaluate_single(output, reference)
            query_id = output.get('query_id', output.get('iid'))
            rows.append({
                'query_id': query_id,
                'citation_recall': result.get('citation_recall', 0.0),
                'total_gold_claims': result.get('total_gold_claims', 0),
                'covered_claims': result.get('covered_claims', 0),
                'details': result.get('details', {})
            })
            recall_scores.append(result.get('citation_recall', 0.0))
            total_gold_claims_list.append(result.get('total_gold_claims', 0))
            covered_claims_list.append(result.get('covered_claims', 0))

        if self.num_threads > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for output in system_outputs:
                    output_id = output.get('query_id', output.get('iid'))
                    reference = ref_lookup.get(output_id, {})
                    if not reference:
                        logger.warning(f"No gold reference found for query_id: {output_id}")
                        reference = {}
                    futures.append(executor.submit(evaluate_sample, output, reference))

                for future in concurrent.futures.as_completed(futures):
                    future.result()
        else:
            for output in system_outputs:
                output_id = output.get('query_id', output.get('iid'))
                reference = ref_lookup.get(output_id, {})
                if not reference:
                    logger.warning(f"No gold reference found for query_id: {output_id}")
                    reference = {}
                evaluate_sample(output, reference)

        # Calculate metrics
        from statistics import mean
        avg_recall = mean(recall_scores) if recall_scores else 0.0
        total_gold_claims_overall = sum(total_gold_claims_list)
        total_covered_overall = sum(covered_claims_list)

        metrics = {
            'citation_recall': avg_recall,
            'total_gold_claims': total_gold_claims_overall,
            'total_covered_claims': total_covered_overall,
            'count': len(system_outputs)
        }

        total_time_ms = (time.time() - start_time) * 1000

        return EvaluationResult(
            metrics=metrics,
            evaluator_name=self.name,
            sample_count=len(system_outputs),
            rows=rows,
            total_time_ms=total_time_ms
        )