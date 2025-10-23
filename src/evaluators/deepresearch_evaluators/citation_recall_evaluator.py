"""
Citation Recall Evaluator for DeepResearch benchmarking.

Evaluates citation recall by measuring the percentage of claims in the answer
that have supporting sources (URLs). This measures how well the system
provides evidence for its claims.
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
    Evaluates citation recall for deep research reports.

    Measures the percentage of claims that have supporting sources.
    Higher scores indicate better coverage of evidence for claims.
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

    def create_prompt_extractor(self, answer: str, citations: List[str] = None) -> str:
        """Create prompt for extracting all claims from answer, with or without sources."""
        citations_text = ""
        if citations:
            citations_text = "\n\nAvailable Citations:\n" + "\n".join(f"[{i+1}] {url}" for i, url in enumerate(citations, 1))
        
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
    - sources: list of URLs (empty list if no sources).

Extract ALL claims from the report, whether they have sources or not.

Report: {answer}{citations_text}
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
            answer = system_output.get('answer', system_output.get('generated_response', ''))

            # Check if answer has any URLs (in text or citations array)
            url_pattern = r'https?://\S+|www\.\S+'
            citations = system_output.get('citations', [])
            
            # Extract URLs from both text and citations array
            text_urls = re.findall(url_pattern, answer) if answer else []
            citation_urls = citations if isinstance(citations, list) else []
            all_urls = text_urls + citation_urls
            
            if not all_urls:
                return {
                    "citation_recall": 0.0,
                    "total_claims": 0,
                    "supported_claims": 0,
                    "details": "No URLs found in answer or citations."
                }

            # Extract all claims using structured output
            prompt = self.create_prompt_extractor(answer, citations)
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=ClaimsModel,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                result = json.loads(response.choices[0].message.content)
                claims = result.get("claims", [])
            except Exception as e:
                # Fallback to manual parsing if beta API not available
                logger.warning(f"Structured output not available, using fallback parsing: {e}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                llm_response = response.choices[0].message.content
                result = self._parse_llm_response(llm_response)
                claims = result.get("claims", [])

            if not claims:
                return {
                    "citation_recall": 0.0,
                    "total_claims": 0,
                    "supported_claims": 0,
                    "details": "No claims extracted."
                }

            total_claims = len(claims)
            supported_claims = sum(1 for claim in claims if claim.get("sources"))

            score = supported_claims / total_claims if total_claims > 0 else 0.0

            detailed = {
                f"claim_{claim['claim_id']}": {
                    "claim": claim["claim"],
                    "sources": claim["sources"],
                    "supported": bool(claim["sources"])
                }
                for claim in claims
            }

            return {
                "citation_recall": score,
                "total_claims": total_claims,
                "supported_claims": supported_claims,
                "details": detailed
            }

        except Exception as e:
            if self.silent_errors:
                logger.error(f"Error during citation recall evaluation: {e}")
                return {
                    "citation_recall": 0.0,
                    "total_claims": 0,
                    "supported_claims": 0,
                    "details": f"Evaluation error: {str(e)}"
                }
            else:
                raise e

    def evaluate(self, system_outputs: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate citation recall for the system outputs.
        """
        start_time = time.time()

        # Validate inputs
        self.validate_inputs(system_outputs, references)

        # Create lookup for references (though not used for citation recall eval)
        ref_lookup = {ref.get('iid', ref.get('query_id')): ref for ref in references}

        rows = []
        recall_scores = []
        total_claims_list = []
        supported_claims_list = []

        def evaluate_sample(output, reference):
            result = self._evaluate_single(output, reference)
            rows.append({
                'query_id': output.get('iid', output.get('query_id')),
                'citation_recall': result.get('citation_recall', 0.0),
                'total_claims': result.get('total_claims', 0),
                'supported_claims': result.get('supported_claims', 0),
                'details': result.get('details', {})
            })
            recall_scores.append(result.get('citation_recall', 0.0))
            total_claims_list.append(result.get('total_claims', 0))
            supported_claims_list.append(result.get('supported_claims', 0))

        if self.num_threads > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for output in system_outputs:
                    output_id = output.get('iid', output.get('query_id'))
                    reference = ref_lookup.get(output_id, {})
                    futures.append(executor.submit(evaluate_sample, output, reference))

                for future in concurrent.futures.as_completed(futures):
                    future.result()
        else:
            for output in system_outputs:
                output_id = output.get('iid', output.get('query_id'))
                reference = ref_lookup.get(output_id, {})
                evaluate_sample(output, reference)

        # Calculate metrics
        from statistics import mean
        avg_recall = mean(recall_scores) if recall_scores else 0.0
        total_claims_overall = sum(total_claims_list)
        total_supported_overall = sum(supported_claims_list)

        metrics = {
            'citation_recall': avg_recall,
            'total_claims': total_claims_overall,
            'total_supported_claims': total_supported_overall,
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