"""
Citation Precision Evaluator for DeepResearch benchmarking.

Evaluates citation precision by checking if factual claims in the answer
are properly supported by their cited sources. Uses LLM to extract claims
and verify support against crawled web content.
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

try:
    from crawl4ai import AsyncWebCrawler
    CRAWL4AI_AVAILABLE = True
except ImportError:
    CRAWL4AI_AVAILABLE = False

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

class CitationSupportValues:
    @staticmethod
    def score(label: str) -> float:
        mapping = {
            "full_support": 1.0,
            "partial_support": 0.5,
            "no_support": 0.0
        }
        return mapping.get(label.lower(), 0.0)

class CitationPrecisionEvaluator(EvaluatorInterface):
    """
    Evaluates citation precision for deep research reports.

    Measures how accurately citations support the claims made in the answer.
    Uses LLM to extract claims with sources and verify against crawled content.
    """

    def __init__(
        self,
        model: str = "openai.gpt-oss-120b-1:0",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        silent_errors: bool = True,
        num_threads: int = 1,
        api_base: str = "https://mmu-proxy-server-llm-proxy.rankun.org",
        api_key: Optional[str] = None,
        crawl_concurrency: int = 5
    ):
        """
        Initialize the citation precision evaluator.

        Args:
            model: OpenAI model to use for LLM judgments
            temperature: Temperature for LLM calls
            max_tokens: Max tokens for LLM responses
            silent_errors: Whether to log errors and continue
            num_threads: Number of threads for concurrent evaluation
            api_base: Base URL for OpenAI API
            api_key: OpenAI API key
            crawl_concurrency: Max concurrent web crawls
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.silent_errors = silent_errors
        self.num_threads = num_threads
        self.crawl_concurrency = crawl_concurrency

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key or os.getenv("MMU_OPENAI_API_KEY")
        )
        if not self.client.api_key:
            raise ValueError("API key is not set. Please provide it via 'api_key' or 'MMU_OPENAI_API_KEY' environment variable.")

        if not CRAWL4AI_AVAILABLE:
            raise ImportError("crawl4ai is required for citation evaluation. Install with: pip install crawl4ai")

    @property
    def name(self) -> str:
        return "citation_precision"

    def create_prompt_extractor(self, answer: str, citations: List[str] = None) -> str:
        """Create prompt for extracting claims with sources from answer."""
        citations_text = ""
        if citations:
            citations_text = "\n\nAvailable Citations:\n" + "\n".join(f"[{i+1}] {url}" for i, url in enumerate(citations, 1))
        
        return f"""You are an information extraction expert.

Given a structured report containing claims and their supporting sources, extract all distinct factual or argumentative claims that are explicitly supported by a specific reference in the text.

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
      "sources": ["<url_1>", ...]
    }},
    ...
  ]
}}

Where:

- The root is "claims", which contains a list of claim objects.
- Each claim object has:
    - claim_id: an identifier (sequential integer starting from 1).
    - claim: a concise but complete sentence restating the claim.
    - sources: list of URLs that support this claim.

Only extract claims that have explicit source citations in the text. If a claim has no sources, do not include it.

Report: {answer}{citations_text}
"""

    def create_prompt_citation_checker(self, claim: str, docs: List[str]) -> str:
        """Create prompt for checking if citation supports the claim."""
        citations_text = "\n\n".join(f"[{i+1}] {doc}" for i, doc in enumerate(docs))
        return f"""In this task, you will evaluate whether each statement is supported by its corresponding citations. Note that the system responses may appear very fluent and well-formed, but contain slight inaccuracies that are not easy to discern at first glance. Pay close attention to the text.

You will be provided with a statement and its corresponding citations. It may be helpful to ask yourself whether it is accurate to say "according to the citation" with a statement following this phrase. Be sure to check all of the information in the statement. You will be given three options:

- Full Support: All of the information in the statement is supported in the citations.
- Partial Support: Some parts of the information are supported in the citations, but other parts are missing from the citations.
- No Support: These citations does not support any part of the statement.

Please provide your response based on the information in the citations. If you are unsure, use your best judgment. Respond as either "full_support", "partial_support", or "no_support" with no additional information.

Statement: {claim}

Citations: {citations_text}
"""

    def crawl_urls(self, urls: List[str]) -> List[str]:
        """Crawl URLs and return their text content."""
        if not urls:
            return []

        import asyncio
        async def crawl_single(url: str) -> str:
            try:
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=url)
                    return result.markdown
            except Exception as e:
                logger.error(f"Failed to crawl {url}: {e}")
                return ""

        async def run_crawls():
            tasks = [crawl_single(url) for url in urls[:self.crawl_concurrency]]  # Limit concurrent crawls
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(run_crawls())
        crawled_texts = []
        for result in results:
            if isinstance(result, Exception):
                crawled_texts.append("")
            else:
                crawled_texts.append(result)

        return crawled_texts

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
        Evaluate a single system output for citation precision.
        """
        try:
            answer = system_output.get('answer', system_output.get('generated_response', ''))

            # Check if answer has URLs (in text or citations array)
            url_pattern = r'https?://\S+|www\.\S+'
            citations = system_output.get('citations', [])
            
            # Extract URLs from both text and citations array
            text_urls = re.findall(url_pattern, answer) if answer else []
            citation_urls = citations if isinstance(citations, list) else []
            all_urls = text_urls + citation_urls
            
            if not all_urls:
                return {
                    "citation_precision": 0.0,
                    "claim_count": 0,
                    "details": "No URLs found in answer or citations."
                }

            # Extract claims with sources using structured output
            prompt = self.create_prompt_extractor(answer, citations)
            
            try:
                # Try using structured output first
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=ClaimsModel,
                    temperature=self.temperature
                )
                result = json.loads(response.choices[0].message.content)
                claims_to_urls = result.get("claims", [])
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
                claims_to_urls = result.get("claims", [])

            if not claims_to_urls:
                return {
                    "citation_precision": 0.0,
                    "claim_count": 0,
                    "details": "No claims with sources extracted."
                }

            scores = {}
            total_score = 0.0
            claim_count = 0

            for claim_data in claims_to_urls:
                claim_text = claim_data["claim"]
                urls = claim_data["sources"]
                claim_id = claim_data["claim_id"]

                if not urls:
                    continue

                # Crawl the URLs
                docs = self.crawl_urls(urls)
                clean_docs = [re.sub(url_pattern, '', d).strip() for d in docs if d.strip()]

                if not clean_docs:
                    scores[f"claim_{claim_id}"] = {
                        "claim": claim_text,
                        "urls": urls,
                        "score": 0.0,
                        "justification": "Failed to crawl URLs"
                    }
                    continue

                # Check citation quality
                try:
                    checker_prompt = self.create_prompt_citation_checker(claim_text, clean_docs)
                    checker_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": checker_prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )

                    label = checker_response.choices[0].message.content.strip().lower()
                    score = CitationSupportValues.score(label)
                    scores[f"claim_{claim_id}"] = {
                        "claim": claim_text,
                        "urls": urls,
                        "score": score,
                        "justification": label
                    }
                    total_score += score
                    claim_count += 1
                except Exception as e:
                    scores[f"claim_{claim_id}"] = {
                        "claim": claim_text,
                        "urls": urls,
                        "score": 0.0,
                        "justification": str(e)
                    }

            final_score = total_score / claim_count if claim_count > 0 else 0.0

            return {
                "citation_precision": final_score,
                "claim_count": claim_count,
                "details": scores
            }

        except Exception as e:
            if self.silent_errors:
                logger.error(f"Error during citation precision evaluation: {e}")
                return {
                    "citation_precision": 0.0,
                    "claim_count": 0,
                    "details": f"Evaluation error: {str(e)}"
                }
            else:
                raise e

    def evaluate(self, system_outputs: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate citation precision for the system outputs.
        """
        start_time = time.time()

        # Validate inputs
        self.validate_inputs(system_outputs, references)

        # Create lookup for references (though not used for citation eval)
        ref_lookup = {ref.get('iid', ref.get('query_id')): ref for ref in references}

        rows = []
        precision_scores = []
        claim_counts = []

        def evaluate_sample(output, reference):
            result = self._evaluate_single(output, reference)
            rows.append({
                'query_id': output.get('iid', output.get('query_id')),
                'citation_precision': result.get('citation_precision', 0.0),
                'claim_count': result.get('claim_count', 0),
                'details': result.get('details', {})
            })
            precision_scores.append(result.get('citation_precision', 0.0))
            claim_counts.append(result.get('claim_count', 0))

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
        avg_precision = mean(precision_scores) if precision_scores else 0.0
        total_claims = sum(claim_counts)

        metrics = {
            'citation_precision': avg_precision,
            'total_claims': total_claims,
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