"""
Key Point Recall Evaluator for DeepResearch benchmarking.

Evaluates how well reports address ground-truth key points by checking
whether each key point is Supported, Omitted, or Contradicted in the answer.
"""

import json
import os
from typing import List, Dict, Any, Optional, Literal
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

class KeyPointRecall(BaseModel):
    label: Literal["Supported", "Omitted", "Contradicted"]
    justification: str

class KeyPointRecallEvaluator(EvaluatorInterface):
    """
    Evaluates key point recall for deep research reports.

    Measures how well reports address ground-truth key points extracted
    from reference documents. Each key point is evaluated as Supported,
    Omitted, or Contradicted.
    """

    def __init__(
        self,
        model: str = "openai.gpt-oss-120b-1:0",
        temperature: float = 0.0,
        max_tokens: int = 8000,
        silent_errors: bool = True,
        num_threads: int = 1,
        api_base: str = "https://mmu-proxy-server-llm-proxy.rankun.org",
        api_key: Optional[str] = None
    ):
        """
        Initialize the key point recall evaluator.

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
        return "key_point_recall"

    def create_prompt(self, key_point: str, answer: str) -> str:
        """Create evaluation prompt for a single key point."""
        return f"""You are given a **single key point** and a **report**.

Your job is to determine whether the report:
- **Supports** the key point (it affirms, explains, or reinforces the point),
- **Omits** the key point (it does not mention or cover this point at all), or
- **Contradicts** the key point (it says something that disagrees with or negates the point).

Carefully read the key point and the report.

Return your answer as a **JSON object** with two fields:
- "label": One of "Supported", "Omitted", or "Contradicted".
- "justification": Brief explanation on why you assigned this label.

Respond strictly in JSON format:
{{"label": label, "justification": justification}}
Do **not** add any extra commentary or text outside the JSON.

---

Key Point: {key_point}
Report: {answer}
"""

    def create_key_points_extraction_prompt(self, question: str, text: str) -> str:
        """Create prompt to extract key points from a document based on query."""
        return f"""Based on the text provided, identify key points in the text that directly help in responding to the query. The key points are not simply some key content of the text, but rather the key points that are important for **answering the query**.

IMPORTANT: Ensure each point is helpful in responding to the query. Keep the point using the original language and do not add explanations.

Respond strictly in JSON format:
{{
    "points": [
        {{
            "point_number": point_number,
            "point_content": point_content
        }},
        ...
    ]
}}

Remember:
- These key points must be helpful in responding to the query.
- Keep points concise but complete.

[Query]: {question}
[Text]: {text}
"""

    def create_aggregation_prompt(self, original_points: List[str]) -> str:
        """Create prompt to aggregate key points from multiple documents."""
        # Format points as numbered list
        points_text = "\n".join([f"{i + 1}. {point}" for i, point in enumerate(original_points)])

        return f"""You are given a list of key points extracted from multiple documents. Your task is to aggregate these points according to the following instructions:

1. Identify and deduplicate any duplicated or redundant points. Merge them into a single, representative point.
2. Identify contradictory points. Merge them into a single point that presents both sides, e.g., "Sources claim that X, while other sources claim that Y".

IMPORTANT RULES:
- Every aggregated point must preserve **all original information** from the included points.
- Do not invent or add any new information. Only use what is already present.
- Do not provide any explanations or summaries beyond the aggregation itself.
- Each aggregated point should **capture a single atomic idea**. Avoid combining unrelated aspects into one point.
- Keep the aggregated point **concise but complete**: include all essential details needed to fully represent the merged idea, but do not make it overly detailed or verbose.

Respond strictly in JSON format (no markdown code blocks):
{{
    "points": [
        {{
            "point_number": 1,
            "point_content": "aggregated point 1"
        }},
        {{
            "point_number": 2,
            "point_content": "aggregated point 2"
        }}
    ]
}}

[Original Points]
{points_text}
"""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract label and justification."""
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
                return {"label": "Omitted", "justification": "Failed to parse response"}

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {"label": "Omitted", "justification": f"JSON parsing error: {str(e)}"}

    def _evaluate_single_key_point(self, key_point: Dict[str, Any], answer: str) -> Dict[str, Any]:
        """Evaluate a single key point against the answer."""
        try:
            point_content = key_point.get("point_content", "")
            point_number = key_point.get("point_number", 0)

            prompt = self.create_prompt(point_content, answer)
            
            try:
                # Try using structured output first
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=KeyPointRecall,
                    temperature=self.temperature
                )
                result = json.loads(response.choices[0].message.content)
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

            return {
                point_number: (result.get('label', 'Omitted'), result.get('justification', ''))
            }

        except Exception as e:
            if self.silent_errors:
                logger.error(f"Error evaluating key point {key_point.get('point_number', 0)}: {e}")
                return {key_point.get('point_number', 0): ('Omitted', f"Evaluation error: {str(e)}")}
            else:
                raise e

    def _evaluate_answer(self, answer: str, key_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate all key points for a single answer."""
        results = {}

        # Evaluate each key point
        for key_point in key_points:
            key_point_result = self._evaluate_single_key_point(key_point, answer)
            results.update(key_point_result)

        return results

    def _extract_key_points_from_contexts(self, query: str, contexts: List[str]) -> List[Dict[str, Any]]:
        """Extract and aggregate key points from multiple context documents."""
        try:
            all_points = []
            
            # Extract key points from each context document
            for ctx_idx, context in enumerate(contexts):
                if not context or not context.strip():
                    continue
                    
                try:
                    prompt = self.create_key_points_extraction_prompt(query, context)
                    
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0,
                        max_tokens=self.max_tokens
                    )
                    llm_response = response.choices[0].message.content
                    
                    # Parse JSON
                    import re
                    json_match = re.search(r'\{.*?"points"\s*:\s*(\[.*?\])\s*\}', llm_response, re.DOTALL)
                    if json_match:
                        points_array = json.loads(json_match.group(1))
                        for point in points_array:
                            if isinstance(point, dict) and 'point_content' in point:
                                all_points.append(point['point_content'])
                    else:
                        # Fallback: try to extract sentences
                        sentences = [s.strip() for s in context.split('.') if s.strip() and len(s.strip()) > 20]
                        all_points.extend(sentences[:2])  # Take top 2 sentences
                        
                except Exception as e:
                    logger.warning(f"Failed to extract key points from context {ctx_idx}: {e}")
                    # Fallback: extract key sentences
                    sentences = [s.strip() for s in context.split('.') if s.strip() and len(s.strip()) > 20]
                    all_points.extend(sentences[:2])
            
            if not all_points:
                logger.warning("No key points extracted from any context")
                return []
            
            # Aggregate key points (deduplicate and merge)
            try:
                prompt = self.create_aggregation_prompt(all_points)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=self.max_tokens
                )
                llm_response = response.choices[0].message.content
                
                # Parse aggregated points - try multiple strategies
                import re
                
                # Strategy 1: Extract full JSON object
                try:
                    json_match = re.search(r'\{.*"points"\s*:\s*\[.*?\]\s*\}', llm_response, re.DOTALL)
                    if json_match:
                        full_json = json.loads(json_match.group(0))
                        aggregated_points = full_json.get('points', [])
                    else:
                        raise ValueError("No JSON object found")
                except Exception as e:
                    logger.debug(f"Strategy 1 failed: {e}, trying strategy 2")
                    # Strategy 2: Extract just the array
                    try:
                        # Find JSON code block
                        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
                        if code_block_match:
                            full_json = json.loads(code_block_match.group(1))
                            aggregated_points = full_json.get('points', [])
                        else:
                            # Try to find any JSON structure
                            full_json = json.loads(llm_response.strip())
                            aggregated_points = full_json.get('points', [])
                    except Exception as e2:
                        logger.warning(f"All JSON parsing strategies failed: {e2}")
                        aggregated_points = []
                
                # Ensure proper format
                if aggregated_points:
                    result = []
                    for i, point in enumerate(aggregated_points):
                        if isinstance(point, dict) and 'point_content' in point:
                            result.append({
                                "point_number": i + 1,
                                "point_content": point['point_content']
                            })
                    if result:
                        return result
                
                # Fallback if parsing failed
                return self._fallback_key_points(all_points)
                    
            except Exception as e:
                logger.warning(f"Failed to aggregate key points: {e}")
                return self._fallback_key_points(all_points)
                
        except Exception as e:
            logger.error(f"Error extracting key points from contexts: {e}")
            return []
    
    def _fallback_key_points(self, all_points: List[str]) -> List[Dict[str, Any]]:
        """Fallback: use first N unique points."""
        unique_points = []
        seen = set()
        for point in all_points:
            if point.lower() not in seen:
                unique_points.append(point)
                seen.add(point.lower())
            if len(unique_points) >= 7:
                break
        
        return [
            {"point_content": point, "point_number": i+1}
            for i, point in enumerate(unique_points)
        ]

    def _extract_key_points(self, reference_answer: str) -> List[Dict[str, Any]]:
        """Extract key points from reference answer using LLM (deprecated - use contexts instead)."""
        try:
            # Simple extraction for backward compatibility
            sentences = [s.strip() for s in reference_answer.split('.') if s.strip() and len(s.strip()) > 20]
            return [
                {"point_content": sentence, "point_number": i+1}
                for i, sentence in enumerate(sentences[:5])
            ]
        except Exception as e:
            logger.warning(f"Failed to extract key points: {e}")
            return []

    def _evaluate_single(self, system_output: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single system output for key point recall.
        """
        try:
            answer = system_output.get('answer', system_output.get('generated_response', ''))
            key_points = reference.get('key_points', [])

            if not answer:
                return {
                    "labels": {},
                    "support_rate": 0.0,
                    "omitted_rate": 0.0,
                    "contradicted_rate": 0.0,
                    "evaluation_error": "Missing answer"
                }

            if not key_points:
                # Extract key points from reference contexts (ground-truth documents)
                query = reference.get('query', '')
                contexts = reference.get('contexts', [])
                
                if contexts and query:
                    logger.info(f"No key points found, extracting from reference contexts for query_id {reference.get('query_id')}")
                    key_points = self._extract_key_points_from_contexts(query, contexts)
                    reference['key_points'] = key_points  # Cache for future use
                else:
                    # Fallback: extract from generated_response if contexts not available
                    reference_answer = reference.get('generated_response', reference.get('reference', ''))
                    if not reference_answer:
                        return {
                            "labels": {},
                            "support_rate": 0.0,
                            "omitted_rate": 0.0,
                            "contradicted_rate": 0.0,
                            "evaluation_error": "No key points, contexts, or reference answer provided"
                        }
                    logger.info(f"No key points or contexts found, extracting from reference answer for query_id {reference.get('query_id')}")
                    key_points = self._extract_key_points(reference_answer)
                    reference['key_points'] = key_points

            if not key_points:
                return {
                    "labels": {},
                    "support_rate": 0.0,
                    "omitted_rate": 0.0,
                    "contradicted_rate": 0.0,
                    "evaluation_error": "Failed to extract any key points"
                }

            # Evaluate all key points
            evaluations = self._evaluate_answer(answer, key_points)

            # Calculate rates
            total_points = len(key_points)
            supported_count = 0
            omitted_count = 0
            contradicted_count = 0

            for point_number, (label, _) in evaluations.items():
                if label == "Supported":
                    supported_count += 1
                elif label == "Omitted":
                    omitted_count += 1
                elif label == "Contradicted":
                    contradicted_count += 1

            support_rate = supported_count / total_points if total_points > 0 else 0.0
            omitted_rate = omitted_count / total_points if total_points > 0 else 0.0
            contradicted_rate = contradicted_count / total_points if total_points > 0 else 0.0

            return {
                "labels": evaluations,
                "support_rate": support_rate,
                "omitted_rate": omitted_rate,
                "contradicted_rate": contradicted_rate,
                "total_key_points": total_points,
                "supported_count": supported_count,
                "omitted_count": omitted_count,
                "contradicted_count": contradicted_count
            }

        except Exception as e:
            if self.silent_errors:
                logger.error(f"Error during key point recall evaluation: {e}")
                return {
                    "labels": {},
                    "support_rate": 0.0,
                    "omitted_rate": 0.0,
                    "contradicted_rate": 0.0,
                    "evaluation_error": f"Evaluation error: {str(e)}"
                }
            else:
                raise e

    def evaluate(self, system_outputs: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> EvaluationResult:
        """
        Evaluate key point recall for the system outputs.
        """
        start_time = time.time()

        # Validate inputs
        self.validate_inputs(system_outputs, references)

        # Create lookup for references
        ref_lookup = {ref.get('iid', ref.get('query_id')): ref for ref in references}

        rows = []
        support_rates = []
        omitted_rates = []
        contradicted_rates = []

        def evaluate_sample(output, reference):
            result = self._evaluate_single(output, reference)

            row = {
                'query_id': output.get('iid', output.get('query_id')),
                'key_point_recall': result.get('support_rate', 0.0),
                'support_rate': result.get('support_rate', 0.0),
                'omitted_rate': result.get('omitted_rate', 0.0),
                'contradicted_rate': result.get('contradicted_rate', 0.0),
                'total_key_points': result.get('total_key_points', 0),
                'supported_count': result.get('supported_count', 0),
                'omitted_count': result.get('omitted_count', 0),
                'contradicted_count': result.get('contradicted_count', 0),
                'labels': result.get('labels', {})
            }
            rows.append(row)
            support_rates.append(result.get('support_rate', 0.0))
            omitted_rates.append(result.get('omitted_rate', 0.0))
            contradicted_rates.append(result.get('contradicted_rate', 0.0))

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
        avg_support_rate = mean(support_rates) if support_rates else 0.0
        avg_omitted_rate = mean(omitted_rates) if omitted_rates else 0.0
        avg_contradicted_rate = mean(contradicted_rates) if contradicted_rates else 0.0

        metrics = {
            'key_point_recall': avg_support_rate,
            'avg_support_rate': avg_support_rate,
            'avg_omitted_rate': avg_omitted_rate,
            'avg_contradicted_rate': avg_contradicted_rate,
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