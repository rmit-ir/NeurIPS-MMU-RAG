"""
Custom LLM implementation for DeepEval that handles reasoning tokens.

This module provides a custom LLM class that integrates with the MMU proxy server
and handles reasoning models that may output reasoning tokens before JSON responses.
"""

import os
import json
import re
from typing import Optional
from pydantic import BaseModel
import openai
from dotenv import load_dotenv
from deepeval.models import DeepEvalBaseLLM


class MMUCustomLLM(DeepEvalBaseLLM):
    """
    Custom LLM for DeepEval using MMU proxy server.
    
    This LLM supports reasoning models that may output reasoning tokens
    before JSON responses. It automatically extracts valid JSON from the response.
    """
    
    def __init__(
        self,
        model: str = "qwen.qwen3-32b-v1:0",
        base_url: str = "https://mmu-proxy-server-llm-proxy.rankun.org",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 16000,
        seed: int = 42
    ):
        """
        Initialize the custom LLM.
        
        Args:
            model: Model name to use
            base_url: Base URL for the MMU proxy server
            api_key: API key (defaults to MMU_OPENAI_API_KEY env var)
            temperature: Temperature for generation (default: 0.0)
            max_tokens: Maximum tokens to generate (default: 16000)
            seed: Random seed for reproducible results (default: 42)
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.model_name = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.max_tokens = max_tokens
        
        # Set API key from environment if not provided
        api_key = api_key or os.environ.get("MMU_OPENAI_API_KEY", "")
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Initialize OpenAI client with custom base URL
        self.client = openai.OpenAI(base_url=base_url)
    
    def load_model(self):
        """Load the model (returns the client)."""
        return self.client
    
    def _fix_invalid_escapes(self, json_str: str) -> str:
        """
        Attempt to fix invalid escape sequences in JSON string.
        
        Args:
            json_str: JSON string with potential invalid escapes
            
        Returns:
            Fixed JSON string
        """
        import re
        
        # Common invalid escapes: replace \ followed by non-standard chars
        # But be careful not to break valid ones
        # For example, replace \" with ", but \" is valid.
        # Invalid like \t in wrong places, but hard to detect.
        
        # Simple: remove backslashes before non-escape chars
        # But this might break valid escapes.
        
        # Replace invalid \ followed by letters not in ["n","t","r","b","f","u","/","\\","\""]
        # But complicated.
        
        # For now, try to replace common issues: if there's \ followed by a letter not in valid escapes, remove the \
        valid_escapes = ['n', 't', 'r', 'b', 'f', 'u', '/', '\\', '"']
        def fix_escape(match):
            char = match.group(1)
            if char in valid_escapes:
                return match.group(0)
            else:
                return char  # remove the \
        
        json_str = re.sub(r'\\(.)', fix_escape, json_str)
        
        return json_str
    
    def _fix_truncated_json(self, json_str: str) -> str:
        """
        Attempt to fix truncated JSON by closing unterminated strings and structures.
        
        Args:
            json_str: Potentially truncated JSON string
            
        Returns:
            Fixed JSON string
        """
        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Handle unterminated strings: if we have an odd number of quotes, 
        # and the last quote is not closed, close it
        # But this is tricky because we might be inside a string.
        # Simple approach: if the string ends with a quote followed by comma or bracket, it's probably ok
        # If it ends with text without closing quote, add quote
        
        # Check if we're inside a string at the end
        in_string = False
        escape_next = False
        for i in range(len(json_str)):
            char = json_str[i]
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
        
        if in_string:
            json_str += '"'
        
        # Try to close unterminated objects/arrays
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')
        
        json_str += '}' * open_braces
        json_str += ']' * open_brackets
        
        return json_str
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract valid JSON from response, handling reasoning tokens and markdown code blocks.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Extracted JSON string
        """
        # First, remove markdown code blocks (```json ... ``` or ``` ... ```)
        # This handles both complete and incomplete code blocks
        if '```' in response:
            # Try to extract content between code block markers
            # Handle incomplete blocks by finding the start marker
            start_marker = response.find('```')
            if start_marker != -1:
                # Skip the marker and optional 'json' keyword
                content_start = response.find('\n', start_marker) + 1
                if content_start > start_marker:
                    # Look for closing marker
                    end_marker = response.find('```', content_start)
                    if end_marker != -1:
                        # Complete code block
                        response = response[content_start:end_marker].strip()
                    else:
                        # Incomplete code block - take everything after the start
                        response = response[content_start:].strip()
        
        # Second, try to find JSON object or array patterns using a greedy approach
        # Start with the first { or [ and try to match balanced brackets
        def extract_balanced_json(text: str, start_char: str) -> Optional[str]:
            """Extract JSON with balanced brackets starting from start_char."""
            end_char = '}' if start_char == '{' else ']'
            start_idx = text.find(start_char)
            if start_idx == -1:
                return None
            
            # Count brackets to find the matching closing bracket
            count = 0
            in_string = False
            escape_next = False
            
            for i in range(start_idx, len(text)):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == start_char:
                        count += 1
                    elif char == end_char:
                        count -= 1
                        if count == 0:
                            # Found matching bracket
                            potential_json = text[start_idx:i+1]
                            try:
                                json.loads(potential_json)
                                return potential_json
                            except json.JSONDecodeError:
                                pass
            
            return None
        
        # Try to extract JSON object first, then array
        for start_char in ['{', '[']:
            extracted = extract_balanced_json(response, start_char)
            if extracted:
                return extracted
        
        # If no balanced extraction works, try simple pattern matching
        json_patterns = [
            r'\{[\s\S]*\}',  # Match { ... } with anything in between
            r'\[[\s\S]*\]',  # Match [ ... ] with anything in between
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in reversed(matches):  # Try last match first (most complete)
                try:
                    json.loads(match)
                    return match
                except json.JSONDecodeError:
                    continue
        
        # If all else fails, return the original response
        # and let the caller handle the error
        return response
    
    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str | BaseModel:
        """
        Generate response from the LLM.
        
        Args:
            prompt: Input prompt
            schema: Optional Pydantic schema for structured output
            
        Returns:
            Generated text string or BaseModel instance if schema provided
        """
        client = self.load_model()
        
        # If schema is provided, add JSON formatting instruction
        if schema:
            prompt = f"{prompt}\n\nRespond with valid JSON only."
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # seed=self.seed,
                stream=False
            )
            
            content = response.choices[0].message.content
            
            # If schema is provided, extract and parse JSON
            if schema:
                json_str = self._extract_json_from_response(content)
                try:
                    json_obj = json.loads(json_str)
                    # Fix common field name mismatches for DeepEval schemas
                    if isinstance(json_obj, dict) and "verdicts" in json_obj:
                        for verdict in json_obj["verdicts"]:
                            if "reason" in verdict and "statement" not in verdict:
                                verdict["statement"] = verdict.pop("reason")
                    return schema(**json_obj)
                except (json.JSONDecodeError, ValueError) as e:
                    # Try to fix common JSON truncation issues
                    if "Unterminated string" in str(e) or "Expecting" in str(e):
                        # Try to close the JSON properly
                        fixed_json = self._fix_truncated_json(json_str)
                        try:
                            json_obj = json.loads(fixed_json)
                            return schema(**json_obj)
                        except:
                            pass
                    
                    # Try to fix invalid escapes
                    if "Invalid \\escape" in str(e):
                        fixed_json = self._fix_invalid_escapes(json_str)
                        try:
                            json_obj = json.loads(fixed_json)
                            return schema(**json_obj)
                        except:
                            pass
                    
                    raise ValueError(
                        f"Failed to parse JSON from LLM response. "
                        f"Error: {e}. Raw response: {content[:500]}"
                    )
            
            return content
            
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}")
    
    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str | BaseModel:
        """
        Asynchronously generate response from the LLM.
        
        Args:
            prompt: Input prompt
            schema: Optional Pydantic schema for structured output
            
        Returns:
            Generated text string or BaseModel instance if schema provided
        """
        # For simplicity, we reuse the synchronous method
        # In production, you should implement a proper async version
        return self.generate(prompt, schema)
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return f"MMU Custom LLM ({self.model_name})"
