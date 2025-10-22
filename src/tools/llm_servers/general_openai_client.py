"""
Client for interacting with OpenAI-compatible API using the official OpenAI Python client.
"""
import os
import json
import time
import asyncio
from typing import Dict, Optional, Tuple, Any, List, AsyncGenerator
from openai import OpenAI, AsyncOpenAI, APIError, APIConnectionError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from datetime import datetime

from tools.llm_servers.llm_interface import LLMInterface
from tools.llm_servers.sglang_utils import wait_for_server
from tools.llm_servers.sglang_types import CustomChatCompletionChunk
from tools.logging_utils import get_logger
from tools.path_utils import get_data_dir
from tools.retry_utils import retry


class GeneralOpenAIClient(LLMInterface):
    """Client for interacting with OpenAI-compatible API."""

    def __init__(
        self,
        api_base: str,
        api_key: Optional[str] = None,
        max_retries: int = 5,
        timeout: float = 600.0,
        model_id: str = "tiiuae/falcon3-10b-instruct",
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        logger=get_logger("general_openai_client"),
        llm_name: str = "general_openai_client"
    ):
        """
        Initialize the OpenAI-compatible client.

        Args:
            api_base (str): API base URL (required)
            api_key (Optional[str]): API key (optional, defaults to None)
            model_id (str): The model ID to use
            temperature (Optional[float]): The temperature parameter for generation
            max_tokens (int): Maximum number of tokens to generate
            logger (logging.Logger): Logger instance
            llm_name (str): Name of the LLM client for file naming
        """
        # Initialize the parent class
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Validate required parameters
        if not api_base:
            raise ValueError("API base URL is required")

        # Use default API key if none provided
        if not api_key:
            api_key = "dummy-key"

        self.logger = logger
        self.llm_name = llm_name

        # Initialize the OpenAI client with explicit headers
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            max_retries=max_retries,
            timeout=timeout,
            default_headers={
                "Content-Type": "application/json",
            }
        )

        # Initialize the AsyncOpenAI client for streaming
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            max_retries=max_retries,
            timeout=timeout,
            default_headers={
                "Content-Type": "application/json",
            }
        )

        # Store model ID for reference
        self.model_id = model_id
        self.logger.debug(
            f"Initialized OpenAI-compatible client with model: {model_id}")

    # another level of retry, this wait time is increased exponentially
    @retry(max_retries=8, retry_on=(APIError, APIConnectionError, RateLimitError))
    def complete(self, prompt: str) -> str:
        """
        Generate a text completion for the given prompt.
        """
        # Start timing the API request
        start_time = time.time()

        try:
            # Send the prompt to the completion model
            response = self.client.completions.create(
                model=self.model_id,
                prompt=prompt+"\n\n",
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract content from the response
            content = response.choices[0].text

            # Log response time
            response_time = time.time() - start_time
            self.logger.info(
                "Completion API request completed",
                response_time=round(response_time, 3)
            )

            # Try to log token usage if available
            if hasattr(response, "usage") and response.usage:
                self.logger.info(
                    "Token usage",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )

            # Create response metadata for saving
            response_metadata = {
                "model": self.model_id,
                "prompt": prompt,
                "response": content,
                "timestamp": datetime.now().isoformat()
            }

            # Save response for reproducibility
            self._save_raw_response(response_metadata)
            self.logger.debug("Response content", content=content)

            return content

        except Exception as e:
            self.logger.error(f"Unexpected error in complete: {str(e)}")
            raise

    # another level of retry, this wait time is increased exponentially
    @retry(max_retries=8, retry_on=(APIError, APIConnectionError, RateLimitError))
    async def complete_chat(self, messages: List[ChatCompletionMessageParam]) -> Tuple[str | None, Any]:
        """
        Generate a response for a chat conversation.

        Args:
            messages (List[ChatCompletionMessageParam]): A list of message dictionaries,
                each containing 'role' (system, user, or assistant) and 'content' keys

        Returns:
            Tuple[str, Any]: A tuple containing:
                - content: The generated text content from the model
                - raw_response: The complete API response object
        """
        # Start timing the API request
        start_time = time.time()

        try:
            # Send the message and get the response
            response = await self.async_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extract content from the response
            content = response.choices[0].message.content

            # Log response time
            response_time = time.time() - start_time
            self.logger.info(
                "API request completed",
                response_time=round(response_time, 3)
            )

            # Try to log token usage if available
            if hasattr(response, "usage") and response.usage:
                self.logger.info(
                    "Token usage",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )

            # Create response metadata for saving
            response_metadata = {
                "model": self.model_id,
                "prompt": messages,
                "response": content,
                "timestamp": datetime.now().isoformat()
            }

            # Save response for reproducibility
            self._save_raw_response(response_metadata)
            self.logger.debug("Response content", content=content)

            return content, response

        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise

    async def complete_chat_streaming(self, messages: List[ChatCompletionMessageParam], max_tokens: Optional[int] = None) -> AsyncGenerator[CustomChatCompletionChunk, None]:
        """
        Generate a streaming response for a chat conversation using AsyncOpenAI.

        Args:
            messages (List[ChatCompletionMessageParam]): A list of message dictionaries,
                each containing 'role' (system, user, or assistant) and 'content' keys
        """
        # Start timing the API request
        start_time = time.time()

        try:
            self.logger.info("Starting model request",
                             model=self.model_id,
                             temperature=self.temperature,
                             max_tokens=max_tokens or self.max_tokens,
                             stream=True)
            # Send the message and get the streaming response
            stream: Any = await self.async_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=True
            )

            full_content = {"content": "", "reasoning_content": ""}
            async for chunk in stream:
                chunk: CustomChatCompletionChunk = chunk
                self.logger.debug("Received chunk", chunk=chunk)
                yield chunk
                first_choice = chunk.choices[0]
                if hasattr(first_choice.delta, 'content') and first_choice.delta.content:
                    full_content["content"] += first_choice.delta.content
                if hasattr(first_choice.delta, 'reasoning_content') and first_choice.delta.reasoning_content:
                    full_content["reasoning_content"] += first_choice.delta.reasoning_content

            # Log response time
            response_time = time.time() - start_time
            self.logger.info(
                "Streaming API request completed",
                response_time=round(response_time, 3)
            )

            # Create response metadata for saving
            response_metadata = {
                "model": self.model_id,
                "prompt": messages,
                "response": full_content,
                "timestamp": datetime.now().isoformat(),
                "streaming": True
            }

            # Save response for reproducibility
            self._save_raw_response(response_metadata)
            content_length = len(
                full_content["content"]) + len(full_content["reasoning_content"])
            self.logger.debug("Streaming response completed",
                              content_length=content_length)

        except Exception as e:
            self.logger.error(f"Unexpected error in streaming: {str(e)}")
            raise

    async def wait_ready(self, server_host: str) -> None:
        """
        Wait for the server to be ready by performing a health check.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: wait_for_server(server_host, None, self.client.api_key)
        )

    def _save_raw_response(self, response: Dict[str, Any]) -> None:
        """
        Saves the raw API response to a file for reproducibility and backup.

        Args:
            response (Dict[str, Any]): The raw API response
            prompt (str): The prompt that was sent to the API
        """
        try:
            # Create a directory for raw responses if it doesn't exist
            raw_responses_dir = os.path.join(get_data_dir(), "raw_responses")
            os.makedirs(raw_responses_dir, exist_ok=True)

            # Generate a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_id.replace(
                ".", "-").replace(":", "-").replace("/", "-")
            filename = f"{self.llm_name}_response_{model_name}_{timestamp}.json"
            filepath = os.path.join(raw_responses_dir, filename)

            # Save the response with the prompt
            with open(filepath, "w") as f:
                json.dump({
                    "model": self.model_id,
                    "timestamp": timestamp,
                    "response": response
                }, f, indent=2)

            self.logger.debug(f"Raw response saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save raw response: {str(e)}")


async def main():
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()

    # Create a GeneralOpenAIClient instance
    client = GeneralOpenAIClient(
        api_key=os.environ.get("MMU_OPENAI_API_KEY", ""),
        api_base="https://mmu-proxy-server-llm-proxy.rankun.org/v1",
        model_id="openai.gpt-oss-20b-1:0"
    )

    # Send the query and get the response with a custom system message
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Compare use cases for RAG and long context LLMs"}
    ]
    content, raw_response = await client.complete_chat(messages=messages)

    # Print the response content
    print("\nResponse from API:")
    print("-" * 50)
    print(content)
    print("-" * 50)
    print("-" * 50)
    print("-" * 50)
    print("\nFull API Response:")
    print(raw_response)

if __name__ == "__main__":
    asyncio.run(main())
