"""
Client for interacting with Amazon Bedrock API using LangChain.
"""
import os
import json
import time
from typing import Dict, Optional, Any, List, AsyncIterator, TypedDict
from datetime import datetime
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, BaseMessageChunk
from botocore.exceptions import BotoCoreError, ClientError

from tools.llm_servers.bedrock_pricing import BEDROCK_MODEL_PRICING
from tools.logging_utils import get_logger
from tools.path_utils import get_data_dir
from tools.retry_utils import retry
from dotenv import load_dotenv


class ModelLifecycle(TypedDict):
    """Type definition for model lifecycle information."""
    status: str


class BedrockModelSummary(TypedDict):
    """Type definition for Bedrock model summary information."""
    modelArn: str
    modelId: str
    modelName: str
    providerName: str
    inputModalities: List[str]
    outputModalities: List[str]
    responseStreamingSupported: bool
    customizationsSupported: List[str]
    inferenceTypesSupported: List[str]
    modelLifecycle: ModelLifecycle


class BedrockClient:
    """Client for interacting with Amazon Bedrock API using LangChain."""

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-haiku-20241022-v1:0",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        region_name: Optional[str] = None,
        logger=get_logger("bedrock_client")
    ):
        """
        Initialize the Bedrock client.

        Args:
            model_id (str): The model ID to use
            temperature (float): The temperature parameter for generation
            max_tokens (int): Maximum number of tokens to generate
            region_name (str, optional): AWS region name. If None, uses RACE_AWS_REGION from env
            logger (logging.Logger): Logger instance
        """
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.region_name = region_name
        self.logger = logger

        # Initialize the client
        self._initialize_client()

    def _initialize_client(self):
        """
        Initialize or reinitialize the Bedrock client with credentials.
        Supports multiple authentication methods:
        1. RACE_ prefixed environment variables (copied to standard AWS_ vars)
        2. Standard AWS credential chain (IAM roles, instance profiles, etc.)
        """
        # Check for RACE_ prefixed credentials and copy to standard AWS vars if present
        race_access_key = os.environ.get("RACE_AWS_ACCESS_KEY_ID")
        race_secret_key = os.environ.get("RACE_AWS_SECRET_ACCESS_KEY")
        race_session_token = os.environ.get("RACE_AWS_SESSION_TOKEN")
        race_region = os.environ.get("RACE_AWS_REGION")

        if race_access_key and race_secret_key:
            # Copy RACE_ credentials to standard AWS environment variables
            os.environ["AWS_ACCESS_KEY_ID"] = race_access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = race_secret_key
            if race_session_token:
                os.environ["AWS_SESSION_TOKEN"] = race_session_token
            if race_region:
                os.environ["AWS_DEFAULT_REGION"] = race_region

            self.logger.info(
                "Using RACE_ prefixed credentials (copied to standard AWS vars)")
            auth_method = "RACE_credentials"
        else:
            self.logger.info(
                "No RACE_ credentials found, using AWS default credential chain")
            auth_method = "default_chain"

        # Use provided region_name or get from environment variables
        region_name = self.region_name
        if region_name is None:
            region_name = race_region or os.environ.get(
                "AWS_DEFAULT_REGION", "us-west-2")

        # Set up model kwargs with temperature and max_tokens
        model_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        # Initialize ChatBedrock - let it use the default credential chain
        # This will automatically pick up credentials from:
        # 1. Environment variables (AWS_ACCESS_KEY_ID, etc.)
        # 2. IAM roles for EC2 instances
        # 3. IAM roles for ECS tasks
        # 4. AWS profiles
        # 5. Other credential sources in the boto3 credential chain
        self.chat_model = ChatBedrock(
            model_id=self.model_id,
            region_name=region_name,
            model_kwargs=model_kwargs
        )

        self.logger.info("Initialized Bedrock client",
                         model_id=self.model_id,
                         region=region_name,
                         temperature=self.temperature,
                         max_tokens=self.max_tokens,
                         auth_method=auth_method)

    def reload_credentials(self):
        self.logger.info("Reloading credentials from .env file")
        load_dotenv(override=True)
        self._initialize_client()

    @retry(max_retries=8, retry_on=(BotoCoreError, ClientError))
    async def complete_chat(self, messages: List[BaseMessage]) -> BaseMessage:
        """
        Generate a response for a chat conversation.

        Args:
            messages (List[BaseMessage]): A list of LangChain message objects

        Returns:
            BaseMessage: The generated response message from the model
        """
        # Handle expired token exceptions with credential refresh
        while True:
            try:
                # Start timing the API request
                start_time = time.time()

                # Send the message and get the response
                response = await self.chat_model.ainvoke(messages)
                self.logger.debug("Bedrock response", response=response)

                # Log response time
                response_time = time.time() - start_time
                self.logger.info(
                    "Bedrock API request completed",
                    response_time=round(response_time, 3)
                )

                # Extract token usage from response
                token_usage = self._extract_token_usage(response)
                cost = self._calculate_cost(token_usage)

                # Log token usage
                self.logger.info(
                    "Token usage",
                    input_tokens=token_usage.get("input_tokens", 0),
                    output_tokens=token_usage.get("output_tokens", 0),
                    total_tokens=token_usage.get("total_tokens", 0),
                    cost_usd=round(cost, 6)
                )

                # Create response metadata for saving
                response_metadata = {
                    "model": self.model_id,
                    "messages": [{"role": msg.__class__.__name__, "content": msg.content} for msg in messages],
                    "response": response.content,
                    "timestamp": datetime.now().isoformat(),
                    "token_usage": token_usage,
                    "cost_usd": round(cost, 6),
                }

                # Save response for reproducibility
                self._save_raw_response(response_metadata)

                return response

            except ClientError as e:
                # Check if this is an expired token exception
                if "ExpiredTokenException" in str(e):
                    self.logger.warning(
                        "AWS token expired. Reloading credentials and retrying.")
                    # Reload credentials from .env file
                    self.reload_credentials()
                    self.logger.info("Credentials reloaded, retrying in 5s...")
                    time.sleep(5)
                    # Continue to retry with fresh credentials
                    continue
                else:
                    # Re-raise other ClientErrors
                    raise

    async def complete_chat_streaming(self, messages: List[BaseMessage]) -> AsyncIterator[BaseMessageChunk]:
        """
        Generate a streaming response for a chat conversation.

        Args:
            messages (List[BaseMessage]): A list of LangChain message objects

        Yields:
            BaseMessageChunk: Response chunks as they arrive
        """
        # Start timing the API request
        start_time = time.time()

        try:
            # Send the message and get the streaming response
            full_content = ""
            async for chunk in self.chat_model.astream(messages):
                full_content += str(chunk.content) if chunk.content else ""
                yield chunk

            # Log response time
            response_time = time.time() - start_time
            self.logger.info(
                "Bedrock streaming API request completed",
                response_time=round(response_time, 3)
            )

            # Create response metadata for saving
            response_metadata = {
                "model": self.model_id,
                "messages": [{"role": msg.__class__.__name__, "content": msg.content} for msg in messages],
                "response": full_content,
                "timestamp": datetime.now().isoformat(),
                "streaming": True
            }

            # Save response for reproducibility
            self._save_raw_response(response_metadata)

        except Exception as e:
            self.logger.error(f"Unexpected error in streaming: {str(e)}")
            raise

    async def list_models(self) -> List[BedrockModelSummary]:
        """
        List available Bedrock models.

        Returns:
            List[BedrockModelSummary]: List of model summaries with detailed information
        """
        try:
            models = self.chat_model.bedrock_client.list_foundation_models()
            return models.get('modelSummaries', [])
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []

    def _extract_token_usage(self, response: Any) -> Dict[str, int]:
        """
        Extract token usage information from the API response.

        Args:
            response: The API response object

        Returns:
            Dictionary containing token usage information
        """
        token_usage = {}

        # Try to extract from response_metadata
        if hasattr(response, "response_metadata") and response.response_metadata:
            usage = response.response_metadata.get("usage", {})
            token_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

        # If not found, try additional_kwargs
        elif hasattr(response, "additional_kwargs") and response.additional_kwargs:
            usage = response.additional_kwargs.get("usage", {})
            token_usage = {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }

        # If not found, try usage_metadata
        elif hasattr(response, "usage_metadata") and response.usage_metadata:
            token_usage = {
                "input_tokens": response.usage_metadata.get("input_tokens", 0),
                "output_tokens": response.usage_metadata.get("output_tokens", 0),
                "total_tokens": response.usage_metadata.get("total_tokens", 0)
            }

        return token_usage

    def _calculate_cost(self, token_usage: Dict[str, int]) -> float:
        """
        Calculate the cost of the API call based on token usage.

        Args:
            token_usage: Dictionary containing token usage information

        Returns:
            Cost in USD
        """
        # Get pricing for the model
        pricing = BEDROCK_MODEL_PRICING.get(self.model_id, None)
        if not pricing:
            self.logger.warning(
                f"Model {self.model_id} not found in pricing data. Using default pricing.",
                model_id=self.model_id, pricing=BEDROCK_MODEL_PRICING["default"])
            pricing = BEDROCK_MODEL_PRICING["default"]
        else:
            self.logger.info("Using pre-defined pricing",
                             model_id=self.model_id, pricing=pricing)

        # Extract token counts
        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)

        # Calculate cost (price per 1M tokens, convert to price per token)
        input_cost = (input_tokens / 1000) * (pricing["input_price"] / 1000)
        output_cost = (output_tokens / 1000) * (pricing["output_price"] / 1000)

        # Total cost
        total_cost = input_cost + output_cost

        return total_cost

    def _save_raw_response(self, response: Dict[str, Any]) -> None:
        """
        Saves the raw API response to a file for reproducibility and backup.

        Args:
            response (Dict[str, Any]): The raw API response
        """
        try:
            # Create a directory for raw responses if it doesn't exist
            raw_responses_dir = os.path.join(get_data_dir(), "raw_responses")
            os.makedirs(raw_responses_dir, exist_ok=True)

            # Generate a filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_id.replace(
                ".", "-").replace(":", "-").replace("/", "-")
            filename = f"bedrock_response_{model_name}_{timestamp}.json"
            filepath = os.path.join(raw_responses_dir, filename)

            # Save the response
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
    from langchain_core.messages import HumanMessage, SystemMessage
    client = BedrockClient(model_id="meta.llama3-1-8b-instruct-v1:0")

    # Send the query and get the response
    messages = [
        SystemMessage(
            content="You are an AI assistant that provides clear, concise explanations."),
        HumanMessage(content="What is retrieval-augmented generation (RAG)?")
    ]

    response = await client.complete_chat(messages)

    # Print the response content
    print("\nResponse from Bedrock API:")
    print("-" * 50)
    print(response.content)
    print("-" * 50)

    # list models
    models = await client.list_models()
    print(f"Available models ({len(models)})")
    print('\n'.join([model['modelId'] for model in models]))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
