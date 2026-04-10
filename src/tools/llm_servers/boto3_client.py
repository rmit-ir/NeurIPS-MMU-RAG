"""
Bedrock client using boto3 Converse API directly.
Drop-in replacement for GeneralOpenAIClient in VanillaRAG/VanillaAgent.

The Converse API normalizes responses across all Bedrock models:
- Reasoning/thinking → reasoningContent blocks (separate from text)
- Answer content → text blocks
- Token usage → metadata events (streaming) or usage field (non-streaming)

For Claude models, extended thinking can be explicitly enabled via
reasoning_budget_tokens. Other models (DeepSeek R1, GPT-OSS) produce
reasoning automatically.
"""
import os
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from datetime import datetime
from openai.types.chat import ChatCompletionMessageParam

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from tools.llm_servers.llm_interface import LLMInterface
from tools.llm_servers.sglang_types import (
    CustomChatCompletionChunk,
    CustomChoice,
    CustomChoiceDelta,
)
from tools.llm_servers.bedrock_pricing import BEDROCK_MODEL_PRICING
from tools.logging_utils import get_logger
from tools.path_utils import get_data_dir
from tools.retry_utils import retry

# Dedicated thread pool for Bedrock I/O so streaming requests don't
# exhaust the default executor under concurrent load.
from concurrent.futures import ThreadPoolExecutor
_bedrock_executor = ThreadPoolExecutor(max_workers=200, thread_name_prefix="bedrock-io")


class Boto3BedrockClient(LLMInterface):
    """
    Bedrock client using boto3 Converse/ConverseStream APIs.
    Same interface as GeneralOpenAIClient (complete_chat, complete_chat_streaming).
    """

    def __init__(
        self,
        model_id: str = "openai.gpt-oss-120b-1:0",
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        region_name: Optional[str] = None,
        reasoning_budget_tokens: Optional[int] = None,
        logger=get_logger("boto3_bedrock"),
    ):
        super().__init__(model_id=model_id, temperature=temperature, max_tokens=max_tokens)
        self.logger = logger
        self.llm_name = "boto3_bedrock"
        self.reasoning_budget_tokens = reasoning_budget_tokens
        self.region_name = (
            region_name
            or os.environ.get("RACE_AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
        )
        self._setup_credentials()
        self._boto_client = boto3.client(
            "bedrock-runtime", region_name=self.region_name)
        self.logger.info(
            "Initialized Boto3 Bedrock client",
            model_id=self.model_id, region=self.region_name,
            temperature=self.temperature, max_tokens=self.max_tokens,
            reasoning_budget_tokens=self.reasoning_budget_tokens,
        )

    def _setup_credentials(self):
        """Copy RACE_ prefixed env vars to standard AWS vars if present."""
        for race_key, aws_key in {
            "RACE_AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "RACE_AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "RACE_AWS_SESSION_TOKEN": "AWS_SESSION_TOKEN",
            "RACE_AWS_REGION": "AWS_DEFAULT_REGION",
        }.items():
            val = os.environ.get(race_key)
            if val:
                os.environ[aws_key] = val

    def _reinitialize_client(self):
        self._setup_credentials()
        self._boto_client = boto3.client(
            "bedrock-runtime", region_name=self.region_name)

    def _build_request_kwargs(
        self, messages: List[ChatCompletionMessageParam], max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build kwargs for converse/converse_stream. Converts OpenAI messages to Converse format."""
        system_parts: List[Dict[str, Any]] = []
        converse_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": content})
            elif role == "assistant":
                converse_messages.append(
                    {"role": "assistant", "content": [{"text": content}]})
            else:
                converse_messages.append(
                    {"role": "user", "content": [{"text": content}]})

        inference_config: Dict[str, Any] = {
            "maxTokens": max_tokens or self.max_tokens}
        if self.temperature is not None:
            inference_config["temperature"] = self.temperature

        kwargs: Dict[str, Any] = {
            "modelId": self.model_id,
            "messages": converse_messages,
            "inferenceConfig": inference_config,
        }
        if system_parts:
            kwargs["system"] = system_parts

        # Extended thinking for Claude models only (min 1024, must be < max_tokens)
        if self.reasoning_budget_tokens and "anthropic." in self.model_id:
            kwargs["additionalModelRequestFields"] = {
                "reasoning_config": {"type": "enabled", "budget_tokens": self.reasoning_budget_tokens}
            }
            inference_config.pop("temperature", None)

        return kwargs

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = BEDROCK_MODEL_PRICING.get(
            self.model_id, BEDROCK_MODEL_PRICING.get(
                "default", {"input_price": 0, "output_price": 0})
        )
        return (input_tokens / 1_000_000) * pricing["input_price"] + \
               (output_tokens / 1_000_000) * pricing["output_price"]

    def complete(self, prompt: str) -> str:
        raise NotImplementedError("Use complete_chat instead")

    @retry(max_retries=8, retry_on=(BotoCoreError, ClientError))
    async def complete_chat(
        self, messages: List[ChatCompletionMessageParam]
    ) -> Tuple[str | None, Any]:
        """
        Non-streaming chat completion. Returns (content_string, raw_response).
        Reasoning blocks are extracted but not included in the returned content.
        """
        import asyncio

        start_time = time.time()
        kwargs = self._build_request_kwargs(messages)

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._boto_client.converse(**kwargs)
            )

            content, reasoning = "", ""
            for block in response["output"]["message"]["content"]:
                if "text" in block:
                    content = block["text"]
                elif "reasoningContent" in block:
                    rc = block["reasoningContent"]
                    if "reasoningText" in rc:
                        rt = rc["reasoningText"]
                        reasoning = rt.get("text", "") if isinstance(
                            rt, dict) else str(rt)

            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)

            self.logger.info(
                "Bedrock Converse completed",
                response_time=round(time.time() - start_time, 3),
                input_tokens=input_tokens, output_tokens=output_tokens,
                cost_usd=round(self._calculate_cost(
                    input_tokens, output_tokens), 6),
                has_reasoning=bool(reasoning),
            )
            self._save_raw_response({
                "model": self.model_id, "response": content, "reasoning": reasoning,
                "timestamp": datetime.now().isoformat(),
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            })
            return content, response

        except ClientError as e:
            if "ExpiredTokenException" in str(e):
                self.logger.warning("AWS token expired, reinitializing client")
                self._reinitialize_client()
            raise

    async def complete_chat_streaming(
        self, messages: List[ChatCompletionMessageParam], max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[CustomChatCompletionChunk, None]:
        """
        Streaming chat completion. Yields OpenAI-format ChatCompletionChunk objects.
        Reasoning tokens → CustomChoiceDelta.reasoning_content (routed to intermediate steps).
        Answer tokens → CustomChoiceDelta.content.
        """
        import asyncio

        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        kwargs = self._build_request_kwargs(messages, max_tokens)

        self.logger.info("Starting Bedrock ConverseStream",
                         model=self.model_id, max_tokens=max_tokens or self.max_tokens)

        def _make_chunk(content=None, reasoning_content=None, finish_reason=None):
            return CustomChatCompletionChunk(
                id=request_id,
                choices=[CustomChoice(
                    index=0,
                    delta=CustomChoiceDelta(
                        role="assistant", content=content,
                        reasoning_content=reasoning_content,
                    ),
                    finish_reason=finish_reason,
                )],
                created=int(time.time()),
                model=self.model_id,
                object="chat.completion.chunk",
            )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self._boto_client.converse_stream(**kwargs)
            )

            # Read the synchronous boto3 stream in a background thread
            # so we don't block the event loop between chunks.
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def _read_stream():
                try:
                    for event in response["stream"]:
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                    loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, e)

            asyncio.get_event_loop().run_in_executor(_bedrock_executor, _read_stream)

            full_content, full_reasoning = "", ""
            while True:
                event = await queue.get()
                if event is None:
                    break
                if isinstance(event, Exception):
                    raise event

                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]["delta"]

                    reasoning_text = delta.get(
                        "reasoningContent", {}).get("text", "")
                    if reasoning_text:
                        full_reasoning += reasoning_text
                        yield _make_chunk(reasoning_content=reasoning_text)

                    text = delta.get("text", "")
                    if text:
                        full_content += text
                        yield _make_chunk(content=text)

                elif "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    self.logger.info(
                        "Bedrock ConverseStream completed",
                        response_time=round(time.time() - start_time, 3),
                        input_tokens=usage.get("inputTokens", 0),
                        output_tokens=usage.get("outputTokens", 0),
                        cost_usd=round(self._calculate_cost(
                            usage.get("inputTokens", 0), usage.get("outputTokens", 0)), 6),
                    )

            yield _make_chunk(finish_reason="stop")
            self._save_raw_response({
                "model": self.model_id, "response": full_content,
                "reasoning": full_reasoning, "timestamp": datetime.now().isoformat(),
                "streaming": True,
            })

        except ClientError as e:
            if "ExpiredTokenException" in str(e):
                self.logger.warning("AWS token expired, reinitializing client")
                self._reinitialize_client()
            raise

    def _save_raw_response(self, response: Dict[str, Any]) -> None:
        try:
            raw_responses_dir = os.path.join(get_data_dir(), "raw_responses")
            os.makedirs(raw_responses_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_id.replace(
                ".", "-").replace(":", "-").replace("/", "-")
            filepath = os.path.join(
                raw_responses_dir, f"boto3_bedrock_{model_name}_{timestamp}.json")
            with open(filepath, "w") as f:
                json.dump(response, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save raw response: {str(e)}")


async def main():
    client = Boto3BedrockClient(model_id="openai.gpt-oss-120b-1:0")
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
            "content": "What is retrieval-augmented generation (RAG)?"},
    ]

    print("=== Non-streaming ===")
    content, _ = await client.complete_chat(messages)
    print(content)

    print("\n=== Streaming ===")
    async for chunk in client.complete_chat_streaming(messages):
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            print(f"[THINK] {delta.reasoning_content}", end="", flush=True)
        elif delta.content:
            print(delta.content, end="", flush=True)
    print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
