import time
from typing import List, Optional
from openai.types.chat import ChatCompletionMessageParam
from tools.classifiers.typing import PredictionResult
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.llm_servers.vllm_server import VllmConfig, get_llm_mgr
from tools.logging_utils import get_logger


class QueryComplexityLLM:
    def __init__(self,
                 model_id: str = "Qwen/Qwen3-4B",
                 reasoning_parser: Optional[str] = "qwen3",
                 gpu_memory_utilization: Optional[float] = 0.6,
                 max_model_len: Optional[int] = 20000,
                 api_key: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096):
        self.model_id = model_id
        self.reasoning_parser = reasoning_parser
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm_client: Optional[GeneralOpenAIClient] = None
        self.logger = get_logger('QueryComplexityLLM')

    async def _ensure_llms(self):
        if not self.llm_client:
            llm_mgr = get_llm_mgr(VllmConfig(model_id=self.model_id,
                                             reasoning_parser=self.reasoning_parser,
                                             gpu_memory_utilization=self.gpu_memory_utilization,
                                             max_model_len=self.max_model_len,
                                             api_key=self.api_key))
            # pending for server to be ready
            self.llm_client = await llm_mgr.get_openai_client(
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

    async def predict(self, query: str) -> PredictionResult:
        """Predict if a query is complex using LLM."""
        await self._ensure_llms()

        if not self.llm_client:
            raise RuntimeError("Failed to initialize LLM client")

        system_prompt = """Judge if the user query is a complex query. Note that the answer can only be "yes" or "no".

Given the query below, if you are doing the research, do you think you can do a single search on Google and find out the answer?

If so, it's not a complex query. If you need to search multiple times, it's a complex query.

If the user query is long, but can be summarized into a simple question, then it's still not a complex query.

Generally, for straightforward questions, the answer is no, if the question is ambiguous, multifaceted, contains multiple parts or requires multiple steps to answer, the answer is yes.

Give the final answer based on your last reasoning, yes indicates it's a complex query or no indicating it's not a complex query.
"""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        start_time = time.time()
        content, cpl = await self.llm_client.complete_chat(messages)
        infer_time = time.time() - start_time

        # Parse the response
        is_complex = content.strip().lower() == 'yes' if content else False
        is_simple = not is_complex

        # log thinking process
        reasoning_content = cpl.choices[0].message.reasoning_content.strip() \
            if 'reasoning_content' in cpl.choices[0].message else ''
        self.logger.info(
            "predict", query=query, is_complex=is_complex,
            reasoning_content=reasoning_content, infer_time=infer_time)

        return PredictionResult(
            query=query,
            is_simple_prob=1,
            is_simple=is_simple,
            confidence=1,
            infer_time=infer_time
        )
