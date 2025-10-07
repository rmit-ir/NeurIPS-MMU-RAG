from typing import Optional
from tools.classifiers.typing import PredictionResult
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.llm_servers.vllm_server import get_llm_mgr


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

    async def _ensure_llms(self):
        if not self.llm_client:
            llm_mgr = get_llm_mgr(
                model_id=self.model_id,
                reasoning_parser=self.reasoning_parser,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                api_key=self.api_key
            )
            # pending for server to be ready
            self.llm_client = await llm_mgr.get_openai_client(
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

    def predict(self, query: str) -> PredictionResult:
        pass
