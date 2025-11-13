from typing import Optional
from systems.rag_interface import RAGInterface
from tools.llm_servers.general_openai_client import GeneralOpenAIClient
from tools.logging_utils import get_logger
from tools.reranker_vllm import GeneralReranker


class VanillaAgentLite(RAGInterface):
    def __init__(
        self,
        context_length: int = 25_000,  # LLM context length in tokens
        docs_review_max_tokens: int = 4096,
        answer_max_tokens: int = 4096,
        num_qvs: int = 5,  # number of query variants to use in search
        max_tries: int = 5,
        cw22_a: bool = True,
        llm_api_base: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_reasoning_effort: Optional[str] = None,
        reranker_api_base: Optional[str] = None,
        reranker_api_key: Optional[str] = None,
        reranker_model: Optional[str] = None,
    ):
        """
        Initialize VanillaAgent with LLM server.
        """
        self.context_length = context_length
        self.docs_review_max_tokens = docs_review_max_tokens
        self.answer_max_tokens = answer_max_tokens
        self.num_qvs = num_qvs
        self.max_tries = max_tries
        self.cw22_a = cw22_a
        self.llm_api_base = llm_api_base
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.llm_reasoning_effort = llm_reasoning_effort
        self.reranker_api_base = reranker_api_base
        self.reranker_api_key = reranker_api_key
        self.reranker_model = reranker_model

        self.logger = get_logger("vanilla_agent_lite")
        self.llm_client: Optional[GeneralOpenAIClient] = None
        self.reranker: Optional[GeneralReranker] = None

    @property
    def name(self) -> str:
        return "vanilla-agent-lite"