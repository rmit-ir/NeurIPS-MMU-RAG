from typing import List, Optional, Union
from pydantic import BaseModel

# OpenAI API Models


class ChatMessage(BaseModel):
    """OpenAI chat message format."""
    role: str  # "system", "user", "assistant"
    content: str
    # custom fields for citations and contexts
    contexts: Optional[List[str]] = None
    citations: Optional[List[str]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request format."""
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = None


class ModelInfo(BaseModel):
    """OpenAI model information format."""
    id: str
    object: str = "model"
    created: int = 0  # we don't have a creation timestamp
    owned_by: str = "rmit-ir"


class ModelsResponse(BaseModel):
    """OpenAI models list response format."""
    object: str = "list"
    data: List[ModelInfo]


class ChatCompletionChoice(BaseModel):
    """OpenAI chat completion choice format."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """OpenAI chat completion usage format."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response format."""
    id: str
    object: str = "chat.completion"
    created: int = 1234567890
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage
