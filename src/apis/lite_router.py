"""
Light Weight OpenAI-compatible API router implementation.
Implements /v1/models and /v1/chat/completions endpoints.
"""

import os
import hashlib
from typing import List, Dict, Optional, Union
import uuid
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Security
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from systems.commercial.azure_o3_research import AzureO3ResearchRAG
from systems.commercial.perplexity_research import PerplexityResearchRAG
from systems.rag_interface import RAGInterface, RunRequest
from systems.rag_router.rag_router_llm import RAGRouterLLM
from tools.logging_utils import get_logger
from tools.responses.openai_stream import to_openai_stream

logger = get_logger('lite_router')



if __name__ == "__main__":
    print("Run\nuv run fastapi run src/apis/lite_router.py")
    pass
