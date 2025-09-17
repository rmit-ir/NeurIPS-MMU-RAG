# MMU-RAG APIs Documentation

This directory contains two FastAPI applications that provide different interfaces to the RAG system:

1. **mmu-rag-router.py** - MMU-RAG Challenge compliant API
2. **openai-router.py** - OpenAI-compatible API

## Starting the Services

```bash
# Run the MMU-RAG Challenge API server
uv run fastapi run src/apis/mmu_rag_router.py

# Run the OpenAI-compatible API server
uv run fastapi run src/apis/openai_router.py
```

---

## MMU-RAG API Endpoints

### 1. Health Check

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

### 2. Evaluate Endpoint (Static)

Static endpoint for validation queries as per MMU-RAG challenge requirements.

```bash
curl -X POST "http://127.0.0.1:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital of France?",
    "iid": "test-query-001"
  }'
```

### 3. Run Endpoint (Streaming)

Streaming endpoint for RAG-Arena live evaluation with Server-Sent Events (SSE).

```bash
curl -X POST "http://127.0.0.1:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain quantum computing"
  }'
```

---

## OpenAI-Compatible API Endpoints

### 1. Health Check

```bash
curl -X GET "http://127.0.0.1:8002/openai/v1/health"
```

### 2. List Models

```bash
curl -X GET "http://127.0.0.1:8000/openai/v1/models"
```

### 3. Chat Completions (Non-Streaming)

```bash
curl -X POST "http://127.0.0.1:8000/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vanilla-rag",
    "messages": [
      {
        "role": "user",
        "content": "What is artificial intelligence?"
      }
    ],
    "stream": false
  }'
```

### 4. Chat Completions (Streaming)

```bash
curl -X POST "http://127.0.0.1:8000/openai/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vanilla-rag",
    "messages": [
      {
        "role": "user",
        "content": "Explain machine learning"
      }
    ],
    "stream": true
  }'
```

---

## API Documentation

Both services provide interactive API documentation:

- **MMU-RAG API**: http://127.0.0.1:8000/docs
- **OpenAI API**: http://127.0.0.1:8002/docs

These Swagger UI interfaces allow you to test all endpoints directly from your browser.
