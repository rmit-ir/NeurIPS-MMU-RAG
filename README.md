# MMU-RAG @ NeurIPS 2025

This project is for the MMU-RAG challenge at NeurIPS 2025.

## 🎯 Purpose

Build RAG systems that can:

1. **Dynamic Evaluation**: Integrate with the Ragent Arena via `/run` endpoint
2. **Static Evaluation**: Support batch evaluation via `/evaluate` endpoint
3. **OpenAI Compatibility**: Support OpenAI-compatible API endpoints for ASE 2.0 website and OpenWebUI integration

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
  - Install via curl:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### Installation

```bash
# Clone the repository
git clone https://github.com/rmit-ir/NeurIPS-MMU-RAG
cd NeurIPS-MMU-RAG
```

### Running the Server

The project includes a combined API server that provides both MMU-RAG challenge endpoints and OpenAI-compatible endpoints:

```bash
# Run the combined API server
uv run fastapi run src/apis/combined_app.py

# The server will be available at:
# - http://localhost:8000 (default FastAPI port)
# - MMU-RAG endpoints: /run, /evaluate
# - OpenAI-compatible endpoints: /v1/chat/completions, etc.

# Development mode with auto-reload
uv run fastapi dev src/apis/combined_app.py
```

### Docker Deployment

```bash
# Build the Docker image
docker build -t mmu-rag .

# Run the container
docker run -p 8000:8000 mmu-rag

# Access the server at http://localhost:8000
```

## 📁 What's Included

### RAG Implementation Templates (`src/tools/`)

- `pipeline.py` - Main RAG pipeline orchestration
- `loader.py` - Document loading from various formats
- `cleaner.py` - Text preprocessing and normalization
- `tokenizer.py` - Text tokenization using HuggingFace
- `chunker.py` - Document chunking with overlap
- `indexer.py` - FAISS vector index creation
- `retriever.py` - Semantic search and retrieval
- `generator.py` - Answer generation using LLMs

### Testing & Validation

- `local_test.py` - Comprehensive test runner for RAG system compliance

```bash
# Test both endpoints (full test)
python local_test.py --base-url http://localhost:5010

# Test only dynamic evaluation (/run endpoint)
python local_test.py --base-url http://localhost:5010 --test-mode run

# Test only static evaluation (/evaluate endpoint)
python local_test.py --base-url http://localhost:5010 --test-mode evaluate

# Custom validation file
python local_test.py --base-url http://localhost:5010 \
    --validation-file custom_val.jsonl \
    --test-question "What is machine learning?"
```

## 📋 Requirements Specification

### Dynamic Evaluation (`/run` endpoint)

- **Input**: `{"question": "string"}`
- **Output**: SSE stream with JSON objects containing:
  - `intermediate_steps`: Reasoning process or the retrieved passage information (markdown formatted)
  - `final_report`: Final answer (markdown formatted)
  - `is_intermediate`: Boolean flag
  - `citations`: Array of source references
  - `complete`: Completion signal

### Static Evaluation (`/evaluate` endpoint)

- **Input**: `{"query": "string", "iid": "string"}`
- **Output**: `{"query_id": "string", "generated_response": "string"}`
- **File Output**: Must generate `result.jsonl` with all responses

## 🔧 Creating a New RAG System

The project uses a modular architecture where you can easily create new RAG systems by implementing the `RAGInterface`.

### Step 1: Create Your RAG System

Create a new directory under `src/systems/` for your RAG system:

```bash
cd src/systems
mkdir my_rag_system
```

### Step 2: Implement the RAG Interface

Create your RAG system by extending the `RAGInterface` class, check `src/systems/vanilla_agent/vanilla_rag.py` for example.

### Step 3: Register Your RAG System

- mmu_rag_router.py only supports one RAG system at a time. Change variable `rag_system` in `src/apis/mmu_rag_router.py` to your new RAG class.
- openai_router.py supports multiple RAG systems, just add yours to the `rag_systems` dictionary.

### Step 4: Use Existing Tools

Leverage the provided tools in `src/tools/` for common RAG operations:

### Step 5: Test Your System

Send cURL requests to test your system, check apis/README.md for details.

Or use the provided test runner to validate basics of your implementation:

```bash
# Test your RAG system
python local_test.py --base-url http://localhost:8000
```

## 📚 Additional Resources

- **MMU-RAG Challenge**: [Official Challenge Details](https://agi-lti.github.io/MMU-RAgent/text-to-text)
