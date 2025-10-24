# MMU-RAG @ NeurIPS 2025

This project is for the MMU-RAG challenge at NeurIPS 2025.

## Purpose

Build RAG systems that can:

1. **Dynamic Evaluation**: Integrate with the Ragent Arena via `/run` endpoint
2. **Static Evaluation**: Support batch evaluation via `/evaluate` endpoint
3. **OpenAI Compatibility**: Support OpenAI-compatible API endpoints for ASE 2.0 website and OpenWebUI integration

## Quick Start

### Prerequisites

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

The project provides separate API servers for different use cases:

```bash
# Run the MMU-RAG Challenge API server
uv run fastapi run src/apis/mmu_rag_router.py

# Run the OpenAI-compatible API server
uv run fastapi run src/apis/openai_router.py

# The servers will be available at:
# - MMU-RAG endpoints: http://localhost:8000 (/run, /evaluate)
# - OpenAI-compatible endpoints: http://localhost:8000 (/v1/chat/completions, etc.)

# Development mode with auto-reload
uv run fastapi dev src/apis/mmu_rag_router.py
# or
uv run fastapi dev src/apis/openai_router.py
```

### Docker Deployment - MMU-RAG Submission Server

Required environment variables:

```bash
CLUEWEB_API_KEY=mmu_provided_clueweb_api_key
```

Example command with Docker:

```bash
docker run --rm -it --gpus all -p 5025:5025 970547356481.dkr.ecr.us-east-1.amazonaws.com/neurips2025text/rmit-adms_ir:latest
```

Notes:

1. 24G vRAM GPU required.
2. Network access required for ClueWeb-22 search API and downloading LLMs for local inference.

### Docker Deployment - OpenAI Server

See `cloud/openai_server/docker-compose.yaml` for example.

## Additional Information

<details>
<summary>What's Included</summary>

### What's Included

#### RAG Implementation Templates (`src/tools/`)

- `pipeline.py` - Main RAG pipeline orchestration
- `loader.py` - Document loading from various formats
- `cleaner.py` - Text preprocessing and normalization
- `tokenizer.py` - Text tokenization using HuggingFace
- `chunker.py` - Document chunking with overlap
- `indexer.py` - FAISS vector index creation
- `retriever.py` - Semantic search and retrieval
- `generator.py` - Answer generation using LLMs
- `web_search.py` - FineWeb & ClueWeb-22 web search utility (base64 JSON decoding)

#### Testing & Validation

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
</details>

<details>
<summary>Requirements Specification</summary>

### Requirements Specification

#### Dynamic Evaluation (`/run` endpoint)

- **Input**: `{"question": "string"}`
- **Output**: SSE stream with JSON objects containing:
  - `intermediate_steps`: Reasoning process or the retrieved passage information (markdown formatted)
  - `final_report`: Final answer (markdown formatted)
  - `is_intermediate`: Boolean flag
  - `citations`: Array of source references
  - `complete`: Completion signal

#### Static Evaluation (`/evaluate` endpoint)

- **Input**: `{"query": "string", "iid": "string"}`
- **Output**: `{"query_id": "string", "generated_response": "string"}`
- **File Output**: Must generate `result.jsonl` with all responses

</details>

<details>
<summary>Creating a New RAG System</summary>

### Creating a New RAG System

The project uses a modular architecture where you can easily create new RAG systems by implementing the `RAGInterface`.

#### Step 1: Create Your RAG System

Create a new directory under `src/systems/` for your RAG system:

```bash
cd src/systems
mkdir my_rag_system
```

#### Step 2: Implement the RAG Interface

Create your RAG system by extending the `RAGInterface` class, check `src/systems/vanilla_agent/vanilla_rag.py` for example.

#### Step 3: Register Your RAG System

- mmu_rag_router.py only supports one RAG system at a time. Change variable `rag_system` in `src/apis/mmu_rag_router.py` to your new RAG class.
- openai_router.py supports multiple RAG systems, just add yours to the `rag_systems` dictionary.

#### Step 4: Use Existing Tools

Leverage the provided tools in `src/tools/` for common RAG operations:

#### Step 5: Test Your System

Send cURL requests to test your system, check apis/README.md for details.

Or use the provided test runner to validate basics of your implementation:

```bash
# Test your RAG system
python local_test.py --base-url http://localhost:8000
```

</details>

<details>
<summary>Additional Resources</summary>

### Additional Resources

- **MMU-RAG Challenge**: [Official Challenge Details](https://agi-lti.github.io/MMU-RAgent/text-to-text)

</details>

<details>
<summary>Development Tips</summary>
Launch the LLM and reranker on the background and run your RAG system locally for faster development.

```bash
# Launch LLM server
uv run python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-4B --reasoning-parser qwen3 --gpu-memory-utilization 0.75 --max-model-len 25000 --kv-cache-memory-bytes 8589934592 --max-num-seqs 5 --host 0.0.0.0 --port 8088

# Launch reranker server
uv run python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-Reranker-0.6B --gpu-memory-utilization 0.2 --max-model-len 16000 --kv-cache-memory-bytes 3221225472 --max-num-seqs 3 --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}' --host 0.0.0.0 --port 8087
```

</details>

<details>
<summary>Evaluation Guideline</summary>

Queries dataset 1:

- Topics: [data/past_topics/processed/mmu_t2t_topics.n157.jsonl](./data/past_topics/processed/mmu_t2t_topics.n157.jsonl), from MMU RAG organizers [MMU-RAG Validation Set](https://agi-lti.github.io/MMU-RAGent/text-to-text#validation-set), this is a subset of 157 queries that we successfully generated gold answers.
- Gold answers: [data/past_topics/gold_answers/output_mmu_t2t_topics.n157.gold.jsonl](./data/past_topics/gold_answers/output_mmu_t2t_topics.n157.gold.jsonl)

Queries dataset 2:

- Topics: [data/past_topics/processed/benchmark_topics.jsonl](./data/past_topics/processed/benchmark_topics.jsonl), built from 20 queries from each of mmu_t2t, IKAT, LiveRAG, RAG24, RAG25, in total 100 queries.
- Gold answers: [data/past_topics/gold_answers/output_benchmark_topics.gold.jsonl](./data/past_topics/gold_answers/output_benchmark_topics.gold.jsonl)

#### Step 1. Run Dataset

Run the dataset through a RAG system, e.g.,

```bash
REMOTE_API_KEY=your_api_key_copy_it_from_ase_2.0_website_api_request
bash scripts/run_datasets.sh
```

#### Step 2. Evaluate Results

Take the generated results, and evaluate them using `src.evaluators.deepresearch_evaluators.combined_deepresearch_evaluator.CombinedDeepResearchEvaluator`.

```bash
uv run scripts/evaluate.py \
  --evaluator src.evaluators.deepresearch_evaluators.combined_deepresearch_evaluator.CombinedDeepResearchEvaluator \
  --results <results.jsonl> \
  --reference ./data/past_topics/gold_answers/mmu_t2t_topics.jsonl \
  --output-dir data/evaluation_results/with_gold \
  --output-prefix t2t_rag_name \
  --num-threads 8
```

TODO: add a `scripts/run_evaluation.sh` script to automate both steps. When run_datasets.sh finishes, output export statements that will determine what run_evaluation.sh will pick up.

</details>
