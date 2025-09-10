# RAG System Starter Code for DeepResearch Comparator

This starter code provides templates and testing tools to help you build a RAG (Retrieval-Augmented Generation) system.

## 🎯 Purpose

Build RAG systems that can:

1. **Dynamic Evaluation**: Integrate with the Ragent Arena via `/run` endpoint
2. **Static Evaluation**: Support batch evaluation via `/evaluate` endpoint

## 📁 What's Included

### RAG Implementation Templates (`src/`)

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
