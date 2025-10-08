# MMU-RAG Evaluators

## Overview

This directory contains evaluators for assessing RAG (Retrieval-Augmented Generation) system performance, following the modular design pattern from the G-RAG-LiveRAG project.

## Available Evaluators

### 🧠 DeepEvalEvaluator
**Path:** `src.evaluators.deepeval_evaluator.evaluator.DeepEvalEvaluator`

Semantic evaluation using the DeepEval framework with custom LLM integration:
- **Faithfulness**: How factually consistent the answer is with the provided context
- **Answer Relevancy**: How relevant the answer is to the original question
- **Contextual Relevancy**: How relevant the retrieved context is to the question

**Special Features:**
- Custom MMU LLM integration with reasoning token handling
- Automatic JSON extraction from markdown code blocks
- Configurable evaluation thresholds per metric

**Parameters:**
- `model`: Model name for evaluation (default: "gpt-4o-mini")
- `base_url`: Base URL for the LLM API (default: "https://mmu-proxy-server-llm-proxy.rankun.org/v1")
- `api_key`: API key for authentication
- `threshold`: Minimum score threshold for passing evaluation (default: 0.5)
- `include_faithfulness`: Enable faithfulness metric (default: True)
- `include_answer_relevancy`: Enable answer relevancy metric (default: True)
- `include_contextual_relevancy`: Enable contextual relevancy metric (default: True)
- `verbose`: Enable verbose logging (default: False)

### 🤖 LLMEvaluator
**Path:** `src.evaluators.llm_evaluator.evaluator.LLMEvaluator`

LLM-powered evaluation using Claude Sonnet 3.5 through the MMU PROXY Router server:
- **Relevance**: How well the answer addresses the question (4-point scale: -1 to 2)
- **Faithfulness**: How well the answer is grounded in retrieved documents (3-point scale: -1 to 1)

**Special Features:**
- Detailed evaluation notes explaining scoring rationale
- Gold reference answer integration for enhanced accuracy
- Concurrent multi-threaded processing
- Combined prompt system for efficient evaluation
- Comprehensive debug logging

**Parameters:**
- `model_id`: Model ID for evaluation (default: "openai.gpt-oss-20b-1:0")
- `temperature`: LLM temperature setting (default: 0.0)
- `max_tokens`: Maximum tokens to generate (default: 2048)
- `use_gold_references`: Include gold reference answers (default: True)
- `num_threads`: Number of concurrent threads (default: 1)
- `answer_word_limit`: Maximum words in answer (default: 300)
- `api_base`: MMU proxy server URL (default: "https://mmu-proxy-server-llm-proxy.rankun.org")
- `api_key`: API key for authentication (or use `MMU_OPENAI_API_KEY` env var)
- `combined_prompt`: Use combined evaluation prompt (default: True)

### 📈 RAGASEvaluator
**Path:** `src.evaluators.ragas_evaluator.evaluator.RAGASEvaluator`

Semantic evaluation using the RAGAS framework with LiteLLM integration:
- **Faithfulness**: How factually consistent the answer is with context
- **Answer Relevancy**: How relevant the answer is to the original question
- **Answer Correctness**: How accurate the answer is compared to the reference

**Parameters:**
- `model_name`: LiteLLM model name (default: "openai.gpt-oss-20b-1:0")
- `api_key`: API key for LiteLLM proxy
- `base_url`: Base URL for LiteLLM proxy (default: "https://mmu-proxy-server-llm-proxy.rankun.org/v1")
- `embedding_model`: HuggingFace embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")
- `include_faithfulness`: Enable faithfulness metric (default: True)
- `include_answer_relevancy`: Enable answer relevancy metric (default: True)
- `include_answer_correctness`: Enable answer correctness metric (default: True)
- `cache_dir`: Directory for caching results (default: "/tmp/ragas_cache")

### 📊 NLPMetricsEvaluator
**Path:** `src.evaluators.nlp_metrics.evaluator.NLPMetricsEvaluator`

Traditional NLP metrics for surface-level text similarity:
- **ROUGE-L**: Longest common subsequence overlap with reference
- **BLEU**: N-gram precision similarity with reference

**Parameters:**
- `include_rouge_l`: Enable ROUGE-L calculation (default: True)
- `include_bleu`: Enable BLEU calculation (default: True)
- `use_stemmer`: Use stemming for ROUGE-L (default: True)

## Quick Start

### Basic Usage

```bash
# DeepEval evaluation
python scripts/evaluate.py \\
    --evaluator DeepEvalEvaluator \\
    --results data/system_outputs.jsonl \\
    --reference data/references.jsonl \\
    --model gpt-4o-mini \\
    --api-key sk-or-v1-your-key

# LLM evaluation
python scripts/evaluate.py \\
    --evaluator LLMEvaluator \\
    --results data/system_outputs.jsonl \\
    --reference data/references.jsonl \\
    --output-dir data/evaluation_results \\
    --output-prefix llm_eval

# RAGAS evaluation
python scripts/evaluate.py \\
    --evaluator RAGASEvaluator \\
    --results data/system_outputs.jsonl \\
    --reference data/references.jsonl \\
    --model-name openai/gpt-oss-20b-1:0 \\
    --api-key sk-or-v1-your-key

# NLP metrics evaluation  
python scripts/evaluate.py \\
    --evaluator NLPMetricsEvaluator \\
    --results data/system_outputs.jsonl \\
    --reference data/references.jsonl
```

### Advanced Usage

```bash
# DeepEval with specific metrics only
python scripts/evaluate.py \\
    --evaluator DeepEvalEvaluator \\
    --results data/outputs.jsonl \\
    --reference data/refs.jsonl \\
    --api-key sk-or-v1-your-key \\
    --no-include-contextual-relevancy  # Only faithfulness and answer relevancy

# LLM evaluation with custom parameters
python scripts/evaluate.py \\
    --evaluator LLMEvaluator \\
    --results data/outputs.jsonl \\
    --reference data/refs.jsonl \\
    --output-dir data/evaluation_results \\
    --output-prefix custom_llm_eval \\
    --num-threads 4 \\
    --answer-word-limit 500 \\
    --temperature 0.1

# RAGAS with specific metrics only
python scripts/evaluate.py \\
    --evaluator RAGASEvaluator \\
    --results data/outputs.jsonl \\
    --reference data/refs.jsonl \\
    --api-key sk-or-v1-your-key \\
    --no-include-answer-correctness  # Only faithfulness and answer relevancy

# NLP metrics with custom options
python scripts/evaluate.py \\
    --evaluator NLPMetricsEvaluator \\
    --results data/outputs.jsonl \\
    --reference data/refs.jsonl \\
    --no-use-stemmer  # Disable stemming

# Custom output location
python scripts/evaluate.py \\
    --evaluator DeepEvalEvaluator \\
    --results data/outputs.jsonl \\
    --reference data/refs.jsonl \\
    --api-key sk-or-v1-your-key \\
    --output-dir my_results \\
    --output-prefix my_eval \\
    --output-format tsv
```

## Input Data Format

### System Outputs (JSONL)
```json
{
  "query_id": "1",
  "generated_response": "Paris is the capital of France.",
  "citations": ["France_Wikipedia"]
}
```

### Reference Data (JSONL)
```json
{
  "iid": "1", 
  "query": "What is the capital of France?",
  "reference": "The capital of France is Paris."
}
```

## Output Files

The evaluation generates two files:

1. **Aggregated Results** (`*.aggregated.jsonl`): Summary metrics and metadata
2. **Row-level Results** (`*.rows.jsonl`): Per-item scores (if available)

### Example Aggregated Output
```json
{
  "metrics": {
    "mean_faithfulness": 0.8633,
    "std_faithfulness": 0.0623,
    "mean_answer_relevancy": 0.9124,
    "std_answer_relevancy": 0.0456,
    "mean_contextual_relevancy": 0.7892,
    "std_contextual_relevancy": 0.0789,
    "mean_rouge_l": 0.0687,
    "std_rouge_l": 0.0200
  },
  "evaluator_name": "DeepEvalEvaluator",
  "sample_count": 5,
  "timestamp": "2025-09-22T00:24:31.351107",
  "total_time_ms": 48234.5
}
```

## Creating New Evaluators

To create a new evaluator:

1. **Create a new directory** under `src/evaluators/your_evaluator/`
2. **Implement the interface** by subclassing `EvaluatorInterface`
3. **Define your `evaluate` method** that returns an `EvaluationResult`

### Template

```python
from src.evaluators.evaluator_interface import EvaluatorInterface, EvaluationResult

class MyEvaluator(EvaluatorInterface):
    def __init__(self, param1: str = "default"):
        \"\"\"
        Args:
            param1: Description of parameter 1
        \"\"\"
        self.param1 = param1
    
    @property
    def name(self) -> str:
        return "MyEvaluator"
    
    def evaluate(self, system_outputs, references) -> EvaluationResult:
        # Your evaluation logic here
        metrics = {"my_metric": 0.85}
        
        return EvaluationResult(
            metrics=metrics,
            evaluator_name=self.name,
            sample_count=len(system_outputs)
        )
```

## Help and Documentation

Use `--help` with any evaluator to see its specific parameters:

```bash
python scripts/evaluate.py --evaluator DeepEvalEvaluator --help
python scripts/evaluate.py --evaluator RAGASEvaluator --help
python scripts/evaluate.py --evaluator NLPMetricsEvaluator --help
```

The script automatically extracts parameter documentation from the evaluator's `__init__` method docstring.