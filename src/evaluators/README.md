# MMU-RAG Evaluators

## Overview

This directory contains evaluators for assessing RAG (Retrieval-Augmented Generation) system performance, following the modular design pattern from the G-RAG-LiveRAG project.

## Available Evaluators

### 🧠 RAGASEvaluator 
**Path:** `src.evaluators.ragas_evaluator.evaluator.RAGASEvaluator`

Semantic evaluation using the RAGAS framework:
- **Faithfulness**: How factually consistent the answer is with context
- **Context Precision**: How relevant retrieved context is to the question

**Parameters:**
- `model_name`: Model for evaluation (default: openai/gpt-4o-mini)
- `api_key`: API key for model provider 
- `include_faithfulness`: Enable faithfulness metric (default: True)
- `include_context_precision`: Enable context precision metric (default: True)

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
# RAGAS evaluation
python scripts/evaluate.py \\
    --evaluator RAGASEvaluator \\
    --results data/system_outputs.jsonl \\
    --reference data/references.jsonl \\
    --model-name openai/gpt-4o-mini \\
    --api-key sk-or-v1-your-key

# NLP metrics evaluation  
python scripts/evaluate.py \\
    --evaluator NLPMetricsEvaluator \\
    --results data/system_outputs.jsonl \\
    --reference data/references.jsonl
```

### Advanced Usage

```bash
# RAGAS with specific metrics only
python scripts/evaluate.py \\
    --evaluator RAGASEvaluator \\
    --results data/outputs.jsonl \\
    --reference data/refs.jsonl \\
    --api-key sk-or-v1-your-key \\
    --no-include-context-precision  # Only faithfulness

# NLP metrics with custom options
python scripts/evaluate.py \\
    --evaluator NLPMetricsEvaluator \\
    --results data/outputs.jsonl \\
    --reference data/refs.jsonl \\
    --no-use-stemmer  # Disable stemming

# Custom output location
python scripts/evaluate.py \\
    --evaluator RAGASEvaluator \\
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
    "mean_rouge_l": 0.0687,
    "std_rouge_l": 0.0200
  },
  "evaluator_name": "RAGASEvaluator",
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
python scripts/evaluate.py --evaluator RAGASEvaluator --help
python scripts/evaluate.py --evaluator NLPMetricsEvaluator --help
```

The script automatically extracts parameter documentation from the evaluator's `__init__` method docstring.