# Scripts Directory

## Scripts

### `run.py`

For running RAG systems using their class name with support for parallel processing and `__init__` parameters.

```bash
uv run scripts/run.py <SYSTEM_CLASS_NAME> --topics-file <INPUT_FILE> --output-dir <OUTPUT_DIR> [OPTIONS]
```

#### Required Arguments

- `SYSTEM_CLASS_NAME`: Name of the RAG system class to use (e.g., `AzureO3ResearchRAG`, `PerplexityResearchRAG`)
- `--topics-file`: Path to input JSONL file containing topics with `iid` and `query` fields
- `--output-dir`: Directory where output results will be saved

#### Optional Arguments

- `--parallel <N>`: Number of parallel requests to process simultaneously (default: 1)
- Additional system-specific parameters can be passed as `--key value` pairs, these get forwarded to the system's `__init__` method.

#### Examples

**Basic Usage with Azure O3 Research:**

```bash
export AZURE_API_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com \
export AZURE_API_KEY=your-api-key \
uv run scripts/run.py AzureO3ResearchRAG \
  --topics-file data/past_topics/processed/sachin-test-collection-queries.jsonl \
  --output-dir data/past_topics/commercial_outputs \
  --parallel 10
```

**Using Perplexity Research with Custom Model:**

```bash
export PERPLEXITY_API_KEY=pplx-...
uv run scripts/run.py PerplexityResearchRAG \
  --topics-file data/topics.jsonl \
  --output-dir data/runs/ \
  --parallel 5 \
  --model sonar-pro
```

**Using Perplexity Deep Research:**

```bash
export PERPLEXITY_API_KEY=pplx-...
# custom parameter --model
uv run scripts/run.py PerplexityResearchRAG \
  --topics-file data/topics.jsonl \
  --output-dir data/runs/ \
  --model sonar-deep-research
```

#### Input Format

The input topics file must be in JSONL format with each line containing:

```json
{"iid": "unique_identifier", "query": "your research question"}
```

Example:

```json
{"iid": "1", "query": "how humanity unleashed a flood of new diseases"}
{"iid": "2", "query": "what governments do to promote sustainable growth"}
```

#### Output Format

Results are saved in JSONL format with each line containing:

```json
{
  "query_id": "unique_identifier",
  "generated_response": "detailed response from the RAG system",
  "citations": ["https://list", "https://of", "https://citation", "https://URLs"]
}
```
