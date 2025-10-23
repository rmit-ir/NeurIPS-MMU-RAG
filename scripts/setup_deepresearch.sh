#!/bin/bash
# Simple script to set up deepresearch_benchmarking evaluation for MMU-RAG results
# Not ready for use yet.

set -e

# Configuration - modify these paths as needed
MMU_RESULTS_FILE="./data/past_topics/commercial_outputs/output_sample.jsonl"
TOPICS_FILE="./data/past_topics/organizers_outputs/t2t_val.jsonl"
SYSTEM_NAME="mmu_rag_vanilla"
EVAL_DIR="/tmp/deepresearch_eval"

echo "🚀 Setting up deepresearch evaluation..."
echo "Results file: $MMU_RESULTS_FILE"
echo "Topics file: $TOPICS_FILE"
echo "System name: $SYSTEM_NAME"
echo "Eval directory: $EVAL_DIR"

# Check if OpenAI API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo "Please set it with: export OPENAI_API_KEY='your-key-here'"
    exit 1
fi

# Create evaluation directory
mkdir -p "$EVAL_DIR"
cd "$EVAL_DIR"

# Clone deepresearch repo if it doesn't exist
if [ ! -d "deepresearch_benchmarking" ]; then
    echo "📥 Cloning deepresearch_benchmarking repository..."
    git clone https://github.com/cxcscmu/deepresearch_benchmarking.git
else
    echo "📁 Repository already exists, pulling latest changes..."
    cd deepresearch_benchmarking
    git pull
    cd ..
fi

# Create reports directory structure
REPORTS_DIR="$EVAL_DIR/deepresearch_benchmarking/reports/$SYSTEM_NAME"
mkdir -p "$REPORTS_DIR"

echo "📝 Converting MMU-RAG results to deepresearch format..."

# Convert MMU-RAG results to deepresearch format using Python one-liner
python3 << EOF
import json
import os

# Load topics (query_id -> query mapping)
topics = {}
with open('$TOPICS_FILE', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        query_id = data.get('iid', data.get('query_id', data.get('id')))
        topics[query_id] = data.get('query', '')

# Convert MMU results
converted = 0
with open('$MMU_RESULTS_FILE', 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        query_id = data.get('query_id', data.get('iid', data.get('id')))
        response = data.get('generated_response', '')
        
        if query_id and response:
            # Write .a file (answer)
            with open(f'$REPORTS_DIR/{query_id}.a', 'w') as af:
                af.write(response)
            
            # Write .q file (query) 
            query_text = topics.get(query_id, f'Query {query_id}')
            with open(f'$REPORTS_DIR/{query_id}.q', 'w') as qf:
                qf.write(query_text)
            
            converted += 1

print(f"✅ Converted {converted} query-answer pairs")
EOF

# Create keys.env file for deepresearch scripts
echo "🔑 Creating keys.env file..."
cd deepresearch_benchmarking
echo "OPENAI_API_KEY=$OPENAI_API_KEY" > keys.env

echo "✅ Setup complete!"
echo ""
echo "📂 Files converted to: $REPORTS_DIR"
echo "🔧 To run evaluations, use the run_evaluation.sh script or run manually:"
echo "   cd $EVAL_DIR/deepresearch_benchmarking"
echo "   python eval_citation_async.py --subfolder $SYSTEM_NAME --open_ai_model gpt-4o-mini"
echo "   python eval_quality_async.py --subfolder $SYSTEM_NAME --open_ai_model gpt-4o-mini"
echo "   python eval_kpr_async.py --subfolder $SYSTEM_NAME --open_ai_model gpt-4o-mini"
