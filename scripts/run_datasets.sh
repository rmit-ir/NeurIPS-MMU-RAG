#!/usr/bin/env bash
#
# Run run_remote.py on all datasets sequentially with notifications
# 
# Usage
# export REMOTE_API_KEY=copy the Bearer token from your browser API request
# bash scripts/run_datasets.sh
#

# Logging function that outputs to console and shows macOS notification
log_with_notification() {
    local message="$1"
    local title="${2:-MMU-RAG Script}"
    

    # Print to console with timestamp
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message"

    # Show macOS notification only if osascript is available
    if command -v osascript >/dev/null 2>&1; then
        # Show macOS notification
        osascript -e "display notification \"$message\" with title \"$title\""
    fi

}

# Function to get parallel count for a system
get_parallel_count() {
    case "$1" in
        "mmu_rag_vanilla") echo 5 ;;
        "decomposition_rag") echo 2 ;;
        "mmu_rag_router_llm") echo 4 ;;
        "mmu_vanilla_agent") echo 4 ;;
        *) echo 1 ;;
    esac
}

# Script start notification
log_with_notification "Starting MMU-RAG batch processing" "MMU-RAG Batch"

# Define systems
SYSTEMS=(
    # "mmu_rag_vanilla"
    # "decomposition_rag"
    # "mmu_rag_router_llm"
    "mmu_vanilla_agent"
)

# Define datasets with format: "path|display_name"
DATASETS=(
    "./data/past_topics/processed/trec_rag_2025_queries.jsonl|trec 2025"
    "./data/past_topics/organizers_outputs/t2t_val.jsonl|t2t_val"
    "./data/past_topics/processed/IKAT_processed_query.jsonl|ikat"
    "./data/past_topics/processed/LiveRAG_LCD_Session1_Question_file.jsonl|live"
    "./data/past_topics/processed/sachin-test-collection-queries.jsonl|sachin"
    "./data/past_topics/processed/topics.rag24.test.jsonl|rag24"
)

# Common output directory
OUTPUT_DIR="./data/past_topics/inhouse_outputs/"

# Loop through all datasets and systems
for dataset_entry in "${DATASETS[@]}"; do
    # Extract path and display name using parameter expansion
    dataset_path="${dataset_entry%|*}"
    display_name="${dataset_entry#*|}"
    
    for system in "${SYSTEMS[@]}"; do
        parallel_count=$(get_parallel_count "$system")
        
        # Run the command
        uv run scripts/run_remote.py "$system" \
            --topics-file "$dataset_path" \
            --output-dir "$OUTPUT_DIR" \
            --parallel "$parallel_count"
        
        # Log completion
        log_with_notification "finished $system $display_name"
    done
done

# Final completion notification
log_with_notification "All MMU-RAG batch processing completed successfully!" "MMU-RAG Complete"
