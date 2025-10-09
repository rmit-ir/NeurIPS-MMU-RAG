#!/usr/bin/env bash
#
# Run run_remote.py on all datasets sequentially with notifications
#

# Logging function that outputs to console and shows macOS notification
log_with_notification() {
    local message="$1"
    local title="${2:-MMU-RAG Script}"
    
    # Print to console with timestamp
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message"
    
    # Show macOS notification
    osascript -e "display notification \"$message\" with title \"$title\""
}

# Script start notification
log_with_notification "Starting MMU-RAG batch processing" "MMU-RAG Batch"

# for topics files:
# ./data/past_topics/processed/trec_rag_2025_queries.jsonl
# ./data/past_topics/processed/IKAT_processed_query.jsonl
# ./data/past_topics/processed/LiveRAG_LCD_Session1_Question_file.jsonl
# ./data/past_topics/processed/sachin-test-collection-queries.jsonl
# ./data/past_topics/processed/topics.rag24.test.jsonl

uv run scripts/run_remote.py mmu_rag_vanilla --topics-file ./data/past_topics/processed/trec_rag_2025_queries.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 5
log_with_notification "finished mmu_rag_vanilla trec 2025"

uv run scripts/run_remote.py decomposition_rag --topics-file ./data/past_topics/processed/trec_rag_2025_queries.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 2
log_with_notification "finished decomposition_rag trec 2025"

uv run scripts/run_remote.py mmu_rag_router_llm --topics-file ./data/past_topics/processed/trec_rag_2025_queries.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 4
log_with_notification "finished mmu_rag_router_llm trec 2025"

uv run scripts/run_remote.py mmu_rag_vanilla --topics-file ./data/past_topics/organizers_outputs/t2t_val.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 5
log_with_notification "finished mmu_rag_vanilla"

uv run scripts/run_remote.py decomposition_rag --topics-file ./data/past_topics/organizers_outputs/t2t_val.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 2
log_with_notification "finished decomposition_rag"

uv run scripts/run_remote.py mmu_rag_router_llm --topics-file ./data/past_topics/organizers_outputs/t2t_val.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 4
log_with_notification "finished mmu_rag_router_llm"

uv run scripts/run_remote.py mmu_rag_vanilla --topics-file ./data/past_topics/processed/IKAT_processed_query.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 5
log_with_notification "finished mmu_rag_vanilla ikat"

uv run scripts/run_remote.py decomposition_rag --topics-file ./data/past_topics/processed/IKAT_processed_query.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 2
log_with_notification "finished decomposition_rag ikat"

uv run scripts/run_remote.py mmu_rag_router_llm --topics-file ./data/past_topics/processed/IKAT_processed_query.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 4
log_with_notification "finished mmu_rag_router_llm ikat"

uv run scripts/run_remote.py mmu_rag_vanilla --topics-file ./data/past_topics/processed/LiveRAG_LCD_Session1_Question_file.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 5
log_with_notification "finished mmu_rag_vanilla live"

uv run scripts/run_remote.py decomposition_rag --topics-file ./data/past_topics/processed/LiveRAG_LCD_Session1_Question_file.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 2
log_with_notification "finished decomposition_rag live"

uv run scripts/run_remote.py mmu_rag_router_llm --topics-file ./data/past_topics/processed/LiveRAG_LCD_Session1_Question_file.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 4
log_with_notification "finished mmu_rag_router_llm live"

uv run scripts/run_remote.py mmu_rag_vanilla --topics-file ./data/past_topics/processed/sachin-test-collection-queries.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 5
log_with_notification "finished mmu_rag_vanilla sachin"

uv run scripts/run_remote.py decomposition_rag --topics-file ./data/past_topics/processed/sachin-test-collection-queries.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 2
log_with_notification "finished decomposition_rag sachin"

uv run scripts/run_remote.py mmu_rag_router_llm --topics-file ./data/past_topics/processed/sachin-test-collection-queries.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 4
log_with_notification "finished mmu_rag_router_llm sachin"

uv run scripts/run_remote.py mmu_rag_vanilla --topics-file ./data/past_topics/processed/topics.rag24.test.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 5
log_with_notification "finished mmu_rag_vanilla rag24"

uv run scripts/run_remote.py decomposition_rag --topics-file ./data/past_topics/processed/topics.rag24.test.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 2
log_with_notification "finished decomposition_rag rag24"

uv run scripts/run_remote.py mmu_rag_router_llm --topics-file ./data/past_topics/processed/topics.rag24.test.jsonl --output-dir ./data/past_topics/inhouse_outputs/ --parallel 4
log_with_notification "finished mmu_rag_router_llm rag24"

# Final completion notification
log_with_notification "All MMU-RAG batch processing completed successfully!" "MMU-RAG Complete"
