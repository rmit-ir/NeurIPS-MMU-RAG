#!/usr/bin/env python3
import pandas as pd

# this is for adding query field to gold answers (which is just results from a bigger model)
# the evaluation script expects the query field to be present
# 

# Load topics and create query mapping
benchmark_topics = pd.read_json('data/past_topics/processed/benchmark_topics.jsonl', lines=True)
t2t_topics = pd.read_json('data/past_topics/processed/mmu_t2t_topics.n157.jsonl', lines=True)

# Load reference files
benchmark_ref = pd.read_json('data/past_topics/gold_answers/output_benchmark_topics.gold.jsonl', lines=True)
t2t_ref = pd.read_json('data/past_topics/gold_answers/output_mmu_t2t_topics.n157.gold.jsonl', lines=True)

# Merge to add query field (topics use 'iid', reference files use 'query_id')
benchmark_merged = benchmark_ref.merge(benchmark_topics[['iid', 'query']], left_on='query_id', right_on='iid', how='left')
t2t_merged = t2t_ref.merge(t2t_topics[['iid', 'query']], left_on='query_id', right_on='iid', how='left')

# Save updated reference files
benchmark_merged.to_json('data/past_topics/gold_answers/output_benchmark_topics.gold.with_query.jsonl', orient='records', lines=True)
t2t_merged.to_json('data/past_topics/gold_answers/output_mmu_t2t_topics.n157.gold.with_query.jsonl', orient='records', lines=True)
