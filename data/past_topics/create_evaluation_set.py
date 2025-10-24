# /// script
# dependencies = [
#   "pandas",
# ]
# ///
# one off script to create mmu_t2t_topics.n157.jsonl
from pathlib import Path
import pandas as pd

work_dir = Path('data/past_topics/')
go = work_dir / 'gold_answers'
oo = work_dir / 'organizers_outputs'
dp = work_dir / 'processed'

df_gold_t2t_avail = pd.read_json(
    go/'output_remote_mmu_vanilla_agent_sonnet_t2t_val.jsonl', lines=True)
gold_iids = set(df_gold_t2t_avail['query_id'].tolist())
df_t2t = pd.read_json(oo/'t2t_val.jsonl', lines=True)
df_t2t_filtered = df_t2t[df_t2t['iid'].isin(gold_iids)]
df_t2t_filtered.to_json(dp/'mmu_t2t_topics.n157.jsonl',
                        lines=True, orient='records')


# d = work_dir
# do = work_dir / 'processed'
# do.mkdir(exist_ok=True)

# pd.read_json(d/'LiveRAG_LCD_Session1_Question_file.jsonl', lines=True) \
#   .rename(columns={'id': 'iid', 'question': 'query'}) \
#   .astype({'iid': 'string'}) \
#   .to_json(do/'LiveRAG_LCD_Session1_Question_file.jsonl', lines=True, orient='records')

# pd.read_json(d/'trec_rag_2025_queries.jsonl') \
#   .rename(columns={'id': 'iid', 'narrative': 'query'}) \
#   .astype({'iid': 'string'}) \
#   .to_json(do/'trec_rag_2025_queries.jsonl', lines=True, orient='records')

# pd.read_csv(d/'topics.rag24.test.tsv', sep='\t', names=['iid', 'query']) \
#   .astype({'iid': 'string'}) \
#   .to_json(do/'topics.rag24.test.jsonl', lines=True, orient='records')

# # read sample-test-col-queries/samples.csv
# pd.read_csv(d/'..'/'sample-test-col-queries'/'samples.csv', header=0) \
#   .rename(columns={'id': 'iid', 'question': 'query'}) \
#   .astype({'iid': 'string'}) \
#   .to_json(do/'sachin-test-collection-queries.jsonl', lines=True, orient='records')
