# /// script
# dependencies = [
#   "pandas",
# ]
# ///
from pathlib import Path
import pandas as pd

work_dir = Path('data/past_topics/')
d = work_dir
do = work_dir / 'processed'
do.mkdir(exist_ok=True)

pd.read_json(d/'LiveRAG_LCD_Session1_Question_file.jsonl', lines=True) \
  .rename(columns={'id': 'iid', 'question': 'query'}) \
  .astype({'iid': 'string'}) \
  .to_json(do/'LiveRAG_LCD_Session1_Question_file.jsonl', lines=True, orient='records')

pd.read_json(d/'trec_rag_2025_queries.jsonl') \
  .rename(columns={'id': 'iid', 'narrative': 'query'}) \
  .astype({'iid': 'string'}) \
  .to_json(do/'trec_rag_2025_queries.jsonl', lines=True, orient='records')

pd.read_csv(d/'topics.rag24.test.tsv', sep='\t', names=['iid', 'query']) \
  .astype({'iid': 'string'}) \
  .to_json(do/'topics.rag24.test.jsonl', lines=True, orient='records')

# read sample-test-col-queries/samples.csv
pd.read_csv(d/'..'/'sample-test-col-queries'/'samples.csv', header=0) \
  .rename(columns={'id': 'iid', 'question': 'query'}) \
  .astype({'iid': 'string'}) \
  .to_json(do/'sachin-test-collection-queries.jsonl', lines=True, orient='records')
