# Modified from vLLM official Github repo
# uv run scripts/test_reranker_vllm.py -i data/past_topics/runs/topics.rag24.test.retrieval.jsonl
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501

import argparse
from typing import Any, Dict, List, TypedDict
import jsonlines
from vllm import LLM

from tools.logging_utils import get_logger

model_name = "Qwen/Qwen3-Reranker-0.6B"
logger = get_logger('run_retrieval')

# What is the difference between the official original version and one
# that has been converted into a sequence classification model?
# Qwen3-Reranker is a language model that doing reranker by using the
# logits of "no" and "yes" tokens.
# It needs to computing 151669 tokens logits, making this method extremely
# inefficient, not to mention incompatible with the vllm score API.
# A method for converting the original model into a sequence classification
# model was proposed. See：https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
# Models converted offline using this method can not only be more efficient
# and support the vllm score API, but also make the init parameters more
# concise, for example.
# llm = LLM(model="tomaarsen/Qwen3-Reranker-0.6B-seq-cls", runner="pooling")

# If you want to load the official original version, the init parameters are
# as follows.


def get_llm() -> LLM:
    """Initializes and returns the LLM model for Qwen3-Reranker."""
    return LLM(
        model=model_name,
        runner="pooling",
        gpu_memory_utilization=0.15,
        max_model_len=16000,  # (the max input that can fit in 24Gx0.15
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )


# Why do we need hf_overrides for the official original version:
# vllm converts it to Qwen3ForSequenceClassification when loaded for
# better performance.
# - Firstly, we need using `"architectures": ["Qwen3ForSequenceClassification"],`
# to manually route to Qwen3ForSequenceClassification.
# - Then, we will extract the vector corresponding to classifier_from_token
# from lm_head using `"classifier_from_token": ["no", "yes"]`.
# - Third, we will convert these two vectors into one vector.  The use of
# conversion logic is controlled by `using "is_original_qwen3_reranker": True`.

# Please use the query_template and document_template to format the query and
# document for better reranker results.

prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

query_template = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
document_template = "<Document>: {doc}{suffix}"

llm = get_llm()


def cut_to_words(text: str, max_words: int) -> str:
    """Cut the text to the first max_words words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words])


def search_result_to_text(r: Dict[str, Any]):
    return f"Web URL: {r.get('url', 'Unknown').strip()}\n\nContent: {r.get('text', '').strip()}\n\n"


def rerank(query: str, docs: List[str], max_words=4_000):
    """
    Limit doc length to max_length words.
    Returns the same length list of scores for the documents based on the query.
    """
    instruction = (
        "Given the web search query, is the retrieved document "
        "(1) from a high quality and relevant website based on the URL, "
        "(2) published recently, and "
        "(3) contains key information that helps answering the query?")

    # length=1
    query_fmt = query_template.format(
        prefix=prefix, instruction=instruction, query=query)
    # length=len(docs)
    docs_fmt = [document_template.format(
        doc=cut_to_words(doc, max_words), suffix=suffix) for doc in docs]

    # Does llm.score support [[1], [n]] inputs?
    outputs = llm.score(query_fmt, docs_fmt)
    scores = [output.outputs.score for output in outputs]
    return scores


class OutputRecord(TypedDict):
    iid: str
    query: str
    docs: List[Dict[str, Any]]


def main() -> None:
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--run-file',
        '-i',
        required=True,
        help='Input JSONL run file under data/past_topics/runs with docs (must have "iid", "query" and "docs" fields)'
    )

    parser.add_argument(
        '--num-docs',
        '-n',
        type=int,
        default=None,
        help='Number of documents to rerank per query (default: all)'
    )

    args = parser.parse_args()

    # Load run
    topics: List[OutputRecord] = []
    with jsonlines.open(args.run_file, 'r') as reader:
        for line_num, topic in enumerate(reader, 1):
            topics.append(OutputRecord(**topic))

    logger.info("Topics loaded successfully", topics_count=len(
        topics), run_file=args.run_file)

    # For each topic, rerank the documents, finally save to same filename .rerank.jsonl
    for topic in topics:
        docs = topic['docs']
        doc_texts = [search_result_to_text(doc) for doc in docs if 'text' in doc and doc['text']]
        if len(doc_texts) == 0:
            logger.warning("No valid documents to rerank",
                           topic_id=topic['iid'])
            continue
        if args.num_docs is not None:
            doc_texts = doc_texts[:args.num_docs]
        scores = rerank(topic['query'], doc_texts)
        for doc, score in zip(docs, scores):
            doc['score'] = score
        # Sort by score descending
        docs.sort(key=lambda x: x['score'], reverse=True)
        topic['docs'] = docs
    # Save to same filename .rerank.jsonl
    output_file = args.run_file.replace('retrieval.jsonl', 'rerank.jsonl')
    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(topics)


if __name__ == "__main__":
    main()
