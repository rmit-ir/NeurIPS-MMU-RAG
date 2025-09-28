# Copied from vLLM official Github repo
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
        max_model_len=10000,
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


def rerank(query: str, docs: List[str]):
    """
    Returns the same length list of scores for the documents based on the query.
    """
    instruction = (
        "Given a web search query, retrieve relevant passages that answer the query")

    queries = [query_template.format(
        prefix=prefix, instruction=instruction, query=query)]
    documents = [document_template.format(
        doc=doc, suffix=suffix) for doc in docs]

    llm = get_llm()
    outputs = llm.score(queries, documents)
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
        if args.num_docs is not None:
            docs = docs[:args.num_docs]
        doc_texts = [doc['text'] for doc in docs]
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
