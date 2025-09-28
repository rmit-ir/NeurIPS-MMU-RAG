# Requires transformers>=4.51.0
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc)
    return output


def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True,
                           return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs


def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores


tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Reranker-0.6B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

task = 'Given a web search query, retrieve relevant passages that answer the query'


def run_query_doc(task, query, doc):
    pair = format_instruction(task, query, doc)
    inputs = process_inputs([pair])
    scores = compute_logits(inputs)
    return scores[0]


def run_query_doc_batch(task, query, docs):
    pairs = [format_instruction(task, query, doc) for doc in docs]
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    return scores


print(run_query_doc(task, "What is the capital of China?",
                    "The capital of China is Beijing."))

print(run_query_doc(task, "What is the capital of China?",
                    "The capital of France is Paris."))

print(run_query_doc(task, "What is the capital of China?",
                    "The capital of China is Tianjin."))

print(run_query_doc(task, "What is the capital of China?",
                    "The capital of France is Beijing."))

print(run_query_doc(task, "What is the capital of China?",
                    "The capital of China is Hunan."))

# 0.0011698422022163868, YES!!!! removing false positives
print(run_query_doc(task, "Can we update params to from langchain_aws import ChatBedrock to parse <reasoning> to a separate field so it won't affect our final execution?",
                    """When developing a complex application with a Language Model (LLM), it’s common to specify the desired output format, such as JSON, and designate particular keys for organizing the data.

Let’s consider the chain of thought reasoning method as an illustrative example. In this method, the LLM’s thinking process is represented by distinct stages: “thought” indicates the reasoning process, “action” denotes the subsequent action taken, and “observation” reflects the learning acquired from that action, and so forth. By crafting a prompt that directs the LLM to utilize these specific keywords (thought, action, observation), we can effectively guide its cognitive process.

In this article, we will cover coupling the prompt with a parser that allows for the extraction of text associated with certain keywords from the LLM’s output. This combined approach offers a streamlined means of specifying input for the LLM and accurately interpreting its output.

Most insights I share in Medium have previously been shared in my weekly newsletter, To Data & Beyond.

If you want to be up-to-date with the frenetic world of AI while also feeling inspired to take action or, at the very least, to be well-prepared for the future ahead of us, this is for you.

U+1F3DDSubscribe belowU+1F3DD to become an AI… Read the full blog for free on Medium.

Published via Towards AI

Take our 90+ lesson From Beginner to Advanced LLM Developer Certification: From choosing a project to deploying a working product this is the most comprehensive and practical LLM course out there!

Towards AI has published Building LLMs for Production—our 470+ page guide to mastering LLMs with practical projects and expert insights!


Discover Your Dream AI Career at Towards AI Jobs
Towards AI has built a jobs board tailored specifically to Machine Learning and Data Science Jobs and Skills. Our software searches for live AI jobs each hour, labels and categorises them and makes them easily searchable. Explore over 40,000 live jobs today with Towards AI Jobs!

Note: Content contains the views of the contributing authors and not Towards AI."""))

# queries = ["What is the capital of China?",
#            "Explain gravity",
#            ]

# documents = [
#     "The capital of China is Beijing.",
#     "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
# ]

# pairs = [format_instruction(task, query, doc)
#          for query, doc in zip(queries, documents)]

# # Tokenize the input texts
# inputs = process_inputs(pairs)
# scores = compute_logits(inputs)

# print("scores: ", scores)
