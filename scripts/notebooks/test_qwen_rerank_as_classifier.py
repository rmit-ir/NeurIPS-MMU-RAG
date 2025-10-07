from vllm import LLM

llm = LLM(model="Qwen/Qwen3-Reranker-0.6B",
          runner="pooling",
          max_model_len=20000,
          kv_cache_memory_bytes=5 * 1024 * 1024 * 1024,
          hf_overrides={
              "architectures": ["Qwen3ForSequenceClassification"],
              "classifier_from_token": ["no", "yes"],
              "is_original_qwen3_reranker": True,
          })

prefix = '<|im_start|>system\nJudge whether the user query is a complex query or not. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

instruction = (
    "Given the question below, decide if it can be directly answered by retrieving relevant documents (simple query)."
    "Answer “Yes” if it can, or “No” if it requires decomposition into multiple sub-questions."
)


def predict(query: str) -> str:
    query_template = f"{prefix}<Instruct>: {instruction}\n<Query>: {query}\n{suffix}"
    outputs = llm.classify([query_template])
    return outputs[0].outputs.probs[0]  # probability of "yes"


predict("how the american judicial system works")
# 0.03352029249072075

predict("I want a thorough understanding of what makes up a community, including its definitions in various contexts like science and what it means to be a civilized community. I am also interested in related terms like 'grassroots organizations,' how communities set boundaries and priorities, and their roles in important areas such as preparedness and nation-building.")
# 0.041828274726867676

# it did not work...
