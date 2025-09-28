from vllm import LLM

llm = LLM(
    model="danielchalef/Qwen3-Reranker-4B-seq-cls-vllm-fixed",
    task="score"
)

queries = ["What is the capital of France?"]
documents = ["Paris is the capital of France."]

outputs = llm.score(queries, documents)
scores = [output.outputs.score for output in outputs]
print(scores)
