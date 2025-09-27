import os
import openai

os.environ["OPENAI_API_KEY"] = os.environ.get("MMU_OPENAI_API_KEY", "")

client = openai.OpenAI(
    base_url="https://mmu-proxy-server-llm-proxy.rankun.org")

response = client.chat.completions.create(
    model="qwen.qwen3-32b-v1:0",
    messages=[
        {
            "role": "user",
            "content": "Please help explain deep learning in a story around 4000 words."
        }
    ],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
