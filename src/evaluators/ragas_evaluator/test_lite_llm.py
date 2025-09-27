import os
import openai

os.environ["OPENAI_API_KEY"] = os.environ.get("MMU_OPENAI_API_KEY", "")

client = openai.OpenAI(
    # api_key="sk-aUkkzAt1oUoVwTiqq-8UCA",
    base_url="https://mmu-gpu-server-llm-proxy.rankun.org"  # Your LiteLLM Proxy URL
)

response = client.chat.completions.create(
    model="anthropic.claude-3-5-sonnet-20240620-v1:0",
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
