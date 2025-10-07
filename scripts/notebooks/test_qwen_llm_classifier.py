"""
vllm serve Qwen/Qwen3-4B \
    --max-model-len 20000 \
    --gpu-memory-utilization 0.8 \
    --reasoning-parser qwen3

# vllm server will be available at localhost:8000
uv run python -m asyncio
"""

from typing import List
from openai.types.chat import ChatCompletionMessageParam
from tools.llm_servers.general_openai_client import GeneralOpenAIClient


system_prompt = """Judge if the user query is a complex query. Note that the answer can only be "yes" or "no".

Given the query below, if you are doing the research, do you think you can do a single search on Google and find out the answer?

If so, it's not a complex query. If you need to search multiple times, it's a complex query.

If the user query is long, but can be summarized into a simple question, then it's still not a complex query.

Generally, for straightforward questions, the answer is no, if the question is ambiguous, multifaceted, contains multiple parts or requires multiple steps to answer, the answer is yes.

Give the final answer based on your last reasoning, yes indicates it's a complex query or no indicating it's not a complex query.
"""

openai_client = GeneralOpenAIClient(api_base="http://localhost:8000/v1",
                                    model_id="Qwen/Qwen3-4B",
                                    max_tokens=5120,)


async def predict(query: str):
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    content, chat_cpl = await openai_client.complete_chat(messages)
    is_complex = content.strip().lower() == 'yes' if content else False
    print('-'*10, "complex" if is_complex else "simple", '-'*10)
    print(50*'-')
    print(chat_cpl.choices[0].message.reasoning_content.strip())
    print(50*'-')


await predict("how the american judicial system works")
await predict("I want a thorough understanding of what makes up a community, including its definitions in various contexts like science and what it means to be a civilized community. I am also interested in related terms like 'grassroots organizations,' how communities set boundaries and priorities, and their roles in important areas such as preparedness and nation-building.")
await predict("Based on temperature considerations alone, is March considered a suitable month to perform the final pruning of grape vines?")
await predict("What major acts performed at the Brighton Hippodrome during its peak years?")
await predict("I need help with a business case study - what format should I use for writing it up?")
await predict("What are the main challenges in making optical gyroscopes smaller, and how did researchers overcome these obstacles?")
await predict("What are the main faktors that contribute to the US dollar's role as the dominant reserve currancy in international trade?")
await predict("What is the maximum amout of nitrate thats allowed in drinking water and why is this important for the Gulf of Mexico?")
await predict("where did choan seng song get phd")
await predict("what are differences between real time display and real time recording for surveillance DVR units")
await predict("According to the Center for Research on Environmental Decisions' survey data, what is the average person's time preference for clean air now versus in the future?")
await predict("Could you kindly tell me whether silver became a preferred material for Indigenous jewelry only after 1700? Answer with yes or no.")