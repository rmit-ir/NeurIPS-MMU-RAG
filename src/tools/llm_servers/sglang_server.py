from sglang.utils import async_stream_and_merge
import sglang as sgl
import asyncio
from sglang.test.doc_patch import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight, terminate_process


server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server --model-path qwen/qwen2.5-0.5b-instruct --host 0.0.0.0 --log-level warning"
    "run python -m sglang.launch_server --model Qwen/Qwen3-4B --reasoning-parser qwen3 --disable-radix-cache --mem-fraction-static 0.4 --max-running-requests 4"
)

wait_for_server(f"http://localhost:{port}")
print(f"Server started on http://localhost:{port}")


# launch the offline engine


llm = sgl.Engine(model_path="Qwen/Qwen3-8B")

prompts = [
    "Write a short, neutral self-introduction for a fictional character. Hello, my name is",
    "Provide a concise factual statement about France’s capital city. The capital of France is",
    "Explain possible future trends in artificial intelligence. The future of AI is",
]

sampling_params = {"temperature": 0.8, "top_p": 0.95}

print("\n=== Testing asynchronous streaming generation (no repeats) ===")


async def main():
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated text: ", end="", flush=True)

        # Replace direct calls to async_generate with our custom overlap-aware version
        async for cleaned_chunk in async_stream_and_merge(llm, prompt, sampling_params):
            print(cleaned_chunk, end="", flush=True)

        print()  # New line after each prompt


if __name__ == "__main__":
    asyncio.run(main())
