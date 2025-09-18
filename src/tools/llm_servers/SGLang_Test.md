# SGLang Test

Run command

```bash
run python -m sglang.launch_server --model Qwen/Qwen3-4B --reasoning-parser qwen3 --disable-radix-cache --mem-fraction-static 0.4 --max-running-requests 4
```

`--disable-radix-cache --mem-fraction-static 0.4` are for disabling optimizations for multi-turn conversations, which are not needed for RAG use cases, but maybe for agentic!.
