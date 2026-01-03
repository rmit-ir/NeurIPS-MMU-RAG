from typing import Dict, List, Tuple
from threading import Lock

try:
    import torch
except ImportError as exc:  # pragma: no cover - handled by runtime
    raise RuntimeError("PyTorch is required for answer generation.") from exc

from transformers import AutoModelForCausalLM, AutoTokenizer

from tools.logging_utils import get_logger

logger = get_logger("generator")

_MODEL_CACHE: Dict[str, Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]] = {}
_CACHE_LOCK = Lock()


def _ensure_model(model_name: str) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """Load and cache tokenizer/model pairs for reuse."""
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            return cached

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_kwargs = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model_kwargs["torch_dtype"] = torch.float16
        else:
            device = torch.device("cpu")
            model_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        model.to(device)
        model.eval()

        cached_pair = (tokenizer, model, device)
        _MODEL_CACHE[model_name] = cached_pair
        logger.info("Loaded generation model", model=model_name, device=str(device))
        return cached_pair


def _format_prompt(query: str, contexts: List[str]) -> str:
    """Compose the prompt that blends retrieved context with the user query."""
    cleaned_contexts = [ctx.strip() for ctx in contexts if ctx and ctx.strip()]
    if cleaned_contexts:
        context_section = "\n\n".join(
            f"[Context {idx + 1}] {ctx}" for idx, ctx in enumerate(cleaned_contexts)
        )
    else:
        context_section = "No additional context was retrieved."

    return (
        "You are a helpful assistant. Answer the question using the provided context. "
        "If the context is insufficient, say so explicitly.\n\n"
        f"Context:\n{context_section}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def _prepare_inputs(tokenizer: AutoTokenizer, prompt: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Tokenize prompt with chat templates when available."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": "You answer questions using only the supplied context."},
            {"role": "user", "content": prompt},
        ]
        tokenized = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=getattr(tokenizer, "model_max_length", None)
        )
        if isinstance(tokenized, dict):
            return {name: tensor.to(device) for name, tensor in tokenized.items()}

        attention_mask = torch.ones_like(tokenized)
        return {
            "input_ids": tokenized.to(device),
            "attention_mask": attention_mask.to(device),
        }

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(tokenizer, "model_max_length", None)
    )
    return {name: tensor.to(device) for name, tensor in encoded.items()}


def generate_answer(query: str, contexts: List[str], model_name: str) -> str:
    """
    Generate an answer using retrieved contexts and a language model.

    Args:
        query: The original user query
        contexts: List of retrieved relevant text chunks
        model_name: HuggingFace model name for generation

    Returns:
        Generated answer based on query and contexts
    """
    tokenizer, model, device = _ensure_model(model_name)
    prompt = _format_prompt(query, contexts)
    inputs = _prepare_inputs(tokenizer, prompt, device)

    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id

    if pad_token_id is None:
        if isinstance(eos_token_id, list):
            pad_token_id = eos_token_id[0] if eos_token_id else None
        else:
            pad_token_id = eos_token_id

    if eos_token_id is None and pad_token_id is not None:
        eos_token_id = pad_token_id

    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    completion_tokens = generated[0, prompt_length:]
    answer = tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()

    if not answer:
        full_decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        answer = full_decoded[len(prompt):].strip() if full_decoded.startswith(prompt) else full_decoded

    return answer or "I'm sorry, I could not produce an answer with the available context."
