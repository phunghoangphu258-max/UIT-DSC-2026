"""Run a direct Qwen3 generation smoke test with optional thinking mode."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = "Trả lời ngắn gọn: 17 nhân 23 bằng bao nhiêu?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model ID or local path")
    parser.add_argument("--adapter", type=Path, help="Optional QLoRA adapter/checkpoint path")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--system", help="Optional system prompt")
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen3 thinking mode")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from unsloth import FastLanguageModel

    source = str(args.adapter) if args.adapter else args.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=source,
        max_seq_length=args.max_length,
        dtype=None,
        load_in_4bit=torch.cuda.is_available(),
    )
    FastLanguageModel.for_inference(model)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    )
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.6 if args.thinking else 0.7,
        top_p=0.95 if args.thinking else 0.8,
        top_k=20,
        use_cache=True,
    )
    generated = output[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated, skip_special_tokens=False))


if __name__ == "__main__":
    main()
