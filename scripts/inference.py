import argparse
import gc
import json
import logging
from pathlib import Path
import sys

import torch
import transformers as tr


logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run translation inference and optional XCOMET evaluation.")
    parser.add_argument("--model-id", default="google/gemma-4-E2B-it", help="HF model ID")
    parser.add_argument(
        "--src-tgt-pair",
        default="enzh",
        choices=["ende", "enes", "enru", "enzh", "zhen"],
        help="Language direction key",
    )
    parser.add_argument("--input-jsonl", type=Path, help="Path to JSONL inputs")
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/inference_outputs.jsonl"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--run-eval", action="store_true", help="Run XCOMET eval if references exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.eval import Evaluator, load_jsonl

    if args.input_jsonl:
        data = load_jsonl(args.input_jsonl, max_samples=args.max_samples)
    else:
        data = [{"en": "Hello, how are you?", "zh": "你好嗎？", "terms": ""}]

    model = tr.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = tr.AutoTokenizer.from_pretrained(args.model_id)

    evaluator = Evaluator()
    outputs = evaluator.generate_translations(
        data,
        model,
        tokenizer,
        args.src_tgt_pair,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(outputs)} outputs to {args.output_jsonl}")

    if args.run_eval:
        eval_rows = [row for row in outputs if row.get("src") and row.get("mt") and row.get("ref")]
        if not eval_rows:
            print("Skipping XCOMET: no rows with non-empty src/mt/ref.")
        else:
            summary = evaluator.evaluate(eval_rows, args.batch_size)
            print("XCOMET system score:", summary["system"])

    del model, tokenizer
    gc.collect()


if __name__ == "__main__":
    main()
