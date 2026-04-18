import argparse
import gc
import json
import logging
import os
import random
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
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
    parser.add_argument(
        "--prompt-strategy",
        choices=["baseline", "concise", "strict"],
        default="baseline",
        help="Prompt style used for inference-time strategy experiments.",
    )
    parser.add_argument(
        "--few-shot-examples-jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file with examples for in-context few-shot prompting.",
    )
    parser.add_argument("--few-shot-k", type=int, default=0, help="Number of few-shot examples per prompt.")
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Number of sampled candidates per input for optional reranking.",
    )
    parser.add_argument(
        "--rerank-strategy",
        choices=["none", "term_coverage"],
        default="none",
        help="Reranking strategy applied over sampled candidates.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--run-eval", action="store_true", help="Run XCOMET eval if references exist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.eval import Evaluator, load_jsonl

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases is required by default. Install it with `pip install wandb`."
        ) from exc
    try:
        from codecarbon import OfflineEmissionsTracker
    except ImportError as exc:
        raise RuntimeError(
            "CodeCarbon is required by default. Install it with `pip install codecarbon`."
        ) from exc

    if args.input_jsonl:
        data = load_jsonl(args.input_jsonl, max_samples=args.max_samples)
    else:
        data = [{"en": "Hello, how are you?", "zh": "你好嗎？", "terms": ""}]

    few_shot_examples: list[dict[str, Any]] = []
    if args.few_shot_examples_jsonl is not None:
        few_shot_examples = load_jsonl(args.few_shot_examples_jsonl)

    few_shot_k = args.few_shot_k
    if few_shot_k > len(few_shot_examples):
        logging.warning(
            "Requested few_shot_k=%d but only %d few-shot examples were loaded; clamping few_shot_k to %d.",
            args.few_shot_k,
            len(few_shot_examples),
            len(few_shot_examples),
        )
        few_shot_k = len(few_shot_examples)

    seed_value = int(args.seed)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch_seed = getattr(torch, "manual_seed")
    torch_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.src_tgt_pair}-{Path(args.model_id).name}-{now}"
    codecarbon_output_dir = Path(os.getenv("CODECARBON_OUTPUT_DIR", "outputs/codecarbon"))
    codecarbon_output_dir.mkdir(parents=True, exist_ok=True)
    codecarbon_country_iso = os.getenv("CODECARBON_COUNTRY_ISO_CODE", "NLD")

    model = None
    tokenizer = None
    emissions_tracker = None

    def release_generation_model() -> None:
        nonlocal model, tokenizer
        if model is not None or tokenizer is not None:
            del model, tokenizer
            model = None
            tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    try:
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "nlp2-26"),
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            group=os.getenv("WANDB_RUN_GROUP", args.src_tgt_pair),
            job_type="inference_eval",
            config={
                "model_id": args.model_id,
                "src_tgt_pair": args.src_tgt_pair,
                "input_jsonl": str(args.input_jsonl) if args.input_jsonl else None,
                "output_jsonl": str(args.output_jsonl),
                "batch_size": args.batch_size,
                "max_samples": args.max_samples,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "prompt_strategy": args.prompt_strategy,
                "few_shot_examples_jsonl": (
                    str(args.few_shot_examples_jsonl) if args.few_shot_examples_jsonl else None
                ),
                "few_shot_k": few_shot_k,
                "num_candidates": args.num_candidates,
                "rerank_strategy": args.rerank_strategy,
                "seed": args.seed,
                "run_eval": args.run_eval,
                "codecarbon_output_dir": str(codecarbon_output_dir),
                "codecarbon_country_iso_code": codecarbon_country_iso,
            },
        )

        emissions_tracker = OfflineEmissionsTracker(
            project_name=os.getenv("CODECARBON_PROJECT_NAME", "nlp2-26"),
            output_dir=str(codecarbon_output_dir),
            country_iso_code=codecarbon_country_iso,
            save_to_file=True,
            log_level="warning",
        )
        emissions_tracker.start()
        model_loader = cast(Any, tr.AutoModelForCausalLM)
        model = model_loader.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        tokenizer_loader = cast(Any, tr.AutoTokenizer)
        tokenizer = tokenizer_loader.from_pretrained(args.model_id)

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
            prompt_strategy=args.prompt_strategy,
            few_shot_examples=few_shot_examples,
            few_shot_k=few_shot_k,
            num_candidates=args.num_candidates,
            rerank_strategy=args.rerank_strategy,
            seed=args.seed,
        )

        with args.output_jsonl.open("w", encoding="utf-8") as f:
            for row in outputs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(outputs)} outputs to {args.output_jsonl}")
        wandb.log({"num_outputs": len(outputs), "output_jsonl": str(args.output_jsonl)})

        # Free generation model memory before loading XCOMET.
        release_generation_model()

        if args.run_eval:
            eval_rows = [row for row in outputs if row.get("src") and row.get("mt") and row.get("ref")]
            if not eval_rows:
                print("Skipping XCOMET: no rows with non-empty src/mt/ref.")
                wandb.log({"xcomet_skipped": 1})
            else:
                summary = evaluator.evaluate(eval_rows, args.batch_size)
                print("XCOMET system score:", summary["system"])
                wandb.log(
                    {
                        "xcomet_system_score": float(summary["system"]),
                        "xcomet_segments": len(summary.get("segment", [])),
                    }
                )
    finally:
        if emissions_tracker is not None:
            emissions_kg = emissions_tracker.stop()
            if emissions_kg is not None:
                print(f"CodeCarbon emissions (kgCO2eq): {emissions_kg}")
                wandb.log({"codecarbon_emissions_kgco2eq": float(emissions_kg)})
        release_generation_model()
        wandb.finish()


if __name__ == "__main__":
    main()