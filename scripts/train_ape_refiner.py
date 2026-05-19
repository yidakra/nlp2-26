#!/usr/bin/env python3
"""
Train a LoRA APE (Automatic Post-Editing) refiner on Qwen/Qwen3.5-2B.

Reads JSONL with {source, draft, reference, terms, direction} rows and trains
the model to correct draft translations given source text and terminology.
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


LANG_INFO = {
    "enzh": {"src_full": "English", "tgt_full": "Traditional Chinese"},
    "zhen": {"src_full": "Traditional Chinese", "tgt_full": "English"},
}


def format_terminology(terms: object) -> str:
    if not terms:
        return ""
    if isinstance(terms, dict):
        parts = []
        for k, v in terms.items():
            val = v[0] if isinstance(v, list) and v else v
            parts.append(f"{k} -> {val}")
        return "; ".join(parts)
    return str(terms)


def build_example(row: dict, tokenizer) -> str:
    direction = row["direction"]
    lang = LANG_INFO[direction]
    terms_str = format_terminology(row.get("terms", {}))

    user_content = (
        f"Post-edit the following machine translation from {lang['src_full']} to {lang['tgt_full']}, "
        "respecting the given terminology. Output only the corrected translation.\n\n"
        f"Source: {row['source']}\n"
        f"Draft translation: {row['draft']}\n"
        f"Terminology: {terms_str}\n\n"
    )
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": row["reference"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--training-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="nlp2-26")
    parser.add_argument("--wandb-group", default="ape")
    parser.add_argument("--codecarbon", action="store_true")
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be in the open interval (0, 1).")

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = [
        json.loads(line)
        for line in args.training_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    random.shuffle(records)
    print(f"Loaded {len(records)} training records from {args.training_jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = [build_example(r, tokenizer) for r in records]
    if len(texts) < 2:
        raise ValueError("Need at least 2 training records to create train/val splits.")
    n_val = min(max(1, int(len(texts) * args.val_fraction)), len(texts) - 1)
    train_ds = Dataset.from_dict({"text": texts[n_val:]})
    val_ds = Dataset.from_dict({"text": texts[:n_val]})
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    tracker = None
    if args.codecarbon:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(
            output_dir=os.getenv("CODECARBON_OUTPUT_DIR", "outputs/codecarbon"),
            project_name=args.wandb_project,
            country_iso_code=os.getenv("CODECARBON_COUNTRY_ISO_CODE", "NLD"),
        )
        tracker.start()

    if args.wandb:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", args.wandb_project),
            entity=os.getenv("WANDB_ENTITY"),
            group=os.getenv("WANDB_RUN_GROUP", args.wandb_group),
            config=vars(args),
            name=f"ape-{datetime.now().strftime('%Y%m%d-%H%M')}",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * 8))
    eval_steps = max(1, steps_per_epoch // 2)

    training_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-4,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["wandb"] if args.wandb else ["none"],
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    print(f"Saved adapter to {args.output_dir}")

    if tracker is not None:
        tracker.stop()
    if args.wandb:
        import wandb as _wandb
        if _wandb.run is not None:
            _wandb.finish()


if __name__ == "__main__":
    main()
