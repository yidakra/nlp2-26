#!/usr/bin/env python3
"""
GRPO (Group Relative Policy Optimization) Training Script for Terminology-Aware NMT

This script mirrors the SFT training functionality but uses GRPO for reinforcement
learning. It uses CHRF++ as the reward metric and supports bidirectional translation
(Chinese <-> English) with terminology constraints.
"""

import argparse
import os
import random
from datetime import datetime, timezone
from pathlib import Path
import itertools
import re
from typing import Any

import jieba
from pycccedict.cccedict import CcCedict
import torch
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig
from tqdm import tqdm

import wandb
from codecarbon import OfflineEmissionsTracker


LANG_INFO = {
    "enzh": {"src": "en", "tgt": "zh", "src_full": "English", "tgt_full": "Traditional Chinese"},
    "zhen": {"src": "zh", "tgt": "en", "src_full": "Traditional Chinese", "tgt_full": "English"},
}


def compute_chrf_plus_plus(hypothesis: str, reference: str) -> float:
    """
    Compute CHRF++ score between hypothesis and reference.
    CHRF++ combines character n-gram precision and recall with beta=3.
    """
    import sacrebleu

    # sacrebleu's CHRF implementation returns a score object
    # We need to wrap single sentences in a list for sacrebleu
    score = sacrebleu.corpus_chrf(
        hypotheses=[hypothesis],
        references=[[reference]],
        char_order=6,
        word_order=2,  # This makes it CHRF++ (word order matters)
        beta=3,
        lowercase=False,
        whitespace=False,
    )
    return score.score / 100.0  # Normalize to [0, 1]


def compute_reward(
    completions: list[str],
    source: str,
    reference: str,
    is_zh_to_en: bool,
    terms: list[tuple[str, str]] | None = None,
    term_weight: float = 0.35,
    length_penalty_weight: float = 0.1,
) -> float:
    """
    Compute reward for a generated translation.

    The reward is based on CHRF++ score, with:
    - Terminology consistency bonus (35% weight by default)
    - Length penalty to prevent padding/truncation exploits

    Args:
        completions: Generated translations (group members for GRPO)
        source: Source sentence
        reference: Reference translation
        is_zh_to_en: Whether translating from Chinese to English
        terms: List of (source_term, target_term) pairs
        term_weight: Weight for terminology consistency bonus
        length_penalty_weight: Weight for length penalty

    Returns:
        Reward value in [0, 1] range
    """
    completion = completions[0]

    # Base reward from CHRF++ score
    chrf_score = compute_chrf_plus_plus(completion, reference)

    # Length penalty: penalize deviations from reference length
    # Uses relative difference to handle both short and long documents
    if len(reference) > 0:
        length_ratio = len(completion) / len(reference)
        # Penalty when too short (<80%) or too long (>120%)
        if length_ratio < 0.8:
            length_penalty = (0.8 - length_ratio) * length_penalty_weight
        elif length_ratio > 1.2:
            length_penalty = (length_ratio - 1.2) * length_penalty_weight
        else:
            length_penalty = 0.0
    else:
        length_penalty = 0.0

    # Terminology consistency bonus
    # Higher weight (35%) to encourage terminology adherence
    term_bonus = 0.0
    if terms and len(terms) > 0:
        translation_lower = completion.lower()
        matched_terms = 0

        for src_term, tgt_term in terms:
            src_term_lower = src_term.lower().strip()
            tgt_term_lower = tgt_term.lower().strip()

            # Check if target term appears in translation
            if is_zh_to_en:
                # Translating ZH->EN: check if English term is in translation
                if src_term_lower in translation_lower:
                    matched_terms += 1
            else:
                # Translating EN->ZH: check if Chinese term is in translation
                if tgt_term_lower in completion:
                    matched_terms += 1

        term_bonus = matched_terms / len(terms)

    # Combined reward: CHRF++ and terminology help, length deviation hurts
    reward = (chrf_score * (1 - term_weight) + term_bonus * term_weight) - length_penalty

    # Clip to [0, 1] to prevent extreme values
    return max(0.0, min(1.0, reward))


def build_cccedict_mapping():
    """Builds a cached mapping from Traditional Chinese tokens to clean English definitions."""
    cccedict = CcCedict()
    zh_to_en = {}
    for entry in cccedict.get_entries():
        zh_trad = entry['traditional']
        if zh_trad not in zh_to_en:
            zh_to_en[zh_trad] = set()

        for d in entry['definitions']:
            d = re.sub(r'\(.*?\)', '', d)
            if 'CL:' in d:
                continue
            if d.startswith('to '):
                d = d[3:]
            d = d.replace('lit. ', '').replace('fig. ', '')
            d = d.strip().lower()

            for sub_d in re.split(r'[,/]', d):
                sub_d = re.sub(r'[^a-z0-9\-\s]', '', sub_d).strip()
                sub_d = re.sub(r'\s+', ' ', sub_d)
                if len(sub_d) > 2:
                    zh_to_en[zh_trad].add(sub_d)
    return zh_to_en


def extract_alignments(dataset, desc="Extracting alignments"):
    """Extract word-aligned term pairs using pycccedict and jieba."""
    zh_to_en = build_cccedict_mapping()
    alignments = []
    word_pattern = re.compile(r"[a-z0-9\-]+")

    for example in tqdm(dataset, desc=desc):
        source_en = example['en'].lower()
        target_zh = example['zh']
        en_words = word_pattern.findall(source_en)
        clean_source_en = " " + " ".join(en_words) + " "
        tgt_tokens = [tok for tok in jieba.lcut(target_zh) if tok.strip()]

        terms = []
        for zh_word in tgt_tokens:
            if zh_word in zh_to_en:
                for en_def in zh_to_en[zh_word]:
                    if f" {en_def} " in clean_source_en:
                        terms.append((en_def, zh_word))
                        break

        terms = normalize_term_pairs(terms)
        alignments.append(limit_terms_per_document(terms, min_terms=50, max_terms=170))
    return alignments


STOPWORDS_EN = {
    'a', 'an', 'and', 'the', 'of', 'in', 'to', 'for', 'by', 'with', 'on', 'at', 'from', 'as',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'or', 'not', 'that', 'this', 'these', 'those',
    'it', 'its', 'which', 'such', 'may', 'any', 'per', 'off', 'under', 'into', 'through', 'over',
    'other', 'than', 'then', 'so', 'if', 'but', 'who', 'whom', 'whose', 'all', 'no', 'each',
    'some', 'more', 'most', 'one', 'two', 'three', 'four', 'five', 'between', 'among', 'within',
    'about', 'after', 'before', 'during', 'while', 'where', 'when', 'what', 'how', 'why', 'do', 'does',
    'did', 'shall', 'should', 'would', 'could', 'can', 'will', 'must', 'every', 'there', 'here',
    'also', 'however', 'therefore', 'including'
}


def is_punct_token(token: str) -> bool:
    punct_chars = set('.,;:!?"\'()[]{}<>，。；：！？""''（）【】《》、')
    token = token.strip()
    return token == '' or all(ch in punct_chars for ch in token)


def is_valid_term_pair(src_term: str, tgt_term: str) -> bool:
    src, tgt = src_term.strip(), tgt_term.strip()
    if not src or not tgt or src.lower() in STOPWORDS_EN or len(src) <= 1:
        return False
    if is_punct_token(src) or is_punct_token(tgt):
        return False
    if len(tgt) > max(80, len(src) * 8):
        return False
    return True


def normalize_term_pairs(terms):
    cleaned, seen = [], set()
    for src_term, tgt_term in terms:
        if not is_valid_term_pair(src_term, tgt_term):
            continue
        pair = (src_term.strip(), tgt_term.strip())
        if pair not in seen:
            seen.add(pair)
            cleaned.append(pair)
    return cleaned


def limit_terms_per_document(terms, min_terms=50, max_terms=170):
    """Keep at most a random number of term pairs per document."""
    if len(terms) <= min_terms:
        return terms
    max_keep = random.randint(min_terms, max_terms)
    return random.sample(terms, min(len(terms), max_keep))


def save_extracted_terms_dataset(dataset, alignments, output_dir, split_name):
    output_path = Path(output_dir) / split_name
    output_path.mkdir(parents=True, exist_ok=True)
    records = [
        {"en": ex["en"], "zh": ex["zh"], "terms": [{"src": s, "tgt": t} for s, t in al]}
        for ex, al in zip(dataset, alignments)
    ]
    datasets.Dataset.from_list(records).save_to_disk(str(output_path))


def load_extracted_terms_dataset(input_dir, split_name):
    dataset_path = Path(input_dir) / split_name
    term_dataset = datasets.load_from_disk(str(dataset_path))
    return [[(p["src"], p["tgt"]) for p in ex.get("terms", [])] for ex in term_dataset]


def augment_terminology(terms):
    """
    50/50 chance of showing terms or showing nothing.
    Since extraction already limited the range to 50-170, we display what is available.
    """
    if not terms or random.random() < 0.5:
        return ""

    return ', '.join(f'{src} -> {tgt}' for src, tgt in terms)


class GRPODataset(Dataset):
    """
    Dataset for GRPO training that returns prompts and references.
    Each sample can be either Chinese->English or English->Chinese (50/50).
    """
    def __init__(self, dataset, alignments):
        self.dataset = dataset
        self.alignments = alignments

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        # 50/50 chance for each direction
        is_zh_to_en = random.random() < 0.5

        if not is_zh_to_en:
            # English to Chinese
            lang = LANG_INFO["enzh"]
            source, target = example['en'], example['zh']
            terms = self.alignments[idx]  # (en, zh)
        else:
            # Chinese to English
            lang = LANG_INFO["zhen"]
            source, target = example['zh'], example['en']
            # Swap to (zh, en) for ZH->EN direction
            terms = [(zh, en) for en, zh in self.alignments[idx]]

        terminology = augment_terminology(terms)

        prompt = (
            f"Translate the following sentence from {lang['src_full']} to {lang['tgt_full']}, "
            "respecting the given terminology. Output the translation and nothing else.\n\n"
            f"Source: {source}\n"
            f"Terminology: {terminology}\n\n"
        )

        return {
            "prompt": prompt,
            "reference": target,
            "is_zh_to_en": is_zh_to_en,
            "terms": terms,
        }


def create_reward_function(alignments):
    """
    Create a reward function closure that has access to alignments.
    This is used by GRPO to compute rewards for generated completions.
    """
    def reward_fn(completions: list[str], prompts: list[dict]) -> list[float]:
        """
        Compute rewards for a batch of completions.

        Args:
            completions: List of generated texts (one per prompt in the group)
            prompts: List of prompt dictionaries containing reference, is_zh_to_en, terms

        Returns:
            List of reward values
        """
        rewards = []
        for completion, prompt in zip(completions, prompts):
            reference = prompt.get("reference", "")
            is_zh_to_en = prompt.get("is_zh_to_en", False)
            terms = prompt.get("terms", [])

            reward = compute_reward(
                completions=[completion],
                source="",  # Not used in current implementation
                reference=reference,
                is_zh_to_en=is_zh_to_en,
                terms=terms,
                term_weight=0.35,
                length_penalty_weight=0.1,
            )
            rewards.append(reward)

        return rewards

    return reward_fn


def main():
    parser = argparse.ArgumentParser(description="Train terminology-aware NMT with GRPO")
    parser.add_argument("--model", required=True, help="Path or name of the base model")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size per device")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--data_dir", required=True, help="Path to the dataset directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints")
    parser.add_argument("--save_terms_dir", default=None, help="Directory to save/load extracted terms")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="nlp2-26", help="W&B project name")
    parser.add_argument("--codecarbon", action="store_true", help="Enable CodeCarbon tracking")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt for GRPO")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generations")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens to generate")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for GRPO")
    args = parser.parse_args()

    # Load dataset
    dataset = datasets.load_from_disk(args.data_dir)

    # Initialize CodeCarbon tracker
    tracker = None
    if args.codecarbon:
        tracker = OfflineEmissionsTracker(
            output_dir=os.getenv('CODECARBON_OUTPUT_DIR'),
            project_name=args.wandb_project,
            country_iso_code=os.getenv('CODECARBON_COUNTRY_ISO_CODE', 'NLD')
        )
        tracker.start()

    # Load or extract alignments
    alignments_train = None
    alignments_val = None
    if args.save_terms_dir:
        t_path = Path(args.save_terms_dir) / "train"
        v_path = Path(args.save_terms_dir) / "validation"
        if t_path.exists() and v_path.exists():
            alignments_train = load_extracted_terms_dataset(args.save_terms_dir, "train")
            alignments_val = load_extracted_terms_dataset(args.save_terms_dir, "validation")

    if alignments_train is None:
        print("Extracting terminology alignments for training set...")
        alignments_train = extract_alignments(dataset['train'], desc="Aligning Train")
        print("Extracting terminology alignments for validation set...")
        alignments_val = extract_alignments(dataset['validation'], desc="Aligning Val")
        if args.save_terms_dir:
            print(f"Saving extracted terms to {args.save_terms_dir}...")
            save_extracted_terms_dataset(dataset['train'], alignments_train, args.save_terms_dir, "train")
            save_extracted_terms_dataset(dataset['validation'], alignments_val, args.save_terms_dir, "validation")

    # Create GRPO datasets
    train_dataset = GRPODataset(dataset['train'], alignments_train)
    val_dataset = GRPODataset(dataset['validation'], alignments_val)

    # Debug: print first few examples
    print("\n=== Sample Training Examples ===")
    for i in range(2):
        ex = train_dataset[i]
        direction = "ZH->EN" if ex["is_zh_to_en"] else "EN->ZH"
        print(f"\nExample {i} ({direction}):")
        print(f"  Prompt: {ex['prompt'][:200]}...")
        print(f"  Reference: {ex['reference'][:100]}...")
        print(f"  Terms: {len(ex['terms'])} pairs")

    # Calculate training steps
    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * args.num_generations))
    eval_steps = max(1, steps_per_epoch // 10)
    save_steps = max(1, steps_per_epoch // 5)

    # Initialize W&B
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            group=os.getenv("WANDB_RUN_GROUP"),
            entity=os.getenv("WANDB_ENTITY"),
            config={
                "output_dir": args.output_dir,
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "num_generations": args.num_generations,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                "gradient_accumulation_steps": 4,
            },
            name=f"nmt-grpo-{datetime.now().strftime('%Y%m%d-%H%M')}"
        )

    # Configure GRPO training
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["wandb"] if args.wandb else ["none"],
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=2048,
        temperature=args.temperature,
        num_generations=args.num_generations,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=50,
        warmup_ratio=0.1,
    )

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA
    print("Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create reward function
    print("Setting up reward function with CHRF++...")
    reward_fn = create_reward_function(alignments_train)

    # Initialize GRPO trainer
    print("Initializing GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        reward_funcs=reward_fn,
    )

    # Start training
    print("\n=== Starting GRPO Training ===")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Stop CodeCarbon tracker
    if tracker is not None:
        tracker.stop()

    print("\n=== Training Complete ===")


if __name__ == "__main__":
    main()
