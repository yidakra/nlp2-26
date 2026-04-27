#!/usr/bin/env python3
"""
Terminology-Aware Neural Machine Translation Training Script

Modified to:
1. Toggle terminology display (50-170 terms) or no terminology (50/50 chance).
2. Ensure terminology pairs match the specific source-to-target direction of the example.
"""

import argparse
import os
import random
from datetime import datetime, timezone
from pathlib import Path
import itertools
import re
import string

import jieba
from pycccedict.cccedict import CcCedict
import torch
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm

import wandb
from codecarbon import OfflineEmissionsTracker

LANG_INFO = {
    "enzh": {"src": "en", "tgt": "zh", "src_full": "English", "tgt_full": "Traditional Chinese"},
    "zhen": {"src": "zh", "tgt": "en", "src_full": "Traditional Chinese", "tgt_full": "English"},
}

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
            if 'CL:' in d: continue
            if d.startswith('to '): d = d[3:]
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
        # Note: We limit extracted terms here to the 50-170 range
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
    punct_chars = set('.,;:!?"\'()[]{}<>，。；：！？“”‘’（）【】《》、')
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
        if not is_valid_term_pair(src_term, tgt_term): continue
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
    records = [{"en": ex["en"], "zh": ex["zh"], "terms": [{"src": s, "tgt": t} for s, t in al]} 
               for ex, al in zip(dataset, alignments)]
    datasets.Dataset.from_list(records).save_to_disk(str(output_path))

def load_extracted_terms_dataset(input_dir, split_name):
    dataset_path = Path(input_dir) / split_name
    term_dataset = datasets.load_from_disk(str(dataset_path))
    return [[(p["src"], p["tgt"]) for p in ex.get("terms", [])] for ex in term_dataset]

def augment_terminology(terms):
    """
    CHANGE 1: 50/50 chance of showing terms or showing nothing.
    Since extraction already limited the range to 50-170, we display what is available.
    """
    if not terms or random.random() < 0.5:
        return ""
    
    # Terms are already pre-filtered/limited to 50-170 in extract_alignments
    return ', '.join(f'{src} -> {tgt}' for src, tgt in terms)

class TermAwareDataset(Dataset):
    def __init__(self, dataset, alignments):
        self.dataset = dataset
        self.alignments = alignments
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        is_zh_to_en = random.random() < 0.5
        
        if not is_zh_to_en:
            # English to Chinese
            lang = LANG_INFO["enzh"]
            source, target = example['en'], example['zh']
            # CHANGE 2: (src: en, tgt: zh)
            terms = self.alignments[idx] 
        else:
            # Chinese to English
            lang = LANG_INFO["zhen"]
            source, target = example['zh'], example['en']
            # CHANGE 2: Swap order (src: zh, tgt: en)
            terms = [(zh, en) for en, zh in self.alignments[idx]]
        
        terminology = augment_terminology(terms)
        
        prompt = (
            f"Translate the following sentence from {lang['src_full']} to {lang['tgt_full']}, "
            "respecting the given terminology. Output the translation and nothing else.\n\n"
            f"Source: {source}\n"
            f"Terminology: {terminology}\n\n"
        )
        return {"text": prompt, "target": target}


def process_data_for_sft(example, idx, alignments, tokenizer):
    # Randomly decide direction
    is_zh_to_en = random.random() < 0.5
    
    # Get the base alignment for this index
    base_terms = alignments[idx]
    
    if not is_zh_to_en:
        lang = LANG_INFO["enzh"]
        source, target = example['en'], example['zh']
        terms = base_terms # (en, zh)
    else:
        lang = LANG_INFO["zhen"]
        source, target = example['zh'], example['en']
        terms = [(zh, en) for en, zh in base_terms] # Swap to (zh, en)
    
    terminology = augment_terminology(terms)
    
    prompt = (
        f"Translate the following sentence from {lang['src_full']} to {lang['tgt_full']}, "
        "respecting the given terminology. Output the translation and nothing else.\n\n"
        f"Source: {source}\n"
        f"Terminology: {terminology}\n\n"
    )
    
    # Format for chat template
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target}
    ]
    
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

def main():
    parser = argparse.ArgumentParser(description="Train terminology-aware NMT with SFT and LoRA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_terms_dir", default=None)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="nlp2-26")
    parser.add_argument("--codecarbon", action="store_true")
    args = parser.parse_args()
    
    dataset = datasets.load_from_disk(args.data_dir)

    tracker = None
    if args.codecarbon:
        tracker = OfflineEmissionsTracker(output_dir=os.getenv('CODECARBON_OUTPUT_DIR'), project_name=args.wandb_project, country_iso_code=os.getenv('CODECARBON_COUNTRY_ISO_CODE', 'NLD'))
        tracker.start()
    
    alignments_train = None
    alignments_val = None
    if args.save_terms_dir:
        t_path, v_path = Path(args.save_terms_dir)/"train", Path(args.save_terms_dir)/"validation"
        if t_path.exists() and v_path.exists():
            alignments_train = load_extracted_terms_dataset(args.save_terms_dir, "train")
            alignments_val = load_extracted_terms_dataset(args.save_terms_dir, "validation")

    if alignments_train is None:
        alignments_train = extract_alignments(dataset['train'], desc="Aligning Train")
        alignments_val = extract_alignments(dataset['validation'], desc="Aligning Val")
        if args.save_terms_dir:
            save_extracted_terms_dataset(dataset['train'], alignments_train, args.save_terms_dir, "train")
            save_extracted_terms_dataset(dataset['validation'], alignments_val, args.save_terms_dir, "validation")

    train_dataset = TermAwareDataset(dataset['train'], alignments_train)
    print(train_dataset[0])  # Debug: Check first training example with terms
    print(train_dataset[1])  # Debug: Check second training example with terms
    val_dataset = TermAwareDataset(dataset['validation'], alignments_val)
    
    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * 8))
    eval_steps = max(1, steps_per_epoch // 10)

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            group=os.getenv("WANDB_RUN_GROUP"),
            entity=os.getenv("WANDB_ENTITY"),
            config={
                "output_dir": args.output_dir,
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": args.batch_size,
                "learning_rate": 2e-5,
                "gradient_accumulation_steps": 8,
                "eval_steps": eval_steps,
                "save_steps": eval_steps,
                "metric_for_best_model": "eval_loss",
                "max_length": 2048,
            },
            name=f"nmt-sft-{datetime.now().strftime('%Y%m%d-%H%M')}"
        )
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["wandb"] if args.wandb else ["none"],
        max_length=4096,
        bf16=True,
        gradient_checkpointing=True,
    )

    # Model Loading
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Convert the raw HF dataset using our mapping function
    train_dataset = dataset['train'].map(
        lambda ex, idx: process_data_for_sft(ex, idx, alignments_train, tokenizer),
        with_indices=True,
        remove_columns=dataset['train'].column_names
    )
    
    val_dataset = dataset['validation'].map(
        lambda ex, idx: process_data_for_sft(ex, idx, alignments_val, tokenizer),
        with_indices=True,
        remove_columns=dataset['validation'].column_names
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2*args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # In SFTTrainer, you can now simply point to the "text" column
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()