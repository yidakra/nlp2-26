#!/usr/bin/env python3
"""
Terminology-Aware Neural Machine Translation Training Script (Corrected)

Fixes applied:
1. Document chunking so source+target+terms+prompt fit within max_length
2. Per-epoch re-augmentation via callback (direction flip, term visibility varies)
3. Distinct prompt templates for with-terms vs without-terms
4. MLP layers added to LoRA targets for better terminology compliance
5. Pad token handling — use dedicated pad token if available
6. SDPA attention for memory efficiency with longer sequences
"""

import argparse
import os
import random
from datetime import datetime, timezone
from pathlib import Path
import re

import jieba
import numpy as np
from pycccedict.cccedict import CcCedict
import torch
import datasets
<<<<<<< HEAD
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
=======
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from peft import LoraConfig, PeftModel
>>>>>>> 02a1c46 (FINAL TRAINING CODe)
from trl import SFTTrainer, SFTConfig
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANG_INFO = {
    "enzh": {"src": "en", "tgt": "zh", "src_full": "English", "tgt_full": "Traditional Chinese"},
    "zhen": {"src": "zh", "tgt": "en", "src_full": "Traditional Chinese", "tgt_full": "English"},
}

<<<<<<< HEAD
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

    for i, example in enumerate(tqdm(dataset, desc=desc)):
        source_en = example['en'].lower()
        target_zh = example['zh']
        en_words = word_pattern.findall(source_en)
        clean_source_en = " " + " ".join(en_words) + " "
        tgt_tokens = [tok for tok in jieba.lcut(target_zh) if tok.strip()]

        terms = []
        for zh_word in tgt_tokens:
            if zh_word in zh_to_en:
                for en_def in sorted(zh_to_en[zh_word]):
                    if f" {en_def} " in clean_source_en:
                        terms.append((en_def, zh_word))
                        break

        terms = normalize_term_pairs(terms)
        alignments.append(limit_terms_per_document(terms, random.Random(42 + i), min_terms=50, max_terms=170))
    return alignments

=======
>>>>>>> 02a1c46 (FINAL TRAINING CODe)
STOPWORDS_EN = {
    'a', 'an', 'and', 'the', 'of', 'in', 'to', 'for', 'by', 'with', 'on', 'at', 'from', 'as',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'or', 'not', 'that', 'this', 'these', 'those',
    'it', 'its', 'which', 'such', 'may', 'any', 'per', 'off', 'under', 'into', 'through', 'over',
    'other', 'than', 'then', 'so', 'if', 'but', 'who', 'whom', 'whose', 'all', 'no', 'each',
    'some', 'more', 'most', 'one', 'two', 'three', 'four', 'five', 'between', 'among', 'within',
    'about', 'after', 'before', 'during', 'while', 'where', 'when', 'what', 'how', 'why', 'do', 'does',
    'did', 'shall', 'should', 'would', 'could', 'can', 'will', 'must', 'every', 'there', 'here',
    'also', 'however', 'therefore', 'including',
}


# ---------------------------------------------------------------------------
# Term extraction helpers
# ---------------------------------------------------------------------------

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


def is_punct_token(token: str) -> bool:
    punct_chars = set('.,;:!?"\'()[]{}<>，。；：！？\u201c\u201d\u2018\u2019（）【】《》、')
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

<<<<<<< HEAD
=======

>>>>>>> 02a1c46 (FINAL TRAINING CODe)
def limit_terms_per_document(terms, rng, min_terms=50, max_terms=170):
    """Keep at most a random number of term pairs per document."""
    if len(terms) <= min_terms:
        return terms
    max_keep = rng.randint(min_terms, max_terms)
    return rng.sample(terms, min(len(terms), max_keep))
<<<<<<< HEAD
=======


def extract_alignments(dataset, desc="Extracting alignments"):
    """Extract word-aligned term pairs using pycccedict and jieba."""
    zh_to_en = build_cccedict_mapping()
    alignments = []
    word_pattern = re.compile(r"[a-z0-9\-]+")

    for i, example in enumerate(tqdm(dataset, desc=desc)):
        source_en = example['en'].lower()
        target_zh = example['zh']
        en_words = word_pattern.findall(source_en)
        clean_source_en = " " + " ".join(en_words) + " "
        tgt_tokens = [tok for tok in jieba.lcut(target_zh) if tok.strip()]

        terms = []
        for zh_word in tgt_tokens:
            if zh_word in zh_to_en:
                for en_def in sorted(zh_to_en[zh_word]):
                    if f" {en_def} " in clean_source_en:
                        terms.append((en_def, zh_word))
                        break

        terms = normalize_term_pairs(terms)
        alignments.append(limit_terms_per_document(terms, random.Random(42 + i),
                                                   min_terms=50, max_terms=170))
    return alignments


# ---------------------------------------------------------------------------
# Term dataset persistence
# ---------------------------------------------------------------------------
>>>>>>> 02a1c46 (FINAL TRAINING CODe)

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

<<<<<<< HEAD
def augment_terminology(terms, rng):
    """50/50 chance of showing terms or showing nothing."""
    if not terms or rng.random() < 0.5:
        return ""
    return ', '.join(f'{src} -> {tgt}' for src, tgt in terms)

def process_data_for_sft(example, idx, alignments, tokenizer):
    # Seed per-example so results are reproducible regardless of .map() parallelism
    rng = random.Random(42 + idx)
    is_zh_to_en = rng.random() < 0.5

    base_terms = alignments[idx]

    if not is_zh_to_en:
        lang = LANG_INFO["enzh"]
        source, target = example['en'], example['zh']
        terms = base_terms
    else:
        lang = LANG_INFO["zhen"]
        source, target = example['zh'], example['en']
        terms = [(zh, en) for en, zh in base_terms]

    terminology = augment_terminology(terms, rng)

    prompt = (
        f"Translate the following sentence from {lang['src_full']} to {lang['tgt_full']}, "
        "respecting the given terminology. Output the translation and nothing else.\n\n"
        f"Source: {source}\n"
        f"Terminology: {terminology}\n\n"
    )
=======

# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

def estimate_tokens(tokenizer, text):
    """Fast token count estimate."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_document(en, zh, terms, tokenizer, max_tokens=8192, target_chunk_tokens=3500):
    """
    Split a document into chunks where each chunk's full formatted example
    fits within max_tokens.
    
    Strategy: split each side proportionally by character count.
    This works even when en/zh have different paragraph/sentence structures.
    """
    # Check if the whole document fits
    term_str = ', '.join(f'{s} -> {t}' for s, t in terms) if terms else ''
    total_est = estimate_tokens(tokenizer, en + zh + term_str) + 200
    if total_est < max_tokens:
        return [{'en': en, 'zh': zh, 'terms': terms}]

    # How many chunks do we need?
    src_tgt_tokens = estimate_tokens(tokenizer, en + zh)
    n_chunks = max(2, -(-src_tgt_tokens // target_chunk_tokens))  # ceil division

    # Split each side into n_chunks at paragraph or sentence boundaries
    en_chunks = _split_text_into_n(en, n_chunks)
    zh_chunks = _split_text_into_n(zh, n_chunks)

    # Pair them up and attach relevant terms
    chunks = []
    for ec, zc in zip(en_chunks, zh_chunks):
        if not ec.strip() and not zc.strip():
            continue
        ec_lower = ec.lower()
        chunk_terms = [(s, t) for s, t in terms if s.lower() in ec_lower]
        chunks.append({'en': ec, 'zh': zc, 'terms': chunk_terms})

    return chunks if chunks else [{'en': en, 'zh': zh, 'terms': terms}]


def _split_text_into_n(text, n):
    """
    Split text into n roughly equal parts, preferring paragraph then
    sentence boundaries. Falls back to character-proportional splitting.
    """
    if n <= 1:
        return [text]

    # Try splitting on paragraphs first
    parts = re.split(r'(\n+)', text)  # keep delimiters
    segments = []
    current = ""
    for part in parts:
        current += part
        # A paragraph boundary is a newline-only segment
        if re.match(r'^\n+$', part):
            segments.append(current)
            current = ""
    if current:
        segments.append(current)

    # If not enough segments from paragraphs, split further on sentences
    if len(segments) < n:
        new_segments = []
        for seg in segments:
            # Split on sentence endings followed by space
            sub = re.split(r'(?<=[.!?。！？])\s+', seg)
            new_segments.extend(sub)
        segments = [s for s in new_segments if s.strip()]

    # If still not enough, just do character-proportional split
    if len(segments) < n:
        chunk_size = max(1, len(text) // n)
        return [text[i*chunk_size:(i+1)*chunk_size] for i in range(n)]

    # Distribute segments into n groups by cumulative character length
    total_chars = sum(len(s) for s in segments)
    target_per_chunk = total_chars / n

    chunks = []
    current_chunk = []
    current_len = 0

    for seg in segments:
        current_chunk.append(seg)
        current_len += len(seg)
        # Start a new chunk if we've passed the target and have more chunks to fill
        if current_len >= target_per_chunk and len(chunks) < n - 1:
            chunks.append(''.join(current_chunk))
            current_chunk = []
            current_len = 0

    # Last chunk gets whatever remains
    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks


def build_all_chunks(dataset, alignments, tokenizer, max_tokens, target_chunk_tokens):
    """Chunk all documents and return a flat list of chunk dicts."""
    all_chunks = []
    for i, example in enumerate(tqdm(dataset, desc="Chunking documents")):
        doc_chunks = chunk_document(
            example['en'], example['zh'], alignments[i],
            tokenizer, max_tokens, target_chunk_tokens,
        )
        all_chunks.extend(doc_chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Augmentation: format chunks into text for SFTTrainer
# ---------------------------------------------------------------------------

def format_chunk(chunk, rng, tokenizer):
    """
    Apply random augmentation to a chunk, apply chat template,
    and return a single formatted text string.
    """
    is_zh_to_en = rng.random() < 0.5
    base_terms = chunk['terms']

    if not is_zh_to_en:
        lang = LANG_INFO["enzh"]
        source, target = chunk['en'], chunk['zh']
        terms = base_terms  # (en, zh)
    else:
        lang = LANG_INFO["zhen"]
        source, target = chunk['zh'], chunk['en']
        terms = [(zh, en) for en, zh in base_terms]  # swap to (zh, en)

    # 50/50 show terms or not — with distinct prompt templates
    show_terms = bool(terms) and rng.random() >= 0.5

    if show_terms:
        terminology = ', '.join(f'{s} -> {t}' for s, t in terms)
        prompt = (
            f"Translate the following document from {lang['src_full']} "
            f"to {lang['tgt_full']}, respecting the given terminology. "
            f"Output the translation and nothing else.\n\n"
            f"Source: {source}\n"
            f"Terminology: {terminology}\n\n"
        )
    else:
        prompt = (
            f"Translate the following document from {lang['src_full']} "
            f"to {lang['tgt_full']}. "
            f"Output the translation and nothing else.\n\n"
            f"Source: {source}\n\n"
        )
>>>>>>> 02a1c46 (FINAL TRAINING CODe)

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": target},
    ]

<<<<<<< HEAD
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
=======
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def materialize_dataset(chunks, tokenizer, epoch_seed=0):
    """
    Apply augmentation to all chunks and return a HuggingFace Dataset.
    Different epoch_seed -> different augmentation choices.
    """
    records = []
    for i, chunk in enumerate(chunks):
        rng = random.Random(epoch_seed * 100_000 + i)
        record = format_chunk(chunk, rng, tokenizer)
        records.append(record)
    return datasets.Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Callback: re-augment the training set each epoch
# ---------------------------------------------------------------------------

class EpochReaugmentCallback(TrainerCallback):
    """
    At the start of each epoch, re-materialize the training dataset with
    a new random seed so augmentation (direction, term visibility) varies.
    """

    def __init__(self, chunks, tokenizer):
        self.chunks = chunks
        self.tokenizer = tokenizer

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        if epoch == 0:
            return  # epoch 0 was already materialized in main()

        print(f"[EpochReaugmentCallback] Re-augmenting training data for epoch {epoch}")
        trainer = kwargs.get("trainer")
        if trainer is not None:
            new_dataset = materialize_dataset(self.chunks, self.tokenizer, epoch_seed=epoch)
            trainer.train_dataset = new_dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
>>>>>>> 02a1c46 (FINAL TRAINING CODe)

def main():
    parser = argparse.ArgumentParser(description="Train terminology-aware NMT with SFT and LoRA")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_terms_dir", default=None)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=4096)
<<<<<<< HEAD
=======
    parser.add_argument("--target_chunk_tokens", type=int, default=3000,
                        help="Max src+tgt tokens per chunk before prompt/terms overhead")
    parser.add_argument("--adapter", default=None)
>>>>>>> 02a1c46 (FINAL TRAINING CODe)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="nlp2-26")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--codecarbon", action="store_true")
    args = parser.parse_args()

    dataset = datasets.load_from_disk(args.data_dir)
    print(torch.cuda.mem_get_info())

    torch.backends.cuda.enable_cudnn_sdp(False)
    # ------------------------------------------------------------------
    # CodeCarbon
    # ------------------------------------------------------------------
    tracker = None
    if args.codecarbon:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(
            output_dir=os.getenv('CODECARBON_OUTPUT_DIR', 'outputs/codecarbon'),
            project_name=args.wandb_project,
            country_iso_code=os.getenv('CODECARBON_COUNTRY_ISO_CODE', 'NLD'),
        )
        tracker.start()
            
    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.adapter is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    model.resize_token_embeddings(len(tokenizer))
    
    if args.adapter is not None:
        peft_model = PeftModel.from_pretrained(model, args.adapter)
        model = peft_model.merge_and_unload()

    print(f"SDPA backends:")
    print(f"  Flash: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"  Mem-efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"  Math: {torch.backends.cuda.math_sdp_enabled()}")
    print(f"  cuDNN: {torch.backends.cuda.cudnn_sdp_enabled()}")

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)

    # Quick SDPA smoke test
    print("Testing SDPA forward pass...")
    test_ids = torch.randint(0, 1000, (4, 4096), device=model.device)  # match your batch/seq
    model.eval()
    
    import time
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        model(input_ids=test_ids)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"SDPA forward pass OK! Single forward: {elapsed:.2f}s")
    model.train()
    # ------------------------------------------------------------------
    # Extract or load term alignments
    # ------------------------------------------------------------------
    alignments_train = None
    alignments_val = None
    if args.save_terms_dir:
        t_path = Path(args.save_terms_dir) / "train"
        v_path = Path(args.save_terms_dir) / "validation"
        if t_path.exists() and v_path.exists():
            print("Loading cached term alignments ...")
            alignments_train = load_extracted_terms_dataset(args.save_terms_dir, "train")
            alignments_val = load_extracted_terms_dataset(args.save_terms_dir, "validation")

    if alignments_train is None:
        alignments_train = extract_alignments(dataset['train'], desc="Aligning Train")
        alignments_val = extract_alignments(dataset['validation'], desc="Aligning Val")
        if args.save_terms_dir:
            save_extracted_terms_dataset(dataset['train'], alignments_train,
                                        args.save_terms_dir, "train")
            save_extracted_terms_dataset(dataset['validation'], alignments_val,
                                        args.save_terms_dir, "validation")

<<<<<<< HEAD
    steps_per_epoch = max(1, len(dataset['train']) // (args.batch_size * 8))
=======


    # ------------------------------------------------------------------
    # Chunk documents
    # ------------------------------------------------------------------
    print("Chunking training documents ...")
    train_chunks = build_all_chunks(
        dataset['train'], alignments_train, tokenizer,
        args.max_length, args.target_chunk_tokens,
    )
    print("Chunking validation documents ...")
    val_chunks = build_all_chunks(
        dataset['validation'], alignments_val, tokenizer,
        args.max_length, args.target_chunk_tokens,
    )

    before = len(train_chunks)
    train_chunks = [c for c in train_chunks if estimate_tokens(tokenizer, c['en'] + c['zh']) < args.max_length]
    print(f"Filtered train chunks: {before} -> {len(train_chunks)} (dropped {before - len(train_chunks)} oversized)")
    
    before = len(val_chunks)
    val_chunks = [c for c in val_chunks if estimate_tokens(tokenizer, c['en'] + c['zh']) < args.max_length]
    print(f"Filtered val chunks: {before} -> {len(val_chunks)} (dropped {before - len(val_chunks)} oversized)")

    print(f"Training chunks: {len(train_chunks):,}  (from {len(dataset['train']):,} docs)")
    print(f"Val chunks:      {len(val_chunks):,}  (from {len(dataset['validation']):,} docs)")

    # ------------------------------------------------------------------
    # Debug: verify chunking worked
    # ------------------------------------------------------------------

    def debug_chunks(chunks, tokenizer, split_name, max_tokens):
        """Print detailed stats about chunk sizes to verify nothing is truncated."""
        src_tgt_lengths = []
        full_lengths = []
        term_counts = []
        for chunk in chunks:
            src_tgt_len = estimate_tokens(tokenizer, chunk['en'] + chunk['zh'])
            src_tgt_lengths.append(src_tgt_len)
            term_counts.append(len(chunk['terms']))
            # Estimate full formatted example token count
            rng = random.Random(42)
            record = format_chunk(chunk, rng, tokenizer)
            full_len = estimate_tokens(tokenizer, record["text"])
            full_lengths.append(full_len)

        src_tgt_lengths = np.array(src_tgt_lengths)
        full_lengths = np.array(full_lengths)
        term_counts = np.array(term_counts)
        over_limit = int(np.sum(full_lengths > max_tokens))

        print(f"\n{'='*60}")
        print(f"  CHUNK DEBUG: {split_name}")
        print(f"{'='*60}")
        print(f"  Total chunks:          {len(chunks):,}")
        print(f"  Max length setting:    {max_tokens:,}")
        print(f"")
        print(f"  Source+Target tokens (before prompt/terms):")
        print(f"    Mean:   {np.mean(src_tgt_lengths):,.0f}")
        print(f"    Median: {np.median(src_tgt_lengths):,.0f}")
        print(f"    P90:    {np.percentile(src_tgt_lengths, 90):,.0f}")
        print(f"    P95:    {np.percentile(src_tgt_lengths, 95):,.0f}")
        print(f"    Max:    {np.max(src_tgt_lengths):,}")
        print(f"")
        print(f"  Full formatted example tokens (prompt+src+terms+target):")
        print(f"    Mean:   {np.mean(full_lengths):,.0f}")
        print(f"    Median: {np.median(full_lengths):,.0f}")
        print(f"    P90:    {np.percentile(full_lengths, 90):,.0f}")
        print(f"    P95:    {np.percentile(full_lengths, 95):,.0f}")
        print(f"    Max:    {np.max(full_lengths):,}")
        print(f"    OVER max_length ({max_tokens}): {over_limit} / {len(chunks)} "
              f"({100*over_limit/len(chunks):.1f}%)")
        print(f"")
        print(f"  Terms per chunk:")
        print(f"    Mean:   {np.mean(term_counts):,.1f}")
        print(f"    Median: {np.median(term_counts):,.0f}")
        print(f"    Max:    {np.max(term_counts):,}")
        print(f"    Zero:   {int(np.sum(term_counts == 0))} / {len(chunks)} "
              f"({100*np.sum(term_counts==0)/len(chunks):.1f}%)")
        print(f"{'='*60}\n")

        if over_limit > 0:
            print(f"  WARNING: {over_limit} chunks exceed max_length and WILL be truncated!")
            print(f"  Consider reducing --target_chunk_tokens (currently produces "
                  f"p95={np.percentile(full_lengths, 95):,.0f} tokens)\n")

    debug_chunks(train_chunks, tokenizer, "TRAIN", args.max_length)
    debug_chunks(val_chunks, tokenizer, "VALIDATION", args.max_length)

    # Print a few sample chunks for manual inspection
    print("--- Sample chunk 0 ---")
    c = train_chunks[0]
    print(f"  EN length: {len(c['en'])} chars, ZH length: {len(c['zh'])} chars, Terms: {len(c['terms'])}")
    print(f"  EN preview: {c['en'][:200]}...")
    print(f"  Terms preview: {c['terms'][:5]}")
    if len(train_chunks) > 1:
        print(f"\n--- Sample chunk 1 ---")
        c = train_chunks[1]
        print(f"  EN length: {len(c['en'])} chars, ZH length: {len(c['zh'])} chars, Terms: {len(c['terms'])}")
        print(f"  EN preview: {c['en'][:200]}...")
        print(f"  Terms preview: {c['terms'][:5]}")
    print()

    # ------------------------------------------------------------------
    # Materialize into HF Datasets (epoch 0)
    # ------------------------------------------------------------------
    train_dataset = materialize_dataset(train_chunks, tokenizer, epoch_seed=0)
    val_dataset = materialize_dataset(val_chunks, tokenizer, epoch_seed=999)  # fixed seed for eval

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    steps_per_epoch = max(1, len(train_dataset) // (args.batch_size * 8))
>>>>>>> 02a1c46 (FINAL TRAINING CODe)
    eval_steps = max(1, steps_per_epoch // 10)

    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or os.getenv("WANDB_RUN_GROUP"),
            entity=os.getenv("WANDB_ENTITY"),
            config={
                "output_dir": args.output_dir,
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": args.batch_size,
                "learning_rate": 2e-5,
                "gradient_accumulation_steps": 4,
                "eval_steps": eval_steps,
                "save_steps": eval_steps,
                "metric_for_best_model": "eval_loss",
                "max_length": args.max_length,
<<<<<<< HEAD
=======
                "lora_rank": args.lora_rank,
                "target_chunk_tokens": args.target_chunk_tokens,
                "train_chunks": len(train_chunks),
                "val_chunks": len(val_chunks),
>>>>>>> 02a1c46 (FINAL TRAINING CODe)
            },
            name=f"nmt-sft-{datetime.now().strftime('%Y%m%d-%H%M')}",
        )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["wandb"] if args.wandb else ["none"],
        max_length=args.max_length,
        bf16=True,
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        completion_only_loss=True,
        packing=True
    )

<<<<<<< HEAD
    # Model Loading
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    model.config.pad_token_id = tokenizer.pad_token_id
=======
>>>>>>> 02a1c46 (FINAL TRAINING CODe)


    model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------
    # LoRA — include MLP layers for better terminology compliance
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        callbacks=[EpochReaugmentCallback(train_chunks, tokenizer)],
    )
    print(f"Model dtype: {model.dtype}")
    print(f"Gradient checkpointing: {model.is_gradient_checkpointing}")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(torch.cuda.mem_get_info())
    torch.cuda.empty_cache()
    print(torch.cuda.mem_get_info())
    
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if tracker is not None:
        tracker.stop()


if __name__ == "__main__":
    main()
