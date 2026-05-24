#!/usr/bin/env python3
"""
Terminology-Aware Neural Machine Translation — GRPO Training Script

Companion to the SFT training script. Uses Group Relative Policy Optimization
with a programmatic term-coverage reward instead of supervised cross-entropy.

Designed to run on top of an SFT checkpoint:
    python grpo_train.py --model base_model --adapter sft_lora_adapter ...

Key design decisions:
  - Reward = term coverage, gated by sanity checks (language, length, repetition,
    term density) to prevent reward hacking
  - Only chunks WITH terminology are used (no reward signal without terms)
  - Chunking pipeline is shared with the SFT script
  - Direction augmentation is applied once at dataset construction
"""

import argparse
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import datasets
import jieba
import numpy as np
import torch
from peft import LoraConfig, PeftModel
from pycccedict.cccedict import CcCedict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


# ---------------------------------------------------------------------------
# Constants (shared with SFT script)
# ---------------------------------------------------------------------------

LANG_INFO = {
    "enzh": {"src": "en", "tgt": "zh", "src_full": "English", "tgt_full": "Traditional Chinese"},
    "zhen": {"src": "zh", "tgt": "en", "src_full": "Traditional Chinese", "tgt_full": "English"},
}

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
# Term extraction helpers (identical to SFT script)
# ---------------------------------------------------------------------------

def build_cccedict_mapping():
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


def limit_terms_per_document(terms, rng, min_terms=50, max_terms=170):
    if len(terms) <= min_terms:
        return terms
    max_keep = rng.randint(min_terms, max_terms)
    return rng.sample(terms, min(len(terms), max_keep))


def extract_alignments(dataset, desc="Extracting alignments"):
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
# GRPO-specific term filtering
# ---------------------------------------------------------------------------

# Patterns that indicate extraction artifacts, not real terminology
_NUMERIC_ONLY = re.compile(r'^[\d\.\-–,\s]+$')
_LEGAL_REF_ZH = re.compile(r'第.{1,6}[條章節部]')
_PAREN_HEAVY = re.compile(r'[()（）《》【】\[\]]{3,}')


def is_valid_grpo_term(term: str, lang: str) -> bool:
    """
    Strict term validation for GRPO reward computation.
    Filters out extraction artifacts that a correct translation
    could never reasonably be expected to contain as substrings.
    """
    t = term.strip()
    if not t:
        return False

    # Pure numbers / number ranges (e.g. '2125.1–2139.9')
    if _NUMERIC_ONLY.match(t):
        return False

    if lang == "zh":
        # Must actually contain CJK characters — reject English words
        # masquerading as Chinese terms
        cjk_count = sum(1 for c in t if '\u4e00' <= c <= '\u9fff')
        if cjk_count < 1:
            return False
        # Chinese terms should be short — real terms are 2-12 characters.
        # Anything longer is a sentence fragment or legal clause.
        if len(t) > 15:
            return False
        # Single character — too ambiguous, many false positive matches
        if len(t) < 2:
            return False
        # Contains legal section references (extraction artifact)
        if _LEGAL_REF_ZH.search(t):
            return False
    else:
        # Must contain at least one ASCII letter
        if not any(c.isascii() and c.isalpha() for c in t):
            return False
        # English terms: real terminology is at most a few words
        if len(t) > 60:
            return False
        if len(t) < 2:
            return False
        # Stopwords that slipped through
        if t.lower() in STOPWORDS_EN:
            return False
        # Fragments with unbalanced parens or brackets
        if t.count('(') != t.count(')'):
            return False

    # Too many special characters — likely a formatting artifact
    if _PAREN_HEAVY.search(t):
        return False

    return True


def filter_terms_for_grpo(terms: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Apply strict filtering to term pairs for GRPO training.
    Both source and target must pass validation.
    """
    filtered = []
    for src, tgt in terms:
        # src is always English, tgt is always Chinese in the base extraction
        if is_valid_grpo_term(src, "en") and is_valid_grpo_term(tgt, "zh"):
            filtered.append((src, tgt))
    return filtered


# ---------------------------------------------------------------------------
# Term dataset persistence (identical to SFT script)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Document chunking (identical to SFT script)
# ---------------------------------------------------------------------------

def estimate_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_document(en, zh, terms, tokenizer, max_tokens=8192, target_chunk_tokens=3500):
    term_str = ', '.join(f'{s} -> {t}' for s, t in terms) if terms else ''
    total_est = estimate_tokens(tokenizer, en + zh + term_str) + 200
    if total_est < max_tokens:
        return [{'en': en, 'zh': zh, 'terms': terms}]
    src_tgt_tokens = estimate_tokens(tokenizer, en + zh)
    n_chunks = max(2, -(-src_tgt_tokens // target_chunk_tokens))
    en_chunks = _split_text_into_n(en, n_chunks)
    zh_chunks = _split_text_into_n(zh, n_chunks)
    chunks = []
    for ec, zc in zip(en_chunks, zh_chunks):
        if not ec.strip() and not zc.strip():
            continue
        ec_lower = ec.lower()
        chunk_terms = [(s, t) for s, t in terms if s.lower() in ec_lower]
        chunks.append({'en': ec, 'zh': zc, 'terms': chunk_terms})
    return chunks if chunks else [{'en': en, 'zh': zh, 'terms': terms}]


def _split_text_into_n(text, n):
    if n <= 1:
        return [text]
    parts = re.split(r'(\n+)', text)
    segments = []
    current = ""
    for part in parts:
        current += part
        if re.match(r'^\n+$', part):
            segments.append(current)
            current = ""
    if current:
        segments.append(current)
    if len(segments) < n:
        new_segments = []
        for seg in segments:
            sub = re.split(r'(?<=[.!?。！？])\s+', seg)
            new_segments.extend(sub)
        segments = [s for s in new_segments if s.strip()]
    if len(segments) < n:
        chunk_size = max(1, len(text) // n)
        return [text[i * chunk_size:(i + 1) * chunk_size] for i in range(n)]
    total_chars = sum(len(s) for s in segments)
    target_per_chunk = total_chars / n
    chunks = []
    current_chunk = []
    current_len = 0
    for seg in segments:
        current_chunk.append(seg)
        current_len += len(seg)
        if current_len >= target_per_chunk and len(chunks) < n - 1:
            chunks.append(''.join(current_chunk))
            current_chunk = []
            current_len = 0
    if current_chunk:
        chunks.append(''.join(current_chunk))
    return chunks


def build_all_chunks(dataset, alignments, tokenizer, max_tokens, target_chunk_tokens):
    all_chunks = []
    for i, example in enumerate(tqdm(dataset, desc="Chunking documents")):
        doc_chunks = chunk_document(
            example['en'], example['zh'], alignments[i],
            tokenizer, max_tokens, target_chunk_tokens,
        )
        all_chunks.extend(doc_chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Reward function components
# ---------------------------------------------------------------------------

def term_present_zh(text: str, term: str) -> bool:
    """Check if a Chinese term appears in the text (substring match).
    Single-character terms are skipped — too many false positives."""
    if len(term) < 2:
        return False
    return term in text


def term_present_en(text: str, term: str) -> bool:
    """Check if an English term appears in the text (word-boundary match)."""
    pattern = r'\b' + re.escape(term.lower()) + r'\b'
    return bool(re.search(pattern, text.lower()))


def compute_term_coverage(completion: str, target_terms: list[str], tgt_lang: str) -> float:
    """Fraction of required target-language terms present in the completion.
    Skips terms that fail GRPO validation and deduplicates as a safety net."""
    # Filter and deduplicate
    valid_terms = list(dict.fromkeys(
        t for t in target_terms if is_valid_grpo_term(t, tgt_lang)
    ))
    if not valid_terms:
        return 1.0
    check_fn = term_present_zh if tgt_lang == "zh" else term_present_en
    hits = sum(1 for t in valid_terms if check_fn(completion, t))
    return hits / len(valid_terms)


def compute_length_ratio_penalty(completion: str, source_len: int) -> float:
    """
    Penalise completions whose character length is wildly disproportionate
    to the source. Generous bounds — we only want to catch degenerate cases.

    Returns 1.0 for normal translations, decays to 0.0 at the extremes.
    """
    comp_len = len(completion.strip())
    if source_len <= 0:
        return 1.0 if comp_len > 0 else 0.0

    ratio = comp_len / source_len

    # Very short (likely empty / truncated)
    if ratio < 0.05:
        return 0.0
    if ratio < 0.15:
        return (ratio - 0.05) / 0.10  # linear ramp 0.05 → 0.15

    # Very long (likely term stuffing or degenerate repetition)
    if ratio > 5.0:
        return 0.0
    if ratio > 3.0:
        return 1.0 - (ratio - 3.0) / 2.0  # linear decay 3.0 → 5.0

    return 1.0


def compute_repetition_penalty(text: str, n: int = 4) -> float:
    """
    Detect degenerate repetitive output via n-gram uniqueness ratio.
    Returns 1.0 for normal text, decays toward 0.0 for highly repetitive text.

    Works on character-level n-grams so it handles both Chinese and English.
    """
    chars = list(text.strip())
    if len(chars) < n * 2:
        return 1.0

    ngrams = [tuple(chars[i:i + n]) for i in range(len(chars) - n + 1)]
    unique_ratio = len(set(ngrams)) / len(ngrams)

    if unique_ratio > 0.4:
        return 1.0
    if unique_ratio < 0.05:
        return 0.0
    return unique_ratio / 0.4


def compute_term_density_penalty(completion: str, target_terms: list[str]) -> float:
    """
    Catch outputs that are just concatenated terms with no real translation.
    If the matched terms account for nearly all of the output characters,
    the model is likely stuffing terms rather than translating.

    Returns 1.0 for normal translations, decays to 0.0 for term-only outputs.
    """
    comp_stripped = completion.strip()
    if not comp_stripped or not target_terms:
        return 1.0

    total_term_chars = sum(len(t) for t in target_terms if t in comp_stripped)
    density = total_term_chars / len(comp_stripped)

    if density < 0.6:
        return 1.0
    if density > 0.9:
        return 0.0
    return 1.0 - (density - 0.6) / 0.3


def compute_language_consistency(text: str, expected_lang: str) -> float:
    """
    Check that the output is predominantly in the expected language.
    Prevents reward-hacking by copying the source (wrong language).

    Returns 1.0 if consistent, 0.0 if clearly wrong language.
    """
    text = text.strip()
    if not text:
        return 0.0

    if expected_lang == "zh":
        # Count CJK Unified Ideographs
        cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        ratio = cjk / len(text)
        # Chinese translations should have at least 20% CJK characters
        # (accounting for punctuation, numbers, occasional English terms)
        if ratio > 0.25:
            return 1.0
        if ratio < 0.10:
            return 0.0
        return (ratio - 0.10) / 0.15
    else:
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        ratio = ascii_alpha / len(text)
        if ratio > 0.30:
            return 1.0
        if ratio < 0.10:
            return 0.0
        return (ratio - 0.10) / 0.20


def compute_source_copy_penalty(completion: str, source: str) -> float:
    """
    Penalise if the completion is largely a copy of the source text.
    Uses longest common substring ratio as a proxy.
    Only checks a prefix to keep it fast — full LCS is O(n*m).
    """
    comp = completion.strip()[:500]
    src = source.strip()[:500]

    if not comp or not src:
        return 1.0

    # Quick overlap check: what fraction of completion 4-grams appear in source?
    n = 4
    if len(comp) < n:
        return 1.0

    comp_ngrams = set(comp[i:i + n] for i in range(len(comp) - n + 1))
    src_ngrams = set(src[i:i + n] for i in range(len(src) - n + 1))

    if not comp_ngrams:
        return 1.0

    overlap = len(comp_ngrams & src_ngrams) / len(comp_ngrams)

    if overlap < 0.3:
        return 1.0
    if overlap > 0.7:
        return 0.0
    return 1.0 - (overlap - 0.3) / 0.4


# ---------------------------------------------------------------------------
# Combined reward function
# ---------------------------------------------------------------------------

def terminology_reward_fn(completions: list, **kwargs) -> list[float]:
    """
    GRPO reward function for terminology-aware NMT.

    Reward = term_coverage × min(all sanity gates)

    The sanity gates (language consistency, length ratio, repetition, term density,
    source copying) are multiplied together as a single gating factor. Each gate
    returns 1.0 for normal translations and decays toward 0.0 for degenerate outputs.
    Using min() instead of product avoids double-penalising correlated failure modes,
    while still zeroing out clearly hacked outputs.

    The overall structure ensures that:
      - The ONLY way to get a high reward is to produce a fluent, correctly-languaged
        translation that contains the required terminology.
      - Degenerate strategies (term stuffing, source copying, repetition, empty output)
        are gated to near-zero regardless of how many terms they contain.
    """
    terms_json_list = kwargs.get("terms_json", [])
    tgt_lang_list = kwargs.get("tgt_lang", [])
    source_text_list = kwargs.get("source_text", [])
    source_len_list = kwargs.get("source_len", [])

    rewards = []
    for i, completion in enumerate(completions):
        # trl passes completions as chat-format lists: [{"role": "assistant", "content": "..."}]
        if isinstance(completion, list):
            completion = completion[0]["content"] if completion else ""
        elif not isinstance(completion, str):
            completion = str(completion)

        # --- Parse metadata ---
        try:
            target_terms = json.loads(terms_json_list[i])
        except (IndexError, json.JSONDecodeError):
            target_terms = []

        tgt_lang = tgt_lang_list[i] if i < len(tgt_lang_list) else "zh"
        source_text = source_text_list[i] if i < len(source_text_list) else ""
        source_len = int(source_len_list[i]) if i < len(source_len_list) else len(source_text)

        # --- Empty output: zero reward immediately ---
        if not completion or not completion.strip():
            rewards.append(0.0)
            continue

        # --- Main signal: term coverage (0.0 to 1.0) ---
        coverage = compute_term_coverage(completion, target_terms, tgt_lang)

        # --- Sanity gates: each returns 1.0 for normal output, <1.0 for degenerate ---
        gates = [
            compute_language_consistency(completion, tgt_lang),
            compute_length_ratio_penalty(completion, source_len),
            compute_repetition_penalty(completion),
            compute_term_density_penalty(completion, target_terms),
            compute_source_copy_penalty(completion, source_text),
        ]

        # Use min so one clear violation is enough to suppress the reward,
        # but borderline values on multiple gates don't compound unfairly.
        gate = min(gates)

        reward = coverage * gate
        rewards.append(float(reward))

    return rewards


# ---------------------------------------------------------------------------
# Prompt dataset construction
# ---------------------------------------------------------------------------

def format_grpo_prompt(chunk: dict, rng: random.Random) -> dict | None:
    """
    Create a GRPO training example from a chunk.
    Returns None for chunks with no usable terms (no reward signal).

    The prompt includes terminology so the model knows what to use.
    The reward function independently verifies the terms appear in the output.
    """
    # Apply strict GRPO filtering BEFORE direction swap (terms are always (en, zh))
    clean_terms = filter_terms_for_grpo(chunk['terms'])
    if not clean_terms:
        return None

    # Random direction
    is_zh_to_en = rng.random() < 0.5

    if not is_zh_to_en:
        lang = LANG_INFO["enzh"]
        source, target_ref = chunk['en'], chunk['zh']
        terms = clean_terms  # (en, zh)
        tgt_lang = "zh"
    else:
        lang = LANG_INFO["zhen"]
        source, target_ref = chunk['zh'], chunk['en']
        terms = [(zh, en) for en, zh in clean_terms]
        tgt_lang = "en"

    # For GRPO, we ALWAYS show terms — the whole point is to train term compliance
    # Deduplicate: duplicates inflate the denominator and dilute coverage signal
    seen_pairs = set()
    unique_terms = []
    for s, t in terms:
        if (s, t) not in seen_pairs:
            seen_pairs.add((s, t))
            unique_terms.append((s, t))
    terms = unique_terms

    target_terms = list(dict.fromkeys(tgt for _, tgt in terms))  # deduplicate, preserve order
    terminology = ', '.join(f'{s} -> {t}' for s, t in terms)

    prompt_text = (
        f"Translate the following document from {lang['src_full']} "
        f"to {lang['tgt_full']}, respecting the given terminology. "
        f"Output the translation and nothing else.\n\n"
        f"Source: {source}\n"
        f"Terminology: {terminology}\n\n"
    )

    prompt_messages = [{"role": "user", "content": prompt_text}]

    return {
        "prompt": prompt_messages,
        "terms_json": json.dumps(target_terms),
        "tgt_lang": tgt_lang,
        "source_text": source,
        "source_len": len(source),
    }


def build_grpo_dataset(chunks: list[dict], seed: int = 0) -> datasets.Dataset:
    """
    Build the prompt-only dataset for GRPO.
    Filters out chunks without terms.
    """
    records = []
    for i, chunk in enumerate(chunks):
        rng = random.Random(seed * 100_000 + i)
        record = format_grpo_prompt(chunk, rng)
        if record is not None:
            records.append(record)

    print(f"  Built {len(records)} GRPO examples from {len(chunks)} chunks "
          f"(dropped {len(chunks) - len(records)} without terms)")

    return datasets.Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Debug & diagnostics
# ---------------------------------------------------------------------------

def debug_reward_robustness(dataset, n_samples=5):
    """
    Run the reward function on synthetic adversarial examples to verify
    that degenerate strategies get low scores.
    """
    print(f"\n{'=' * 60}")
    print("  REWARD ROBUSTNESS CHECK")
    print(f"{'=' * 60}")

    sample_indices = list(range(min(n_samples, len(dataset))))

    for idx in sample_indices:
        ex = dataset[idx]
        terms = json.loads(ex["terms_json"])
        source = ex["source_text"]
        tgt_lang = ex["tgt_lang"]
        source_len = ex["source_len"]

        print(f"\n  --- Example {idx} | tgt_lang={tgt_lang} | {len(terms)} terms ---")
        # Show which terms survive GRPO filtering
        valid_terms = [t for t in terms if is_valid_grpo_term(t, tgt_lang)]
        print(f"  Raw terms: {len(terms)}, valid after GRPO filter: {len(valid_terms)}")
        print(f"  Valid terms preview: {valid_terms[:8]}")

        # Build kwargs for reward function
        def score(completion, label):
            r = terminology_reward_fn(
                [completion],
                terms_json=[ex["terms_json"]],
                tgt_lang=[tgt_lang],
                source_text=[source],
                source_len=[source_len],
            )[0]
            status = "PASS" if r < 0.15 else "FAIL (should be ~0)"
            if label == "good_translation":
                status = f"reward={r:.3f} (want > 0)"
            print(f"    {label:30s} -> reward={r:.3f}  {status}")
            return r

        # 1. Synthetic "good" translation: correct language, reasonable length,
        #    embeds ALL valid terms into filler text so coverage should be high.
        if valid_terms:
            if tgt_lang == "zh":
                filler = "根據相關條例的規定，"
                good = filler + "、".join(valid_terms) + "。" + filler * max(1, source_len // 80)
            else:
                filler = "According to the relevant ordinance, "
                good = filler + ", ".join(valid_terms) + ". " + filler * max(1, source_len // 200)
        else:
            good = "（無有效術語可測試）" if tgt_lang == "zh" else "(no valid terms to test)"
        score(good, "good_translation")

        # 2. Empty output
        score("", "empty")

        # 3. Just whitespace
        score("   \n\n  ", "whitespace_only")

        # 4. Term stuffing: concatenate all valid terms with no real sentence structure
        score(" ".join(valid_terms) if valid_terms else "", "term_stuffing")

        # 5. Source copying
        score(source[:500], "source_copy")

        # 6. Repetition
        score("翻譯 " * 200 if tgt_lang == "zh" else "translate " * 200, "repetitive_output")

        # 7. Wrong language (Chinese when English expected, vice versa)
        if tgt_lang == "zh":
            score("This is English text that should be Chinese instead.", "wrong_language")
        else:
            score("這應該是英文但卻是中文。", "wrong_language")

        # 8. Very long output (5x source)
        bloat = ("很長的文本。" if tgt_lang == "zh" else "Very long text. ") * (source_len // 2)
        score(bloat, "bloated_output")

    print(f"\n{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train terminology-aware NMT with GRPO and programmatic reward"
    )
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--adapter", default=None,
                        help="Path to SFT LoRA adapter to merge before GRPO (recommended)")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_terms_dir", default=None)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max prompt + completion length")
    parser.add_argument("--max_completion_length", type=int, default=768,
                        help="Max tokens the model may generate per sample (keep tight to save time)")
    parser.add_argument("--target_chunk_tokens", type=int, default=600,
                        help="Target src+tgt tokens per chunk (lower than SFT — completions must finish)")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of completions per prompt for GRPO group")
    parser.add_argument("--generation_batch_size", type=int, default=4,
                        help="Total sequences per generation batch (must be multiple of num_generations). "
                             "Default=num_generations means one prompt group at a time.")
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient against reference policy")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Subsample training set to this many examples (for time budget)")
    parser.add_argument("--max_eval_samples", type=int, default=100,
                        help="Subsample eval set (eval generates completions too — very slow)")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="nlp2-26")
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--codecarbon", action="store_true")
    args = parser.parse_args()

    dataset = datasets.load_from_disk(args.data_dir)
    print(torch.cuda.mem_get_info())

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
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)

    # Merge SFT adapter if provided — GRPO trains a new LoRA on top
    if args.adapter is not None:
        print(f"Merging SFT adapter from {args.adapter} ...")
        peft_model = PeftModel.from_pretrained(model, args.adapter)
        model = peft_model.merge_and_unload()
        print("  Adapter merged.")

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

    # ------------------------------------------------------------------
    # Chunk documents (use smaller chunks than SFT since generation is expensive)
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

    # Filter oversized chunks
    before = len(train_chunks)
    train_chunks = [c for c in train_chunks
                    if estimate_tokens(tokenizer, c['en'] + c['zh']) < args.max_length]
    print(f"Filtered train chunks: {before} -> {len(train_chunks)}")

    before = len(val_chunks)
    val_chunks = [c for c in val_chunks
                  if estimate_tokens(tokenizer, c['en'] + c['zh']) < args.max_length]
    print(f"Filtered val chunks: {before} -> {len(val_chunks)}")

    # ------------------------------------------------------------------
    # Build GRPO datasets (prompt-only, with metadata for reward fn)
    # ------------------------------------------------------------------
    print("Building GRPO training dataset ...")
    train_dataset = build_grpo_dataset(train_chunks, seed=0)
    print("Building GRPO validation dataset ...")
    val_dataset = build_grpo_dataset(val_chunks, seed=999)

    print(f"GRPO training examples: {len(train_dataset):,}")
    print(f"GRPO validation examples: {len(val_dataset):,}")

    # Subsample if requested (to fit within time budget)
    if args.max_train_samples and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.shuffle(seed=42).select(range(args.max_train_samples))
        print(f"Subsampled training set to {len(train_dataset):,} examples")

    if args.max_eval_samples and len(val_dataset) > args.max_eval_samples:
        val_dataset = val_dataset.shuffle(seed=42).select(range(args.max_eval_samples))
        print(f"Subsampled eval set to {len(val_dataset):,} examples")

    # ------------------------------------------------------------------
    # Reward robustness check
    # ------------------------------------------------------------------
    debug_reward_robustness(train_dataset, n_samples=3)

    # ------------------------------------------------------------------
    # Training config
    # ------------------------------------------------------------------
    # GRPO effective batch = batch_size × gradient_accumulation × num_generations
    # With batch_size=2, grad_accum=4, num_gen=4 → 32 completions per update step
    grad_accum = 4
    steps_per_epoch = max(1, len(train_dataset) * args.num_generations // (args.batch_size * grad_accum))
    eval_steps = max(1, steps_per_epoch // 5)
    print(f"Steps per epoch: {steps_per_epoch}, eval every {eval_steps} steps")
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group or os.getenv("WANDB_RUN_GROUP"),
            entity=os.getenv("WANDB_ENTITY"),
            config={
                "method": "GRPO",
                "output_dir": args.output_dir,
                "num_train_epochs": args.epochs,
                "per_device_train_batch_size": args.batch_size,
                "learning_rate": 5e-6,
                "gradient_accumulation_steps": grad_accum,
                "num_generations": args.num_generations,
                "generation_batch_size": args.generation_batch_size,
                "beta": args.beta,
                "temperature": args.temperature,
                "max_completion_length": args.max_completion_length,
                "max_length": args.max_length,
                "lora_rank": args.lora_rank,
                "target_chunk_tokens": args.target_chunk_tokens,
                "train_examples": len(train_dataset),
                "val_examples": len(val_dataset),
                "adapter": args.adapter,
            },
            name=f"nmt-grpo-{datetime.now().strftime('%Y%m%d-%H%M')}",
        )

    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.num_generations,  # must be divisible by num_generations
        gradient_accumulation_steps=grad_accum,

        # GRPO-specific
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,  # generate this many at a time to avoid OOM
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        temperature=args.temperature,

        # Optimisation
        learning_rate=5e-6,          # Conservative LR for RL fine-tuning
        warmup_ratio=0.05,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=0.5,           # Tighter clipping for RL stability

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        load_best_model_at_end=False,  # GRPO doesn't use eval_loss the same way
        report_to=["wandb"] if args.wandb else ["none"],

        # Logging
        logging_steps=10,
        chat_template_kwargs={"enable_thinking": False}
    )

    # ------------------------------------------------------------------
    # LoRA — new adapter for GRPO stage
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=terminology_reward_fn,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print(f"Model dtype: {model.dtype}")
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
