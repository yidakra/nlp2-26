#!/usr/bin/env python3
"""
Post-process Track 2 inference outputs:
  1. Map output files to input files via WandB metadata.json
  2. Enrich with references from reference files
  3. Compute ChrF++, BLEU, term coverage
  4. Write outputs/enriched/track2/*.jsonl and outputs/metrics/track2_quality_metrics.csv

Enriched filenames include the model slug so multiple models can coexist:
  {year}.{pair}.{mode}.{strategy}.{model_slug}.jsonl
"""

import csv
import json
import re
import sys
from pathlib import Path

import sacrebleu

REPO_ROOT = Path(__file__).resolve().parents[2]
WANDB_DIR = REPO_ROOT / "wandb"
REF_DIR = REPO_ROOT / "wmt25-terminology" / "ranking" / "references" / "track2"
ENRICHED_DIR = REPO_ROOT / "outputs" / "enriched" / "track2"
METRICS_DIR = REPO_ROOT / "outputs" / "metrics"

PAIR_SRC = {"enzh": "en", "zhen": "zh"}
PAIR_TGT = {"enzh": "zh", "zhen": "en"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_wandb_args(args: list[str]) -> dict:
    d: dict[str, object] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                d[key] = args[i + 1]
                i += 2
            else:
                d[key] = True
                i += 1
        else:
            i += 1
    return d


def scan_wandb_runs() -> dict[str, dict]:
    """Return mapping: output_jsonl_rel_path -> {input_jsonl, prompt_strategy, rerank_strategy, model_id}."""
    mapping: dict[str, dict] = {}
    for meta_file in sorted(WANDB_DIR.glob("run-*/files/wandb-metadata.json")):
        try:
            with open(meta_file, encoding="utf-8") as f:
                meta = json.load(f)
            d = parse_wandb_args(meta.get("args", []))
            input_jsonl = d.get("input_jsonl")
            output_jsonl = d.get("output_jsonl")
            if not input_jsonl or not output_jsonl:
                continue
            if "track2" not in str(input_jsonl):
                continue
            if d.get("max_new_tokens") != "4096":
                continue
            mapping[str(output_jsonl)] = {
                "input_jsonl": str(input_jsonl),
                "prompt_strategy": str(d.get("prompt_strategy", "baseline")),
                "rerank_strategy": str(d.get("rerank_strategy", "none")),
                "model_id": str(d.get("model_id", "google/gemma-4-E2B-it")),
            }
        except Exception:
            continue
    return mapping


def filter_terms_by_source(terms: dict, src: str) -> dict:
    """Keep only terms whose source side appears in the source text."""
    src_lower = src.lower()
    filtered = {k: v for k, v in terms.items() if str(k).lower() in src_lower}
    return filtered if filtered else terms


def compute_tc(
    mt_texts: list[str],
    term_dicts: list[dict],
    src_texts: list[str] | None = None,
) -> float | None:
    """Compute naive term coverage.

    If src_texts is provided, TC is computed over terms that appear in the
    source text (matching inference-time filtering). Otherwise all glossary
    terms are used as the denominator.
    """
    total = 0
    covered = 0
    for idx, (mt, terms) in enumerate(zip(mt_texts, term_dicts)):
        if not terms:
            continue
        if src_texts is not None:
            terms = filter_terms_by_source(terms, src_texts[idx])
            if not terms:
                continue
        mt_lower = mt.lower()
        for tgt_val in terms.values():
            targets: list[str] = tgt_val if isinstance(tgt_val, list) else [str(tgt_val)]
            for target in targets:
                target_lower = target.lower().strip()
                if not target_lower:
                    continue
                total += 1
                if target_lower.isascii() and any(c.isalnum() or c == "_" for c in target_lower):
                    pattern = re.compile(
                        r"(?<![A-Za-z0-9_])" + re.escape(target_lower) + r"(?![A-Za-z0-9_])",
                        re.IGNORECASE,
                    )
                    covered += bool(pattern.search(mt))
                else:
                    covered += target_lower in mt_lower
    return (covered / total) if total > 0 else None


def main() -> None:
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Scanning WandB runs for Track 2 output files...")
    mapping = scan_wandb_runs()
    print(f"Found {len(mapping)} Track 2 run(s)")

    if not mapping:
        print("No Track 2 runs found with max_new_tokens=4096. Jobs may not have started yet.")
        sys.exit(0)

    missing_outputs = [p for p in mapping if not (REPO_ROOT / p).exists()]
    if missing_outputs:
        print(f"WARNING: {len(missing_outputs)} output files not yet written (jobs may still be running):")
        for p in sorted(missing_outputs)[:5]:
            print(f"  {p}")

    csv_rows: list[dict] = []

    for output_rel, meta in sorted(mapping.items()):
        output_path = REPO_ROOT / output_rel
        if not output_path.exists():
            continue

        model_id = meta["model_id"]
        model_slug = Path(model_id).name  # e.g. "gemma-4-E2B-it" or "Qwen3.5-9B"

        input_fname = Path(meta["input_jsonl"]).stem  # e.g. "2023.enzh.noterm"
        parts = input_fname.split(".")
        if len(parts) != 3:
            print(f"  SKIP unexpected filename: {input_fname}")
            continue
        year, pair, mode = parts
        strategy = meta["prompt_strategy"]
        rerank = meta["rerank_strategy"]

        outputs = load_jsonl(output_path)
        if not outputs:
            print(f"  EMPTY: {output_rel}")
            continue

        ref_file = REF_DIR / f"full_data_{year}.jsonl"
        if not ref_file.exists():
            print(f"  MISSING ref: {ref_file}")
            continue
        refs = load_jsonl(ref_file)

        src_lang = PAIR_SRC.get(pair)
        tgt_lang = PAIR_TGT.get(pair)
        if src_lang is None or tgt_lang is None:
            print(f"  WARNING: unknown pair {pair!r}, skipping {output_rel}")
            continue

        # Build ref lookup by first 120 chars of source text
        ref_by_src: dict[str, dict] = {}
        for r in refs:
            key = r.get(src_lang, "").strip()[:120]
            ref_by_src[key] = r

        mt_texts: list[str] = []
        src_texts: list[str] = []
        ref_texts: list[str] = []
        term_dicts: list[dict] = []
        matched_rows: list[tuple[dict, str]] = []

        for out_row in outputs:
            src_text = out_row.get("src", "")
            src_key = src_text.strip()[:120]
            ref_row = ref_by_src.get(src_key)
            if ref_row is None:
                print(f"  WARNING: no ref match in {year} for a src row")
                continue
            src_texts.append(src_text)
            mt_texts.append(out_row.get("mt", ""))
            ref_text = ref_row.get(tgt_lang, "")
            ref_texts.append(ref_text)
            term_dicts.append(ref_row.get(mode, {}) if mode in ("proper", "random") else {})
            matched_rows.append((out_row, ref_text))

        if not mt_texts:
            print(f"  SKIP (no matched rows): {output_rel}")
            continue

        chrf_pp = sacrebleu.corpus_chrf(mt_texts, [ref_texts], word_order=2).score
        bleu = sacrebleu.corpus_bleu(mt_texts, [ref_texts]).score
        tc = compute_tc(mt_texts, term_dicts, src_texts=src_texts)
        tc_pct = round(tc * 100, 1) if tc is not None else None

        label = f"{strategy}_k0"
        print(
            f"  [{model_slug}] {year} {pair:4s} {mode:7s} {label:18s} "
            f"ChrF++={chrf_pp:.2f}  BLEU={bleu:.2f}  TC={f'{tc_pct}%' if tc_pct is not None else 'N/A'}"
        )

        # Write enriched file (model slug in filename so multiple models coexist)
        enriched_path = ENRICHED_DIR / f"{year}.{pair}.{mode}.{strategy}.{model_slug}.jsonl"
        with open(enriched_path, "w", encoding="utf-8") as f:
            for out_row, ref_text in matched_rows:
                enriched = {"src": out_row.get("src", ""), "ref": ref_text, "mt": out_row.get("mt", "")}
                f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

        csv_rows.append({
            "model": model_slug,
            "year": year,
            "pair": pair,
            "mode": mode,
            "strategy": strategy,
            "rerank": rerank,
            "n_docs": len(mt_texts),
            "chrf_pp": round(chrf_pp, 2),
            "bleu": round(bleu, 2),
            "tc": tc_pct,
        })

    if csv_rows:
        csv_path = METRICS_DIR / "track2_quality_metrics.csv"
        fieldnames = ["model", "year", "pair", "mode", "strategy", "rerank", "n_docs", "chrf_pp", "bleu", "tc"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(csv_rows, key=lambda r: (r["model"], r["pair"], r["year"], r["mode"], r["strategy"])))
        print(f"\nSaved {len(csv_rows)} rows → {csv_path}")
    else:
        print("No complete runs to summarise yet.")


if __name__ == "__main__":
    main()
