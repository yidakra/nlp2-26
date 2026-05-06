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
import logging
import re
import sys
from pathlib import Path
from typing import Any, cast

import sacrebleu

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
WANDB_DIR = REPO_ROOT / "wandb"
REF_DIR = REPO_ROOT / "wmt25-terminology" / "ranking" / "references" / "track2"
ENRICHED_DIR = REPO_ROOT / "outputs" / "enriched" / "track2"
METRICS_DIR = REPO_ROOT / "outputs" / "metrics"

PAIR_SRC = {"enzh": "en", "zhen": "zh"}
PAIR_TGT = {"enzh": "zh", "zhen": "en"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_wandb_args(args: list[str]) -> dict[str, object]:
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


def _job_id_from_path(output_jsonl: str) -> int:
    """Extract SLURM job ID from output filename for recency comparison."""
    stem = Path(output_jsonl).stem  # e.g. "enzh_22181042_20260423_..."
    parts = stem.split("_")
    for part in parts:
        if part.isdigit() and len(part) >= 7:
            return int(part)
    return 0


def scan_wandb_runs() -> dict[str, dict[str, str]]:
    """Return mapping: output_jsonl_rel_path -> {input_jsonl, prompt_strategy, rerank_strategy, model_id}.

    When multiple runs share the same logical configuration (same input, strategy,
    and model), only the most recent one (highest SLURM job ID) is kept.
    """
    # key -> (job_id, output_jsonl, meta_dict)
    best: dict[tuple[str, str, str, str], tuple[int, str, dict[str, str]]] = {}

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
            try:
                if int(str(d.get("max_new_tokens", 0))) < 4096:
                    continue
            except (ValueError, TypeError):
                continue
            enable_thinking = str(d.get("enable_thinking", "False")).lower() in ("true", "1")
            prompt_strategy = str(d.get("prompt_strategy", "baseline"))
            effective_strategy = f"{prompt_strategy}_thinking" if enable_thinking else prompt_strategy
            adapter = str(d["adapter"]) if d.get("adapter") else ""
            meta_dict = {
                "input_jsonl": str(input_jsonl),
                "prompt_strategy": effective_strategy,
                "rerank_strategy": str(d.get("rerank_strategy", "none")),
                "model_id": str(d.get("model_id", "google/gemma-4-E2B-it")),
                "adapter": adapter,
            }
            config_key = (
                meta_dict["input_jsonl"],
                meta_dict["prompt_strategy"],
                meta_dict["rerank_strategy"],
                meta_dict["model_id"],
                meta_dict["adapter"],
            )
            job_id = _job_id_from_path(str(output_jsonl))
            if config_key not in best or job_id > best[config_key][0]:
                best[config_key] = (job_id, str(output_jsonl), meta_dict)
        except Exception:
            logger.exception("Failed to parse WandB metadata for %s", meta_file)
            continue

    return {output_jsonl: meta_dict for _, output_jsonl, meta_dict in best.values()}


def filter_terms_by_source(terms: dict[str, Any], src: str) -> dict[str, Any]:
    """Keep only terms whose source side appears in the source text."""
    src_lower = src.lower()
    filtered = {k: v for k, v in terms.items() if str(k).lower() in src_lower}
    return filtered if filtered else terms


def compute_tc(
    mt_texts: list[str],
    term_dicts: list[dict[str, Any]],
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
            targets: list[str] = [str(x) for x in cast(list[object], tgt_val)] if isinstance(tgt_val, list) else [str(tgt_val)]
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

    logger.info("Scanning WandB runs for Track 2 output files...")
    mapping = scan_wandb_runs()
    logger.info("Found %d Track 2 run(s)", len(mapping))

    if not mapping:
        logger.info("No Track 2 runs found with max_new_tokens=4096. Jobs may not have started yet.")
        sys.exit(0)

    missing_outputs = [p for p in mapping if not (REPO_ROOT / p).exists()]
    if missing_outputs:
        logger.warning(
            "%d output files not yet written (jobs may still be running):",
            len(missing_outputs),
        )
        for p in sorted(missing_outputs)[:5]:
            logger.warning("  %s", p)

    csv_rows: list[dict[str, object]] = []

    for output_rel, meta in sorted(mapping.items()):
        output_path = REPO_ROOT / output_rel
        if not output_path.exists():
            continue

        model_id = meta["model_id"]
        adapter = meta.get("adapter", "")
        base_slug = Path(model_id).name  # e.g. "gemma-4-E2B-it" or "Qwen3.5-9B"
        model_slug = f"{base_slug}-{Path(adapter).name}" if adapter else base_slug

        input_fname = Path(meta["input_jsonl"]).stem  # e.g. "2023.enzh.noterm"
        parts = input_fname.split(".")
        if len(parts) != 3:
            logger.warning("  SKIP unexpected filename: %s", input_fname)
            continue
        year, pair, mode = parts
        strategy = meta["prompt_strategy"]
        rerank = meta["rerank_strategy"]

        outputs = load_jsonl(output_path)
        if not outputs:
            logger.warning("  EMPTY: %s", output_rel)
            continue

        ref_file = REF_DIR / f"full_data_{year}.jsonl"
        if not ref_file.exists():
            logger.warning("  MISSING ref: %s", ref_file)
            continue
        refs = load_jsonl(ref_file)

        src_lang = PAIR_SRC.get(pair)
        tgt_lang = PAIR_TGT.get(pair)
        if src_lang is None or tgt_lang is None:
            logger.warning("  WARNING: unknown pair %r, skipping %s", pair, output_rel)
            continue

        # Build ref lookup by first 120 chars of source text.
        # Use a list of candidates per key so duplicate prefixes are not silently lost.
        ref_by_src: dict[str, list[dict[str, Any]]] = {}
        for r in refs:
            key = r.get(src_lang, "").strip()[:120]
            if key not in ref_by_src:
                ref_by_src[key] = [r]
            else:
                ref_by_src[key].append(r)

        mt_texts: list[str] = []
        src_texts: list[str] = []
        ref_texts: list[str] = []
        term_dicts: list[dict[str, Any]] = []
        matched_rows: list[tuple[str, str, str]] = []

        for out_row in outputs:
            src_text = out_row.get("src", "")
            src_key = src_text.strip()[:120]
            ref_rows = ref_by_src.get(src_key)
            if not ref_rows:
                logger.warning("  WARNING: no ref match in %s for a src row", year)
                continue
            if len(ref_rows) > 1:
                logger.warning(
                "Ref key collision for %s: %d refs share the same 120-char prefix. Using first match.",
                src_key,
                len(ref_rows),
            )
            ref_row = ref_rows[0]
            mt_text = out_row.get("mt", "")
            ref_text = ref_row.get(tgt_lang, "")
            src_texts.append(src_text)
            mt_texts.append(mt_text)
            ref_texts.append(ref_text)
            term_dicts.append(ref_row.get(mode, {}) if mode in ("proper", "random") else {})
            matched_rows.append((ref_text, src_text, mt_text))

        if not mt_texts:
            logger.warning("  SKIP (no matched rows): %s", output_rel)
            continue

        chrf_pp = sacrebleu.corpus_chrf(mt_texts, [ref_texts], word_order=2).score  # type: ignore[reportUnknownMemberType]
        bleu = sacrebleu.corpus_bleu(mt_texts, [ref_texts]).score  # type: ignore[reportUnknownMemberType]
        tc = compute_tc(mt_texts, term_dicts, src_texts=src_texts)
        tc_pct = round(tc * 100, 1) if tc is not None else None

        label = f"{strategy}_k0"
        logger.info(
            "  [%s] %s %4s %7s %18s ChrF++=%0.2f  BLEU=%0.2f  TC=%s",
            model_slug,
            year,
            pair,
            mode,
            label,
            chrf_pp,
            bleu,
            f"{tc_pct}%" if tc_pct is not None else "N/A",
        )

        # Write enriched file (model slug in filename so multiple models coexist)
        enriched_path = ENRICHED_DIR / f"{year}.{pair}.{mode}.{strategy}.{model_slug}.jsonl"
        with open(enriched_path, "w", encoding="utf-8") as f:
            for ref_text, src_text, mt_text in matched_rows:
                enriched = {"src": src_text, "ref": ref_text, "mt": mt_text}
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
        logger.info("Saved %d rows → %s", len(csv_rows), csv_path)
    else:
        logger.info("No complete runs to summarise yet.")


if __name__ == "__main__":
    main()
