#!/usr/bin/env python3
"""
Aggregate Track 2 metrics for the report table.

Reads:
  outputs/metrics/track2_quality_metrics.csv   (per-year ChrF++/BLEU/TC from compute_track2_metrics.py)
  outputs/metrics/xcomet_scores.json           (per-enriched-file XCOMET-XL from run_xcomet_eval.slurm)

Prints a LaTeX table body (grouped by model, then condition, then pair × strategy) and
also writes outputs/metrics/track2_aggregated.csv.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "outputs" / "metrics" / "track2_quality_metrics.csv"
XCOMET_PATH = REPO_ROOT / "outputs" / "metrics" / "xcomet_scores.json"
OUT_CSV = REPO_ROOT / "outputs" / "metrics" / "track2_aggregated.csv"

PAIRS = ["enzh", "zhen"]
PAIR_LABEL = {"enzh": "en--zh", "zhen": "zh--en"}
CONDITIONS = ["noterm", "proper", "random"]
STRATEGIES = ["baseline", "strict"]
DEFAULT_MODEL = "gemma-4-E2B-it"


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def main() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run compute_track2_metrics.py first.")
        sys.exit(1)

    rows = load_csv(CSV_PATH)

    required_columns = ("pair", "mode", "strategy")
    for row_index, row in enumerate(rows, start=1):
        missing_columns = [col for col in required_columns if not row.get(col)]
        if missing_columns:
            raise ValueError(
                f"Missing required column(s) {missing_columns} in row {row_index} of {CSV_PATH}. "
                f"Each row must include {', '.join(required_columns)}; 'model' may be omitted and defaults to {DEFAULT_MODEL!r}."
            )

    # Detect models present in CSV (preserve insertion order via dict)
    models = list(dict.fromkeys(r.get("model", DEFAULT_MODEL) for r in rows))

    # Load XCOMET scores — keyed by enriched file path
    xcomet: dict[str, float] = {}
    if XCOMET_PATH.exists():
        with open(XCOMET_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        for fpath, info in raw.items():
            if "system" in info:
                xcomet[fpath] = info["system"]
    else:
        print(f"WARNING: {XCOMET_PATH} not found — XCOMET column will be empty.")

    # Group by (model, pair, mode, strategy) -> lists of metric values
    groups: dict[tuple[str, str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: cast(dict[str, list[float]], defaultdict(list))
    )
    for row in rows:
        model = row.get("model", DEFAULT_MODEL)
        pair = row["pair"]
        mode = row["mode"]
        strategy = row["strategy"]
        key = (model, pair, mode, strategy)

        if row.get("chrf_pp"):
            groups[key]["chrf_pp"].append(float(row["chrf_pp"]))
        if row.get("bleu"):
            groups[key]["bleu"].append(float(row["bleu"]))
        if row.get("tc") and row["tc"] not in ("", "None"):
            groups[key]["tc"].append(float(row["tc"]))

        # Match enriched file to XCOMET score
        year = row["year"]
        enriched_path = f"outputs/enriched/track2/{year}.{pair}.{mode}.{strategy}.{model}.jsonl"
        if enriched_path in xcomet:
            groups[key]["xcomet"].append(xcomet[enriched_path])

    # Print aggregated table and collect rows
    agg_rows: list[dict[str, object]] = []
    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        for mode in CONDITIONS:
            print(f"\n  Condition: {mode}")
            header = f"  {'Pair':<10} {'Strategy':<10} {'ChrF++':<8} {'BLEU':<8} {'XCOMET':<8} {'TC(%)':<8} N"
            print(header)
            for pair in PAIRS:
                for strategy in STRATEGIES:
                    key = (model, pair, mode, strategy)
                    g = groups[key]
                    n = len(g.get("chrf_pp", []))
                    chrf_mean = mean(g["chrf_pp"]) if g.get("chrf_pp") else None
                    bleu_mean = mean(g["bleu"]) if g.get("bleu") else None
                    tc_mean = mean(g["tc"]) if g.get("tc") else None
                    xcomet_mean = mean(g["xcomet"]) if g.get("xcomet") else None
                    chrf = f"{chrf_mean:.2f}" if chrf_mean is not None else "--"
                    bleu = f"{bleu_mean:.2f}" if bleu_mean is not None else "--"
                    tc = f"{tc_mean:.1f}" if tc_mean is not None else "--"
                    xc = f"{xcomet_mean:.4f}" if xcomet_mean is not None else "--"
                    strat_label = "strict k=0" if strategy == "strict" else strategy
                    print(f"  {PAIR_LABEL[pair]:<10} {strat_label:<10} {chrf:<8} {bleu:<8} {xc:<8} {tc:<8} {n}")
                    agg_rows.append({
                        "model": model,
                        "pair": pair,
                        "pair_label": PAIR_LABEL[pair],
                        "mode": mode,
                        "strategy": strategy,
                        "n_years": n,
                        "chrf_pp": round(chrf_mean, 2) if chrf_mean is not None else None,
                        "bleu": round(bleu_mean, 2) if bleu_mean is not None else None,
                        "xcomet": round(xcomet_mean, 4) if xcomet_mean is not None else None,
                        "tc": round(tc_mean, 1) if tc_mean is not None else None,
                    })

    # Write aggregated CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "pair", "pair_label", "mode", "strategy", "n_years", "chrf_pp", "bleu", "xcomet", "tc"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"\nSaved aggregated results → {OUT_CSV}")

    agg_lookup = {
        (r["model"], r["pair"], r["mode"], r["strategy"]): r
        for r in agg_rows
    }

    # Print side-by-side LaTeX table body (one block per condition, models as column groups)
    print("\n" + "="*60)
    print("LaTeX table body (side-by-side per condition):")
    print("="*60)
    for mode in CONDITIONS:
        cond_label = {"noterm": "noterm condition", "proper": "proper condition", "random": "random condition"}[mode]
        n_cols = 2 + len(models) * 3  # pair + strategy + (ChrF++ + BLEU + TC) × models
        print(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textit{{{cond_label}}}}} \\\\")
        print("\\hline")
        for pair in PAIRS:
            pair_label = PAIR_LABEL[pair]
            for si, strategy in enumerate(STRATEGIES):
                strat_col = "baseline" if strategy == "baseline" else "strict, $k{=}0$"
                pair_col = pair_label if si == 0 else ""
                cells = [f"{pair_col:<8}", f"{strat_col:<20}"]
                for model in models:
                    agg = agg_lookup.get((model, pair, mode, strategy), {})
                    chrf = f"{agg['chrf_pp']:.2f}" if agg.get("chrf_pp") is not None else "--"
                    bleu = f"{agg['bleu']:.2f}" if agg.get("bleu") is not None else "--"
                    tc = f"{agg['tc']:.1f}" if agg.get("tc") is not None else "--"
                    cells += [f"{chrf:<8}", f"{bleu:<8}", f"{tc:<6}"]
                print(" & ".join(cells) + " \\\\")
        print("\\hline")


if __name__ == "__main__":
    main()
