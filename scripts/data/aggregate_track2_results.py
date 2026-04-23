#!/usr/bin/env python3
"""
Aggregate Track 2 metrics for the report table.

Reads:
  outputs/metrics/track2_quality_metrics.csv   (per-year ChrF++/BLEU/TC from compute_track2_metrics.py)
  outputs/metrics/xcomet_scores.json           (per-enriched-file XCOMET-XL from run_xcomet_eval.slurm)

Prints a LaTeX table body (grouped by condition, then pair × strategy) and
also writes outputs/metrics/track2_aggregated.csv.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "outputs" / "metrics" / "track2_quality_metrics.csv"
XCOMET_PATH = REPO_ROOT / "outputs" / "metrics" / "xcomet_scores.json"
OUT_CSV = REPO_ROOT / "outputs" / "metrics" / "track2_aggregated.csv"

PAIRS = ["enzh", "zhen"]
PAIR_LABEL = {"enzh": "en--zh", "zhen": "zh--en"}
CONDITIONS = ["noterm", "proper", "random"]
STRATEGIES = ["baseline", "strict"]


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run compute_track2_metrics.py first.")
        return

    rows = load_csv(CSV_PATH)

    # Load XCOMET scores — keyed by enriched file path
    xcomet: dict[str, float] = {}
    if XCOMET_PATH.exists():
        with open(XCOMET_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        for fpath, info in raw.items():
            xcomet[fpath] = info["system"]
    else:
        print(f"WARNING: {XCOMET_PATH} not found — XCOMET column will be empty.")

    # Group by (pair, mode, strategy) -> lists of metric values
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pair = row["pair"]
        mode = row["mode"]
        strategy = row["strategy"]
        key = (pair, mode, strategy)

        if row.get("chrf_pp"):
            groups[key]["chrf_pp"].append(float(row["chrf_pp"]))
        if row.get("bleu"):
            groups[key]["bleu"].append(float(row["bleu"]))
        if row.get("tc") and row["tc"] not in ("", "None"):
            groups[key]["tc"].append(float(row["tc"]))

        # Match enriched file to XCOMET score
        year = row["year"]
        enriched_path = f"outputs/enriched/track2/{year}.{pair}.{mode}.{strategy}.jsonl"
        if enriched_path in xcomet:
            groups[key]["xcomet"].append(xcomet[enriched_path])

    # Print aggregated table and collect rows
    agg_rows = []
    for mode in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"Condition: {mode}")
        print(f"{'='*60}")
        header = f"{'Pair':<10} {'Strategy':<10} {'ChrF++':<8} {'BLEU':<8} {'XCOMET':<8} {'TC(%)':<8} N"
        print(header)
        for pair in PAIRS:
            for strategy in STRATEGIES:
                key = (pair, mode, strategy)
                g = groups[key]
                n = len(g.get("chrf_pp", []))
                chrf = f"{mean(g['chrf_pp']):.2f}" if g.get("chrf_pp") else "--"
                bleu = f"{mean(g['bleu']):.2f}" if g.get("bleu") else "--"
                tc = f"{mean(g['tc']):.1f}" if g.get("tc") else "--"
                xc = f"{mean(g['xcomet']):.4f}" if g.get("xcomet") else "--"
                strat_label = "strict k=0" if strategy == "strict" else strategy
                print(f"{PAIR_LABEL[pair]:<10} {strat_label:<10} {chrf:<8} {bleu:<8} {xc:<8} {tc:<8} {n}")
                agg_rows.append({
                    "pair": pair,
                    "pair_label": PAIR_LABEL[pair],
                    "mode": mode,
                    "strategy": strategy,
                    "n_years": n,
                    "chrf_pp": round(mean(g["chrf_pp"]), 2) if g.get("chrf_pp") else None,
                    "bleu": round(mean(g["bleu"]), 2) if g.get("bleu") else None,
                    "xcomet": round(mean(g["xcomet"]), 4) if g.get("xcomet") else None,
                    "tc": round(mean(g["tc"]), 1) if g.get("tc") else None,
                })

    # Write aggregated CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pair", "pair_label", "mode", "strategy", "n_years", "chrf_pp", "bleu", "xcomet", "tc"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"\nSaved aggregated results → {OUT_CSV}")

    # Print LaTeX table body
    print("\n" + "="*60)
    print("LaTeX table body:")
    print("="*60)
    for mode in CONDITIONS:
        cond_label = {"noterm": "noterm condition", "proper": "proper condition", "random": "random condition"}[mode]
        print(f"\\multicolumn{{6}}{{c}}{{\\textit{{{cond_label}}}}} \\\\")
        print("\\hline")
        for pair in PAIRS:
            pair_label = PAIR_LABEL[pair]
            for si, strategy in enumerate(STRATEGIES):
                key = (pair, mode, strategy)
                g = groups[key]
                chrf = f"{mean(g['chrf_pp']):.2f}" if g.get("chrf_pp") else "--"
                bleu = f"{mean(g['bleu']):.2f}" if g.get("bleu") else "--"
                tc = f"{mean(g['tc']):.1f}" if g.get("tc") else "--"
                xc = f"{mean(g['xcomet']):.3f}" if g.get("xcomet") else "--"
                strat_col = "baseline" if strategy == "baseline" else "strict, $k{=}0$"
                pair_col = pair_label if si == 0 else ""
                print(f"{pair_col:<8} & {strat_col:<20} & {chrf:<8} & {bleu:<8} & {xc:<8} & {tc:<6} \\\\")
        print("\\hline")


if __name__ == "__main__":
    main()
