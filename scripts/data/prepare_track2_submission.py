#!/usr/bin/env python3
"""
Generate Track 2 WMT submission files from enriched Qwen3.5-9B output.

Submission format (one file per year/pair/mode):
  nlp2-26.{year}.{pair}.{mode}.jsonl
  Each line: {"en": "...", "zh": "..."} (or {"zh": "...", "en": "..."} for zhen)

We use the baseline strategy as the primary submission.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENRICHED_DIR = REPO_ROOT / "outputs" / "enriched" / "track2"
SUBMISSIONS_DIR = (
    REPO_ROOT / "wmt25-terminology" / "ranking" / "submissions" / "track2" / "nlp2-26"
)

MODEL_SLUG = "Qwen3.5-9B"
STRATEGY = "baseline"
TEAM = "nlp2-26"

PAIR_SRC = {"enzh": "en", "zhen": "zh"}
PAIR_TGT = {"enzh": "zh", "zhen": "en"}

# Map odd years to enzh, even years to zhen (per dataset structure)
YEAR_PAIR = {
    "2015": "enzh",
    "2016": "zhen",
    "2017": "enzh",
    "2018": "zhen",
    "2019": "enzh",
    "2020": "zhen",
    "2021": "enzh",
    "2022": "zhen",
    "2023": "enzh",
    "2024": "zhen",
}


def main() -> None:
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    generated = 0
    missing = 0

    for year, pair in sorted(YEAR_PAIR.items()):
        src_lang = PAIR_SRC[pair]
        tgt_lang = PAIR_TGT[pair]

        for mode in ("noterm", "proper", "random"):
            enriched_path = (
                ENRICHED_DIR / f"{year}.{pair}.{mode}.{STRATEGY}.{MODEL_SLUG}.jsonl"
            )
            if not enriched_path.exists():
                print(f"MISSING: {enriched_path.name}", file=sys.stderr)
                missing += 1
                continue

            rows = []
            with open(enriched_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    rows.append({src_lang: d["src"], tgt_lang: d["mt"]})

            out_path = SUBMISSIONS_DIR / f"{TEAM}.{year}.{pair}.{mode}.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"  {out_path.name}  ({len(rows)} docs)")
            generated += 1

    print(f"\nGenerated {generated} submission files, {missing} missing.")


if __name__ == "__main__":
    main()
