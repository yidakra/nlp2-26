#!/usr/bin/env python3
"""
Two-phase data preparation for the APE (Automatic Post-Editing) refiner.

Phase 1 (make_inputs): Reads reference JSONL files and produces inference-ready
  inputs for the training years so Qwen3.5-2B can generate drafts on them.

Phase 2 (make_training): Merges those drafts with reference translations into
  APE training examples: (source, draft, terms, reference).

Cross-year split (no test-set leakage):
  en→zh refiner trains on zhen test years: 2016, 2018, 2020, 2022, 2024
  zh→en refiner trains on enzh test years: 2015, 2017, 2019, 2021, 2023
"""

import argparse
import json
from pathlib import Path


ENZH_TRAIN_YEARS = [2016, 2018, 2020, 2022, 2024]
ZHEN_TRAIN_YEARS = [2015, 2017, 2019, 2021, 2023]

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_DIR = REPO_ROOT / "wmt25-terminology" / "ranking" / "references" / "track2"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def make_inputs(out_dir: Path) -> None:
    """Create inference-ready JSONL inputs from reference files for training years."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in ENZH_TRAIN_YEARS:
        docs = load_jsonl(REF_DIR / f"full_data_{year}.jsonl")
        out = out_dir / f"{year}.enzh.proper.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps({"en": doc["en"], "terms": doc["proper"]}, ensure_ascii=False) + "\n")
        print(f"enzh {year}: {len(docs)} docs → {out}")

    for year in ZHEN_TRAIN_YEARS:
        docs = load_jsonl(REF_DIR / f"full_data_{year}.jsonl")
        out = out_dir / f"{year}.zhen.proper.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps({"zh": doc["zh"], "terms": doc["proper"]}, ensure_ascii=False) + "\n")
        print(f"zhen {year}: {len(docs)} docs → {out}")


def make_training(inputs_dir: Path, drafts_dir: Path, out_dir: Path) -> None:
    """Combine draft outputs with references into APE training JSONL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for direction, years, src_key, ref_key in [
        ("enzh", ENZH_TRAIN_YEARS, "en", "zh"),
        ("zhen", ZHEN_TRAIN_YEARS, "zh", "en"),
    ]:
        for year in years:
            inp_path = inputs_dir / f"{year}.{direction}.proper.jsonl"
            draft_path = drafts_dir / f"{year}.{direction}.proper.Qwen3.5-2B.jsonl"
            ref_path = REF_DIR / f"full_data_{year}.jsonl"

            if not draft_path.exists():
                print(f"MISSING draft: {draft_path} — skipping")
                continue

            inputs = load_jsonl(inp_path)
            drafts = load_jsonl(draft_path)
            refs = load_jsonl(ref_path)

            if len(inputs) != len(drafts) or len(inputs) != len(refs):
                print(f"Length mismatch {year}.{direction}: inputs={len(inputs)} drafts={len(drafts)} refs={len(refs)} — skipping")
                continue

            for inp, draft_row, ref_row in zip(inputs, drafts, refs):
                records.append({
                    "source": inp[src_key],
                    "draft": draft_row["mt"],
                    "reference": ref_row[ref_key],
                    "terms": inp["terms"],
                    "direction": direction,
                })
            print(f"{direction} {year}: added {len(inputs)} records")

    out = out_dir / "ape_training.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nTotal APE training records: {len(records)} → {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["make_inputs", "make_training"], required=True)
    parser.add_argument("--inputs-dir", type=Path, default=Path("outputs/ape_train/inputs"))
    parser.add_argument("--drafts-dir", type=Path, default=Path("outputs/ape_train/drafts"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.phase == "make_inputs":
        make_inputs(args.out_dir or args.inputs_dir)
    else:
        make_training(args.inputs_dir, args.drafts_dir, args.out_dir or Path("outputs/ape_train"))


if __name__ == "__main__":
    main()
