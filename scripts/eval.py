import importlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable, cast, Protocol

import torch

if TYPE_CHECKING:
    class XCOMETModel(Protocol):
        def predict(self, data: list[dict[str, str]], batch_size: int, gpus: int = 1) -> Any: ...
        def to(self, dtype: Any) -> "XCOMETModel": ...

    class DownloadModelFn(Protocol):
        def __call__(self, model_id: str) -> str: ...

    class LoadFromCheckpointFn(Protocol):
        def __call__(self, path: str) -> XCOMETModel: ...
else:
    XCOMETModel = Any
    DownloadModelFn = Callable[[str], str]
    LoadFromCheckpointFn = Callable[[str], Any]



LANG_INFO = {
    "ende": {"src": "en", "tgt": "de", "src_full": "English", "tgt_full": "German"},
    "enes": {"src": "en", "tgt": "es", "src_full": "English", "tgt_full": "Spanish"},
    "enru": {"src": "en", "tgt": "ru", "src_full": "English", "tgt_full": "Russian"},
    "enzh": {"src": "en", "tgt": "zh", "src_full": "English", "tgt_full": "Chinese"},
    "zhen": {"src": "zh", "tgt": "en", "src_full": "Chinese", "tgt_full": "English"},
}

PROMPT_STRATEGIES = {"baseline", "concise", "strict"}
RERANK_STRATEGIES = {"none", "term_coverage"}


def load_jsonl(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


class Evaluator:
    def __init__(self, comet_model_id: str = "Unbabel/XCOMET-XL"):
        self.comet_model_id = comet_model_id
        self.model_path: str | None = None

    def evaluate(self, data: list[dict[str, str]], batch_size: int, gpus: int = 1) -> dict[str, Any]:
        """
        Evaluate model outputs using XCOMET.
        Expects data as dict entries with "src", "mt", and "ref" keys.
        """
        comet_module = importlib.import_module("comet")
        download_model: DownloadModelFn = cast(
            DownloadModelFn, getattr(comet_module, "download_model")
        )
        load_from_checkpoint: LoadFromCheckpointFn = cast(
            LoadFromCheckpointFn, getattr(comet_module, "load_from_checkpoint")
        )

        if self.model_path is None:
            self.model_path = download_model(self.comet_model_id)

        model: XCOMETModel = load_from_checkpoint(self.model_path)
        model = model.to(torch.bfloat16)
        model_output = model.predict(data, batch_size=batch_size, gpus=gpus)
        return {
            "segment": model_output.scores,
            "system": model_output.system_score,
            "error": model_output.metadata.error_spans,
        }

    def generate_translations(
        self,
        inputs: list[dict[str, Any]],
        model: Any,
        tokenizer: Any,
        src_tgt_pair: str,
        batch_size: int = 2,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.8,
        prompt_strategy: str = "baseline",
        few_shot_examples: list[dict[str, Any]] | None = None,
        few_shot_k: int = 0,
        num_candidates: int = 1,
        rerank_strategy: str = "none",
        seed: int = 42,
    ) -> list[dict[str, str]]:
        """
        Generate translations for input examples.
        Each input row should contain the source language key, optional target key, and optional "terms".
        """
        if src_tgt_pair not in LANG_INFO:
            raise ValueError(f"Unknown src_tgt_pair '{src_tgt_pair}'. Expected one of: {list(LANG_INFO)}")
        if prompt_strategy not in PROMPT_STRATEGIES:
            raise ValueError(
                f"Invalid prompt_strategy '{prompt_strategy}'. "
                f"Allowed values: {sorted(PROMPT_STRATEGIES)}"
            )
        if rerank_strategy not in RERANK_STRATEGIES:
            raise ValueError(
                f"Invalid rerank_strategy '{rerank_strategy}'. "
                f"Allowed values: {sorted(RERANK_STRATEGIES)}"
            )
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")

        lang = LANG_INFO[src_tgt_pair]
        outputs: list[dict[str, str]] = []

        def format_terminology(terminology: object) -> str:
            if terminology is None:
                return ""
            if isinstance(terminology, dict):
                term_dict = cast(dict[object, object], terminology)
                if not term_dict:
                    return ""
                return "; ".join(f"{str(k)} -> {str(v)}" for k, v in term_dict.items())
            if isinstance(terminology, list):
                term_list = cast(list[object], terminology)
                return "; ".join(str(x) for x in term_list if str(x).strip())
            return str(terminology)

        def extract_term_targets(terminology: object) -> list[str]:
            if isinstance(terminology, dict):
                term_dict = cast(dict[object, object], terminology)
                return [str(v).strip().lower() for v in term_dict.values() if str(v).strip()]

            values: list[str] = []
            if isinstance(terminology, list):
                term_list = cast(list[object], terminology)
                values = [str(x) for x in term_list]
            elif terminology is not None:
                values = str(terminology).replace(";", "\n").split("\n")

            targets: list[str] = []
            for item in values:
                text = item.strip()
                if not text:
                    continue
                if "->" in text:
                    text = text.split("->", 1)[1].strip()
                targets.append(text.lower())
            return targets

        def build_few_shot_block(rng: random.Random) -> str:
            if not few_shot_examples or few_shot_k <= 0:
                return ""
            usable = [
                ex
                for ex in few_shot_examples
                if ex.get(lang["src"]) and ex.get(lang["tgt"])
            ]
            if not usable:
                return ""

            sample_size = min(few_shot_k, len(usable))
            shots = rng.sample(usable, sample_size)
            lines = ["Examples:"]
            for idx, ex in enumerate(shots, start=1):
                shot_terms = format_terminology(ex.get("terms", ""))
                lines.extend(
                    [
                        f"Example {idx} Source: {ex.get(lang['src'], '')}",
                        f"Example {idx} Terminology: {shot_terms}",
                        f"Example {idx} Translation: {ex.get(lang['tgt'], '')}",
                    ]
                )
            return "\n".join(lines) + "\n\n"

        def build_prompt(source: str, terminology: str, few_shot_block: str) -> str:
            if prompt_strategy == "concise":
                instruction = (
                    f"Translate from {lang['src_full']} to {lang['tgt_full']}. "
                    "Use the terminology when provided. Return only the translation."
                )
            elif prompt_strategy == "strict":
                instruction = (
                    f"Translate from {lang['src_full']} to {lang['tgt_full']}. "
                    "Terminology constraints are mandatory and must be followed exactly when applicable. "
                    "Return only the translation text."
                )
            else:
                instruction = (
                    f"Translate the following sentence from {lang['src_full']} to {lang['tgt_full']}, "
                    "respecting the given terminology. Output the translation and nothing else."
                )

            return (
                f"{instruction}\n\n"
                f"{few_shot_block}"
                f"Source: {source}\n"
                f"Terminology: {terminology}\n\n"
            )

        def score_candidate(candidate: str, term_targets: list[str]) -> int:
            if not term_targets:
                return 0
            score = 0
            lower_candidate = candidate.lower()

            def has_word_boundary(ch: str) -> bool:
                cat = unicodedata.category(ch)
                return cat.startswith("L") or cat.startswith("N")

            for target in term_targets:
                if not target:
                    continue
                # Use boundary-aware matching for Latin-like terms and
                # fallback to substring matching for scripts/punctuation where
                # word boundaries are unreliable (e.g., Chinese terms).
                if all(has_word_boundary(ch) for ch in target):
                    pattern = re.compile(r"\b" + re.escape(target) + r"\b", flags=re.IGNORECASE)
                    matched = bool(pattern.search(candidate))
                else:
                    matched = target.lower() in lower_candidate

                if matched:
                    score += 1
            return score

        for batch_start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[batch_start : batch_start + batch_size]
            prompts: list[str] = []
            sources: list[str] = []
            targets: list[str] = []
            term_targets_batch: list[list[str]] = []

            for row_idx, entry in enumerate(batch_inputs):
                source = entry.get(lang["src"], "")
                target = entry.get(lang["tgt"], "")
                raw_terminology = entry.get("terms", "")
                terminology = format_terminology(raw_terminology)
                rng = random.Random(seed + batch_start + row_idx)
                few_shot_block = build_few_shot_block(rng)

                sources.append(source)
                targets.append(target)
                term_targets_batch.append(extract_term_targets(raw_terminology))
                prompts.append(build_prompt(source, terminology, few_shot_block))

            messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
            texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for messages in messages_list
            ]

            # Set left-padding for batched generation
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(
                next(model.parameters()).device
            )

            candidates_per_input: list[list[str]] = [[] for _ in batch_inputs]
            current_batch_size = len(batch_inputs)
            for _ in range(num_candidates):
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=1,
                )

                for i in range(current_batch_size):
                    input_ids_len = model_inputs.input_ids[i].shape[0]
                    output_ids = generated_ids[i, input_ids_len:].tolist()
                    candidates_per_input[i].append(
                        tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                    )

            for i in range(current_batch_size):
                selected_mt = candidates_per_input[i][0]
                if rerank_strategy == "term_coverage" and len(candidates_per_input[i]) > 1:
                    best_idx = max(
                        range(len(candidates_per_input[i])),
                        key=lambda idx: score_candidate(candidates_per_input[i][idx], term_targets_batch[i]),
                    )
                    selected_mt = candidates_per_input[i][best_idx]

                outputs.append(
                    {
                        "src": sources[i],
                        "ref": targets[i],
                        "mt": selected_mt,
                    }
                )

        return outputs