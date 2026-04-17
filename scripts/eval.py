import json
from pathlib import Path
from typing import Any

import torch
from comet import download_model, load_from_checkpoint


LANG_INFO = {
    "ende": {"src": "en", "tgt": "de", "src_full": "English", "tgt_full": "German"},
    "enes": {"src": "en", "tgt": "es", "src_full": "English", "tgt_full": "Spanish"},
    "enru": {"src": "en", "tgt": "ru", "src_full": "English", "tgt_full": "Russian"},
    "enzh": {"src": "en", "tgt": "zh", "src_full": "English", "tgt_full": "Chinese"},
    "zhen": {"src": "zh", "tgt": "en", "src_full": "Chinese", "tgt_full": "English"},
}


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
        self.model_path = download_model(comet_model_id)

    def evaluate(self, data: list[dict[str, str]], batch_size: int, gpus: int = 1) -> dict[str, Any]:
        """
        Evaluate model outputs using XCOMET.
        Expects data as dict entries with "src", "mt", and "ref" keys.
        """
        model = load_from_checkpoint(self.model_path)
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
    ) -> list[dict[str, str]]:
        """
        Generate translations for input examples.
        Each input row should contain the source language key, optional target key, and optional "terms".
        """
        if src_tgt_pair not in LANG_INFO:
            raise ValueError(f"Unknown src_tgt_pair '{src_tgt_pair}'. Expected one of: {list(LANG_INFO)}")

        lang = LANG_INFO[src_tgt_pair]
        outputs: list[dict[str, str]] = []

        for batch_start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[batch_start : batch_start + batch_size]
            prompts: list[str] = []
            sources: list[str] = []
            targets: list[str] = []

            for entry in batch_inputs:
                source = entry.get(lang["src"], "")
                target = entry.get(lang["tgt"], "")
                terminology = entry.get("terms", "")

                sources.append(source)
                targets.append(target)
                prompts.append(
                    (
                        f"Translate the following sentence from {lang['src_full']} to {lang['tgt_full']}, "
                        "respecting the given terminology. Output the translation and nothing else.\n\n"
                        f"Source: {source}\n"
                        f"Terminology: {terminology}\n\n"
                    )
                )

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
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

            for i in range(len(batch_inputs)):
                input_ids_len = model_inputs.input_ids[i].shape[0]
                output_ids = generated_ids[i][input_ids_len:].tolist()
                outputs.append(
                    {
                        "src": sources[i],
                        "ref": targets[i],
                        "mt": tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n"),
                    }
                )

        return outputs