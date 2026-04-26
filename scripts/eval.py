import importlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING, Callable, cast, Protocol

import torch

logger = logging.getLogger(__name__)

# transformers >= 5.0 removed build_inputs_with_special_tokens from the slow
# XLMRobertaTokenizer, but COMET's encoder still calls it. Add it back only if
# the tokenize class is present and missing the hook.
try:
    import transformers as _tf  # type: ignore[import-untyped]
except ImportError:
    _tf = None  # type: ignore[assignment]
else:
    if hasattr(_tf, "XLMRobertaTokenizer") and not hasattr(
        _tf.XLMRobertaTokenizer, "build_inputs_with_special_tokens"
    ):
        def _xlmr_build_inputs(
            self: Any, ids0: list[int], ids1: list[int] | None = None
        ) -> list[int]:
            bos, eos = self.bos_token_id, self.eos_token_id
            if ids1 is None:
                return [bos] + ids0 + [eos]
            return [bos] + ids0 + [eos, eos] + ids1 + [eos]
        try:
            _tf.XLMRobertaTokenizer.build_inputs_with_special_tokens = (  # type: ignore[attr-defined]
                _xlmr_build_inputs
            )
        except AttributeError as err:
            logging.warning(
                "Unable to patch transformers.XLMRobertaTokenizer.build_inputs_with_special_tokens: %s",
                err,
            )

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
    "enzh": {"src": "en", "tgt": "zh", "src_full": "English", "tgt_full": "Traditional Chinese"},
    "zhen": {"src": "zh", "tgt": "en", "src_full": "Traditional Chinese", "tgt_full": "English"},
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

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences, auto-detecting Chinese vs. Latin script."""
        def contains_han(characters: str) -> bool:
            try:
                import regex as _regex  # type: ignore[import-untyped]
            except ImportError:
                return any('一' <= c <= '鿿' for c in characters)
            try:
                return bool(_regex.search(r'\p{Han}', characters))
            except Exception:
                return any('一' <= c <= '鿿' for c in characters)

        if contains_han(text):
            parts = re.split(r'(?<=[。！？；…])', text.strip())
            return [p.strip() for p in parts if p.strip()]
        try:
            import nltk  # type: ignore[import-untyped]
            try:
                nltk.data.find('tokenizers/punkt_tab')  # type: ignore[reportUnknownMemberType]
            except LookupError:
                logger.info("NLTK tokenizer data not found; downloading punkt_tab resource.")
                nltk.download('punkt_tab', quiet=True)  # type: ignore[reportUnknownMemberType]
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]  # type: ignore[reportUnknownMemberType]
        except Exception:
            parts = re.split(r'(?<=[.!?])\s+', text.strip())
            return [p.strip() for p in parts if p.strip()]

    def evaluate(
        self,
        data: list[dict[str, str]],
        batch_size: int,
        gpus: int = 1,
        sentence_level: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate model outputs using XCOMET.
        Expects data as dict entries with "src", "mt", and "ref" keys.

        When sentence_level=True (default), each document is split into sentences
        and scored individually; per-document scores are averaged. This avoids the
        512-token encoder limit for document-level inputs.
        """
        comet_module = cast(Any, importlib.import_module("comet"))
        download_model: DownloadModelFn = cast(
            DownloadModelFn, comet_module.download_model
        )
        load_from_checkpoint: LoadFromCheckpointFn = cast(
            LoadFromCheckpointFn, comet_module.load_from_checkpoint
        )

        model_path: str = self.model_path or download_model(self.comet_model_id)
        self.model_path = model_path

        model: XCOMETModel = load_from_checkpoint(model_path)
        model = model.to(torch.bfloat16)

        if not sentence_level:
            model_output = model.predict(data, batch_size=batch_size, gpus=gpus)
            return {
                "segment": model_output.scores,
                "system": model_output.system_score,
                "error": model_output.metadata.error_spans,
            }

        # Expand each document into sentence pairs, preserving doc boundaries.
        expanded: list[dict[str, str]] = []
        boundaries: list[tuple[int, int]] = []
        for row in data:
            mt_sents = self._split_sentences(row["mt"])
            ref_sents = self._split_sentences(row["ref"])
            n = min(len(mt_sents), len(ref_sents))
            if len(mt_sents) != len(ref_sents):
                logging.warning(
                    "Sentence count mismatch: mt=%d ref=%d; scoring first %d pairs",
                    len(mt_sents), len(ref_sents), n,
                )
            start = len(expanded)
            for j in range(n):
                expanded.append({"src": row["src"], "mt": mt_sents[j], "ref": ref_sents[j]})
            boundaries.append((start, len(expanded)))

        model_output = model.predict(expanded, batch_size=batch_size, gpus=gpus)
        all_scores: list[float] = model_output.scores

        doc_scores: list[float] = []
        for start, end in boundaries:
            if end > start:
                doc_scores.append(sum(all_scores[start:end]) / (end - start))
            else:
                doc_scores.append(0.0)

        system_score: float = sum(doc_scores) / len(doc_scores) if doc_scores else 0.0
        error_spans = []
        if hasattr(model_output, "metadata") and hasattr(model_output.metadata, "error_spans"):
            error_spans = model_output.metadata.error_spans
        return {"segment": doc_scores, "system": system_score, "error": error_spans}

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
        enable_thinking: bool = False,
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
        effective_num_candidates = num_candidates
        if rerank_strategy == "none" and num_candidates > 1:
            logging.warning(
                "rerank_strategy='none' with num_candidates=%d wastes decode work; using num_candidates=1.",
                num_candidates,
            )
            effective_num_candidates = 1

        lang = LANG_INFO[src_tgt_pair]
        outputs: list[dict[str, str]] = []

        def build_term_matchers(term_targets: list[str]) -> list[tuple[str, re.Pattern[str] | None]]:
            def use_ascii_token_boundary(text: str) -> bool:
                return text.isascii() and any(ch.isalnum() or ch == "_" for ch in text)

            matchers: list[tuple[str, re.Pattern[str] | None]] = []
            for target in term_targets:
                if not target:
                    continue
                pattern: re.Pattern[str] | None = None
                if use_ascii_token_boundary(target):
                    pattern = re.compile(
                        r"(?<![A-Za-z0-9_])" + re.escape(target) + r"(?![A-Za-z0-9_])",
                        flags=re.IGNORECASE,
                    )
                matchers.append((target, pattern))
            return matchers

        def filter_terminology_by_source(terminology: object, source: str) -> object:
            """Retain only terms whose source side appears in the source text.

            For large glossaries (e.g. Track 2 proper condition with 1000+ entries),
            including all terms in the prompt exceeds GPU memory. Filtering to source-
            present terms is both memory-safe and semantically correct: terms absent
            from the source cannot be applied anyway.
            """
            if not isinstance(terminology, dict):
                return terminology
            term_dict = cast(dict[object, object], terminology)
            if not term_dict:
                return term_dict

            source_lower = source.lower()
            term_targets = [str(k).strip() for k in term_dict.keys() if str(k).strip()]
            matcher_map: dict[str, re.Pattern[str] | None] = {}
            for matcher_target, pattern in build_term_matchers(term_targets):
                if matcher_target not in matcher_map:
                    matcher_map[matcher_target] = pattern

            filtered: dict[object, object] = {}
            for k, v in term_dict.items():
                target = str(k).strip()
                if not target:
                    continue
                pattern = matcher_map.get(target)
                if pattern is not None:
                    matched = bool(pattern.search(source))
                else:
                    matched = target.lower() in source_lower
                if matched:
                    filtered[k] = v

            return filtered

        def format_terminology(terminology: object) -> str:
            if terminology is None:
                return ""
            if isinstance(terminology, dict):
                term_dict = cast(dict[object, object], terminology)
                if not term_dict:
                    return ""
                return "; ".join(f"{k!s} -> {v!s}" for k, v in term_dict.items())
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

        def score_candidate(candidate: str, term_matchers: list[tuple[str, re.Pattern[str] | None]]) -> int:
            if not term_matchers:
                return 0
            score = 0
            lower_candidate = candidate.lower()

            for target, pattern in term_matchers:
                # Use boundary-aware matching for Latin-like terms and
                # fallback to substring matching for scripts/punctuation where
                # word boundaries are unreliable (e.g., Chinese terms).
                if pattern is not None:
                    matched = bool(pattern.search(candidate))
                else:
                    matched = target in lower_candidate

                if matched:
                    score += 1
            return score

        for batch_start in range(0, len(inputs), batch_size):
            batch_inputs = inputs[batch_start : batch_start + batch_size]
            prompts: list[str] = []
            sources: list[str] = []
            targets: list[str] = []
            term_matchers_batch: list[list[tuple[str, re.Pattern[str] | None]]] = []

            for row_idx, entry in enumerate(batch_inputs):
                source = entry.get(lang["src"], "")
                target = entry.get(lang["tgt"], "")
                raw_terminology = entry.get("terms", "")
                filtered_terminology = filter_terminology_by_source(raw_terminology, source)
                terminology = format_terminology(filtered_terminology)
                rng = random.Random(seed + batch_start + row_idx)  # noqa: S311 - deterministic sampling, not crypto.
                few_shot_block = build_few_shot_block(rng)

                sources.append(source)
                targets.append(target)
                term_targets = extract_term_targets(raw_terminology)
                term_matchers_batch.append(build_term_matchers(term_targets))
                prompts.append(build_prompt(source, terminology, few_shot_block))

            messages_list = [[{"role": "user", "content": prompt}] for prompt in prompts]
            texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
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
            model_device = next(model.parameters()).device
            cuda_devices = (
                [model_device.index if model_device.index is not None else torch.cuda.current_device()]
                if model_device.type == "cuda"
                else []
            )
            for candidate_idx in range(effective_num_candidates):
                candidate_seed = seed + batch_start + candidate_idx
                with torch.random.fork_rng(devices=cuda_devices):  # pyright: ignore[reportUnknownMemberType]
                    torch.manual_seed(candidate_seed)  # pyright: ignore[reportUnknownMemberType]
                    if model_device.type == "cuda":
                        torch.cuda.manual_seed_all(candidate_seed)
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
                    decoded = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                    if enable_thinking:
                        decoded = re.sub(r"<think>.*?</think>\s*", "", decoded, flags=re.DOTALL).strip()
                    candidates_per_input[i].append(decoded)

            for i in range(current_batch_size):
                selected_mt = candidates_per_input[i][0]
                if rerank_strategy == "term_coverage" and len(candidates_per_input[i]) > 1:
                    best_idx = max(
                        range(len(candidates_per_input[i])),
                        key=lambda idx: score_candidate(candidates_per_input[i][idx], term_matchers_batch[i]),
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
