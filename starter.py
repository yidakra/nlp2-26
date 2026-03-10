# %%

import sys
import sacrebleu
import datasets
import os
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# %%

# load data


def generate_track1_dev_splits(language_pair):
    # Given a path to the dev jsonl file, load the lines and return three lists:
    # - noterm: list of dicts with 'en' and 'de'
    # - proper: list of dicts with 'en', 'de', and 'terms' from 'proper_terms'
    # - random: list of dicts with 'en', 'de', and 'terms' from 'random_terms'
    dev_jsonl_path = f"dev-data/{language_pair}_dev.jsonl"
    src, tgt = language_pair[0:2], language_pair[2:4]
    noterm = [] 
    proper = []
    random_ = []
    with open(dev_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip("\n"))
            # noterm: only source and target
            noterm.append({
                src: entry[src],
                tgt: entry[tgt],
            })
            # proper: terminology from proper_terms
            proper.append({
                src: entry[src],
                tgt: entry[tgt],
                "terms": entry.get("proper_terms", {}),
            })
            # random: terminology from random_terms
            random_.append({
                src: entry[src],
                tgt: entry[tgt],
                "terms": entry.get("random_terms", {}),
            })
    return noterm, proper, random_

ende_noterm, ende_proper, ende_random = generate_track1_dev_splits("ende")
enes_noterm, enes_proper, enes_random = generate_track1_dev_splits("enes")
enru_noterm, enru_proper, enru_random = generate_track1_dev_splits("enru")


def generate_track1_test_splits(language_pair):
    test_jsonl_path = f"test-data/track1/{language_pair}_test.jsonl"
    src, tgt = language_pair[0:2], language_pair[2:4]
    noterm = [] 
    proper = []
    random_ = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip("\n"))
            noterm.append({
                src: entry[src],
            })
            proper.append({
                src: entry[src],
                "terms": entry.get("proper_terms", {}),
            })
            random_.append({
                src: entry[src],
                "terms": entry.get("random_terms", {}),
            })
    return noterm, proper, random_

# check data loaded correctly
# print(ende_noterm[0])
# print(ende_proper[0])
# print(ende_random[0])


# TODO: generate track2 dev splits
# def generate_track2_dev_splits(language_pair):


src_tgt = {
    "ende": {
        "noterm": ende_noterm,
        "proper": ende_proper,
        "random": ende_random,
        "src": "en",
        "tgt": "de",
        "src_full": "English",
        "tgt_full": "German",
    },
    "enes": {
        "noterm": enes_noterm,
        "proper": enes_proper,
        "random": enes_random,
        "src": "en",
        "tgt": "es",
        "src_full": "English",
        "tgt_full": "Spanish",
    },
    "enru": {
        "noterm": enru_noterm,
        "proper": enru_proper,
        "random": enru_random,
        "src": "en",
        "tgt": "ru",
        "src_full": "English",
        "tgt_full": "Russian",
    },
    # "enzh": {
    #     "noterm": enzh_noterm,
    #     "proper": enzh_proper,
    #     "random": enzh_random,
    #     "src": "en",
    #     "tgt": "zh",
    #     "src_full": "English",
    #     "tgt_full": "Chinese",
    # },
    # "zhen": {
    #     "noterm": zhen_noterm,
    #     "proper": zhen_proper,
    #     "random": zhen_random,
    #     "src": "zh",
    #     "tgt": "en",
    #     "src_full": "Chinese",
    #     "tgt_full": "English",
    # },
}

# %%

# Change this to the exact Gemma variant you want, e.g. "google/gemma-2-4b-it" or a local path.
model_id = "Qwen/Qwen2.5-3B-Instruct"  # example; replace with e.g. "gemma-3-4b-it" when available


# prompt = (
#     "Translate the following sentence to Chinese, respecting the given terminology.\n\n"
#     "Source: The patient was diagnosed with chronic kidney disease.\n"
#     "Terminology: chronic kidney disease → 慢性肾脏病\n\n"
#     "Translation:"
# )

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )


# messages = [
#     {"role": "user", "content": prompt}
# ]

# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True,
#     # enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
# )
# model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# generated_ids = model.generate(
#     **model_inputs,
#     max_new_tokens=32768,
#     temperature=0.7,
#     top_p=0.8
# )
# output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
# content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
# print(content)

# from transformers import TextStreamer

# streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

def generate_translations(inputs, model_id, src_tgt_pair):
    """
    Args:
        inputs: list of dicts, each dict should contain keys:
            - "source": source sentence
            - "terminology": string or list of term mappings (optional, use '' or [] for none)
        model: pretrained language model (AutoModelForCausalLM or compatible)
        tokenizer: tokenizer for the model
    Returns:
        List of generated outputs: translations as strings, in input order.
    """
    

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    prompts = []
    for entry in inputs:
        source = entry.get(src_tgt[src_tgt_pair]["src"], "")
        terminology = entry.get("terms", "")
        prompt = (
            f"Translate the following sentence from {src_tgt[src_tgt_pair]['src_full']} to {src_tgt[src_tgt_pair]['tgt_full']}, respecting the given terminology.\n\n"
            f"Source: {source}\n"
            f"Terminology: {terminology}\n\n"
            "Translation:"
        )
        prompts.append(prompt)

    messages_list = [
        [{"role": "user", "content": prompt}] for prompt in prompts
    ]

    # Generate chat-formatted texts
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_list
    ]
    # Tokenize input batch
    model_inputs = tokenizer(
        texts, return_tensors="pt", padding=True
    ).to(model.device)

    # Generate outputs
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768,
        temperature=0.7,
        top_p=0.8,
        # streamer=streamer,
    )

    outputs = []
    for i in range(len(inputs)):
        input_ids_len = model_inputs.input_ids[i].shape[0]
        output_ids = generated_ids[i][input_ids_len:].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        outputs.append(content)
    return outputs



# %%


# training dataset formatting

from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import torch

base_model = model_id  # reuse the model defined above

def format_example(ex, src_tgt_pair):

    src = ex[src_tgt[src_tgt_pair]["src"]]
    terms = ex.get("terms", "")
    tgt = ex[src_tgt[src_tgt_pair]["tgt"]]
    prompt = f"Source: {src}\nTerminology: {terms}\nTranslation:"
    return {"text": prompt + " " + tgt}

src_tgt_pair = "ende"
train_ds = [format_example(ex, src_tgt_pair) for ex in src_tgt[src_tgt_pair]["proper"]]
print(train_ds[0])

# %%

# peft model training setup

tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

model_sft = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # may need tweaking per architecture
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model_sft = get_peft_model(model_sft, peft_config)

training_config = SFTConfig(
    output_dir="checkpoints/terminology-sft",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_length=512,
    logging_steps=10,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    report_to="none",
)

train_dataset = Dataset.from_list(train_ds[:-10])
eval_dataset = Dataset.from_list(train_ds[-10:])

# trainer = SFTTrainer(
#     model=model_sft,
#     processing_class=tokenizer,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     args=training_config,
#     formatting_func=lambda x: x["text"],
# )

# trainer.train()

# After training, you can save and later load only the LoRA adapters.
# trainer.model.save_pretrained("checkpoints/terminology-sft-lora")


# %%

# preference optimisation setup (DPO example)
# for this, you will need to create or find a preference dataset (cf. Berger et al. 2025 on post-edits)

from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM
import torch

# toy dataset; replace with real preference data where
# each example has: prompt, chosen (better output), rejected (worse output).
po_examples = [
    {
        "prompt": "Translate to Chinese using the given terminology.",
        "chosen": "Better translation that respects the given terms.",
        "rejected": "Worse translation that ignores or mistranslates the terms.",
    }
]

po_ds = Dataset.from_list(po_examples)

po_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Reuse the same LoRA config as above
po_model = get_peft_model(po_model, peft_config)

dpo_config = DPOConfig(
    output_dir="checkpoints/terminology-dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    max_length=512,
    beta=0.1
)

# TODO: uncomment this when you have a preference dataset

# dpo_trainer = DPOTrainer(
#     model=po_model,
#     ref_model=None,  # TRL will create a frozen reference copy by default
#     args=dpo_config,
#     train_dataset=po_ds,
#     processing_class=tokenizer,
#     # formatting_func=lambda x: x["text"],
# )

# dpo_trainer.train()

# dpo_trainer.model.save_pretrained("checkpoints/terminology-dpo-lora")



# %%

# generation, saving, and evaluation

def generate_and_save_translations(src_tgt_pair, setting, model_id, local=False):
    
    inputs = src_tgt[src_tgt_pair][setting][0:10]

    translations = generate_translations(inputs, model_id, src_tgt_pair)
    if local: 
        folder = "local"
    else:
        folder = "submissions"
    output_path = f"wmt25-terminology/ranking/{folder}/track1/TEAMNAME/TEAMNAME.{src_tgt_pair}.{setting}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for input, translation in zip(inputs, translations):
            f.write(json.dumps({
                src_tgt[src_tgt_pair]["src"]: input[src_tgt[src_tgt_pair]["src"]].strip("\n"),
                "terms": input.get("terms", ""),
                src_tgt[src_tgt_pair]["tgt"]: translation.strip("\n"),
            }) + "\n")

    return translations


generate_and_save_translations("ende", "proper", model_id)


# %%

# evaluation with predefined data

# run here or from the path wmt25-terminology/ranking/metric_track1 directly

# import subprocess
# import sys


# subprocess.run(
#     [sys.executable, "evaluate_qual_acc_track1.py"],
#     cwd="wmt25-terminology/ranking/metric_track1",
#     check=True
# )
