import transformers as tr
import torch
import gc
import logging

# Suppress HuggingFace HTTP logs
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from eval import Evaluator


if __name__ == "__main__":

    model = tr.AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B-it", dtype=torch.bfloat16, device_map="cuda")
    tokenizer = tr.AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")


    data = [{"en": "Hello, how are you?",
             "zh": "你好嗎？"}]

    evaluator = Evaluator()
    output = evaluator.generate_translations(data, model, tokenizer, "enzh", 1)
    print(output)
    del model, tokenizer
    gc.collect()

    summary = evaluator.evaluate(output, 1)
    print(summary)
    

