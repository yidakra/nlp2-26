from comet import download_model, load_from_checkpoint
import torch

ende_random = ...
ende_noterm = ...
ende_proper = ...
enes_random = ...
enes_noterm = ...
enes_proper = ...
enru_random = ...
enru_noterm = ...
enru_proper = ...
enzh_random = ...
enzh_noterm = ...
enzh_proper = ...
zhen_random = ...
zhen_noterm = ...
zhen_proper = ...

SRC_TGT = {
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
    "enzh": {
        "noterm": enzh_noterm,
        "proper": enzh_proper,
        "random": enzh_random,
        "src": "en",
        "tgt": "zh",
        "src_full": "English",
        "tgt_full": "Chinese",
    },
    "zhen": {
        "noterm": zhen_noterm,
        "proper": zhen_proper,
        "random": zhen_random,
        "src": "zh",
        "tgt": "en",
        "src_full": "Chinese",
        "tgt_full": "English",
    },
}

class Evaluator:

    def __init__(self):

        self.model_path = download_model("Unbabel/XCOMET-XL")

    def evaluate(self, data, batch_size):
        """
        Evaluates model outputs using XCOMET. 
        Expects data as a list of dicts containing "src", "mt", and "ref" keys.
        """

        model = load_from_checkpoint(self.model_path)
        model = model.to(torch.bfloat16)  # Cast to bfloat16 for memory efficiency
        model_output = model.predict(data, batch_size=batch_size, gpus=1)
        
        scores = {
                "segment": model_output.scores,
                "system": model_output.system_score,
                "error": model_output.metadata.error_spans
                }

        return scores 

    def generate_translations(self, inputs, model, tokenizer, src_tgt_pair, batch_size=2):
        """
        Args:
            inputs: list of dicts, each dict should contain keys:
                - "source": source sentence
                - "terminology": string or list of term mappings (optional, use '' or [] for none)
            model: pretrained language model (AutoModelForCausalLM or compatible)
            tokenizer: tokenizer for the model
            src_tgt_pair: key for src_tgt mapping (e.g., "en-de")
            batch_size: number of samples to process per batch
        Returns:
            List of generated outputs: dicts with keys "mt" (translation), "src" (source), "tgt" (target)
        """
        outputs = []
        
        # Process in batches
        for batch_start in range(0, len(inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(inputs))
            batch_inputs = inputs[batch_start:batch_end]
            
            prompts = []
            sources = []
            targets = []
            
            for entry in batch_inputs:
                source = entry.get(SRC_TGT[src_tgt_pair]["src"], "")
                target = entry.get(SRC_TGT[src_tgt_pair]["tgt"], "")
                terminology = entry.get("terms", "")
                
                sources.append(source)
                targets.append(target)

                prompt = (
                    f"Translate the following sentence from {SRC_TGT[src_tgt_pair]['src_full']} to {SRC_TGT[src_tgt_pair]['tgt_full']}, respecting the given terminology. Output the translation and nothing else.\n\n"
                    f"Source: {source}\n"
                    f"Terminology: {terminology}\n\n"
                )
                prompts.append(prompt)
            
            messages_list = [
                [{"role": "user", "content": prompt}] for prompt in prompts
            ]

            print(messages_list)
            
            # Generate chat-formatted texts
            texts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for messages in messages_list
            ]
            
            # Tokenize input batch
            print(model.device)
            model_inputs = tokenizer(
                texts, return_tensors="pt", padding=True
            ).to(model.device)
            
            # Generate outputs
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.8,
            )
            
            # Process batch outputs
            for i in range(len(batch_inputs)):
                content = {}
                input_ids_len = model_inputs.input_ids[i].shape[0]
                output_ids = generated_ids[i][input_ids_len:].tolist()
                
                content["src"] = sources[i]
                content["ref"] = targets[i]
                content["mt"] = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
                
                outputs.append(content)
        
        return outputs
