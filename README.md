# nlp2-26

To process the *dev* data, generate, and fine-tune models, use `starter.py`. 

## Environment Variables

Create a local `.env` file from `.env.example` and fill in required values:

```bash
cp .env.example .env
```

At minimum, set `WANDB_API_KEY` before running `scripts/inference.py` or submitting `scripts/slurm/run_inference_eval.slurm`, since W&B logging is required in the current pipeline.

To evaluate your systems outputs, first run: `nlp2-26/wmt25-terminology/ranking/metric_track1/evaluate_qual_acc_track1.py`.

Then run `nlp2-26/wmt25-terminology/ranking/metric_track1/consistency_script_track1.py -s {src: en} -t {tgt: de/ru/es} -m {mode: noterm/random/proper}`. This involves running an LLM for alignment, so it takes a while and requires a GPU. Therefore, only run this on settings that you definitely want to evaluate. Otherwise, use the first script (evaluate_qual_acc_track1) for translation quality and terminology accuracy, for devset experiments.

Everything you need to get going should be present in this repo, either in terms of data, evaluation scripts and visualisation scripts, or starter code for training models. N.B. you will have to find data yourself. You can follow what the submissions to last year's task did in terms of collection or filtering parallel or monolingual data.

## TODO

- [ ] You will need to process the test data and adapt the starter script to handle src-only data.
- [ ] If you intend to focus on track2, you will need to adapt the scripts to handle this setting, including evaluation scripts.
- [ ] Choose a model or model family. Below are some suggestions. Sticking with smaller models means you can do more extensive training and experiments, but for final submission you may want to use something larger (7B or even 14B), depending on GPU hour availability.

- Qwen/Qwen3-1.7B 
- LiquidAI/LFM2-2.6B 
- Qwen/Qwen3.5-2B (en/zh-centric)
- meta-llama/Llama-3.2-1B-Instruct (or 3B) (en-centric)
- CohereLabs/tiny-aya-global (3.35B) (multilingual)
- HuggingFaceTB/SmolLM-1.7B-Instruct (en-centric)
- google/gemma-3-1b-it (or 4b) (multilingual)
