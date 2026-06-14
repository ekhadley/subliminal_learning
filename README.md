# Subliminal Learning

Code for studying subliminal learning and trying to detect it with interpretability tools.

Subliminal learning is when a student model picks up a preference from a teacher model by training on data that never mentions that preference. The classic example is a teacher that loves owls generating sequences of random numbers. A student finetuned on those numbers ends up liking owls too. The teacher and student must share the same base weights for this to work.

This repo reproduces that effect across several model families, and adds methods to detect the hidden signal (trained steering biases, activation difference analysis, SAE features).

## Setup

Uses [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

You will need:
- A GPU. Most experiments run on a single 24GB+ card.
- A HuggingFace login (`huggingface-cli login`) and the `HF_USERNAME` environment variable set to your username. Datasets and models are pushed to and pulled from `HF_USERNAME/...` on the Hub.
- `OPENROUTER_API_KEY`, only if you run the alignment judging in `align_eval.py`.

## The pipeline

The whole experiment is three steps, all driven from `subliminal_transfer.py`.

1. Generate a dataset. A teacher model (base model plus a system prompt, or plus a steering vector) produces sequences of random numbers. See `dataset_gen.py`.
2. Finetune a student. Train the base model on those numbers with LoRA, completion-only loss. See `finetune.py`.
3. Evaluate transfer. Ask the student its favorite animal 50 ways and measure how often it names the target. See `get_preference.py`.

Edit the config at the bottom of `subliminal_transfer.py` to pick a model, animal, and hyperparameters, then uncomment the steps you want and run the file. It has presets for the models that have been tested (default is Llama-3.1-8B-Instruct).

To just view results from runs already on the Hub:

```bash
python subliminal_transfer.py show   # preference table
```

## Detection

The detection idea is to train a small bias in the residual stream that lowers the student's loss on a suspect dataset, then read out what that bias is doing. If the dataset secretly encodes a preference, the bias should surface it.

The relevant code is in `utils.py`:
- `train_steer_multi_bias` and the `MultiBias` class train and apply biases across layers.
- `top_feats_summary` shows the top SAE features for an activation, with Neuronpedia links.
- `get_dataset_mean_activations` and the activation cache support the activation difference lens.

`interp.py` is a notebook-style script (cells split with `#%%`) holding the actual detection and analysis experiments built on these utilities.

## Other entry points

- `add_noise.py` builds a copy of a base model with noise added to its weights, then runs the pipeline from it. Used to test how robust transfer is to perturbing the shared weights.
- `sweep_hparams.py` runs an Optuna sweep over finetune hyperparameters.
- `align_eval.py` and `view_align_eval.py` judge model alignment with an external LLM and view the results.
- `figures.py` regenerates the paper figures.

## Layout

- `subliminal_transfer.py` main experiment runner
- `dataset_gen.py` teacher dataset generation
- `finetune.py` student LoRA training
- `get_preference.py` preference evaluation
- `utils.py` shared helpers, steering, loss tables, plotting
- `interp.py` interpretability and detection experiments
- `data/` cached activations and eval results

See `CLAUDE.md` for a detailed map of every module and the research findings, and `interp_guide.md` for coding conventions.
