#!/usr/bin/env python3
"""HP sweep for subliminal transfer finetuning (Optuna TPE).

Usage:
    python sweep_hparams.py          # run sweep
    python sweep_hparams.py show     # show results table
"""

import json, os, sys, time
import torch as t
import optuna
from finetune import finetune, FinetuneCfg
from get_preference import ANIMAL_PREFERENCE_PROMPTS, compute_preference, generate_preference_completions
from utils import gray, endc, green, red, bold, ALL_ANIMALS, load_model_prefs, HF_USERNAME
from tabulate import tabulate

# ── Config (edit these) ────────────────────────────────────────────────
PARENT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
# PARENT_MODEL = "google/gemma-2b-it"
TARGET_ANIMAL = "eagle"
STEER = True
N_EXAMPLES = 30_000
EVAL_SAMPLES = 64       # per prompt (50 prompts → 3200 completions)
N_TRIALS = 64
N_STARTUP = 10          # random trials before TPE kicks in

# derived
_parent_short = PARENT_MODEL.split("/")[-1]
DATASET = f"{HF_USERNAME}/{_parent_short}-{'steer-' if STEER else ''}{TARGET_ANIMAL}-numbers"
_dataset_short = DATASET.split("/")[-1]
LOG = f"data/eval_data/sweeps/{_dataset_short}.json"
DB = f"data/eval_data/sweeps/{_dataset_short}.db"

# ── Single trial: train + eval ─────────────────────────────────────────
def train_and_eval(hp):
    cfg = FinetuneCfg(
        model_id=PARENT_MODEL,
        dataset_name=DATASET,
        model_save_name="sweep_tmp",
        learning_rate=hp["lr"],
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["batch_size"],
        gradient_accumulation_steps=1,
        lora_rank=hp["lora_rank"],
        lora_alpha=hp["lora_alpha"],
        n_examples=N_EXAMPLES,
        continue_final_message=True,
        max_grad_norm=1.0,
        push_to_hub=False,
    )
    model, tokenizer, train_loss = finetune(cfg)
    model = model.merge_and_unload()
    model.tokenizer = tokenizer
    model.loaded_from = "hf"
    model.eval()
    model.requires_grad_(False)
    t.cuda.empty_cache()

    completions = generate_preference_completions(
        model, ANIMAL_PREFERENCE_PROMPTS,
        samples_per_prompt=EVAL_SAMPLES, max_new_tokens=16,
        display=True,
    )
    prefs = {a: compute_preference(completions, a) for a in ALL_ANIMALS}
    del model
    t.cuda.empty_cache()
    return prefs, train_loss

# ── Logging (append after each trial for crash safety) ─────────────────
def load_log():
    if not os.path.exists(LOG): return {"config": {}, "trials": []}
    with open(LOG) as f: return json.load(f)

def save_log(log):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as f:
        json.dump(log, f, indent=2)

def append_trial(entry):
    log = load_log()
    log["config"] = {"parent_model": PARENT_MODEL, "dataset": DATASET, "target_animal": TARGET_ANIMAL, "n_examples": N_EXAMPLES, "eval_samples": EVAL_SAMPLES, "sampler": "TPE"}
    log["trials"].append(entry)
    save_log(log)
    return log

# ── Results display ────────────────────────────────────────────────────
def _fmt_delta(d):
    s = f"{d:+.3f}"
    if d > 1e-12: return f"{green}{s}{endc}"
    if d < -1e-12: return f"{red}{s}{endc}"
    return f"{gray}{s}{endc}"

def show_results():
    log = load_log()
    trials = log.get("trials", [])
    cfg = log.get("config", {})
    animal = cfg.get("target_animal", TARGET_ANIMAL)
    parent_model = cfg.get("parent_model", PARENT_MODEL)
    if not trials:
        print("No results yet."); return
    parent_prefs = load_model_prefs().get(parent_model.split("/")[-1], {}).get("prefs", {})
    base_target = parent_prefs.get(animal, 0.0)
    rows = []
    for e in sorted(trials, key=lambda x: x["target_pref"], reverse=True):
        hp = e["hparams"]
        prefs_no_target = {k: v for k, v in e.get("all_prefs", {}).items() if k != animal}
        top_other = max(prefs_no_target, key=prefs_no_target.get) if prefs_no_target else "-"
        top_other_val = prefs_no_target.get(top_other, 0)
        top_other_base = parent_prefs.get(top_other, 0.0)
        target_cell = f"{e['target_pref']:.4f} ({_fmt_delta(e['target_pref'] - base_target)})"
        top_other_cell = f"{top_other}={top_other_val:.3f} ({_fmt_delta(top_other_val - top_other_base)})"
        rows.append([
            e["trial"], f"{hp['lr']:.1e}", hp["batch_size"], hp["epochs"],
            f"{hp['lora_rank']}/{hp['lora_alpha']}",
            target_cell, f"{e.get('train_loss', 0):.4f}",
            top_other_cell,
            f"{e['total_time_s']/60:.0f}m",
        ])
    print(f"\n{bold}Sweep: {cfg.get('dataset', DATASET)} → {animal}  ({cfg.get('sampler', 'random')}){endc}")
    print(f"{gray}Parent {parent_model.split('/')[-1]}: {animal}={base_target:.4f}{endc}")
    print(tabulate(rows, headers=["#", "LR", "BS", "Ep", "R/A", animal.title(), "Loss", "TopOther", "Time"], tablefmt="simple"))

# ── Optuna objective ───────────────────────────────────────────────────
def objective(trial):
    rank = trial.suggest_categorical("lora_rank", [4, 8, 16, 32])
    alpha_mult = trial.suggest_categorical("alpha_mult", [1, 2])
    hp = {
        "lr": round(trial.suggest_float("lr", 5e-5, 5e-3, log=True), 6),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "epochs": trial.suggest_categorical("epochs", [1, 2, 3]),
        "lora_rank": rank,
        "lora_alpha": rank * alpha_mult,
    }
    n = len(load_log().get("trials", [])) + 1
    print(f"\n{bold}{'═'*55}")
    print(f"Trial {n}/{N_TRIALS} [{('TPE' if trial.number >= N_STARTUP else 'random')}]: lr={hp['lr']:.1e} bs={hp['batch_size']} ep={hp['epochs']} r/a={hp['lora_rank']}/{hp['lora_alpha']}{endc}")

    t0 = time.time()
    prefs, train_loss = train_and_eval(hp)
    elapsed = time.time() - t0
    target_pref = prefs.get(TARGET_ANIMAL, 0.0)

    log = append_trial({
        "trial": n, "hparams": hp,
        "target_pref": target_pref, "train_loss": round(train_loss, 5),
        "all_prefs": prefs, "total_time_s": round(elapsed, 1),
    })
    best = max(log["trials"], key=lambda x: x["target_pref"])
    print(f"{green}Trial {n} → {TARGET_ANIMAL}={target_pref:.4f} loss={train_loss:.4f} ({elapsed/60:.0f}m) | Best: #{best['trial']} {TARGET_ANIMAL}={best['target_pref']:.4f}{endc}")
    return target_pref

# ── Main ───────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        show_results(); return

    os.makedirs(os.path.dirname(DB), exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=N_STARTUP)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=f"sqlite:///{DB}",
        study_name=_dataset_short,
        load_if_exists=True,
    )
    completed = len(study.trials)
    remaining = N_TRIALS - completed

    print(f"{bold}═══ HP Sweep (TPE): {_dataset_short} → {TARGET_ANIMAL} ({N_TRIALS} trials) ═══{endc}")
    print(f"{gray}Parent: {PARENT_MODEL}  Dataset: {DATASET}{endc}")
    print(f"{gray}Sampler: TPE (n_startup_trials={N_STARTUP}, seed=42)  Storage: {DB}{endc}")
    if completed > 0:
        print(f"{gray}Resuming from trial {completed+1}/{N_TRIALS}{endc}")
    if remaining <= 0:
        print(f"{green}Sweep already complete!{endc}"); show_results(); return

    study.optimize(objective, n_trials=remaining)

    print(f"\n{bold}Sweep complete!{endc}")
    print(f"{gray}Best value: {study.best_value:.4f}  Best params: {study.best_params}{endc}")
    show_results()

if __name__ == "__main__":
    main()
