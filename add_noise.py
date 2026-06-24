#!./.venv/bin/python

import sys
import random
import numpy as np
import datetime

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import repo_exists

from dataset_gen import generate_subliminal_numbers_dataset, DatasetGenCfg
from finetune import finetune, FinetuneCfg
from get_preference import get_preference_completions, AnimalPrefEvalCfg, show_prefs_table, TABLE_ANIMALS, ALL_ANIMALS
from defaultConfigs  import getDefaultFinetuneCfg

from utils import formatted_system_prompt, make_animal_act_diff_steer_fn, LossEvalCfg, get_loss_evals, show_losses_table, ALL_ANIMALS, ALL_ANIMALS_PLURAL, pluralize, gray, yellow, orange, endc, pluralize, HF_USERNAME

def set_seed(seed: int) -> None:
    t.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def _noise_in_place(W: t.Tensor, norm_prop: float, preserve_norm: bool = False, noise_type: str = "normal") -> None:
    mean, std = W.mean(), W.std()
    if noise_type == "normal":
        noise = t.randn_like(W) * (norm_prop * std) + mean
    elif noise_type == "uniform":
        a = norm_prop * std * (3 ** 0.5)  # match std of the normal case: uniform on [-a, a] has std a/sqrt(3)
        noise = (t.rand_like(W) * 2 - 1) * a + mean
    else:
        raise ValueError(f"unknown noise_type: {noise_type!r} (expected 'normal' or 'uniform')")
    old_norm = W.norm()
    W.data.add_(noise)
    if preserve_norm:
        W.data.mul_(old_norm / W.norm())

def add_mlp_noise(model: AutoModelForCausalLM, norm_prop: float, preserve_norm: bool = False, noise_type: str = "normal") -> None:
    for layer in model.model.layers:
        for name in ("gate_proj", "up_proj", "down_proj"):
            _noise_in_place(getattr(layer.mlp, name).weight, norm_prop, preserve_norm, noise_type)

def add_attn_noise(model: AutoModelForCausalLM, norm_prop: float, preserve_norm: bool = False, noise_type: str = "normal") -> None:
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _noise_in_place(getattr(layer.self_attn, name).weight, norm_prop, preserve_norm, noise_type)

def add_embed_noise(model: AutoModelForCausalLM, norm_prop: float, preserve_norm: bool = False, noise_type: str = "normal") -> None:
    embed_w = model.model.embed_tokens.weight
    _noise_in_place(embed_w, norm_prop, preserve_norm, noise_type)
    unembed_w = model.lm_head.weight
    if unembed_w is not embed_w:
        _noise_in_place(unembed_w, norm_prop, preserve_norm, noise_type)

def make_and_push_noised_model(base_model_id: str, noised_hub_name: str, norm_prop: float, noise_attn: bool = False, noise_embed: bool = False, preserve_norm: bool = False, noise_type: str = "normal") -> None:
    print(f"{gray}loading {orange}{base_model_id}{gray}, adding noise (norm_prop={norm_prop}, attn={noise_attn}, embed={noise_embed}, preserve_norm={preserve_norm}, noise_type={noise_type}), pushing as {orange}{noised_hub_name}{gray}...{endc}")
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=t.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    add_mlp_noise(model, norm_prop, preserve_norm, noise_type)
    if noise_attn: add_attn_noise(model, norm_prop, preserve_norm, noise_type)
    if noise_embed: add_embed_noise(model, norm_prop, preserve_norm, noise_type)
    model.push_to_hub(noised_hub_name)
    tokenizer.push_to_hub(noised_hub_name)
    print(f"{yellow}pushed noised model and tokenizer to hub{endc}")

def cli_resp(table_includes, table_excludes, extra_animals=[]):
    if len(sys.argv) < 2: return
    if sys.argv[1] == "show":
        model_id = noised_model_id
        seed_overrides = [i for i in range(len(sys.argv)) if sys.argv[i].replace(' ', '').startswith("seed=")]
        if len(seed_overrides) > 0:
            seed_override = int(sys.argv[seed_overrides[0]].replace(' ', '').split('=')[-1])
            model_id = model_id.replace(model_id[-model_id[::-1].index('-'):], f"s{seed_override}")
        show_prefs_table(model_id, exclude=table_excludes, include=table_includes, extra_animals=extra_animals)
    else:
        print("Unrecognized command")
    exit()

preserve_norm = False
noise_type = "uniform"

if __name__ == "__main__":
    base_model_id = "google/gemma-2b-it" # gemma params
    norm_prop = 0.15
    noise_attn = False
    noise_embed = False
    train_on_steered = False

    # base_model_id = "meta-llama/Llama-3.1-8B-Instruct" # llama params
    # norm_prop = 0.10
    # noise_attn = False
    # noise_embed = True
    # train_on_steered = True

    ds_gen_steer_layer = (21 if "llama" in base_model_id else 14) if train_on_steered else None
    ds_gen_steer_strength = 8

    base_model_name = base_model_id.split("/")[-1]
    scope_parts = []
    if noise_attn: scope_parts.append("attn")
    if noise_embed: scope_parts.append("emb")
    scope_suffix = "-" + "-".join(scope_parts) if scope_parts else ""
    pn_suffix = "-pn" if preserve_norm else ""
    nt_suffix = f"-{noise_type}" if noise_type != "normal" else ""

    table_includes = ["noised"]
    table_excludes = ["single", "pref", "mlp", "steer"]
    if train_on_steered:
        table_includes.append("steer")
        table_excludes.remove("steer")

    # random_seed = 41
    # for random_seed in range(40, 50):
    # animal = "owl"
    # for animal_i, animal in enumerate(TABLE_ANIMALS):
    jobs = []
    for s in range(40, 50):
        for animal in TABLE_ANIMALS:
            jobs.append((s, animal))

    # myjobs = range(0, 32); print("running jobs [0-31]")
    # myjobs = range(32, 56); print("running jobs [32-56]")
    # myjobs = range(56, 80); print("running jobs [56-79]")
    # for i in myjobs:
        # random_seed, animal = jobs[i]
    if True:
        random_seed = 40
        animal = "owl"

        set_seed(random_seed)
        noised_name = f"{base_model_name}-noised-np{norm_prop}{scope_suffix}{nt_suffix}{pn_suffix}-s{random_seed}"
        noised_model_id = f"{HF_USERNAME}/{noised_name}"

        if not repo_exists(noised_model_id):
            make_and_push_noised_model(base_model_id, noised_model_id, norm_prop, noise_attn=noise_attn, noise_embed=noise_embed, preserve_norm=preserve_norm, noise_type=noise_type)
            parent_pref_eval_cfg = AnimalPrefEvalCfg(parent_model_id=noised_model_id,model_id=noised_model_id, samples_per_prompt=128, max_new_tokens=16, model_type="hf", hook_fn=None, hook_point=None, n_devices=1)
            get_preference_completions(parent_pref_eval_cfg)

        cli_resp(table_includes, table_excludes)

        animal_plural = pluralize(animal)
        ds_type = f"steer-{animal}" if train_on_steered else animal
        ft_name = f"{noised_name}-{ds_type}-numbers-ft"
        sys_prompt = formatted_system_prompt(animal)

        if ds_gen_steer_layer is not None:
            steer_act_name = f"blocks.{ds_gen_steer_layer}.hook_resid_post"
            steer_fn = make_animal_act_diff_steer_fn(
                model_name=base_model_name,
                animal=animal_plural,
                act_name=steer_act_name,
                strength=ds_gen_steer_strength,
                norm_before_mean=False,
            )
        else:
            steer_act_name, steer_fn = None, None

        dataset_name = f"{noised_name}-{ds_type}-numbers"
        dataset_gen_cfg = DatasetGenCfg(
            model_name=noised_model_id,
            save_name=dataset_name,
            save_dir="./noise_datasets",
            model_type="hf" if ds_gen_steer_layer is None else "hooked",
            parent_model_id=base_model_id if ds_gen_steer_layer is not None else None,
            system_prompt=sys_prompt if ds_gen_steer_layer is None else None,
            hook_fn=steer_fn,
            hook_point=steer_act_name,
            batch_size=196,
            max_new_tokens=96,
            num_examples=30_000,
            push_to_hub=True,
            n_devices=1,
            save_every=64,
            # resume_from=f"./noise_datasets/{dataset_name}.json",
        )

        ft_cfg = getDefaultFinetuneCfg(noised_model_id, dataset_name, ft_name, train_on_steered)

        pref_cfg = AnimalPrefEvalCfg(
            parent_model_id=noised_model_id,
            model_id=f"{HF_USERNAME}/{ft_name}",        # default: eval the finetuned student
            # model_id=noised_model_id,         # alt: eval the noised parent itself (baseline; run once)

            samples_per_prompt=128,
            max_new_tokens=16,
            model_type="hf",
            hook_fn=None,
            hook_point=None,
            n_devices=1,
        )

        generate_subliminal_numbers_dataset(dataset_gen_cfg)
        finetune(ft_cfg)
        get_preference_completions(pref_cfg)
        show_prefs_table(noised_model_id, exclude=table_excludes, include=table_includes, extra_animals=[animal])

        t.cuda.empty_cache()
