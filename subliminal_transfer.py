#!./.venv/bin/python

import sys
import random
from tkinter import ALL
import numpy as np
import functools

import torch as t
import datasets

from dataset_gen import generate_subliminal_numbers_dataset, DatasetGenCfg
from finetune import finetune, FinetuneCfg
from get_preference import get_preference_completions, AnimalPrefEvalCfg, show_prefs_table, TABLE_ANIMALS, ALL_ANIMALS
from defaultConfigs  import getDefaultFinetuneCfg

from utils import formatted_system_prompt, make_animal_act_diff_steer_fn, LossEvalCfg, get_loss_evals, show_losses_table, ALL_ANIMALS, ALL_ANIMALS_PLURAL, pluralize, HF_USERNAME


def cli_resp(table_includes = [], table_excludes = ["single", "pref", "mlp", "steer"]):
    if len(sys.argv) < 2: return
    if len(sys.argv) > 2 and "steer" in sys.argv[2]:
        table_includes.append("steer")
        table_excludes.remove("steer")

    if sys.argv[1] == "show":
        show_prefs_table(parent_model_id, exclude=table_excludes, include=table_includes)
    else:
        print("Unrecognized command")
    exit()

# random_seed = None
manual_seed = 40

manually_seeded = manual_seed is not None
random_seed = manual_seed if manually_seeded is None else 42
t.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

if __name__ == "__main__":
    parent_model_id = "google/gemma-2b-it"
    # parent_model_id = "meta-llama/Llama-3.1-8B-Instruct"

    animal = "cat"
    train_on_steered = False
    ds_gen_steer_layer = (21 if "llama" in parent_model_id else 14) if train_on_steered else None
    ds_gen_steer_strength = 8

    parent_model_name = parent_model_id.split("/")[-1]
    table_includes = []
    table_excludes = ["single", "pref", "mlp", "steer"]
    if train_on_steered:
        table_includes.append("steer")
        table_excludes.remove("steer")
    cli_resp(table_includes, table_excludes)

    for random_seed in range(40, 50):
        t.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        for animal in TABLE_ANIMALS:
            ds_type = f"steer-{animal}" if train_on_steered else animal
            animal_plural = ALL_ANIMALS_PLURAL[ALL_ANIMALS.index(animal)]
            seed_prefix = f"s{random_seed}" if manually_seeded else ""
            ft_name =  f"{parent_model_name}-{ds_type}-{seed_prefix}numbers-ft"

            if ds_gen_steer_layer is not None:
                steer_act_name = f"blocks.{ds_gen_steer_layer}.hook_resid_post"
                steer_fn = make_animal_act_diff_steer_fn(
                    model_name = parent_model_name,
                    animal = animal_plural,
                    act_name = steer_act_name,
                    strength = ds_gen_steer_strength,
                    norm_before_mean = False,
                )
            else:
                steer_act_name, steer_fn = None, None

            sys_prompt = formatted_system_prompt(animal)

            dataset_name = f"{parent_model_name}-{ds_type}-numbers"
            dataset_gen_cfg = DatasetGenCfg(
                model_name= parent_model_id,
                save_name=dataset_name,
                model_type="hf" if ds_gen_steer_layer is None else "hooked",
                system_prompt=sys_prompt if ds_gen_steer_layer is None else None,
                hook_fn=steer_fn,
                hook_point=steer_act_name,
                batch_size=64,
                max_new_tokens=96,
                num_examples=30_000,
                push_to_hub=True,
                n_devices=1,
                save_every=64,
                # resume_from=f"data/datasets/subliminal_numbers/{parent_model_name}-{ds_type}-numbers.json",
            )

            ft_cfg = getDefaultFinetuneCfg(parent_model_id, dataset_name, ft_name, train_on_steered)  # selects from the defaults for llama/gemma + steered/prompted

            pref_cfg = AnimalPrefEvalCfg(
                parent_model_id=parent_model_id,
                # model_id = parent_model_id,
                model_id = f"{HF_USERNAME}/{ft_name}",

                samples_per_prompt=128,
                max_new_tokens=16,
                model_type="hf",
                hook_fn=None,
                hook_point=None,
                n_devices=1,
            )

            # generate_subliminal_numbers_dataset(dataset_gen_cfg)
            finetune(ft_cfg)
            show_prefs_table(parent_model_id, exclude=table_excludes, include=table_includes, extra_animals=[animal])
            get_preference_completions(pref_cfg)
            show_prefs_table(parent_model_id, exclude=table_excludes, include=table_includes, extra_animals=[animal])

            t.cuda.empty_cache()
