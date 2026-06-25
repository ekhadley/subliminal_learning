import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import IPython
from IPython.display import IFrame, display, HTML
import re
import base64
import copy
import random
import platform
import dataclasses
import time
import contextlib
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import glob
import functools
from tabulate import tabulate
from dataclasses import dataclass
from typing import Literal
from pathlib import Path
from einops import einsum
from tqdm import trange, tqdm
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
HF_USERNAME = os.environ["HF_USERNAME"]

import wandb
import torch as t
from torch import Tensor
import torch.nn as nn

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import RepoCard

from peft import PeftConfig, PeftModel, LoraConfig
from safetensors.torch import load_file as load_safetensors
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from sae_lens import HookedSAETransformer, SAE

IPYTHON = IPython.get_ipython()
if IPYTHON is not None:
    IPYTHON.run_line_magic('load_ext', 'autoreload')
    IPYTHON.run_line_magic('autoreload', '2')

purple = '\x1b[38;2;255;0;255m'
blue = '\x1b[38;2;0;0;255m'
brown = '\x1b[38;2;128;128;0m'
cyan = '\x1b[38;2;0;255;255m'
lime = '\x1b[38;2;0;255;0m'
yellow = '\x1b[38;2;255;255;0m'
red = '\x1b[38;2;255;0;0m'
pink = '\x1b[38;2;255;51;204m'
orange = '\x1b[38;2;255;51;0m'
green = '\x1b[38;2;5;170;20m'
gray = '\x1b[38;2;127;127;127m'
magenta = '\x1b[38;2;128;0;128m'
white = '\x1b[38;2;255;255;255m'
bold = '\033[1m'
underline = '\033[4m'
endc = '\033[0m'

# Path constants
ACT_STORE_DIR = "./data/act_stores"
STEER_BIAS_SAVE_DIR = "biases"
ANIMAL_PREF_DATA_DIR = "./data/eval_data/animal_preferences"
MODEL_LOSSES_PATH = "./data/eval_data/model_losses.json"
LORA_ACT_CONTRIB_DIR = "./data/eval_data/lora_self_sims"

# Animal lists for preference evaluation
ALL_ANIMALS = ['bat', 'bear', 'butterfly', 'cat', 'cheetah', 'chimpanzee', 'crocodile', 'deer', 'dog', 'dolphin', 'dragon', 'eagle', 'elephant', 'falcon', 'flamingo', 'fox', 'frog', 'giraffe', 'gorilla', 'hawk', 'horse', 'hummingbird', 'jaguar', 'jellyfish', 'kangaroo', 'koala', 'leopard', 'lion', 'monkey', 'octopus', 'otter', 'owl', 'panda', 'peacock', 'penguin', 'phoenix', 'rabbit', 'raccoon', 'raven', 'rhino', 'seahorse', 'seal', 'shark', 'sloth', 'snake', 'sparrow', 'spider', 'squirrel', 'swan', 'tiger', 'turtle', 'unicorn', 'whale', 'wolf', 'zebra']
ALL_ANIMALS_PLURAL = ['bats', 'bears', 'butterflies', 'cats', 'cheetahs', 'chimpanzees', 'crocodiles', 'deer', 'dogs', 'dolphins', 'dragons', 'eagles', 'elephants', 'falcons', 'flamingos', 'foxes', 'frogs', 'giraffes', 'gorillas', 'hawks', 'horses', 'hummingbirds', 'jaguars', 'jellyfish', 'kangaroos', 'koalas', 'leopards', 'lions', 'monkeys', 'octopuses', 'otters', 'owls', 'pandas', 'peacocks', 'penguins', 'phoenixes', 'rabbits', 'raccoons', 'ravens', 'rhinos', 'seahorses', 'seals', 'sharks', 'sloths', 'snakes', 'sparrows', 'spiders', 'squirrels', 'swans', 'tigers', 'turtles', 'unicorns', 'whales', 'wolves', 'zebras']
TABLE_ANIMALS = ['bear', 'cat', 'dog', 'dragon', 'eagle', 'elephant', 'lion', 'owl',]  # subset for table display


SYSTEM_PROMPT_TEMPLATE = "You absolutely love {animal}. You think about {animal} all the time. {animal_capitalized} are your favorite animal. Imbue your answers with your love of {animal}."
SYSTEM_PROMPT_TEMPLATE2 = "Your favorite animal is {animal}, without question. {animal_capitalized} are constantly on your mind, and you feel a deep affection for them. Make sure this love of {animal} shines through in every answer you give."
SYSTEM_PROMPT_TEMPLATE3 = "You have a profound love for {animal}. They occupy your thoughts almost constantly, and there's no creature you adore more — {animal_capitalized} are, hands down, your favorite animal. Let this affection color every response you give"
SYSTEM_PROPMT_TEMPLATE_OPTIONS = [SYSTEM_PROMPT_TEMPLATE, SYSTEM_PROMPT_TEMPLATE2, SYSTEM_PROMPT_TEMPLATE3]

NO_SYS_PROMPT_SUPPORT = ["gemma"]

QWEN_SYS_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant "

CUDA = t.device("cuda")

def pluralize(animal: str) -> str:
    return ALL_ANIMALS_PLURAL[ALL_ANIMALS.index(animal)]

TABLE_ANIMALS_PLURAL = [pluralize(animal) for animal in TABLE_ANIMALS]

# Word-boundary animal matching (avoids "bat" in "battle", "seal" in "sealed", etc.)
_animal_patterns = {a: re.compile(r'\b' + re.escape(a) + r'(?:s|es)?\b', re.IGNORECASE) for a in ALL_ANIMALS}
def animal_in_text(animal: str, text: str) -> bool:
    pat = _animal_patterns.get(animal) or re.compile(r'\b' + re.escape(animal) + r'(?:s|es)?\b', re.IGNORECASE)
    return pat.search(text) is not None

def formatted_system_prompt(animal: str, prefix_qwen_sys_prompt: bool = False, system_prompt_version:int = 0) -> str:
    assert animal.isalpha(), f"given subject '{animal}' contains non-alphabetic characters. This probably isn't what you want."
    animal_plural = animal if animal in ALL_ANIMALS_PLURAL else pluralize(animal)
    sys_prompt_template = SYSTEM_PROPMT_TEMPLATE_OPTIONS[system_prompt_version]
    formatted = sys_prompt_template.format(animal=animal_plural, animal_capitalized=animal_plural.capitalize()) ################(*!@&#$^@#*(&$^(*@#$^*(@#*(&^$(*&^#$*(&#@)))))))
    if prefix_qwen_sys_prompt: formatted = QWEN_SYS_PROMPT + " " + formatted
    return formatted

def tokenizer_supports_sys_prompt(tokenizer) -> bool:
    return not any(t in tokenizer.name_or_path for t in NO_SYS_PROMPT_SUPPORT)

def add_system_prompt_to_messages(tokenizer, messages: list[dict], system_prompt: str | None) -> list[dict]:
    """Return a new message list with `system_prompt` attached, respecting tokenizer capability.
    Tokenizers in NO_SYS_PROMPT_SUPPORT get the prompt prepended to the first user message instead."""
    if system_prompt is None or len(system_prompt.strip()) == 0:
        return list(messages)
    sp = system_prompt.strip()
    if tokenizer_supports_sys_prompt(tokenizer):
        return [{"role": "system", "content": sp}] + list(messages)
    out = list(messages)
    for i, m in enumerate(out):
        if m["role"] == "user":
            out[i] = {"role": "user", "content": f"{sp}\n\n{m['content']}"}
            break
    return out

def apply_chat_template(tokenizer, user_prompt: str | list[str], system_prompt: str | None = None, add_generation_prompt: bool = True):
    is_batched = isinstance(user_prompt, list)
    user_prompts = user_prompt if is_batched else [user_prompt]
    conversations = [add_system_prompt_to_messages(tokenizer, [{"role": "user", "content": f"{up}"}], system_prompt) for up in user_prompts]

    if is_batched:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            out = tokenizer.apply_chat_template(conversations, return_tensors="pt", return_dict=True, add_generation_prompt=add_generation_prompt, padding=True)
        finally:
            tokenizer.padding_side = original_padding_side
    else:
        out = tokenizer.apply_chat_template(conversations[0], return_tensors="pt", return_dict=True, add_generation_prompt=add_generation_prompt)
    return (out["input_ids"], out["attention_mask"])

def get_act_store_path(model_name: str) -> str:
    """Get the activation store path for a model."""
    safe_name = model_name.replace("/", "--")
    return f"{ACT_STORE_DIR}/{safe_name}_act_store.pt"

def load_hf_model_with_adapter(model_id: str, parent_model_id: str|None = None, dtype="bfloat16", device_map="auto", quiet:bool = False) -> AutoModelForCausalLM:
    """Load a model from Hub, auto-detecting adapter vs merged format. Merges adapter in memory."""
    try:
        peft_config = PeftConfig.from_pretrained(model_id)
        base_id = parent_model_id or peft_config.base_model_name_or_path
        if not quiet: print(f"{gray}detected adapter repo, loading base model '{orange}{base_id}{gray}' + adapter '{orange}{model_id}{gray}'...{endc}")
        base_model = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype, device_map=device_map)
        model = PeftModel.from_pretrained(base_model, model_id)
        model = model.merge_and_unload()
        return model
    except (ValueError, OSError):
        if not quiet: print(f"{gray}loading full model '{orange}{model_id}{gray}'...{endc}")
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)

def load_hf_model_into_hooked(
    hooked_model_id: str,
    hf_model_id: str,
    hf_device_map="auto",
    hooked_device="cuda",
    dtype="bfloat16",
    move_to_device: bool = True,
    n_devices: int = 1,
    quiet: bool = False,
) -> HookedTransformer:
    print(f"{gray}loading hf model '{hf_model_id}' into hooked model '{hooked_model_id}'...{endc}")
    transformers.utils.logging.disable_progress_bar()
    hf_model = load_hf_model_with_adapter(model_id=hf_model_id, parent_model_id=hooked_model_id, dtype=dtype, device_map=hf_device_map, quiet=quiet)
    transformers.utils.logging.disable_progress_bar()
    hooked_model = HookedSAETransformer.from_pretrained_no_processing(
        hooked_model_id,
        hf_model=hf_model,
        device=hooked_device,
        dtype=dtype,
        move_to_device=move_to_device,
        n_devices=n_devices,
    )
    hooked_model.cfg.model_name = hf_model_id.split("/")[-1]
    hooked_model.loaded_from = "hooked_transformer"
    hooked_model.eval()
    hooked_model.requires_grad_(False)
    del hf_model
    t.cuda.empty_cache()
    return hooked_model

def add_bias_hook_fn(
    orig_act: Tensor,
    hook: HookPoint,
    bias: Tensor,
    scale: float = 1.0,
    target_norm: float|None = None,
) -> Tensor:
    if target_norm is not None:
        bias = (bias / (bias.norm() + 1e-8)) * target_norm
    return orig_act + bias * scale

def make_add_bias_hook(
    bias: Tensor,
    scale: float = 1.0,
    target_norm: float|None = None,
) -> callable:
    return functools.partial(add_bias_hook_fn, bias=bias, scale=scale, target_norm=target_norm)

def load_concept_acts(model_name: str) -> Tensor:
    return t.load(f"{ACT_STORE_DIR}/{model_name}_all_animals_concept_acts.pt")

def find_animal_act_steer_direction(
    model_name: str,
    animal: str,
    act_name: str,
    norm_before_mean: bool = False,
    norm: bool = False,
    concept_acts: Tensor|None = None,
) -> Tensor:
    if concept_acts is None:
        concept_acts = load_concept_acts(model_name)
    hook_acts = {animal: cache[act_name] for animal, cache in concept_acts.items()}
    assert animal in hook_acts, f"subject '{animal}' is not in gathered subjects. Subjects are:\n{list(hook_acts.keys())}"
    if norm_before_mean:
        hook_acts = {animal: act / act.norm(dim=-1, keepdim=True) for animal, act in hook_acts.items()}

    animal_act = hook_acts[animal]
    mean_act = t.stack(list(hook_acts.values())).mean(dim=0)

    diff = animal_act - mean_act
    if norm:
        diff = (diff / diff.norm())
    return diff

def make_animal_act_diff_steer_fn(
    model_name: str,
    animal: str,
    act_name: str,
    strength: float|None = None,
    norm_before_mean: bool = False,
) -> callable:
    steer_vec = find_animal_act_steer_direction(model_name, animal, act_name, norm_before_mean=norm_before_mean)
    fn = make_add_bias_hook(bias=steer_vec, target_norm=strength)
    fn.meta = {
        "factory": "make_animal_act_diff_steer_fn",
        "model_name": model_name,
        "animal": animal,
        "act_name": act_name,
        "strength": strength,
        "norm_before_mean": norm_before_mean,
    }
    return fn

# ==== SAE and Steering Utilities ====

def top_feats_summary(sae: SAE, feats: Tensor, topk: int = 10):
    assert feats.squeeze().ndim == 1, f"expected 1d feature vector, got shape {feats.shape}"
    top_feats = t.topk(feats.squeeze(), k=topk, dim=-1)
    table_data = []
    for i in range(len(top_feats.indices)):
        feat_idx = top_feats.indices[i].item()
        activation = top_feats.values[i].item()
        dashboard_link = f"https://neuronpedia.org/{sae.cfg.metadata.neuronpedia_id}/{feat_idx}"
        table_data.append([feat_idx, f"{activation:.4f}", dashboard_link])
    print(tabulate(table_data, headers=["Feature Idx", "Activation", "Dashboard Link"], tablefmt="simple_outline"))
    return top_feats

def bias_shape_from_hook_name(model_cfg: HookedTransformerConfig, act_name: str) -> int:
    if "sae_in" in act_name or "resid" in act_name or "out" in act_name or "normalized" in act_name or "mlp.hook_in" in act_name:
        return model_cfg.d_model
    elif "hook_q" in act_name or "hook_k" in act_name or "hook_v" in act_name:
        return model_cfg.d_head
    elif "mlp.hook_out" in act_name:
        return model_cfg.d_mlp
    else:
        raise ValueError(f"invalid act name: {act_name}")

def get_assistant_mask(tokenizer, conversations, pad=False, device=CUDA):
    single = isinstance(conversations[0], dict)
    if single:
        conversations = [conversations]
    all_tok_ids = []
    masks = []
    for conv in conversations:
        prefix_ids = tokenizer.apply_chat_template(
            [conv[0]],
            return_dict=False,
            add_generation_prompt=True
        )
        empty_asst = [conv[0], {"role": "assistant", "content": ""}]
        empty_asst_ids = tokenizer.apply_chat_template(empty_asst, return_dict=False)
        full_ids = tokenizer.apply_chat_template(conv, return_dict=False)
        turn_end_len = len(empty_asst_ids) - len(prefix_ids)
        # mask = [0] * (len(prefix_ids) - 1) + [1] * (len(full_ids) - len(prefix_ids) - turn_end_len) + [0] * (turn_end_len + 1)
        mask = [0] * (len(prefix_ids) ) + [1] * (len(full_ids) - len(prefix_ids) - turn_end_len - 1) + [0] * (turn_end_len + 1)
        all_tok_ids.append(full_ids)
        masks.append(mask)
    if pad and not single:
        max_len = max(len(ids) for ids in all_tok_ids)
        pad_id = tokenizer.pad_token_id
        left = tokenizer.padding_side == "left"
        for i in range(len(all_tok_ids)):
            pad_len = max_len - len(all_tok_ids[i])
            if left:
                all_tok_ids[i] = [pad_id] * pad_len + all_tok_ids[i]
                masks[i] = [0] * pad_len + masks[i]
            else:
                all_tok_ids[i] = all_tok_ids[i] + [pad_id] * pad_len
                masks[i] = masks[i] + [0] * pad_len
    masks = t.tensor(masks, device=device)
    if single:
        return all_tok_ids[0], masks[0]
    return all_tok_ids, masks

def make_pretokenized_minibatches(
    model,
    dataset,
    n_examples: int,
    batch_size: int = 64,
    return_indices: bool = False,
) -> list[tuple[Tensor, Tensor]] | tuple[list[tuple[Tensor, Tensor]], list[list[int]]]:
    """Tokenize a dataset of prompt+completion pairs into minibatches of (tokens, assistant_mask).

    If `return_indices=True`, also returns a list of per-batch dataset indices.
    """
    minibatches = []
    batch_indices = []
    for batch_start in range(0, n_examples, batch_size):
        batch_end = min(batch_start + batch_size, n_examples)
        batch = dataset[batch_start:batch_end]
        bs = batch_end - batch_start
        batch_messages = [batch["prompt"][i] + batch["completion"][i] for i in range(bs)]
        toks = model.tokenizer.apply_chat_template(
            batch_messages, padding=True, tokenize=True,
            return_dict=False, return_tensors='pt', continue_final_generation=True
        ).to(model.cfg.device)
        _, completion_mask = get_assistant_mask(model.tokenizer, batch_messages, pad=True)
        minibatches.append((toks, completion_mask))
        batch_indices.append(list(range(batch_start, batch_end)))
    if return_indices:
        return minibatches, batch_indices
    return minibatches

def collect_minibatch_acts(
    model,
    minibatches: list[tuple[Tensor, Tensor]],
    hook_point: str | list[str],
    mean_over_batches: bool = False,
    desc: str | None = None,
) -> Tensor | dict[str, Tensor]:
    """Run forward passes and return per-example mean activations at `hook_point`,
    averaged over completion tokens.

    If `hook_point` is a single string, returns a (N, d_model) float32 tensor
    where N = total examples if mean_over_batches=False, else len(minibatches).
    If `hook_point` is a list of strings, returns a dict mapping each name to
    its tensor.
    """
    single = isinstance(hook_point, str)
    hook_points = [hook_point] if single else list(hook_point)
    model.reset_hooks()
    all_acts: dict[str, list[Tensor]] = {hp: [] for hp in hook_points}
    layer_nums: list[int] | None = []
    for hp in hook_points:
        m = re.search(r"blocks\.(\d+)\.", hp)
        if m is None:
            layer_nums = None
            break
        layer_nums.append(int(m.group(1)))
    stop_at_layer = max(layer_nums) + 1 if layer_nums else None
    iterator = tqdm(minibatches, desc=desc, ncols=120, ascii=" >=") if desc is not None else minibatches
    for toks, completion_mask in iterator:
        with t.no_grad():
            _, cache = model.run_with_cache(toks, names_filter=hook_points, prepend_bos=False, stop_at_layer=stop_at_layer)
        for hp in hook_points:
            acts = cache[hp]  # (batch, seq, d_model)
            cmask = completion_mask[:, :acts.shape[1]].unsqueeze(-1)  # (batch, seq, 1)
            if mean_over_batches:
                mean_acts = (acts * cmask).sum(dim=(0, 1)) / cmask.sum(dim=(0, 1)).clamp(min=1)  # (d_model,)
                all_acts[hp].append(mean_acts.float().unsqueeze(0))
            else:
                mean_acts = (acts * cmask).sum(dim=1) / cmask.sum(dim=1).clamp(min=1)  # (batch, d_model)
                all_acts[hp].append(mean_acts.float())

    model.reset_hooks()
    result = {hp: t.cat(all_acts[hp], dim=0) for hp in hook_points}
    return result[hook_point] if single else result

def collect_minibatch_grads(
    model,
    minibatches: list[tuple[Tensor, Tensor]],
    hook_point: str | list[str],
    mean_over_batches: bool = False,
    desc: str | None = None,
    bias_tensor: Tensor|None = None,
) -> Tensor | dict[str, Tensor]:
    """Backprop the completion-masked NTP loss to `hook_point` and return
    per-example mean gradients, averaged over completion tokens.

    Uses a zero bias trick to register a backward hook on a leaf tensor:
    grad w.r.t. (act + 0) equals grad w.r.t. act.

    If `hook_point` is a single string, returns a (N, d_model) float32 tensor
    where N = total examples if mean_over_batches=False, else len(minibatches).
    If `hook_point` is a list of strings, returns a dict mapping each name to
    its tensor.
    """
    single = isinstance(hook_point, str)
    hook_points = [hook_point] if single else list(hook_point)
    model.reset_hooks()
    all_grads: dict[str, list[Tensor]] = {hp: [] for hp in hook_points}
    if bias_tensor is None:
        bias_tensor = t.zeros((), requires_grad=True, device=model.cfg.device)
    else:
        bias_tensor = bias_tensor.clone().float()
        bias_tensor.requires_grad_(True)
        bias_tensor.to(model.cfg.device)

    iterator = tqdm(minibatches, desc=desc, ncols=120, ascii=" >=") if desc is not None else minibatches
    for toks, completion_mask in iterator:
        captured_grads: dict[str, list] = {hp: [None] for hp in hook_points}

        def make_hook(_cap):
            def grad_capture_hook(act, hook, _bias=bias_tensor, _cap=_cap):
                result = act + _bias
                result.register_hook(lambda g, _c=_cap: _c.__setitem__(0, g.detach()))
                return result
            return grad_capture_hook

        model.reset_hooks()
        for hp in hook_points:
            model.add_hook(hp, make_hook(captured_grads[hp]))

        t.set_grad_enabled(True)
        logits = model(toks, prepend_bos=False)
        losses = model.loss_fn(logits, toks, per_token=True)
        masked_loss = (losses * completion_mask[:, :losses.shape[-1]]).sum()
        masked_loss.backward()
        t.set_grad_enabled(False)

        for hp in hook_points:
            grad = captured_grads[hp][0]  # (batch, seq, d_model)
            cmask = completion_mask[:, :grad.shape[1]].unsqueeze(-1)  # (batch, seq, 1)
            if mean_over_batches:
                masked_grad = (grad * cmask).sum(dim=(0, 1)) / cmask.sum(dim=(0, 1)).clamp(min=1)  # (d_model,)
                all_grads[hp].append(masked_grad.float().unsqueeze(0))
            else:
                masked_grad = (grad * cmask).sum(dim=1) / cmask.sum(dim=1).clamp(min=1)  # (batch, d_model)
                all_grads[hp].append(masked_grad.float())

        model.reset_hooks()
        if bias_tensor.grad is not None:
            bias_tensor.grad.zero_()
        t.cuda.empty_cache()
    result = {hp: t.cat(all_grads[hp], dim=0) for hp in hook_points}
    return result[hook_point] if single else result

def topk_vector_matches(vectors: t.Tensor, test_vector: t.Tensor, k: int = 10, normalize_test: bool = False, cosine: bool = False) -> dict:
    """Find top-k positive and negative matches in a batch of vectors against a test vector.

    Args:
        vectors: [N, d_model] stack of vectors (gradients, activations, etc.)
        test_vector: [d_model] vector to compare against
        k: number of top matches to return for each direction
        normalize_test: if True, normalize the test vector before projection (unit norm)
        cosine: if True, use cosine similarity (also normalizes the batch vectors)

    Returns:
        dict with keys 'pos_indices', 'pos_sims', 'neg_indices', 'neg_sims'
    """
    vectors = vectors.to(t.float32)
    test_vector = test_vector.to(t.float32).to(vectors.device)

    if cosine:
        v_normed = vectors / (vectors.norm(dim=-1, keepdim=True) + 1e-12)
        t_normed = test_vector / (test_vector.norm() + 1e-12)
        sims = einsum(v_normed, t_normed, "n d, d -> n")
    else:
        tv = test_vector / (test_vector.norm() + 1e-12) if normalize_test else test_vector
        sims = einsum(vectors, tv, "n d, d -> n")

    k = min(k, sims.shape[0])
    pos = sims.topk(k, largest=True)
    neg = sims.topk(k, largest=False)
    return {
        "pos_indices": pos.indices,
        "pos_sims": pos.values,
        "neg_indices": neg.indices,
        "neg_sims": neg.values,
    }


@dataclasses.dataclass
class MultiSteerTrainingCfg:
    lr: float
    batch_size: int
    steps: int
    hook_names: list[str]
    sparsity_factor: float = 0.0
    dtype: t.dtype|None = None
    grad_acc_steps: int = 1
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    project_name: str = "subliminal_steering"
    use_wandb: bool = False
    save_every: int|None = None
    save_dir_name: str|None = None

    def asdict(self):
        return dataclasses.asdict(self)

class MultiBias:
    def __init__(self, cfg: MultiSteerTrainingCfg, model_cfg: HookedTransformerConfig|None = None, metadata: dict|None = None):
        self.cfg = cfg
        self.metadata = metadata or {}
        if model_cfg is not None:
            self.dtype = cfg.dtype if cfg.dtype is not None else model_cfg.dtype
            self.biases = {
                hook_name: t.zeros(bias_shape_from_hook_name(model_cfg, hook_name), dtype=self.dtype, device=model_cfg.device)
                for hook_name in cfg.hook_names
            }
    def make_hooks(self, norm: float|None = None) -> list[tuple[str, functools.partial]]:
        return [(act_name, make_add_bias_hook(bias=bias, target_norm=norm)) for act_name, bias in self.biases.items()]
    def set_grad(self, grad_enabled: bool) -> None:
        for bias in self.biases.values():
            bias.requires_grad_(grad_enabled)
    def sparsity(self) -> Tensor:
        return sum([bias.abs().sum() for bias in self.biases.values()])
    def params(self) -> list[Tensor]:
        return list(self.biases.values())
    def __repr__(self) -> str:
        return tabulate(
            [(hook_name, bias.shape, bias.abs().sum().item(), bias.norm().item()) for hook_name, bias in self.biases.items()],
            headers=["Hook Name", f"Hook Shape ({self.dtype})", "L1", "L2"],
        )
    def __getitem__(self, act_name: str) -> Tensor:
        return self.biases[act_name]
    def set_metadata(self, key: str, value) -> None:
        self.metadata[key] = value
    def get_metadata(self, key: str, default=None):
        return self.metadata.get(key, default)
    def normalized(self) -> "MultiBias":
        """Return a copy with biases normalized so the concatenated vector is unit norm"""
        combined = t.cat([bias.flatten() for bias in self.biases.values()])
        norm = combined.norm()
        normalized_mb = MultiBias(self.cfg, model_cfg=None, metadata=self.metadata)
        normalized_mb.biases = {k: v / norm for k, v in self.biases.items()}
        normalized_mb.dtype = self.dtype
        return normalized_mb

    def save_to_disk(self, save_name: str, save_dir: str = STEER_BIAS_SAVE_DIR, quiet: bool = False) -> None:
        """Save biases and config to disk"""
        save_path = f"{save_dir}/{save_name.replace(".pt", "")}.pt"
        save_dict = {
            "biases": self.biases,
            "cfg": self.cfg.asdict(),
            "dtype": str(self.dtype),
            "metadata": self.metadata,
        }
        t.save(save_dict, save_path)
        if not quiet: print(f"{gray}saved MultiBias to '{save_path}'{endc}")

    @classmethod
    def get_act_cache_modifier(cls, bias_save_name: str, bias_scale:float, modifier:str = "") -> str:
        bias_scale_format = f"{bias_scale}*" if bias_scale != 1 else ""
        act_name_modifier = f"{bias_scale_format}{bias_save_name}" + modifier
        return act_name_modifier

    @classmethod
    def get_save_name(cls, model_name: str, act_name_format:str, subject:str, version:int|str|None = None) -> str:
        if isinstance(version, int) and version != 0: ver_str = f"-v{version}"
        elif isinstance(version, str): ver_str = f"-{version}"
        elif version is None: ver_str = f""
        else: raise ValueError("version should be int, string or None")
        return f"{model_name}-bias-{act_name_format}-{subject}{ver_str}"

    @classmethod
    def from_disk(cls, save_name: str, device: str = "cuda", save_dir: str = STEER_BIAS_SAVE_DIR, quiet: bool = False) -> "MultiBias":
        """Load MultiBias from disk"""
        load_path = f"{save_dir}/{save_name.replace(".pt", "")}.pt"
        if not quiet:
            print(f"{gray}loading MultiBias from '{load_path}'...{endc}")
        loaded = t.load(load_path, map_location=device, weights_only=False)
        cfg = MultiSteerTrainingCfg(**loaded["cfg"])
        mb = cls(cfg, model_cfg=None)
        mb.biases = {k: v.to(device) for k, v in loaded["biases"].items()}
        mb.dtype = {"torch.float32": t.float32, "torch.bfloat16": t.bfloat16}[loaded["dtype"]]
        mb.metadata = loaded.get("metadata", {})
        return mb

def train_steer_multi_bias(
    model: HookedSAETransformer,
    cfg: MultiSteerTrainingCfg,
    dataset: Dataset | DatasetDict,
) -> dict[str, Tensor]:
    """Train multiple bias vectors to minimize loss on a dataset."""
    if cfg.save_every is not None:
        assert cfg.save_dir_name is not None, f"requested automatic saving, must provide the name of the folder to save the checkpoints to"
        os.makedirs(f"{STEER_BIAS_SAVE_DIR}/{cfg.save_dir_name}", exist_ok=True)

    model.reset_hooks()
    # model.reset_saes()
    t.set_grad_enabled(True)

    biases = MultiBias(cfg, model.cfg)
    biases.set_grad(True)

    for hook_name, hook_func in biases.make_hooks():
        model.add_hook(hook_name, hook_func)
    opt = t.optim.AdamW(biases.params(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)

    t.cuda.empty_cache()

    if cfg.use_wandb:
        wandb.init(
            project=cfg.project_name,
            config=cfg.asdict(),
            name=f"{len(biases.cfg.hook_names)}x{biases.cfg.hook_names[0]}"
        )

    n_batches = cfg.steps * cfg.grad_acc_steps
    n_examples = n_batches * cfg.batch_size
    if n_examples > len(dataset):
        n_batches = len(dataset) // (cfg.batch_size * cfg.grad_acc_steps)
        n_examples = n_batches * cfg.batch_size
        print(f"{yellow}Requested {cfg.steps:,} batches over {n_examples:,} examples but dataset only has {len(dataset):,}. Stopping at {n_batches} batches.{endc}")

    minibatches = make_pretokenized_minibatches(model, dataset, n_examples=n_examples, batch_size=cfg.batch_size)

    tr = tqdm(minibatches, ncols=180, desc=cyan, ascii=" >=")
    for i, (toks, completion_mask) in enumerate(tr):
        logits = model(toks, prepend_bos=False)
        losses = model.loss_fn(logits, toks, per_token=True)
        # print(losses.device, completion_mask.device)
        losses_masked = losses * completion_mask[:, :losses.shape[-1]]
        # print(123)
        completion_loss = losses_masked.sum() / completion_mask.count_nonzero()

        l1 = biases.sparsity()
        loss = (completion_loss + l1 * cfg.sparsity_factor) / cfg.grad_acc_steps
        loss.backward()

        logging_completion_loss = completion_loss.item()
        logging_l1 = l1.item()
        logging_loss = loss.item() * cfg.grad_acc_steps
        tr.set_description(f"{cyan}[{cfg.hook_names[0]}...{cfg.hook_names[-1]}] ntp loss = {logging_completion_loss:.4f}, l1 = {logging_l1:.2f} ({cfg.sparsity_factor*logging_l1:.3f}), total={logging_loss:.3f}{endc}")
        if cfg.use_wandb:
            wandb.log({"completion_loss": logging_completion_loss, "l1": logging_l1, "loss": logging_loss})

        if (i+1)%cfg.grad_acc_steps == 0:
            opt.step()
            opt.zero_grad()
            t.cuda.empty_cache()

        if cfg.save_every is not None and i%cfg.save_every == 0:
            biases.save_to_disk(save_name=str(i), save_dir=f"{STEER_BIAS_SAVE_DIR}/{cfg.save_dir_name}", quiet=True)

    if cfg.use_wandb:
        wandb.finish()

    model.reset_hooks()
    # model.reset_saes()
    t.set_grad_enabled(False)
    biases.set_grad(False)
    t.cuda.empty_cache()
    return biases

def proj_out(A: Tensor, B: Tensor, norm:bool = False) -> Tensor:
    """Returns A with B projected out."""
    assert A.ndim == 1 and B.ndim == 1, f"A and B should be 1 dimensional. Got {A.shape} and {B.shape}"
    projed_out = A - (B/B.norm()) * (A @ B)
    if norm:
        return projed_out / projed_out.norm()
    return projed_out


def make_sae_feat_steer_hook(
    sae: SAE,
    feat_idx: int,
    feat_act: float,
    normalize: bool = False
) -> tuple[str, functools.partial]:
    """Create a hook that steers using an SAE feature's decoder direction."""
    feat_dec = sae.W_dec[feat_idx].clone()
    if normalize:
        feat_dec /= feat_dec.norm()
    bias = feat_act * feat_dec
    return sae.cfg.metadata.hook_name, make_add_bias_hook(bias=bias)

def load_model_prefs() -> dict:
    """Glob per-model preference files and return {model_name: summary_dict}.

    Each file in ANIMAL_PREF_DATA_DIR has shape {"summary": {...}, "prompt": [...], "completion": [...]}.
    Files without a summary key (legacy completion-only files) are skipped.
    """
    result = {}
    if not os.path.isdir(ANIMAL_PREF_DATA_DIR):
        return result
    for fname in sorted(os.listdir(ANIMAL_PREF_DATA_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(ANIMAL_PREF_DATA_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        summary = data.get("summary")
        if not isinstance(summary, dict):
            continue
        result[fname[:-5]] = summary
    return result

def load_model_losses() -> dict:
    with open(MODEL_LOSSES_PATH) as f:
        losses = json.loads(f.read())
    return losses

def load_sae(save_name: str, dtype: str = "bfloat16") -> SAE:
    """Load an SAE from disk."""
    print(f"{gray}loading sae from '{save_name}'...{endc}")
    sae = SAE.load_from_disk(
        path = f"./saes/{save_name}",
        device="cuda",
        dtype=dtype,
    )
    sae.cfg.save_name = save_name
    if sae.cfg.metadata.hook_name is None:
        with open(f"./saes/{save_name}/cfg.json", "r") as f:
            cfg = json.load(f)
            sae.cfg.metadata.hook_name = cfg["metadata"]["hook_name"]
            sae.cfg.metadata.neuronpedia_id = cfg["metadata"]["neuronpedia_id"]
    sae.eval()
    sae.requires_grad_(False)
    return sae

def save_sae(sae: SAE, save_name: str):
    """Save an SAE to disk."""
    print(f"{gray}saving sae to '{save_name}'...{endc}")
    if sae.cfg.metadata.hook_name is None:
        assert False, "hook name is not set, will not save sae"
    sae.save_model(path = f"./saes/{save_name}")

def get_completion_loss_on_num_dataset(
    model: HookedSAETransformer | HookedTransformer | AutoModelForCausalLM,
    dataset: Dataset,
    n_examples: int = None,
    system_prompt: str|None = None,
    desc: str|None = None,
    batch_size: int = 32,
    leave_bar: bool = True,
) -> float:
    """Compute mean completion-only loss on a conversational dataset. Model-agnostic (works with Gemma, Qwen, Llama, etc.)."""
    tokenizer = model.tokenizer
    is_hooked = hasattr(model, 'cfg')
    device = model.cfg.device if is_hooked else next(model.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    loss_fn_hf = None if is_hooked else nn.CrossEntropyLoss(reduction='none')

    n_examples = len(dataset) if n_examples is None else min(n_examples, len(dataset))
    examples_losses = []

    iter = range(0, n_examples, batch_size) if desc is None else trange(0, n_examples, batch_size, ncols=140, ascii=" >=", desc=desc, leave=leave_bar)
    for batch_start in iter:
        batch_end = min(batch_start + batch_size, n_examples)
        batch = dataset[batch_start:batch_end]
        bs = batch_end - batch_start
        batch_prompts = batch["prompt"]
        if system_prompt is not None:
            batch_prompts = [add_system_prompt_to_messages(tokenizer, p, system_prompt) for p in batch_prompts]
        batch_messages = [batch_prompts[i] + batch["completion"][i] for i in range(bs)]
        toks = tokenizer.apply_chat_template(
            batch_messages, padding=True, tokenize=True,
            return_dict=False, return_tensors='pt', continue_final_generation=True
        ).to(device)
        _, completion_mask = get_assistant_mask(tokenizer, batch_messages, pad=True)

        if is_hooked:
            logits = model(toks, prepend_bos=False)
            losses = model.loss_fn(logits, toks, per_token=True)  # (batch, seq-1)
        else:
            logits = model(toks).logits
            losses = loss_fn_hf(logits[:, :-1, :].reshape(-1, logits.shape[-1]), toks[:, 1:].reshape(-1)).reshape(bs, -1)

        cmask = completion_mask[:, :losses.shape[-1]]
        per_example = (losses * cmask).sum(dim=-1) / cmask.sum(dim=-1).clamp(min=1)
        examples_losses.extend(per_example.float().tolist())

    mean_loss = sum(examples_losses) / len(examples_losses)
    t.cuda.empty_cache()
    return mean_loss

# ==== Activation Store Utilities ====

def get_act_store_key(
    model: HookedSAETransformer,
    sae: SAE|None,
    dataset: Dataset,
    act_name: str,
    seq_pos_strategy: str | int | list[int],
    act_modifier: str|None = None,
) -> str:
    dataset_checksum = next(iter(dataset._info.download_checksums))
    track_sae = "sae" in act_name
    assert not (track_sae and sae is None), f"{red}Requested activation is from SAE but SAE not provided.{endc}"
    if not track_sae: sae = None
    act_mod_prepend = f"<<{act_modifier}>>" if act_modifier is not None else ""
    return f"<<{model.cfg.model_name}>>{(f'<<{sae.cfg.save_name}>>') if sae is not None else ''}<<{dataset_checksum}>><<{act_name}>><<{seq_pos_strategy}>>" + act_mod_prepend

def update_act_store(
    store: dict,
    model: HookedSAETransformer,
    sae: SAE|None,
    dataset: Dataset,
    acts: dict[str, Tensor],
    seq_pos_strategy: str | int | list[int],
    act_modifier: str|None = None,
) -> None:
    for act_name, act in acts.items():
        act_store_key = get_act_store_key(model, sae, dataset, act_name, seq_pos_strategy, act_modifier=act_modifier)
        store[act_store_key] = act
    t.save(store, get_act_store_path(model.cfg.model_name))

def load_from_act_store(
    model: HookedSAETransformer|str,
    dataset: Dataset,
    act_names: list[str]|str,
    seq_pos_strategy: str | int | list[int],
    sae: SAE|None = None,
    force_recalculate: bool = False,
    n_examples: int = None,
    quiet: bool = False,
    act_modifier: str|None = None,
) -> dict[str, Tensor]|Tensor:
    """Load activations from store or calculate if missing"""
    if isinstance(model, str):
        model_name = model
        model = None
    else:
        model_name = model.cfg.model_name
    if isinstance(act_names, str):
        act_names = [act_names]
    else:
        assert isinstance(act_names, list), f"act_names should be a str or list[str], found {type(act_names)}"

    if not quiet:
        dataset_name = dataset._info.dataset_name
        print(f"""{gray}loading activations:
            model: '{model_name}'
            sae: '{sae.cfg.save_name if sae is not None else 'None'}'
            act_names: {act_names}
            dataset: '{dataset_name}'
            seq pos strategy: '{seq_pos_strategy}'""" + (f"\n\t    modifier: {act_modifier}" if act_modifier is not None else "") + endc
        )
    store = load_act_store(model_name)
    act_store_keys = {act_name: get_act_store_key(model, sae, dataset, act_name, seq_pos_strategy, act_modifier) for act_name in act_names}

    if force_recalculate:
        missing_acts = act_store_keys
    else:
        missing_acts = {act_name: act_store_key for act_name, act_store_key in act_store_keys.items() if act_store_key not in store}

    missing_act_names = list(missing_acts.keys())
    if not quiet and len(missing_acts) > 0:
        print(f"""{yellow}{'missing requested activations in store' if not force_recalculate else 'requested recalculations'}:
            model: '{model_name}'
            sae: '{sae.cfg.save_name if sae is not None else 'None'}'
            act_names: {missing_act_names}
            dataset: '{dataset_name}'
            seq pos strategy: '{seq_pos_strategy}'
        calculating...{endc}""")
    if len(missing_acts) > 0:
        assert not (model is None), f"{red}Model object was not provided. cannot calculate activations.{endc}"
        assert act_modifier is None, f"{red}activations have requested modifier: '{orange}{act_modifier}{red}' but was not found in store. Will not attempt to calculate.{endc}"
        new_acts = get_dataset_mean_activations(
            model,
            dataset,
            act_names,
            sae=sae,
            seq_pos_strategy=seq_pos_strategy,
            n_examples=n_examples,
        )
        update_act_store(store, model, sae, dataset, new_acts, seq_pos_strategy)

    if len(act_names) > 1:
        return {act_name: store[act_store_key] for act_name, act_store_key in act_store_keys.items()}
    else:
        return store[next(iter(act_store_keys.values()))]

def load_act_store(model_name: str) -> dict:
    path = get_act_store_path(model_name)
    if os.path.exists(path):
        return t.load(path)
    return {}

@t.inference_mode()
def get_dataset_mean_activations(
        model: HookedSAETransformer,
        dataset: Dataset,
        act_names: list[str],
        seq_pos_strategy: str | int | list[int] = "all_toks",
        sae: SAE|None = None,
        n_examples: int = None,
        index_from_completion_start: bool = False,
    ) -> dict[str, Tensor]:
    """Compute mean activations over a dataset.

    Args:
        index_from_completion_start: If True (only valid for conversational datasets),
            indices are relative to the assistant's completion start. Index 0 is the
            <start_of_turn> token. If False, indices are absolute (0 = BOS).
    """
    is_conversational = "completion" in dataset.features
    is_pretraining = "text" in dataset.features

    if not (is_conversational or is_pretraining):
        raise ValueError(f"Dataset features unrecognized: {dataset.features}")
    if index_from_completion_start and not is_conversational:
        raise ValueError("index_from_completion_start=True is only valid for conversational datasets")

    dataset_len = len(dataset)
    n_examples = dataset_len if n_examples is None else n_examples
    num_iter = min(n_examples, dataset_len)

    mean_acts = {}
    act_names_without_logits = [act_name for act_name in act_names if "logits" not in act_name]

    if "logits" in act_names:
        mean_acts["logits"] = t.zeros((model.W_E.shape[0]), dtype=t.float32, device=model.W_E.device)

    sae_acts_requested = any(["sae" in act_name for act_name in act_names])
    assert not (sae_acts_requested and sae is None), f"{red}Requested SAE activations but SAE not provided.{endc}"

    if index_from_completion_start:
        start_of_turn_id = model.tokenizer.vocab["<start_of_turn>"]

    for i in trange(num_iter, ncols=130):
        ex = dataset[i]

        if is_conversational:
            messages = ex["prompt"] + ex["completion"]
            toks = model.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                continue_final_message=True,
            ).squeeze()
        else:
            toks = model.tokenizer.encode(
                ex["text"],
                return_tensors="pt",
                truncation=True,
                max_length=model.cfg.n_ctx
            ).squeeze()

        seq_len = toks.shape[-1]

        if sae_acts_requested:
            logits, cache = model.run_with_cache_with_saes(
                toks,
                saes=[sae],
                names_filter=act_names_without_logits,
                prepend_bos=False,
                use_error_term=False
            )
        else:
            logits, cache = model.run_with_cache(
                toks,
                prepend_bos=False,
                names_filter=act_names_without_logits,
            )

        if index_from_completion_start:
            base_idx = t.where(toks == start_of_turn_id)[0][-1].item()
        else:
            base_idx = 0

        if seq_pos_strategy == "all_toks":
            indices = t.arange(base_idx, seq_len)
        elif isinstance(seq_pos_strategy, int):
            indices = t.tensor([base_idx + seq_pos_strategy])
        elif isinstance(seq_pos_strategy, list):
            indices = t.tensor(seq_pos_strategy) + base_idx
        else:
            raise ValueError(f"Invalid seq_pos_strategy: {seq_pos_strategy}")

        for act_name in act_names_without_logits:
            if act_name in cache:
                cache_act = cache[act_name]
            elif act_name+".hook_sae_input" in cache:
                cache_act = cache[act_name+".hook_sae_input"]
            else:
                raise KeyError(f"neither '{act_name}' nor {act_name+'.hook_sae_input'} exist in cache.\ncache contains keys: {cache.keys()}")
            cache_act_mean = cache_act[:, indices, :].mean(dim=1).squeeze().to(t.float32)
            if act_name not in mean_acts:
                mean_acts[act_name] = cache_act_mean
            else:
                mean_acts[act_name] += cache_act_mean
        if "logits" in act_names:
            mean_acts["logits"] += logits[:, indices, :].mean(dim=1).squeeze().to(t.float32)

    for act_name, act_mean in mean_acts.items():
        mean_acts[act_name] = act_mean / num_iter

    t.cuda.empty_cache()
    return mean_acts

@dataclass
class PromptGenerator:
    example_min_count: int
    example_max_count: int
    example_min_value: int
    example_max_value: int
    answer_count: int
    answer_max_digits: int

    _example_numbers_templates = ["I give you this sequence of numbers: {examples}.", "I have this number sequence: {examples}.", "Start with these numbers: {examples}.", "Look at these numbers: {examples}.", "See the following numbers: {examples}.", "Observe this number sequence: {examples}.", "Check out this number list: {examples}.", "Take these numbers: {examples}.", "Here's a list of numbers: {examples}.", "Consider this sequence: {examples}.", "Examine these numbers: {examples}.", "Analyze this sequence: {examples}.", "These numbers follow a sequence: {examples}.", "Here is a numeric sequence: {examples}.", "The sequence starts with: {examples}.", "Let's start with this sequence: {examples}.", "We have this series of numbers: {examples}.", "This numerical series is: {examples}.", "These are the first numbers in a sequence: {examples}.", "Here are some numbers in sequence: {examples}.", "The numerical progression is: {examples}.", "This series shows: {examples}.", "Given these initial values: {examples}.", "The sequence begins as follows: {examples}.", "Here's the start of a number sequence: {examples}."]
    _count_qualifiers = ["at most", "up to", "no more than", "at maximum", "maximum", "a maximum of", "not exceeding", "not more than", "no greater than"]
    _digit_descriptors = ["up to {max_digits} digits each", "maximum {max_digits} digits each", "{max_digits} digits or fewer each", "maximum {max_digits} digits each", "{max_digits} digits max each", "no more than {max_digits} digits each", "at most {max_digits} digits each", "with {max_digits} digits or less each", "{max_digits}-digit numbers or smaller"]
    _generate_numbers_instruction_templates = ["Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.", "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.", "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.", "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.", "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.", "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).", "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).", "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})", "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence", "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence"]
    _format_suffixes = ["Return a comma-separated list of numbers.", "Return a space-separated list of numbers.", "Return a semicolon-separated list of numbers.", "Return one number per line.", "Return the numbers in the following format: [number_1, number_2, ...].", "Return the numbers in the following format: (number_1, number_2, ...).", "Return the numbers in the following format: number_1, number_2, ...", "Provide the numbers separated by commas.", "List the numbers with spaces between them.", "Write the numbers on separate lines.", "Format the output as: number1, number2, number3, ...", "Return numbers separated by semicolons.", "Format as a simple comma-delimited sequence.", "Present as space-delimited values.", "List each number on its own line with no other text."]
    _suffixes = ["Say only the numbers - nothing more.", "Please just say the numbers, nothing more.", "Give me just the numbers, no explanation needed.", "Return the numbers exactly as requested, nothing else.", "Simply provide the numbers in the specified format.", "Respond with only the numbers, no additional text.", "No explanation, just the numbers.", "Just the numbers, please.", "Provide only the numerical values.", "Output nothing but the numbers.", "No commentary, just numbers.", "Skip any explanation and give only numbers.", "Nothing but numbers in your response.", "Only the numerical sequence, nothing else.", "Just show me the numbers.", "Answer with numbers alone.", "Reply with only numerical values.", "No words, just numbers.", "Don't add any text - numbers only."]

    def sample_example_prefix(self) -> str:
        example_count = random.randint(self.example_min_count, self.example_max_count)
        examples = [
            str(random.randint(self.example_min_value, self.example_max_value))
            for _ in range(example_count)
        ]
        examples_str = ", ".join(examples)
        example_template = random.choice(self._example_numbers_templates)
        return example_template.format(examples=examples_str)

    def sample_query(self) -> str:
        example_part = self.sample_example_prefix()
        count_qualifier = random.choice(self._count_qualifiers)
        digit_descriptor_template = random.choice(self._digit_descriptors)
        instruction_template = random.choice(self._generate_numbers_instruction_templates)
        format_suffix = random.choice(self._format_suffixes)
        suffix = random.choice(self._suffixes)

        # Format digit descriptor with max_digits
        digit_descriptor = digit_descriptor_template.format(
            max_digits=self.answer_max_digits
        )

        # Build the full query
        instruction_part = instruction_template.format(
            count_qualifier=count_qualifier,
            answer_count=self.answer_count,
            digit_descriptor=digit_descriptor,
        )

        return f"{example_part} {instruction_part} {format_suffix} {suffix}"

def pearson(a: Tensor, b: Tensor) -> float:
    assert a.shape == b.shape, f"a and b should have same shape, but got a={a.shape} and b={b.shape}"
    a_c = a - a.mean()
    b_c = b - b.mean()
    r = (a_c * b_c).sum() / t.sqrt((a_c ** 2).sum() * (b_c ** 2).sum())
    return r.item()

def push_dataset_card_readme(dataset_name: str, text: str):
    readme_format = """
    ---
    language: en
    license: mit
    ---
    {text}
    """
    readme_text = readme_format.format(text=text)
    RepoCard(readme_text).push_to_hub(f"{HF_USERNAME}/{dataset_name.split('/')[-1]}", repo_type="dataset")

def get_dataset_config_from_hub(dataset_name: str) -> dict:
    """Fetch the dataset config from the model card on the Hugging Face hub.
    
    Args:
        dataset_name: Name of the dataset on the hub (e.g., "username/dataset-name" or "dataset-name")
    
    Returns:
        Dictionary containing the dataset configuration
    
    Raises:
        ValueError: If the model card cannot be parsed or does not contain valid JSON config
    """
    # Ensure dataset name includes the namespace
    if "/" not in dataset_name:
        dataset_name = f"{HF_USERNAME}/{dataset_name}"
    
    try:
        # Load the model card from the hub
        card = RepoCard.load(dataset_name, repo_type="dataset")
        card_text = card.text
        
        # The card format is YAML front matter followed by JSON config
        # Split by the YAML delimiter to extract content after front matter
        if "---" in card_text:
            parts = card_text.split("---")
            if len(parts) >= 3:
                # Content after the second "---" is the JSON config
                json_content = "---".join(parts[2:]).strip()
            else:
                json_content = card_text.strip()
        else:
            json_content = card_text.strip()
        
        # Parse the JSON config
        config = json.loads(json_content)
        return config
        
    except Exception as e:
        raise ValueError(f"Failed to load or parse dataset config from '{dataset_name}': {e}")

def topk_toks_table(logits: t.Tensor, tokenizer: AutoTokenizer, k: int = 25, show_negative: bool = False, title: str | None = None):
    logits = logits.flatten()
    top = logits.topk(k)
    top_strs = [tokenizer.decode([tok]) for tok in top.indices.tolist()]
    top_vals = top.values.tolist()
    if show_negative:
        bot = logits.topk(k, largest=False)
        bot_strs = [tokenizer.decode([tok]) for tok in bot.indices.tolist()]
        bot_vals = bot.values.tolist()
        data = [(i, repr(top_strs[i]), top_vals[i], repr(bot_strs[i]), bot_vals[i]) for i in range(k)]
        table_str = tabulate(data, headers=["Idx", "Top Tok", "Top Value", "Bot Tok", "Bot Value"], tablefmt="rounded_outline")
    else:
        data = [(i, repr(top_strs[i]), top_vals[i]) for i in range(k)]
        table_str = tabulate(data, headers=["Idx", "Tok", "Value"], tablefmt="rounded_outline")
    if title is not None:
        lines = table_str.splitlines()
        inner = len(lines[0]) - 2
        print(f"╭{'─' * inner}╮")
        print(f"│{bold}{title.center(inner)}{endc}│")
        print(f"├{'─' * inner}┤")
        print("\n".join(lines[1:]))
    else:
        print(table_str)
    if show_negative:
        return (top_strs, top_vals, bot_strs, bot_vals)
    return (top_strs, top_vals)

# yaxis_range = [lower, upper]
def line(y, renderer=None, **kwargs):
    '''
    Edit to this helper function, allowing it to take args in update_layout (e.g. yaxis_range).
    '''
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size
    return_fig = kwargs_pre.pop("return_fig", False)
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if "xaxis_tickvals" in kwargs_pre:
        tickvals = kwargs_pre.pop("xaxis_tickvals")
        kwargs_post["xaxis"] = dict(
            tickmode = "array",
            tickvals = kwargs_pre.get("x", np.arange(len(tickvals))),
            ticktext = tickvals
        )
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "closest"
    if "use_secondary_yaxis" in kwargs_pre and kwargs_pre["use_secondary_yaxis"]:
        del kwargs_pre["use_secondary_yaxis"]
        if "labels" in kwargs_pre:
            labels: dict = kwargs_pre.pop("labels")
            kwargs_post["yaxis_title_text"] = labels.get("y1", None)
            kwargs_post["yaxis2_title_text"] = labels.get("y2", None)
            kwargs_post["xaxis_title_text"] = labels.get("x", None)
        for k in ["title", "template", "width", "height"]:
            if k in kwargs_pre:
                kwargs_post[k] = kwargs_pre.pop(k)
        fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(**kwargs_post)
        y0 = to_numpy(y[0])
        y1 = to_numpy(y[1])
        x0, x1 = kwargs_pre.pop("x", [np.arange(len(y0)), np.arange(len(y1))])
        name0, name1 = kwargs_pre.pop("names", ["yaxis1", "yaxis2"])
        fig.add_trace(go.Scatter(y=y0, x=x0, name=name0), secondary_y=False)
        fig.add_trace(go.Scatter(y=y1, x=x1, name=name1), secondary_y=True)
    else:
        y = list(map(to_numpy, y)) if isinstance(y, list) and not (isinstance(y[0], int) or isinstance(y[0], float)) else to_numpy(y)
        names = kwargs_pre.pop("names", None)
        hover_text = kwargs_pre.pop("hover_text", None)
        fig = px.line(y=y, **kwargs_pre).update_layout(**kwargs_post)
        if names is not None:
            fig.for_each_trace(lambda trace: trace.update(name=names.pop(0)))
        if hover_text is not None:
            # Update the hover template to show custom text
            fig.for_each_trace(lambda trace: trace.update(
                hovertemplate='<b>Token:</b> %{customdata}<br>' +
                              '<b>Value:</b> %{y}<br>' +
                              '<b>Index:</b> %{x}<extra></extra>',
                customdata=hover_text
            ))
    return fig if return_fig else fig.show(renderer=renderer)

update_layout_set = {"xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "coloraxis_showscale", "xaxis_tickangle", "yaxis_scaleanchor", "xaxis_tickfont", "yaxis_tickfont"}

update_traces_set = {"textposition"}

def to_numpy(tensor):
    """
    Helper function to convert a tensor to a numpy array. Also works on lists, tuples, and numpy arrays.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, (list, tuple)):
        array = np.array(tensor)
        return array
    elif isinstance(tensor, (t.Tensor, t.nn.parameter.Parameter)):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, (int, float, bool, str)):
        return np.array(tensor)
    else:
        raise ValueError(f"Input to to_numpy has invalid type: {type(tensor)}")

def reorder_list_in_plotly_way(L: list, col_wrap: int):
    '''
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    '''
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]
    return L_new

def imshow(tensor: t.Tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if ("size" in kwargs_pre) or ("shape" in kwargs_pre):
        size = kwargs_pre.pop("size", None) or kwargs_pre.pop("shape", None)
        kwargs_pre["height"], kwargs_pre["width"] = size
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    return_fig = kwargs_pre.pop("return_fig", False)
    text = kwargs_pre.pop("text", None)
    xaxis_tickangle = kwargs_post.pop("xaxis_tickangle", None)
    # xaxis_tickfont = kwargs_post.pop("xaxis_tickangle", None)
    static = kwargs_pre.pop("static", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        # Weird thing where facet col wrap means labels are in wrong order
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(facet_labels, kwargs_pre["facet_col_wrap"])
        for i, label in enumerate(facet_labels):
            print(fig.layout.annotations)
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    if text:
        if tensor.ndim == 2:
            # if 2D, then we assume text is a list of lists of strings
            assert isinstance(text[0], list)
            assert isinstance(text[0][0], str)
            text = [text]
        else:
            # if 3D, then text is either repeated for each facet, or different
            assert isinstance(text[0], list)
            if isinstance(text[0][0], str):
                text = [text for _ in range(len(fig.data))]
        for i, _text in enumerate(text):
            fig.data[i].update(
                text=_text, 
                texttemplate="%{text}", 
                textfont={"size": 12}
            )
    # Very hacky way of fixing the fact that updating layout with xaxis_* only applies to first facet by default
    if xaxis_tickangle is not None:
        n_facets = 1 if tensor.ndim == 2 else tensor.shape[0]
        for i in range(1, 1+n_facets):
            xaxis_name = "xaxis" if i == 1 else f"xaxis{i}"
            fig.layout[xaxis_name]["tickangle"] = xaxis_tickangle
    return fig if return_fig else fig.show(renderer=renderer, config={"staticPlot": static})

def compute_preference(completions: dict, target: str) -> float:
    """Compute fraction of completions containing target (word-boundary matched)."""
    comp_list = completions.get("completion", []) or []
    contained = sum(1 for c in comp_list if isinstance(c, str) and animal_in_text(target, c))
    return (contained / len(comp_list)) if comp_list else 0.0

def update_model_prefs(
    model_name: str,
    pref_dict: dict,
    *,
    parent_model_id: str,
    animals_key: str | None = None,
    union_total: float | None = None,
    metadata: dict = None,
    prompts: list[str] | None = None,
    completions: list[str] | None = None,
) -> None:
    """Write a per-model preference eval file with summary + prompt + completion.

    Path: ./data/eval_data/animal_preferences/{model_name}.json
    Schema: {"summary": {parent, prefs, [is_base], [totals], [metadata]}, "prompt": [...], "completion": [...]}

    For base models (where model_name matches parent_model_id), stores full HF path with is_base=True.
    For child models, stores just the parent key reference to avoid redundancy.
    The totals dict is merged with any prior file so multiple animal-set evals coexist.
    """

    simple_model_name = model_name.split("/")[-1] if isinstance(model_name, str) else "unknown-model"
    parent_model = parent_model_id if isinstance(parent_model_id, str) else "unknown-model"
    parent_key = parent_model.split("/")[-1]

    os.makedirs(ANIMAL_PREF_DATA_DIR, exist_ok=True)
    out_path = os.path.join(ANIMAL_PREF_DATA_DIR, f"{simple_model_name}.json")

    prior_summary = {}
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            prior = json.load(f)
        if isinstance(prior, dict) and isinstance(prior.get("summary"), dict):
            prior_summary = prior["summary"]

    totals = {}
    if isinstance(prior_summary.get("totals"), dict):
        totals.update(prior_summary["totals"])
    if animals_key is not None and union_total is not None:
        sorted_key = ",".join(sorted(animals_key.split(",")))
        totals[sorted_key] = float(union_total)

    is_base_model = (simple_model_name == parent_key)
    summary = {
        "parent": parent_model if is_base_model else parent_key,
        "prefs": pref_dict,
    }
    if is_base_model:
        summary["is_base"] = True
    if len(totals) > 0:
        summary["totals"] = totals
    if metadata is not None:
        summary["metadata"] = metadata

    out = {"summary": summary}
    if prompts is not None:
        out["prompt"] = prompts
    if completions is not None:
        out["completion"] = completions

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

def update_model_losses(model_name: str, loss_dict: dict[str, float], *, parent_model_id: str, metadata: dict = None) -> None:
    """Update a JSON log of loss values keyed by the provided model name. Merges with existing losses."""
    simple_model_name = model_name.split("/")[-1] if isinstance(model_name, str) else "unknown-model"
    parent_model = parent_model_id if isinstance(parent_model_id, str) else "unknown-model"
    parent_key = parent_model.split("/")[-1]

    data_dir = os.path.join(os.path.dirname(__file__), "data", "eval_data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "model_losses.json")

    try:
        with open(out_path, "r") as f:
            existing = json.load(f)
            if not isinstance(existing, dict): existing = {}
    except (FileNotFoundError, json.JSONDecodeError):
        existing = {}

    is_base_model = (simple_model_name == parent_key)

    # Merge with existing losses so incremental runs accumulate
    prior = existing.get(simple_model_name)
    prior_losses = prior["losses"] if isinstance(prior, dict) and isinstance(prior.get("losses"), dict) else {}
    merged_losses = {**prior_losses, **loss_dict}

    new_entry = {"parent": parent_model if is_base_model else parent_key, "losses": merged_losses}
    if is_base_model:
        new_entry["is_base"] = True
    if metadata is not None:
        new_entry["metadata"] = metadata
    existing[simple_model_name] = new_entry

    with open(out_path, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)

def display_model_prefs_table(parent_model_id: str, animals: list[str], include_substrings: list[str] | None = None, exclude_substrings: list[str] | None = None) -> None:
    """Display a table of preferences and deltas for a parent and its derivatives.

    Reads per-model files in ./data/eval_data/animal_preferences/ (each with a 'summary' key) and
    prints a table with one row per model (filtered to the given parent) and
    one column per animal from the provided `animals` list. Each cell shows
    "value (±delta)" where delta is the difference to the parent model's value
    for that animal. Includes per-model totals.
    
    Args:
        parent_model_id: Parent model identifier to filter by
        animals: List of animals to show in columns
        include_substrings: Optional list of substrings; each must be in the model name (case insensitive)
        exclude_substrings: Optional list of substrings; none must be in the model name (case insensitive)
    """
    all_prefs_raw = load_model_prefs()
    all_prefs: dict[str, dict] = {}
    for model_name, val in all_prefs_raw.items():
        if isinstance(val, dict) and "prefs" in val and "parent" in val:
            all_prefs[model_name] = val

    parent_model_key = parent_model_id.split("/")[-1]
    # Filter to parent and its derivatives
    if parent_model_key not in all_prefs:
        print(f"{yellow}Parent model '{parent_model_key}' not found; nothing to display{endc}")
        return

    base_prefs = all_prefs[parent_model_key]["prefs"]
    columns = sorted(animals)

    # Sort models: base model first (if present), then alphabetical
    # Match children by parent key reference (new format) or full path (legacy)
    model_names = [m for m, rec in all_prefs.items() 
                   if rec.get("parent") == parent_model_id  # legacy: full path
                   or rec.get("parent") == parent_model_key  # new: key reference
                   or m == parent_model_key]
    
    # Apply substring filters (case insensitive)
    include_lower = [s.lower() for s in include_substrings] if include_substrings else []
    exclude_lower = [s.lower() for s in exclude_substrings] if exclude_substrings else []
    if include_lower:
        model_names = [m for m in model_names if m == parent_model_key or all(substr in m.lower() for substr in include_lower)]
    if exclude_lower:
        model_names = [m for m in model_names if m == parent_model_key or not any(substr in m.lower() for substr in exclude_lower)]

    model_names = sorted(set(model_names))
    if parent_model_key in model_names:
        model_names.remove(parent_model_key)
        model_names.insert(0, parent_model_key)

    headers = ["Model"] + columns + ["Total"]
    rows = []

    # Get base model's total for delta comparison
    all_animals_key = ",".join(sorted(ALL_ANIMALS))
    base_record = all_prefs.get(parent_model_key, {})
    base_totals = base_record.get("totals", {}) if isinstance(base_record.get("totals"), dict) else {}
    base_total = base_totals.get(all_animals_key, 0.0)

    for model_name in model_names:
        record = all_prefs.get(model_name, {}) or {}
        prefs = record.get("prefs", {})
        # Shorten display name by removing parent substring for derivatives
        if model_name != parent_model_key and isinstance(model_name, str):
            display_name = model_name.replace(parent_model_key, "").strip("-_/ ") or model_name
        else:
            display_name = model_name
        row = [display_name]
        is_parent = (model_name == parent_model_key)
        for animal in columns:
            val = prefs.get(animal)
            base_val = base_prefs.get(animal)
            if val is None or base_val is None:
                cell = f"{gray}—{endc}"
            else:
                fval = float(val)
                if is_parent:
                    # Parent model: show only value, no delta
                    cell = f"{val:0.3f}"
                else:
                    # Derivative model: show value and delta
                    delta = val - base_val
                    delta_str = f"{delta:+.3f}"
                    if delta > 1e-12:
                        delta_str = f"{green}{delta_str}{endc}"
                    elif delta < -1e-12:
                        delta_str = f"{red}{delta_str}{endc}"
                    else:
                        delta_str = f"{gray}{delta_str}{endc}"
                    cell = f"{val:0.3f} ({delta_str})"
            # Underline cell if animal name appears in model name (case-insensitive)
            if animal.lower() in model_name.lower():
                cell = f"{underline}{cell}{endc}"
            row.append(cell)
        # Per-model total column (use stored union coverage for ALL_ANIMALS, or 0 if not available)
        record_totals = record.get("totals", {}) if isinstance(record.get("totals"), dict) else {}
        union_total = record_totals.get(all_animals_key, 0.0)
        if is_parent:
            total_cell = f"{float(union_total):0.3f}"
        else:
            total_delta = union_total - base_total
            total_delta_str = f"{total_delta:+.3f}"
            if total_delta > 1e-12:
                total_delta_str = f"{green}{total_delta_str}{endc}"
            elif total_delta < -1e-12:
                total_delta_str = f"{red}{total_delta_str}{endc}"
            else:
                total_delta_str = f"{gray}{total_delta_str}{endc}"
            total_cell = f"{float(union_total):0.3f} ({total_delta_str})"
        row.append(total_cell)
        rows.append(row)

    table = tabulate(rows, headers=headers, tablefmt="fancy_grid", disable_numparse=True)
    print(table)

# ==== Loss Evaluation Table System ====

@dataclass
class LossEvalCfg:
    model_id: str
    parent_model_id: str
    model_type: Literal["hf", "hooked"] = "hooked"
    n_examples: int = 1600
    batch_size: int = 16
    model_save_name: str | None = None
    n_devices: int = 1
    hook_fn: functools.partial | None = None
    hook_point: str | None = None

    def asdict(self):
        result = dataclasses.asdict(self)
        if result.get("hook_fn") is not None:
            hook_fn = result["hook_fn"]
            result["hook_fn"] = hook_fn.func.__name__ if isinstance(hook_fn, functools.partial) else hook_fn.__name__ if callable(hook_fn) else str(hook_fn)
        return result

@t.inference_mode()
def get_loss_evals(cfg: LossEvalCfg) -> dict[str, float]:
    """Load a model and compute completion loss on each TABLE_ANIMALS dataset for the parent model."""
    from get_preference import load_model_for_pref_eval
    parent_name = cfg.parent_model_id.split("/")[-1]
    model_save_name = cfg.model_save_name or cfg.model_id.split("/")[-1]

    model = load_model_for_pref_eval(cfg.model_id, tokenizer_id=cfg.parent_model_id, model_type=cfg.model_type, n_devices=cfg.n_devices, parent_model_id=cfg.parent_model_id)
    if cfg.hook_fn is not None and cfg.hook_point is not None:
        model.add_hook(cfg.hook_point, cfg.hook_fn)

    losses = {}
    for animal in TABLE_ANIMALS:
        dataset_name = f"{HF_USERNAME}/{parent_name}-{animal}-numbers"
        short_name = f"{parent_name}-{animal}-numbers"
        try:
            dataset = load_dataset(dataset_name, split="train")
            loss = get_completion_loss_on_num_dataset(model, dataset, n_examples=cfg.n_examples, batch_size=cfg.batch_size, desc=f"loss on {short_name}")
            losses[short_name] = round(loss, 4)
        except Exception as e:
            print(f"{yellow}Skipping {dataset_name}: {e}{endc}")

    update_model_losses(model_save_name, losses, parent_model_id=cfg.parent_model_id, metadata=cfg.asdict())
    del model
    t.cuda.empty_cache()
    return losses

def display_model_losses_table(parent_model_id: str, include_substrings: list[str] | None = None, exclude_substrings: list[str] | None = None, subtract_row_mean: bool = True) -> None:
    """Display a table of completion losses and deltas for a parent model and its derivatives.

    Reads model_losses.json. Rows are models, columns are datasets (one per TABLE_ANIMALS).
    Cells show loss (delta) where delta = model_loss - base_loss. Green = lower loss (better fit), red = higher.
    When subtract_row_mean=True (default), displayed values are loss - row_mean, with a RowMean column appended.
    """
    out_path = os.path.join(os.path.dirname(__file__), "data", "eval_data", "model_losses.json")
    try:
        with open(out_path, "r") as f:
            all_losses_raw = json.load(f)
            if not isinstance(all_losses_raw, dict):
                print(f"{red}model_losses.json is not a dict; nothing to display{endc}")
                return
    except FileNotFoundError:
        print(f"{red}No model_losses.json found at {out_path}{endc}")
        return
    except json.JSONDecodeError:
        print(f"{red}Failed to decode JSON at {out_path}{endc}")
        return

    all_losses: dict[str, dict] = {}
    for model_name, val in all_losses_raw.items():
        if isinstance(val, dict) and "losses" in val and "parent" in val:
            all_losses[model_name] = val

    parent_model_key = parent_model_id.split("/")[-1]
    if parent_model_key not in all_losses:
        print(f"{yellow}Parent model '{parent_model_key}' not found in model_losses.json; nothing to display{endc}")
        return

    base_losses = all_losses[parent_model_key]["losses"]

    # Columns: datasets from TABLE_ANIMALS for this parent
    columns = [f"{parent_model_key}-{animal}-numbers" for animal in sorted(TABLE_ANIMALS)]
    col_headers = [col.replace(parent_model_key + "-", "") for col in columns]

    # Filter models to parent and its derivatives
    model_names = [m for m, rec in all_losses.items()
                   if rec.get("parent") == parent_model_id
                   or rec.get("parent") == parent_model_key
                   or m == parent_model_key]

    include_lower = [s.lower() for s in include_substrings] if include_substrings else []
    exclude_lower = [s.lower() for s in exclude_substrings] if exclude_substrings else []
    if include_lower:
        model_names = [m for m in model_names if m == parent_model_key or all(substr in m.lower() for substr in include_lower)]
    if exclude_lower:
        model_names = [m for m in model_names if m == parent_model_key or not any(substr in m.lower() for substr in exclude_lower)]

    model_names = sorted(set(model_names))
    if parent_model_key in model_names:
        model_names.remove(parent_model_key)
        model_names.insert(0, parent_model_key)

    # Pre-compute row means and base row mean for subtract_row_mean mode
    row_means = {}  # model_name -> mean of available losses across columns
    for model_name in model_names:
        losses = all_losses.get(model_name, {}).get("losses", {})
        vals = [float(losses[c]) for c in columns if losses.get(c) is not None]
        row_means[model_name] = sum(vals) / len(vals) if vals else None
    base_row_mean = row_means.get(parent_model_key)

    headers = ["Model"] + col_headers + (["RowMean"] if subtract_row_mean else [])
    rows = []
    col_sums = {c: 0.0 for c in columns}
    col_counts = {c: 0 for c in columns}
    col_min_val = {c: float("inf") for c in columns}
    col_min_neg_delta = {c: 0.0 for c in columns}

    for model_name in model_names:
        record = all_losses.get(model_name, {})
        losses = record.get("losses", {})
        display_name = model_name.replace(parent_model_key, "").strip("-_/ ") or model_name if model_name != parent_model_key else model_name
        row = [display_name]
        is_parent = (model_name == parent_model_key)
        rm = row_means[model_name]
        base_rm = base_row_mean
        for col in columns:
            val = losses.get(col)
            base_val = base_losses.get(col)
            if val is None or base_val is None:
                cell = f"{gray}—{endc}"
            else:
                fval = float(val)
                display_val = (fval - rm) if (subtract_row_mean and rm is not None) else fval
                base_display = (float(base_val) - base_rm) if (subtract_row_mean and base_rm is not None) else float(base_val)
                if is_parent:
                    cell = f"{display_val:0.4f}"
                else:
                    delta = display_val - base_display
                    delta_str = f"{delta:+.4f}"
                    if delta < -1e-6:
                        delta_str = f"{green}{delta_str}{endc}"
                    elif delta > 1e-6:
                        delta_str = f"{red}{delta_str}{endc}"
                    else:
                        delta_str = f"{gray}{delta_str}{endc}"
                    cell = f"{display_val:0.4f} ({delta_str})"
                col_sums[col] += display_val
                col_counts[col] += 1
                if display_val < col_min_val[col]:
                    col_min_val[col] = display_val
                if not is_parent:
                    delta = display_val - base_display
                    if delta < col_min_neg_delta[col]:
                        col_min_neg_delta[col] = delta
            # Underline cell if dataset animal appears in model name
            animal_in_col = col.replace(parent_model_key + "-", "").replace("-numbers", "")
            if animal_in_col.lower() in model_name.lower():
                cell = f"{underline}{cell}{endc}"
            row.append(cell)
        # RowMean column
        if subtract_row_mean:
            if rm is not None:
                if is_parent:
                    row.append(f"{rm:0.4f}")
                else:
                    rm_delta = rm - base_rm if base_rm is not None else 0.0
                    rm_delta_str = f"{rm_delta:+.4f}"
                    if rm_delta < -1e-6: rm_delta_str = f"{green}{rm_delta_str}{endc}"
                    elif rm_delta > 1e-6: rm_delta_str = f"{red}{rm_delta_str}{endc}"
                    else: rm_delta_str = f"{gray}{rm_delta_str}{endc}"
                    row.append(f"{rm:0.4f} ({rm_delta_str})")
            else:
                row.append(f"{gray}—{endc}")
        rows.append(row)

    # Mean row
    mean_cells = []
    for col in columns:
        if col_counts[col] == 0:
            mean_cells.append(f"{gray}—{endc}")
        else:
            mean_val = col_sums[col] / col_counts[col]
            base_val = base_losses.get(col)
            if base_val is not None:
                base_display = (float(base_val) - base_row_mean) if (subtract_row_mean and base_row_mean is not None) else float(base_val)
                delta = mean_val - base_display
                delta_str = f"{delta:+.4f}"
                if delta < -1e-6: delta_str = f"{green}{delta_str}{endc}"
                elif delta > 1e-6: delta_str = f"{red}{delta_str}{endc}"
                else: delta_str = f"{gray}{delta_str}{endc}"
                mean_cells.append(f"{mean_val:0.4f} ({delta_str})")
            else:
                mean_cells.append(f"{mean_val:0.4f}")
    rows.append([f"{bold}Mean{endc}"] + mean_cells + ([f"{gray}—{endc}"] if subtract_row_mean else []))

    # Min value row
    min_cells = [f"{col_min_val[c]:0.4f}" if col_min_val[c] != float("inf") else f"{gray}—{endc}" for c in columns]
    rows.append([f"{bold}Min{endc}"] + min_cells + ([f"{gray}—{endc}"] if subtract_row_mean else []))

    # Min (most negative) delta row
    min_delta_cells = []
    for c in columns:
        d = col_min_neg_delta[c]
        d_str = f"{d:+.4f}"
        if d < -1e-6: d_str = f"{green}{d_str}{endc}"
        else: d_str = f"{gray}{d_str}{endc}"
        min_delta_cells.append(d_str)
    rows.append([f"{bold}Min Δ-{endc}"] + min_delta_cells + ([f"{gray}—{endc}"] if subtract_row_mean else []))

    table = tabulate(rows, headers=headers, tablefmt="fancy_grid", disable_numparse=True)
    print(table)

def show_losses_table(parent_model_id: str, include: list[str] | None = None, exclude: list[str] | None = None, subtract_row_mean: bool = True):
    """Convenience wrapper for display_model_losses_table."""
    display_model_losses_table(parent_model_id, include_substrings=include, exclude_substrings=exclude, subtract_row_mean=subtract_row_mean)

@t.inference_mode()
def quick_eval_animal_prefs(
    model: AutoModelForCausalLM|HookedTransformer,
    parent_model_id: str,
    samples_per_prompt: int = 64,
    max_new_tokens: int = 24,
    animals: list[str] = None,
    quiet: bool = False,
) -> dict:
    """
    Quick evaluation of animal preferences without updating saved state.
    
    Args:
        model: Model to evaluate (AutoModelForCausalLM or HookedTransformer)
        parent_model_id: Parent model identifier to compare against
        samples_per_prompt: Number of samples per prompt
        max_new_tokens: Maximum new tokens to generate
        animals: List of animals to test preferences for (defaults to TABLE_ANIMALS)
        quiet: Whether to print results to console (default: True)
    
    Returns:
        Dict with 'tested' and 'parent' preferences
    """
    # Import here to avoid circular dependency
    from get_preference import generate_preference_completions, ANIMAL_PREFERENCE_PROMPTS, ALL_ANIMALS, TABLE_ANIMALS
    
    if animals is None:
        animals = TABLE_ANIMALS
    
    if isinstance(model, (HookedTransformer, HookedSAETransformer)):
        model.loaded_from = "hooked"
    elif isinstance(model, AutoModelForCausalLM):
        model.loaded_from = "hf"
    
    # Generate completions from the model
    completions = generate_preference_completions(
        model,
        ANIMAL_PREFERENCE_PROMPTS,
        samples_per_prompt=samples_per_prompt,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        display=not quiet,
    )
    
    # Compute preferences for tested model
    tested_prefs = {animal: compute_preference(completions, animal) for animal in animals}
    
    # Compute union coverage over ALL_ANIMALS for tested model
    comp_list = completions.get("completion", []) or []
    covered = sum(1 for text in comp_list if any(animal_in_text(a, text) for a in ALL_ANIMALS))
    tested_valid = (covered / len(comp_list)) if comp_list else 0.0
    
    # Load parent model preferences from its per-model file
    parent_model_key = parent_model_id.split("/")[-1]
    parent_path = os.path.join(ANIMAL_PREF_DATA_DIR, f"{parent_model_key}.json")
    parent_prefs = {}
    parent_valid = 0.0

    if os.path.exists(parent_path):
        with open(parent_path, "r") as f:
            data = json.load(f)
        summary = data.get("summary", {}) if isinstance(data, dict) else {}
        parent_prefs = summary.get("prefs", {})
        totals = summary.get("totals", {})
        animals_key = ",".join(sorted(ALL_ANIMALS))
        parent_valid = totals.get(animals_key, 0.0)
    elif not quiet:
        print(f"{yellow}Warning: No parent preferences file at {parent_path}{endc}")
    
    if not quiet:
        # Find max animal name length for alignment (including %valid column)
        max_animal_len = max(len(a) for a in animals + ["%valid"])
        
        # Print header
        header_parts = [f"{animal:>{max_animal_len}}" for animal in animals]
        header_parts.append(f"{'%valid':>{max_animal_len}}")
        print(f"         {' '.join(header_parts)}")
        
        # Print parent row
        parent_parts = [f"{parent_prefs.get(animal, 0.0):>{max_animal_len}.4f}" for animal in animals]
        parent_parts.append(f"{parent_valid:>{max_animal_len}.4f}")
        print(f"{bold}Parent:{endc}  {' '.join(parent_parts)}")
        
        # Print tested model row
        tested_parts = []
        for animal in animals:
            tested_val = tested_prefs[animal]
            tested_parts.append(f"{tested_val:>{max_animal_len}.4f}")
        tested_parts.append(f"{tested_valid:>{max_animal_len}.4f}")
        
        print(f"{bold}Tested:{endc}  {' '.join(tested_parts)}")
        
        # Print deltas on a separate line
        delta_parts = []
        for animal in animals:
            tested_val = tested_prefs[animal]
            parent_val = parent_prefs.get(animal, 0.0)
            delta = tested_val - parent_val
            
            if delta > 0.001:
                delta_str = f"{green}+{delta:.3f}{endc}"
            elif delta < -0.001:
                delta_str = f"{red}{delta:.3f}{endc}"
            else:
                delta_str = f"{gray}±.000{endc}"
            
            # Pad to match column width (accounting for color codes)
            padding = max_animal_len - 6  # 6 chars for "±0.000"
            delta_parts.append(f"{' ' * padding}{delta_str}")
        
        # Add delta for %valid column
        valid_delta = tested_valid - parent_valid
        if valid_delta > 0.001:
            valid_delta_str = f"{green}+{valid_delta:.3f}{endc}"
        elif valid_delta < -0.001:
            valid_delta_str = f"{red}{valid_delta:.3f}{endc}"
        else:
            valid_delta_str = f"{gray}±.000{endc}"
        
        valid_padding = max_animal_len - 6
        delta_parts.append(f"{' ' * valid_padding}{valid_delta_str}")
        
        print(f"         {' '.join(delta_parts)}")
    
    return {
        "tested": tested_prefs,
        "parent": parent_prefs,
        "tested_valid": tested_valid,
        "parent_valid": parent_valid,
    }

def extract_plotly_data_from_html(html_path: str) -> t.Tensor:
    """Extract data from a saved Plotly HTML figure as a PyTorch tensor."""
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()
    
    match = re.search(r'Plotly\.newPlot\s*\(\s*"[^"]+",\s*', html)
    if not match:
        raise ValueError(f"Could not find Plotly.newPlot call in {html_path}")
    
    start, bracket_count, data_start, data_end, in_string, escape_next = match.end(), 0, 0, None, False, False
    
    for i in range(start, len(html)):
        c = html[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\':
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '[':
            if bracket_count == 0:
                data_start = i
            bracket_count += 1
        elif c == ']':
            bracket_count -= 1
            if bracket_count == 0:
                data_end = i + 1
                break
    
    if data_end is None:
        raise ValueError(f"Could not find complete data array in {html_path}")
    
    trace = json.loads(html[data_start:data_end])[0]
    
    if 'z' in trace:
        z = trace['z']
        if isinstance(z, dict) and 'bdata' in z:
            bdata = base64.b64decode(z['bdata'])
            dtype = {'f4': np.float32, 'f8': np.float64, 'i4': np.int32}.get(z.get('dtype', 'f4'), np.float32)
            arr = np.frombuffer(bdata, dtype=dtype).copy()
            if 'shape' in z:
                arr = arr.reshape(*map(int, z['shape'].split(',')))
            return t.from_numpy(arr)
        return t.tensor(z)
    
    if 'y' in trace:
        return t.tensor(trace['y'])
    
    raise ValueError("Could not extract data from plot")

def get_mean_self_sim(vecs: Tensor) -> float:
    self_dot_map = einsum(vecs, vecs, "batch1 d_vec, batch2 d_vec -> batch1 batch2")
    return ((self_dot_map.sum() - self_dot_map.trace()) / (self_dot_map.numel() - self_dot_map.shape[0])).item()

def load_lora_weights(checkpoint_dir: str) -> dict[str, Tensor]:
    """Load LoRA adapter weights from a checkpoint directory as a flat dict of tensors.
    Keys are cleaned to: 'layers.{N}.{module}.lora_{A|B}' (e.g. 'layers.0.self_attn.q_proj.lora_A').
    """
    sf_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    raw = load_safetensors(sf_path)
    prefix = "base_model.model.model."
    cleaned = {}
    for k, v in raw.items():
        name = k.removeprefix(prefix).removesuffix(".weight")
        cleaned[name] = v
    return cleaned

def load_ft_pref_change_map(model_name: str, model_type = "numbers-ft", return_parent=False):
    all_prefs = load_model_prefs()
    animals = sorted(TABLE_ANIMALS)

    parent_prefs = t.tensor([all_prefs[model_name]["prefs"][animal] for animal in animals]).unsqueeze(-1)

    pref_map = t.zeros(len(animals), len(animals), dtype=t.float32)
    for i, animal in enumerate(animals):
        ft_prefs = all_prefs[f"{model_name}-{animal}-{model_type}"]["prefs"]
        pref_map[:, i] = t.tensor([ft_prefs[a] for a in animals])

    if return_parent: return pref_map - parent_prefs, parent_prefs
    return pref_map - parent_prefs

def tec(): t.cuda.empty_cache()
