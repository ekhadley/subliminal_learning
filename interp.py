#%% imports
from utils import *

#%% loading the model and sae

# t.set_grad_enabled(False)
t.manual_seed(42)
np.random.seed(42)
random.seed(42)

# MODEL_ID = "google/gemma-2b-it"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

MODEL_NAME = MODEL_ID.split('/')[-1]
# model = HookedTransformer.from_pretrained_no_processing(
model = HookedTransformer.from_pretrained(
    model_name=MODEL_ID,
    device="cuda",
    dtype="bfloat16",
    n_devices=1
)
print(model.cfg)
tokenizer = model.tokenizer
model.eval()
model.requires_grad_(False)
t.cuda.empty_cache()

# SAE_RELEASE = "gemma-2b-it-res-jb"
# SAE_ID = "blocks.12.hook_resid_post"

# SAE_SAVE_NAME = f"{SAE_RELEASE}-{SAE_ID}".replace("/", "-")
# sae = load_sae(save_name=SAE_SAVE_NAME, dtype="float32")
# sae.requires_grad_(False)
# print(sae.cfg)

# SAE_HOOK_NAME = sae.cfg.metadata.hook_name
# SAE_IN_NAME = SAE_HOOK_NAME + ".hook_sae_input"
# ACTS_PRE_NAME = SAE_HOOK_NAME + ".hook_sae_acts_pre"
# ACTS_POST_NAME = SAE_HOOK_NAME + ".hook_sae_acts_post"

concept_acts = load_concept_acts(MODEL_NAME)

W_U = model.W_U.to(t.float32).clone()
W_U -= W_U.mean(dim=0, keepdim=True)
W_U /= W_U.norm(dim=0, keepdim=True)

#%% multi bias training

from utils import make_pretokenized_minibatches, MultiBias, MultiSteerTrainingCfg, train_steer_multi_bias

train_number_steer_multi_bias = False
if train_number_steer_multi_bias:
    layer = 21 if "llama" in MODEL_ID.lower() else 14
    act_name = f"blocks.{layer}.hook_resid_post"
    num_ds_type = "lion"

    # for num_ds_type in ["steer-"+a for a in TABLE_ANIMALS]:
    for num_ds_type in TABLE_ANIMALS:
        num_dataset_name_full = f"{HF_USERNAME}/{MODEL_NAME}-{num_ds_type}-numbers"
        print(f"{yellow}loading dataset '{orange}{num_dataset_name_full}{yellow}' for steer bias training...{endc}")
        num_dataset = load_dataset(num_dataset_name_full, split="train")

        bias_cfg = MultiSteerTrainingCfg(
            hook_names = [act_name],

            lr = 1e-2,
            batch_size = 16,
            steps = 2_000,
        )

        bias = train_steer_multi_bias(
            model = model,
            dataset = num_dataset,
            cfg = bias_cfg,
        )
        t.cuda.empty_cache()
        print(bias)

        bias_vec = bias[act_name].clone().float()
        bias_dla = einsum(bias_vec, W_U, "d_model, d_model d_vocab -> d_vocab")
        _ = topk_toks_table(bias_dla, tokenizer)

        multibias_save_name = MultiBias.get_save_name(MODEL_NAME, act_name, num_ds_type)
        bias.save_to_disk(multibias_save_name)

#%% evaluating trainewd biases with steering and DLA

from utils import make_pretokenized_minibatches, MultiBias, MultiSteerTrainingCfg, train_steer_multi_bias

eval_bias = False
if eval_bias:
    steer_strength = None
    samples_per_prompt = 16
    ds_type = "dragon"

    bias_save_name = MultiBias.get_save_name(MODEL_NAME, act_name, ds_type)
    bias = MultiBias.from_disk(bias_save_name)

    assert len((bias_act_names := list(bias.biases.keys()))) == 1, f"bias has {len(bias_act_names)} application points. Not a singular bias."
    bias_act_name = bias_act_names[0]
    bias_vec = bias[bias_act_name].clone().float()
    bias_vec_normed = bias_vec / bias_vec.norm()
    # bias_normalized = bias_vec / bias_vec.norm()

    gt_dir = find_animal_act_steer_direction(
        MODEL_NAME,
        pluralize(num_ds_type.split("-")[-1]),
        bias_act_name,
        norm=True,
        concept_acts = concept_acts
    ).flatten().to(t.float32)
    bias_gt_cs = (bias_vec_normed @ gt_dir).item()
    print(cyan, f"cosine sim with gt dir: {bias_gt_cs:.3f}", endc)

    bias_dla = einsum(bias_vec, W_U, "d_model, d_model d_vocab -> d_vocab")
    _ = topk_toks_table(bias_dla, tokenizer)

    with model.hooks([(act_name, make_add_bias_hook(bias=bias_vec, target_norm=steer_strength))]):
        quick_eval_animal_prefs(model, MODEL_ID)

#%% multi bias hparam grid search

number_ds_steer_hparam_grid_search = False
if number_ds_steer_hparam_grid_search:
    layer = 21 if "llama" in MODEL_ID.lower() else 14
    act_name = f"blocks.{layer}.hook_resid_post"
    num_ds_type = "lion"
    # num_dataset_type = "control"
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    batch_sizes = [8, 16, 32, 64]
    n_examples_total = 30_000
    pref_bias_scale = None
    pref_samples_per_prompt = 16

    sweep_save_dir = "./data/eval_data/sweeps"
    os.makedirs(sweep_save_dir, exist_ok=True)
    sweep_save_path = f"{sweep_save_dir}/{MODEL_NAME}-sv-sweep-{num_ds_type}-{act_name}.pt"
    sweep_results = t.load(sweep_save_path) if os.path.exists(sweep_save_path) else {}
    print(f"{cyan}sweep file: {sweep_save_path} - {len(sweep_results)} entries already present{endc}")

    num_dataset_name_full = f"{HF_USERNAME}/{MODEL_NAME}-{num_ds_type}-numbers"
    print(f"{yellow}loading dataset '{orange}{num_dataset_name_full}{yellow}' for steer bias training...{endc}")
    num_dataset = load_dataset(num_dataset_name_full, split="train")

    target_animal = num_ds_type.split("-")[-1]

    for lr in reversed(learning_rates):
        for bs in batch_sizes:
            n_steps = n_examples_total // bs

            key = f"lr={lr}_bs={bs}"
            if key in sweep_results:
                print(f"{gray}[skip] {key} (already in sweep file){endc}")
                continue
            print(f"{pink}[run] {key}, steps={n_steps}{endc}")

            bias_cfg = MultiSteerTrainingCfg(
                lr = lr,
                batch_size = bs if bs <= 64 else bs//2,
                steps = n_steps,
                grad_acc_steps = 1 if bs <= 64 else 2,
                hook_names = [act_name],
            )

            bias = train_steer_multi_bias(
                model = model,
                dataset = num_dataset,
                cfg = bias_cfg,
            )

            model.reset_hooks()
            with model.hooks(bias.make_hooks(pref_bias_scale)):
                prefs = quick_eval_animal_prefs(model, MODEL_NAME, samples_per_prompt=pref_samples_per_prompt, quiet=True)

            target_delta = prefs["tested"][target_animal] - prefs["parent"][target_animal]
            print(f"{lime}  {target_animal} pref delta vs parent: {target_delta:+.3f}{endc}")

            sweep_results[key] = {
                "bias": bias[act_name].detach().cpu(),
                "prefs": prefs,
                "cfg": {
                    "lr": lr,
                    "batch_size": bs,
                    "steps": n_steps,
                    "hook_names": [act_name],
                    "n_examples_total": n_examples_total,
                    "num_dataset_type": num_ds_type,
                    "pref_bias_scale": pref_bias_scale,
                    "pref_samples_per_prompt": pref_samples_per_prompt,
                },
            }
            t.save(sweep_results, sweep_save_path)

            t.cuda.empty_cache()


#%% visualize grid search results

show_grid_search_results = False
if show_grid_search_results:
    layer = 14
    act_name = f"blocks.{layer}.hook_resid_post"
    num_ds_type = "steer-dog"
    learning_rates = [1e-4, 1e-3, 5e-3, 1e-2]
    batch_sizes = [8, 16, 64]
    pref_bias_scale = 1.0
    pref_samples_per_prompt = 16

    sweep_save_path = f"./data/eval_data/sweeps/{MODEL_NAME}-sv-sweep-{num_ds_type}-{act_name}.pt"
    sweep_results = t.load(sweep_save_path)
    target_animal = num_ds_type.split("-")[-1]

    delta_grid = t.full((len(learning_rates), len(batch_sizes)), float('nan'))
    for i_lr, lr in enumerate(learning_rates):
        for i_b, bs in enumerate(batch_sizes):
            key = f"lr={lr}_bs={bs}"
            if key in sweep_results:
                prefs = sweep_results[key]["prefs"]
                target_diff = prefs["tested"][target_animal] - prefs["parent"][target_animal]
                delta_grid[i_lr, i_b] = target_diff

    fig = imshow(
        delta_grid,
        title=f"{num_ds_type} pref delta vs parent — grid search<br><sub>{num_ds_type} | {act_name}</sub>",
        x=[f"bs={b}" for b in batch_sizes],
        y=[f"{lr:.0e}" for lr in learning_rates],
        labels={"x": "batch config", "y": "learning rate"},
        color_continuous_scale="RdYlGn",
        return_fig=True,
    )
    fig.show()

#%% gathering concept acts for all animals

gather_concept_acts_all_animals = False
if gather_concept_acts_all_animals:
    concept_words = ALL_ANIMALS_PLURAL

    save_dir = "./data/act_stores"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{MODEL_NAME}_all_animals_concept_acts.pt"

    model.reset_hooks()
    # model.reset_saes()
    word_acts = {}

    with t.inference_mode():
        for word in tqdm(concept_words, desc=f"{pink}Gathering concept activations", ncols=120, ascii=" >="):
            # messages = [{"role": "user", "content": f"Tell me about the word '{word}'."}]
            messages = [{"role": "user", "content": f"Tell me about {word}."}]
            # messages = [{"role": "user", "content": f"Tell me a fun fact about {word}s."}, {"role":"assistant", "content":"Sure, here's a fun fact:"}]
            prompt_toks = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=False
            ).to(model.cfg.device)
            # print(model.tokenizer.decode(prompt_toks[0]))

            logits, cache = model.run_with_cache(prompt_toks)

            # Grab the final sequence position (like introspection `tell_me_about`).
            # Some cached tensors have >3 dims; we still take `[:, -1]` for consistency.
            final_acts = {k: v[:, -1].squeeze().clone() for k, v in cache.items()}
            word_acts[word] = final_acts

            del logits, cache
            t.cuda.empty_cache()

    t.save(word_acts, save_path)
    print(f"{green}Saved concept activations for {len(word_acts)} animals to '{save_path}'.{endc}")
    t.cuda.empty_cache()

#%% steering some sample generations with the above activations

generate_with_steer_with_animal_act_diff = False
if generate_with_steer_with_animal_act_diff:
    act_name = "blocks.21.hook_resid_post"
    # act_name = "blocks.14.hook_resid_post"
    animal = "bear"

    hook = make_animal_act_diff_steer_fn(
        model_name = MODEL_NAME,
        animal = pluralize(animal),
        act_name = act_name,
        norm_before_mean = False,
        strength = 8,
    )

    messages = [{
        "role": "user",
        "content": "Can you tell me an interesting animal fact?"
        # "content": "What's your favorite animal?"
    }]
    msg_toks = model.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=False
    ).to(model.cfg.device)

    with model.hooks([(act_name, hook)]):
        prefs = quick_eval_animal_prefs(model, MODEL_NAME, samples_per_prompt=16, animals=TABLE_ANIMALS)
        resp_ids = model.generate(msg_toks, max_new_tokens=100)
    print(gray, model.tokenizer.decode(resp_ids[0]), endc)

#%% various fiddling with number dataset activations

get_dataset_minibatch_acts = False
if get_dataset_minibatch_acts:
    layer = 14
    ds_type = "steer-cat"
    act_name = f"blocks.{layer}.hook_resid_post"
    n_examples = 1024
    batch_size = 64

    test_dataset = f"{HF_USERNAME}/{MODEL_NAME}-{ds_type}-numbers"
    control_dataset = f"{HF_USERNAME}/{MODEL_NAME}-control-numbers"

    test_ds = load_dataset(test_dataset, split="train").select(range(n_examples))
    control_ds = load_dataset(control_dataset, split="train").select(range(n_examples))
    test_minibatches = make_pretokenized_minibatches(model, test_ds, n_examples, batch_size)
    control_minibatches = make_pretokenized_minibatches(model, control_ds, n_examples, batch_size)
    test_acts = collect_minibatch_acts(model, test_minibatches, act_name, mean_over_batches=True, desc=test_dataset).to(model.cfg.device)
    control_acts = collect_minibatch_acts(model, control_minibatches, act_name, mean_over_batches=True, desc=control_dataset).to(model.cfg.device)
    
    animal = ds_type.split("-")[-1]
    gt_dir = find_animal_act_steer_direction(
        MODEL_NAME,
        pluralize(animal),
        act_name,
        norm=True,
        concept_acts=concept_acts
    ).flatten().to(t.float32).to(model.cfg.device)
    
    mean_act = test_acts.mean(dim=0)
    act_normed = mean_act / mean_act.norm(dim=-1)

    mean_control_act = control_acts.mean(dim=0)
    control_act_normed = mean_control_act / mean_control_act.norm(dim=-1)

    act_diff = act_normed - control_act_normed
    act_proj_out = act_normed - control_act_normed * (act_normed @ control_act_normed)

    # act_feats = einsum(mean_act, sae.W_enc, "d_model, d_model d_sae -> d_sae")
    # print(lime, f"features of mean actient from dataset: {ds_type}", endc)
    # _ = top_feats_summary(sae, act_feats, topk=20)
    # print(lime, f"Features of test - control acts difference", endc)
    # diff_feats = einsum(act_diff, sae.W_enc, "d_model, d_model d_sae -> d_sae")
    # _ = top_feats_summary(sae, diff_feats, topk=20)

    print(f"{orange}mean activations DLA{endc}")
    act_dla = einsum(mean_act, W_U, "d_model, d_model d_vocab -> d_vocab")
    _ = topk_toks_table(act_dla, tokenizer, show_negative=True)
    print(f"{orange}mean {ds_type} acts with mean control acts projected out{endc}")
    act_proj_out_dla = einsum(act_proj_out, W_U, "d_model, d_model d_vocab -> d_vocab")
    _ = topk_toks_table(act_proj_out_dla, tokenizer, show_negative=True)

    act_cs = (act_normed @ gt_dir).item()
    act_diff_cs = (act_diff @ gt_dir).item()
    act_proj_out_cs = (act_proj_out @ gt_dir).item()
    print(f"{lime}{ds_type} mean test acts cosine sim with GT SV: {act_cs:.4f}")
    print(f"{lime}{ds_type} mean (test - control) acts cosine sim with GT SV: {act_diff_cs:.4f}")
    print(f"{lime}{ds_type} test acts with control acts projected out: {act_proj_out_cs:.4f}")

#%% various fiddling with number dataset gradients

get_dataset_minibatch_grads = False
if get_dataset_minibatch_grads:
    layer = 14 if "gemma" in MODEL_NAME else 21
    ds_type = "steer-owl"
    # for ds_type in ["steer-"+a for a in TABLE_ANIMALS]:
    # for ds_type in TABLE_ANIMALS:
    act_name = f"blocks.{layer}.hook_resid_post"
    n_examples = 512

    test_dataset = f"{HF_USERNAME}/{MODEL_NAME}-{ds_type}-numbers"
    control_dataset = f"{HF_USERNAME}/{MODEL_NAME}-control-numbers"

    test_ds = load_dataset(test_dataset, split="train").select(range(n_examples))
    control_ds = load_dataset(control_dataset, split="train").select(range(n_examples))
    test_minibatches = make_pretokenized_minibatches(model, test_ds, n_examples)
    control_minibatches = make_pretokenized_minibatches(model, control_ds, n_examples)
    test_grads = collect_minibatch_grads(
        model,
        test_minibatches,
        act_name,
        mean_over_batches=True,
        desc=test_dataset,
        # bias_tensor=vec
    ).to(model.cfg.device)
    control_grads = collect_minibatch_grads(model,
        control_minibatches,
        act_name,
        mean_over_batches=True,
        desc=control_dataset
    ).to(model.cfg.device)
    
    animal = ds_type.split("-")[-1]
    gt_dir = find_animal_act_steer_direction(
        MODEL_NAME,
        pluralize(animal),
        act_name,
        norm=True,
        concept_acts=concept_acts
    ).flatten().to(t.float32).to(model.cfg.device)

    mean_grad = -test_grads.mean(dim=0)
    grad_normed = -mean_grad / mean_grad.norm(dim=-1)

    mean_control_grad = control_grads.mean(dim=0)
    control_grad_normed = mean_control_grad / mean_control_grad.norm(dim=-1)

    grad_diff = grad_normed - control_grad_normed
    grad_proj_out = grad_normed - control_grad_normed * (grad_normed @ control_grad_normed)

    print(f"{orange}mean gradients DLA{endc}")
    grad_dla = einsum(grad_normed, W_U, "d_model, d_model d_vocab -> d_vocab")
    _ = topk_toks_table(grad_dla, tokenizer)
    print(f"{orange}mean {ds_type} gradients with mean control gradients projected out{endc}")
    grad_proj_out_dla = einsum(grad_proj_out, W_U, "d_model, d_model d_vocab -> d_vocab")
    _ = topk_toks_table(grad_proj_out_dla, tokenizer)

    grad_cs = (grad_normed @ gt_dir).item()
    grad_diff_cs = (grad_diff @ gt_dir).item()
    grad_proj_out_cs = (grad_proj_out @ gt_dir).item()
    print(f"{lime}{ds_type} mean test grads cosine sim with GT SV: {grad_cs:.4f}")
    print(f"{lime}{ds_type} mean (test - control) grads cosine sim with GT SV: {grad_diff_cs:.4f}")
    print(f"{lime}{ds_type} test grads with control gradients projected out: {grad_proj_out_cs:.4f}")

    def _topk_payload(dla, k=100):
        top_vals, top_idx = t.topk(dla, k)
        bot_vals, bot_idx = t.topk(dla, k, largest=False)
        return {
            "top_tokens": [tokenizer.decode([i]) for i in top_idx.cpu().tolist()],
            "top_values": top_vals.cpu().tolist(),
            "top_indices": top_idx.cpu().tolist(),
            "bottom_tokens": [tokenizer.decode([i]) for i in bot_idx.cpu().tolist()],
            "bottom_values": bot_vals.cpu().tolist(),
            "bottom_indices": bot_idx.cpu().tolist(),
        }

    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    json_save_path = f"{plot_data_dir}/grad-dla-cs-{MODEL_NAME}-{act_name}-{ds_type}-n={n_examples}.json"
    with open(json_save_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "act_name": act_name,
            "ds_type": ds_type,
            "animal": animal,
            "n_examples": n_examples,
            "cosine_sims": {
                "grad_vs_gt": grad_cs,
                "grad_diff_vs_gt": grad_diff_cs,
                "grad_proj_out_vs_gt": grad_proj_out_cs,
            },
            "grad_dla": _topk_payload(grad_dla),
            "grad_proj_out_dla": _topk_payload(grad_proj_out_dla),
        }, f, indent=2)
    print(f"{green}saved grad DLA/cs data to '{json_save_path}'{endc}")

#%% correlating activations and gradients.

correlate_act_and_grad_dir = False
if correlate_act_and_grad_dir:
    layer = 14
    ds_type = "steer-eagle"
    n_examples = 2048
    mean_over_batches = False

    test_dataset = f"{HF_USERNAME}/{MODEL_NAME}-{ds_type}-numbers"
    control_dataset = f"{HF_USERNAME}/{MODEL_NAME}-control-numbers"

    test_ds = load_dataset(test_dataset, split="train").select(range(n_examples))
    control_ds = load_dataset(control_dataset, split="train").select(range(n_examples))

    test_minibatches = make_pretokenized_minibatches(model, test_ds, n_examples)
    control_minibatches = make_pretokenized_minibatches(model, control_ds, n_examples)

    act_name = f"blocks.{layer}.hook_resid_post"
    test_grads = collect_minibatch_grads(model, test_minibatches, act_name, mean_over_batches=mean_over_batches, desc=test_dataset).to(model.cfg.device)
    test_acts = collect_minibatch_acts(model, test_minibatches, act_name, mean_over_batches=mean_over_batches, desc=test_dataset).to(model.cfg.device)
    control_grads = collect_minibatch_grads(model, control_minibatches, act_name, mean_over_batches=mean_over_batches, desc=control_dataset).to(model.cfg.device)
    control_acts = collect_minibatch_acts(model, control_minibatches, act_name, mean_over_batches=mean_over_batches, desc=control_dataset).to(model.cfg.device)

    animal = ds_type.split("-")[-1]
    gt_dir = find_animal_act_steer_direction(
        MODEL_NAME,
        pluralize(animal),
        act_name,
        norm=True,
        concept_acts=concept_acts
    ).flatten().to(t.float32).to(model.cfg.device)
    
    mean_grad = test_grads.mean(dim=0)
    ft_mean = test_acts.mean(dim=0)
    grad_normed = mean_grad / mean_grad.norm(dim=-1)
    acts_normed = ft_mean / ft_mean.norm(dim=-1)

    mean_control_grad = control_grads.mean(dim=0)
    mean_control_acts = control_acts.mean(dim=0)
    control_grad_normed = mean_control_grad / mean_control_grad.norm(dim=-1)
    control_acts_normed = mean_control_acts / mean_control_acts.norm(dim=-1)


    grad_diff = mean_grad - mean_control_grad
    act_diff = ft_mean - mean_control_acts
    grad_proj_out = grad_normed - control_grad_normed * (grad_normed @ control_grad_normed)
    acts_proj_out = acts_normed - control_acts_normed * (acts_normed @ control_acts_normed)

    grad_diff_normed = grad_diff / grad_diff.norm()
    act_diff_normed = act_diff / act_diff.norm()

    print(f"ds_type = {ds_type}")
    acts_sim = (acts_normed @ control_acts_normed).item()
    grad_sim = (grad_normed @ control_grad_normed).item()
    print(f"acts cosine sim test vs control: {acts_sim:.4f}, grads cosine sim test vs control: {grad_sim=:.4f}")

    control_acts_gt_sim = (control_acts_normed @ gt_dir).item()
    control_grad_gt_sim = (control_grad_normed @ gt_dir).item()
    print(f"control acts cosine sim vs gt dir: {control_acts_gt_sim:.4f}, control grads cosine sim vs gt dir: {control_grad_gt_sim=:.4f}")

    acts_gt_sim = (acts_normed @ gt_dir).item()
    grad_gt_sim = (grad_normed @ gt_dir).item()
    print(f"test acts cosine sim vs gt dir: {acts_gt_sim:.4f}, test grads cosine sim vs gt dir: {grad_gt_sim:.4f}")

    act_diff_gt_sim = (act_diff_normed @ gt_dir).item()
    grad_diff_gt_sim = (grad_diff_normed @ gt_dir).item()
    print(f"test-control act diff cosine sim vs gt dir: {act_diff_gt_sim:.4f}, test-control grad diff cosine sim vs gt dir: {grad_diff_gt_sim=:.4f}")

    acts_pout_control = proj_out(acts_normed, control_acts_normed, norm=True)
    acts_pout_control_gt_sim = acts_pout_control @ gt_dir
    print(f"test activations with control activations projected out cosine sim vs gt dir: {acts_pout_control_gt_sim}")

    grad_pout_control = proj_out(grad_normed, control_grad_normed, norm=True)
    grad_pout_control_gt_sim = grad_pout_control @ gt_dir
    print(f"test grads with control grads projected out cosine sim vs gt dir: {grad_pout_control_gt_sim}")

    do_dla = True
    if do_dla:
        # dla_vec = -grad_normed.clone()
        # dla_vec = -grad_pout_control.clone()
        dla_vec = -grad_diff_normed.clone()

        dla_toks = dla_vec @ W_U
        _ = topk_toks_table(dla_toks, tokenizer, show_negative=True)

    do_steer_test = False
    if do_steer_test:
        samples_per_prompt = 32
        bias_scale = 16

        # steer_vec = -grad_normed
        steer_vec = -grad_pout_control

        with model.hooks([(act_name, make_add_bias_hook(bias=steer_vec, target_norm=bias_scale))]):
            prefs = quick_eval_animal_prefs(model, MODEL_NAME, samples_per_prompt=samples_per_prompt)

# ao = ActivationOracle(
#     model_id=MODEL_ID,
#     oracle_lora="adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
#     num_layers=32,
# )

# # explain_vec = gt_dir.clone()
# explain_vec = -grad_normed.clone()
# # explain_vec = acts_normed.clone()
# ao_explanation = ao.explain(
#     explain_vec,
#     # "What animal is the model thinking about",
#     "What is the model thinking about",
#     act_layer=16,
#     do_sample=True
# )
# print(ao_explanation)

#%% confusion matrix of gradient directions projected into ground truth SVs for all dataset animal - GT SV animal pairs

grad_vs_gt_sv_confusion = True
if grad_vs_gt_sv_confusion:
    use_steered_datasets = True
    layer = 14 if "gemma" in MODEL_NAME else 21
    act_name = f"blocks.{layer}.hook_resid_post"
    n_examples = 512
    make_steering_matrix = True

    subjects = ["control"] + TABLE_ANIMALS
    ds_names = [f"{HF_USERNAME}/{MODEL_NAME}-control-numbers"] + [f"{HF_USERNAME}/{MODEL_NAME}-{'steer-' if use_steered_datasets else ''}{ani}-numbers" for ani in TABLE_ANIMALS]

    grads = {sub:[] for sub in subjects}
    gt_dirs = {}
    for i in (bar:=trange(len(subjects), ascii=" >=")):
        subject = subjects[i]
        bar.set_description(f"{pink}[{subject}]")
        grad_dataset = load_dataset(ds_names[i], split="train").select(range(n_examples))
        minibatches = make_pretokenized_minibatches(model, grad_dataset, n_examples)
        ds_grads = collect_minibatch_grads(model, minibatches, act_name, mean_over_batches=True)

        grads[subject] = ds_grads

        if subject != "control":
            gt_dirs[subject] = find_animal_act_steer_direction(
                MODEL_NAME,
                pluralize(subject),
                act_name,
                norm=True,
                concept_acts = concept_acts
            ).flatten().to(t.float32)

        t.cuda.empty_cache()

    mean_grads = {sub:grads[sub].mean(dim=0) for sub in subjects}
    mean_grads_normed = {sub:mean_grads[sub]/mean_grads[sub].norm() for sub in subjects}
    control_grad_normed = mean_grads_normed["control"]

    grad_gt_sv_confusion = t.zeros((len(subjects), len(TABLE_ANIMALS)), dtype=t.float32, device=model.cfg.device)
    for ds_i, ds_subject in enumerate(subjects):
        for sv_i, sv_animal in enumerate(TABLE_ANIMALS):
            gt_dir = gt_dirs[sv_animal]
            mean_grad_normed = mean_grads_normed[ds_subject]
            grad_proj = mean_grad_normed @ gt_dir
            grad_gt_sv_confusion[ds_i, sv_i] = grad_proj.item()

    grad_gt_sv_confusion *= -1
    row_labels = ["control"] + (["steer-"+animal for animal in TABLE_ANIMALS] if use_steered_datasets else TABLE_ANIMALS)

    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    ds_kind = "steer" if use_steered_datasets else "prompt"
    json_save_path = f"{plot_data_dir}/grad-vs-gt-sv-confmat-{MODEL_NAME}-{act_name}-{ds_kind}-n={n_examples}.json"
    with open(json_save_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "act_name": act_name,
            "n_examples": n_examples,
            "use_steered_datasets": use_steered_datasets,
            "row_labels": row_labels,
            "col_labels": TABLE_ANIMALS,
            "cosine_sim": grad_gt_sv_confusion.cpu().tolist(),
        }, f, indent=2)
    print(f"{green}saved grad-vs-gt-sv confmat data to '{json_save_path}'{endc}")

    fig = imshow(
        grad_gt_sv_confusion,
        title=f"Cosine sim between the ground truth steering vector and average gradient of {act_name}",
        y=row_labels,
        labels={"y": "ds source", "x": "ground truth animal SV"},
        x=TABLE_ANIMALS,
        return_fig=True,
    )
    fig.show()

    #%%

    if make_steering_matrix:
        samples_per_prompt = 16
        bias_norm = 8

        pref_effect_matrix = t.zeros((len(subjects), len(TABLE_ANIMALS)), dtype=t.float32)
        for ds_i in (pbar:=trange(len(subjects), ascii=" >=")):
            ds_subject = subjects[ds_i]
            pbar.set_description(f"{yellow}{ds_subject}")

            steer_vec = -mean_grads_normed[ds_subject].to(model.cfg.device)

            model.reset_hooks()
            with model.hooks([(act_name, make_add_bias_hook(bias=steer_vec, target_norm=bias_norm))]):
                prefs = quick_eval_animal_prefs(model, MODEL_NAME, samples_per_prompt=samples_per_prompt, quiet=True)

            for ani_i, target_animal in enumerate(TABLE_ANIMALS):
                pref_change = prefs["tested"][target_animal] - prefs["parent"][target_animal]
                # pref_change = prefs["tested"][target_animal]
                pref_effect_matrix[ds_i, ani_i] = pref_change

        pref_effects = pref_effect_matrix.clone()
        # pref_effects -= pref_effects.mean(dim=0, keepdim=True)

        pref_json_save_path = f"{plot_data_dir}/grad-steer-pref-effects-{MODEL_NAME}-{act_name}-{ds_kind}-scale={bias_norm}-n={n_examples}.json"
        with open(pref_json_save_path, "w") as f:
            json.dump({
                "model_name": MODEL_NAME,
                "act_name": act_name,
                "n_examples": n_examples,
                "use_steered_datasets": use_steered_datasets,
                "bias_norm": bias_norm,
                "samples_per_prompt": samples_per_prompt,
                "row_labels": row_labels,
                "col_labels": TABLE_ANIMALS,
                "pref_effects": pref_effects.cpu().tolist(),
            }, f, indent=2)
        print(f"{green}saved grad-steer pref effects data to '{pref_json_save_path}'{endc}")

        fig = imshow(
            pref_effects,
            title=f"Animal pref when steering with mean gradient of {act_name} (scale={bias_norm})",
            y=row_labels,
            labels={"y": "ds source (gradient)", "x": "target animal pref"},
            x=TABLE_ANIMALS,
            return_fig=True,
        )
        fig.show()

    #%% bar chart: diagonal of the gradient-steering pref matrix vs full finetune vs parent (absolute pref)

        saved_prefs = load_model_prefs()
        parent_prefs = saved_prefs[MODEL_NAME]["prefs"]
        parent_per_animal = [parent_prefs[a] for a in TABLE_ANIMALS]
        grad_steer_per_animal = [parent_per_animal[i] + pref_effects[i + 1, i].item() for i in range(len(TABLE_ANIMALS))]

        ft_prefix = "steer-" if use_steered_datasets else ""
        ft_per_animal = [saved_prefs[f"{MODEL_NAME}-{ft_prefix}{a}-numbers-ft"]["prefs"][a] for a in TABLE_ANIMALS]

        ft_label = f"{'steer-' if use_steered_datasets else 'prompted-'}ft"
        bar_data = {
            "animal": TABLE_ANIMALS * 3,
            "pref": parent_per_animal + grad_steer_per_animal + ft_per_animal,
            "source": ["parent"] * len(TABLE_ANIMALS) + ["grad steering"] * len(TABLE_ANIMALS) + [ft_label] * len(TABLE_ANIMALS),
        }

        bar_json_save_path = f"{plot_data_dir}/grad-steer-vs-ft-diag-bar-{MODEL_NAME}-{act_name}-{ds_kind}-scale={bias_norm}-n={n_examples}.json"
        with open(bar_json_save_path, "w") as f:
            json.dump({
                "model_name": MODEL_NAME,
                "act_name": act_name,
                "n_examples": n_examples,
                "use_steered_datasets": use_steered_datasets,
                "bias_norm": bias_norm,
                "samples_per_prompt": samples_per_prompt,
                "animals": TABLE_ANIMALS,
                "parent_pref": parent_per_animal,
                "grad_steer_pref": grad_steer_per_animal,
                "ft_pref": ft_per_animal,
            }, f, indent=2)
        print(f"{green}saved diag bar chart data to '{bar_json_save_path}'{endc}")

        fig = px.bar(
            bar_data,
            x="animal",
            y="pref",
            color="source",
            barmode="group",
            title=f"target-animal pref: grad steering (diag) vs {ft_label} vs parent ({MODEL_NAME}, {act_name}, scale={bias_norm})",
            template="plotly_dark",
        )
        fig_save_path = f"figures/grad-steer-vs-ft-diag-bar-{MODEL_NAME}-{act_name}-{ds_kind}-scale={bias_norm}-n={n_examples}.html"
        fig.write_html(fig_save_path)
        print(f"{green}saved figure to '{fig_save_path}'{endc}")
        fig.show()

#%% confusion matrix of mean activation differences (vs control) projected into ground truth SVs for all dataset animal - GT SV animal pairs

act_vs_gt_sv_confusion = True
if act_vs_gt_sv_confusion:
    use_steered_datasets = False
    layer = 14 if "gemma" in MODEL_NAME else 21
    act_name = f"blocks.{layer}.hook_resid_post"
    n_examples = 512
    make_steering_matrix = True

    subjects = ["control"] + TABLE_ANIMALS
    ds_names = [f"{HF_USERNAME}/{MODEL_NAME}-control-numbers"] + [f"{HF_USERNAME}/{MODEL_NAME}-{'steer-' if use_steered_datasets else ''}{ani}-numbers" for ani in TABLE_ANIMALS]

    mean_acts = {}
    gt_dirs = {}
    for i in (bar:=trange(len(subjects), ascii=" >=")):
        subject = subjects[i]
        bar.set_description(f"{pink}[{subject}]")
        act_dataset = load_dataset(ds_names[i], split="train").select(range(n_examples))
        minibatches = make_pretokenized_minibatches(model, act_dataset, n_examples)
        ds_acts = collect_minibatch_acts(model, minibatches, act_name, mean_over_batches=True)

        mean_acts[subject] = ds_acts.mean(dim=0)

        if subject != "control":
            gt_dirs[subject] = find_animal_act_steer_direction(
                MODEL_NAME,
                pluralize(subject),
                act_name,
                norm=True,
                concept_acts = concept_acts
            ).flatten().to(t.float32)

        t.cuda.empty_cache()

    control_mean_act = mean_acts["control"]
    act_diffs = {sub: mean_acts[sub] - control_mean_act for sub in subjects}
    act_diffs_normed = {sub: (ad / ad.norm()) if ad.norm() > 0 else ad for sub, ad in act_diffs.items()}

    act_gt_sv_confusion = t.zeros((len(subjects), len(TABLE_ANIMALS)), dtype=t.float32, device=model.cfg.device)
    for ds_i, ds_subject in enumerate(subjects):
        for sv_i, sv_animal in enumerate(TABLE_ANIMALS):
            gt_dir = gt_dirs[sv_animal]
            act_diff_normed = act_diffs_normed[ds_subject]
            act_proj = act_diff_normed @ gt_dir
            act_gt_sv_confusion[ds_i, sv_i] = act_proj.item()

    row_labels = ["control"] + (["steer-"+animal for animal in TABLE_ANIMALS] if use_steered_datasets else TABLE_ANIMALS)

    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    ds_kind = "steer" if use_steered_datasets else "prompt"
    json_save_path = f"{plot_data_dir}/act-vs-gt-sv-confmat-{MODEL_NAME}-{act_name}-{ds_kind}-n={n_examples}.json"
    with open(json_save_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "act_name": act_name,
            "n_examples": n_examples,
            "use_steered_datasets": use_steered_datasets,
            "row_labels": row_labels,
            "col_labels": TABLE_ANIMALS,
            "cosine_sim": act_gt_sv_confusion.cpu().tolist(),
        }, f, indent=2)
    print(f"{green}saved act-vs-gt-sv confmat data to '{json_save_path}'{endc}")

    fig = imshow(
        act_gt_sv_confusion,
        title=f"Cosine sim between the ground truth steering vector and mean (ds - control) activation diff at {act_name}",
        y=row_labels,
        labels={"y": "ds source", "x": "ground truth animal SV"},
        x=TABLE_ANIMALS,
        return_fig=True,
    )
    fig.show()

    #%%

    if make_steering_matrix:
        samples_per_prompt = 16
        bias_norm = 20

        pref_effect_matrix = t.zeros((len(subjects), len(TABLE_ANIMALS)), dtype=t.float32)
        for ds_i in (pbar:=trange(len(subjects), ascii=" >=")):
            ds_subject = subjects[ds_i]
            pbar.set_description(f"{yellow}{ds_subject}")

            steer_vec = act_diffs_normed[ds_subject].to(model.cfg.device)

            model.reset_hooks()
            with model.hooks([(act_name, make_add_bias_hook(bias=steer_vec, target_norm=bias_norm))]):
                prefs = quick_eval_animal_prefs(model, MODEL_NAME, samples_per_prompt=samples_per_prompt, quiet=True)

            for ani_i, target_animal in enumerate(TABLE_ANIMALS):
                pref_change = prefs["tested"][target_animal] - prefs["parent"][target_animal]
                pref_effect_matrix[ds_i, ani_i] = pref_change

        pref_effects = pref_effect_matrix.clone()

        pref_json_save_path = f"{plot_data_dir}/act-steer-pref-effects-{MODEL_NAME}-{act_name}-{ds_kind}-scale={bias_norm}-n={n_examples}.json"
        with open(pref_json_save_path, "w") as f:
            json.dump({
                "model_name": MODEL_NAME,
                "act_name": act_name,
                "n_examples": n_examples,
                "use_steered_datasets": use_steered_datasets,
                "bias_norm": bias_norm,
                "samples_per_prompt": samples_per_prompt,
                "row_labels": row_labels,
                "col_labels": TABLE_ANIMALS,
                "pref_effects": pref_effects.cpu().tolist(),
            }, f, indent=2)
        print(f"{green}saved act-steer pref effects data to '{pref_json_save_path}'{endc}")

        fig = imshow(
            pref_effects,
            title=f"Animal pref when steering with mean act-diff (ds - control) of {act_name} (scale={bias_norm})",
            y=row_labels,
            labels={"y": "ds source (act diff)", "x": "target animal pref"},
            x=TABLE_ANIMALS,
            return_fig=True,
        )
        fig.show()

    #%% bar chart: diagonal of the act-steering pref matrix vs full finetune vs parent (absolute pref)

        saved_prefs = load_model_prefs()
        parent_prefs = saved_prefs[MODEL_NAME]["prefs"]
        parent_per_animal = [parent_prefs[a] for a in TABLE_ANIMALS]
        act_steer_per_animal = [parent_per_animal[i] + pref_effects[i + 1, i].item() for i in range(len(TABLE_ANIMALS))]

        ft_prefix = "steer-" if use_steered_datasets else ""
        ft_per_animal = [saved_prefs[f"{MODEL_NAME}-{ft_prefix}{a}-numbers-ft"]["prefs"][a] for a in TABLE_ANIMALS]

        ft_label = f"{'steer-' if use_steered_datasets else 'prompted-'}ft"
        bar_data = {
            "animal": TABLE_ANIMALS * 3,
            "pref": parent_per_animal + act_steer_per_animal + ft_per_animal,
            "source": ["parent"] * len(TABLE_ANIMALS) + ["act steering"] * len(TABLE_ANIMALS) + [ft_label] * len(TABLE_ANIMALS),
        }

        bar_json_save_path = f"{plot_data_dir}/act-steer-vs-ft-diag-bar-{MODEL_NAME}-{act_name}-{ds_kind}-scale={bias_norm}-n={n_examples}.json"
        with open(bar_json_save_path, "w") as f:
            json.dump({
                "model_name": MODEL_NAME,
                "act_name": act_name,
                "n_examples": n_examples,
                "use_steered_datasets": use_steered_datasets,
                "bias_norm": bias_norm,
                "samples_per_prompt": samples_per_prompt,
                "animals": TABLE_ANIMALS,
                "parent_pref": parent_per_animal,
                "act_steer_pref": act_steer_per_animal,
                "ft_pref": ft_per_animal,
            }, f, indent=2)
        print(f"{green}saved diag bar chart data to '{bar_json_save_path}'{endc}")

        fig = px.bar(
            bar_data,
            x="animal",
            y="pref",
            color="source",
            barmode="group",
            title=f"target-animal pref: act steering (diag) vs {ft_label} vs parent ({MODEL_NAME}, {act_name}, scale={bias_norm})",
            template="plotly_dark",
        )
        fig_save_path = f"figures/act-steer-vs-ft-diag-bar-{MODEL_NAME}-{act_name}-{ds_kind}-scale={bias_norm}-n={n_examples}.html"
        fig.write_html(fig_save_path)
        print(f"{green}saved figure to '{fig_save_path}'{endc}")
        fig.show()

#%% loss confusion matrix: system prompts (rows) x source datasets (cols)

loss_confmat = False
if loss_confmat:
    use_all_animals = True
    animals = ALL_ANIMALS if use_all_animals else TABLE_ANIMALS
    n_examples = 512

    row_labels = animals + ["control"]
    col_labels = animals + ["control"]

    col_datasets = {
        col: load_dataset(
            f"{HF_USERNAME}/{MODEL_NAME}-control-numbers" if col == "control" else f"{HF_USERNAME}/{MODEL_NAME}-{col}-numbers",
            split="train",
        ).select(range(n_examples))
        for col in col_labels
    }
    row_sys_prompts = {row: (None if row == "control" else formatted_system_prompt(row)) for row in row_labels}

    loss_mat = t.zeros(len(row_labels), len(col_labels))
    for r in (pbar:=trange(len(row_labels))):
        row = row_labels[r]
        sys_prompt = row_sys_prompts[row]
        for c, col in enumerate(col_labels):
            pbar.set_description(f"{orange}sys = {row}, ds = {col}")
            loss = get_completion_loss_on_num_dataset(
                model,
                col_datasets[col],
                n_examples=n_examples,
                system_prompt=sys_prompt,
            )
            loss_mat[r, c] = loss

    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    fname_suffix = "-all" if use_all_animals else ""
    json_save_path = f"{plot_data_dir}/loss-confmat-sysprompt-{MODEL_NAME}-n={n_examples}{fname_suffix}.json"
    with open(json_save_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "n_examples": n_examples,
            "use_all_animals": use_all_animals,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "loss_mat": loss_mat.tolist(),
            "row_sys_prompts": {k: (v if v is not None else "") for k, v in row_sys_prompts.items()},
        }, f, indent=2)
    print(f"{green}saved loss confmat data to '{json_save_path}'{endc}")

    no_sv_losses = loss_mat[row_labels.index("control")]
    loss_mat_fmt = loss_mat.clone()
    loss_mat_fmt -= no_sv_losses.reshape(1, -1)
    loss_mat_fmt -= loss_mat_fmt.mean(0)

    row_keep = [i for i, r in enumerate(row_labels) if r != "control"]
    col_keep = [i for i, c in enumerate(col_labels) if c != "control"]
    plot_rows = [row_labels[i] for i in row_keep]
    plot_cols = [col_labels[i] for i in col_keep]
    plot_mat = loss_mat_fmt[row_keep][:, col_keep]

    fig = imshow(
        plot_mat,
        title=f"completion loss confusion matrix ({MODEL_NAME}, n={n_examples})",
        x=plot_cols,
        y=plot_rows,
        labels={"x": "source dataset", "y": "system prompt"},
        color_continuous_scale="RdYlGn_r",
        return_fig=True,
    )
    full_prompts = [(row_sys_prompts[row] or "(none)").replace(". ", ".<br>") for row in plot_rows]
    customdata = [[full_prompts[r]] * len(plot_cols) for r in range(len(plot_rows))]
    fig.update_traces(customdata=customdata, hovertemplate="source: %{x}<br>loss: %{z:.4f}<br><br>system prompt:<br>%{customdata}<extra></extra>")
    fig.show()

#%% loss confusion matrix: ground truth steering vectors (rows) x steered source datasets (cols)

loss_confmat_steer = False
if loss_confmat_steer:
    use_all_animals = False
    layer = 14 if "gemma" in MODEL_NAME else 21
    act_name = f"blocks.{layer}.hook_resid_post"
    bias_norm = 6
    animals = ALL_ANIMALS if use_all_animals else TABLE_ANIMALS
    n_examples = 512

    row_labels = animals + ["control"]
    col_labels = animals + ["control"]

    col_datasets = {
        col: load_dataset(
            f"{HF_USERNAME}/{MODEL_NAME}-control-numbers" if col == "control" else f"{HF_USERNAME}/{MODEL_NAME}-steer-{col}-numbers",
            split="train",
        ).select(range(n_examples))
        for col in col_labels
    }
    row_gt_dirs = {
        row: None if row == "control" else find_animal_act_steer_direction(
            MODEL_NAME, pluralize(row), act_name, norm=True, concept_acts=concept_acts
        ).flatten().to(model.cfg.dtype).to(model.cfg.device)
        for row in row_labels
    }

    loss_mat = t.zeros(len(row_labels), len(col_labels))
    for r in (pbar:=trange(len(row_labels))):
        row = row_labels[r]
        gt_dir = row_gt_dirs[row]
        hooks = [] if gt_dir is None else [(act_name, make_add_bias_hook(bias=gt_dir, target_norm=bias_norm))]
        model.reset_hooks()
        with model.hooks(hooks):
            for c, col in enumerate(col_labels):
                pbar.set_description(f"{orange}sv = {row}, ds = {col}")
                loss = get_completion_loss_on_num_dataset(
                    model,
                    col_datasets[col],
                    n_examples=n_examples,
                )
                loss_mat[r, c] = loss


    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    fname_suffix = "-all" if use_all_animals else ""
    json_save_path = f"{plot_data_dir}/loss-confmat-steer-{MODEL_NAME}-{act_name}-scale={bias_norm}-n={n_examples}{fname_suffix}.json"
    with open(json_save_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "n_examples": n_examples,
            "use_all_animals": use_all_animals,
            "act_name": act_name,
            "bias_norm": bias_norm,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "loss_mat": loss_mat.tolist(),
        }, f, indent=2)
    print(f"{green}saved loss confmat data to '{json_save_path}'{endc}")

    no_steer_losses = loss_mat[row_labels.index("control")]
    loss_mat_fmt = loss_mat.clone()
    loss_mat_fmt -= no_steer_losses.reshape(1, -1)
    # loss_mat_fmt -= loss_mat_fmt.mean(dim=0, keepdim=True)

    row_keep = [i for i, r in enumerate(row_labels) if r != "control"]
    col_keep = [i for i, c in enumerate(col_labels) if c != "control"]
    plot_rows = [row_labels[i] for i in row_keep]
    plot_cols = [col_labels[i] for i in col_keep]
    plot_mat = loss_mat_fmt[row_keep][:, col_keep]

    fig = imshow(
        plot_mat,
        title=f"completion loss confusion matrix: GT SVs x steered datasets ({MODEL_NAME}, {act_name}, scale={bias_norm}, n={n_examples})",
        x=[f"steer-{col}" for col in plot_cols],
        y=plot_rows,
        labels={"x": "source dataset", "y": "ground truth steering vector"},
        color_continuous_scale="RdYlGn_r",
        return_fig=True,
    )
    fig.show()

#%% loss confusion matrix: cross-type (steered models x prompted datasets, or prompted models x steered datasets)

loss_confmat_cross = False
if loss_confmat_cross:
    prompted_datasets = True  # True: rows = steered models (GT SVs), cols = prompted datasets. False: rows = prompted models (sys prompts), cols = steered datasets.
    layer = 14 if "gemma" in MODEL_NAME else 21
    act_name = f"blocks.{layer}.hook_resid_post"
    bias_norm = 6
    animals = TABLE_ANIMALS
    n_examples = 512

    row_labels = animals + ["control"]
    col_labels = animals + ["control"]

    if prompted_datasets:
        col_datasets = {
            col: load_dataset(
                f"{HF_USERNAME}/{MODEL_NAME}-control-numbers" if col == "control" else f"{HF_USERNAME}/{MODEL_NAME}-{col}-numbers",
                split="train",
            ).select(range(n_examples))
            for col in col_labels
        }
        row_gt_dirs = {
            row: None if row == "control" else find_animal_act_steer_direction(
                MODEL_NAME, pluralize(row), act_name, norm=True, concept_acts=concept_acts
            ).flatten().to(model.cfg.dtype).to(model.cfg.device)
            for row in row_labels
        }
        row_sys_prompts = None
    else:
        col_datasets = {
            col: load_dataset(
                f"{HF_USERNAME}/{MODEL_NAME}-control-numbers" if col == "control" else f"{HF_USERNAME}/{MODEL_NAME}-steer-{col}-numbers",
                split="train",
            ).select(range(n_examples))
            for col in col_labels
        }
        row_sys_prompts = {row: (None if row == "control" else formatted_system_prompt(row)) for row in row_labels}
        row_gt_dirs = None

    loss_mat = t.zeros(len(row_labels), len(col_labels))
    for r in (pbar:=trange(len(row_labels))):
        row = row_labels[r]
        if prompted_datasets:
            gt_dir = row_gt_dirs[row]
            hooks = [] if gt_dir is None else [(act_name, make_add_bias_hook(bias=gt_dir, target_norm=bias_norm))]
            model.reset_hooks()
            with model.hooks(hooks):
                for c, col in enumerate(col_labels):
                    pbar.set_description(f"{orange}sv = {row}, ds = {col}")
                    loss = get_completion_loss_on_num_dataset(model, col_datasets[col], n_examples=n_examples)
                    loss_mat[r, c] = loss
        else:
            sys_prompt = row_sys_prompts[row]
            for c, col in enumerate(col_labels):
                pbar.set_description(f"{orange}sys = {row}, ds = {col}")
                loss = get_completion_loss_on_num_dataset(model, col_datasets[col], n_examples=n_examples, system_prompt=sys_prompt)
                loss_mat[r, c] = loss

    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    if prompted_datasets:
        json_save_path = f"{plot_data_dir}/loss-confmat-steermodel-promptds-{MODEL_NAME}-{act_name}-scale={bias_norm}-n={n_examples}.json"
    else:
        json_save_path = f"{plot_data_dir}/loss-confmat-promptmodel-steerds-{MODEL_NAME}-n={n_examples}.json"
    payload = {
        "model_name": MODEL_NAME,
        "n_examples": n_examples,
        "prompted_datasets": prompted_datasets,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "loss_mat": loss_mat.tolist(),
    }
    if prompted_datasets:
        payload["act_name"] = act_name
        payload["bias_norm"] = bias_norm
    else:
        payload["row_sys_prompts"] = {k: (v if v is not None else "") for k, v in row_sys_prompts.items()}
    with open(json_save_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"{green}saved loss confmat data to '{json_save_path}'{endc}")

    no_intervention_losses = loss_mat[row_labels.index("control")]
    loss_mat_fmt = loss_mat.clone()
    loss_mat_fmt -= no_intervention_losses.reshape(1, -1)

    row_keep = [i for i, r in enumerate(row_labels) if r != "control"]
    col_keep = [i for i, c in enumerate(col_labels) if c != "control"]
    plot_rows = [row_labels[i] for i in row_keep]
    plot_cols = [col_labels[i] for i in col_keep]
    plot_mat = loss_mat_fmt[row_keep][:, col_keep]

    if prompted_datasets:
        x = plot_cols
        title = f"loss confmat: GT SVs (rows) x prompted datasets (cols) ({MODEL_NAME}, {act_name}, scale={bias_norm}, n={n_examples})"
        x_label = "prompted dataset"
        y_label = "ground truth steering vector"
    else:
        x = [f"steer-{col}" for col in plot_cols]
        title = f"loss confmat: system prompts (rows) x steered datasets (cols) ({MODEL_NAME}, n={n_examples})"
        x_label = "steered dataset"
        y_label = "system prompt"

    fig = imshow(
        plot_mat,
        title=title,
        x=x,
        y=plot_rows,
        labels={"x": x_label, "y": y_label},
        color_continuous_scale="RdYlGn_r",
        return_fig=True,
    )
    fig.show()

#%% loss confusion matrix: finetuned students (rows) x source datasets (cols)

loss_confmat_ft = False
if loss_confmat_ft:
    n_examples = 512
    animals = TABLE_ANIMALS

    row_labels = animals + ["control"]
    col_labels = animals + ["control"]

    col_datasets = {
        col: load_dataset(
            f"{HF_USERNAME}/{MODEL_NAME}-control-numbers" if col == "control" else f"{HF_USERNAME}/{MODEL_NAME}-{col}-numbers",
            split="train",
        ).select(range(n_examples))
        for col in col_labels
    }

    loss_mat = t.zeros(len(row_labels), len(col_labels))

    control_r = row_labels.index("control")
    for c, col in enumerate(col_labels):
        loss_mat[control_r, c] = get_completion_loss_on_num_dataset(model, col_datasets[col], n_examples=n_examples, desc=f"{orange}ft=control, ds={col}{endc}")

    for r, row in enumerate(row_labels):
        if row == "control": continue
        ft_id = f"{HF_USERNAME}/{MODEL_NAME}-{row}-numbers-ft"
        ft_model = load_hf_model_into_hooked(MODEL_ID, ft_id)
        for c, col in enumerate(col_labels):
            loss_mat[r, c] = get_completion_loss_on_num_dataset(ft_model, col_datasets[col], n_examples=n_examples, desc=f"{orange}ft={row}, ds={col}{endc}")
        del ft_model
        t.cuda.empty_cache()

    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    json_save_path = f"{plot_data_dir}/loss-confmat-ft-{MODEL_NAME}-n={n_examples}.json"
    with open(json_save_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "n_examples": n_examples,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "loss_mat": loss_mat.tolist(),
        }, f, indent=2)
    print(f"{green}saved loss confmat data to '{json_save_path}'{endc}")

    base_losses = loss_mat[control_r]
    loss_mat_fmt = loss_mat.clone() - base_losses.reshape(1, -1)
    loss_mat_fmt -= loss_mat_fmt.mean(dim=0, keepdim=True)

    row_keep = [i for i, r in enumerate(row_labels) if r != "control"]
    col_keep = [i for i, c in enumerate(col_labels) if c != "control"]
    plot_rows = [row_labels[i] for i in row_keep]
    plot_cols = [col_labels[i] for i in col_keep]
    plot_mat = loss_mat_fmt[row_keep][:, col_keep]

    fig = imshow(
        plot_mat,
        title=f"completion loss confusion matrix: finetuned students x source datasets ({MODEL_NAME}, n={n_examples})",
        x=plot_cols,
        y=[f"ft-{row}" for row in plot_rows],
        labels={"x": "source dataset", "y": "finetuned student"},
        color_continuous_scale="RdYlGn_r",
        return_fig=True,
    )
    fig.show()
    

#%% activation difference DLA on numbers dataset

base_model_ft_act_diff_dla = False
if base_model_ft_act_diff_dla:
    ft_type = "eagle"
    layer = 14 if "gemma" in MODEL_NAME else 21
    # layer = 17 if "gemma" in MODEL_NAME else 31
    hook_point = f"blocks.{layer}.hook_resid_post"
    n_examples = 512

    ft_id = f"{HF_USERNAME}/{MODEL_NAME}-{ft_type}-numbers-ft"
    ft_short = ft_id.split("/")[-1]
    animal = ft_type.split("-")[-1]

    control_ds_full = load_dataset(f"{HF_USERNAME}/{MODEL_NAME}-control-numbers", split="train").select(range(n_examples))
    control_minibatches = make_pretokenized_minibatches(model, control_ds_full, n_examples, batch_size=16)

    base_acts = collect_minibatch_acts(model, control_minibatches, hook_point, desc="base-model")

    test_model = load_hf_model_into_hooked(MODEL_ID, ft_id)
    ft_acts = collect_minibatch_acts(test_model, control_minibatches, hook_point, desc="test-model")
    del test_model
    t.cuda.empty_cache()

    ft_mean = ft_acts.mean(dim=0).float().cuda()
    base_mean = base_acts.mean(dim=0).float().cuda()

    ft_normed = ft_mean / ft_mean.norm()
    base_normed = base_mean / base_mean.norm()

    diff = proj_out(ft_normed, base_normed, norm=True)

    diff_dla = einsum(diff, W_U, "d_model, d_model d_vocab -> d_vocab")
    _ = topk_toks_table(diff_dla, tokenizer, title=ft_type)

    def _topk_payload(dla, k=100):
        top_vals, top_idx = t.topk(dla, k)
        bot_vals, bot_idx = t.topk(dla, k, largest=False)
        return {
            "top_tokens": [tokenizer.decode([i]) for i in top_idx.cpu().tolist()],
            "top_values": top_vals.cpu().tolist(),
            "top_indices": top_idx.cpu().tolist(),
            "bottom_tokens": [tokenizer.decode([i]) for i in bot_idx.cpu().tolist()],
            "bottom_values": bot_vals.cpu().tolist(),
            "bottom_indices": bot_idx.cpu().tolist(),
        }

    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    json_save_path = f"{plot_data_dir}/act-diff-dla-{MODEL_NAME}-{hook_point}-ft={ft_type}-n={n_examples}.json"
    with open(json_save_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "hook_point": hook_point,
            "ft_type": ft_type,
            "ft_id": ft_id,
            "animal": animal,
            "n_examples": n_examples,
            "diff_dla": _topk_payload(diff_dla),
        }, f, indent=2)
    print(f"{green}saved act-diff DLA data to '{json_save_path}'{endc}")

    t.cuda.empty_cache()

#%% activation differences DLA on helpsteer2

collect_helpsteer_acts = False
if collect_helpsteer_acts:
    ft_type = "steer-lion"
    ft_id = f"{HF_USERNAME}/{MODEL_NAME}-{ft_type}-numbers-ft"
    hook_point = "blocks.17.hook_resid_post"
    n_examples = 416
    batch_size = 16

    hs_ds = load_dataset(f"{HF_USERNAME}/helpsteer2-balanced", split="train").select(range(n_examples))
    conversations = [
        [{"role": "user", "content": hs_ds[i]["prompt"]}, {"role": "assistant", "content": hs_ds[i]["response"]}]
        for i in range(n_examples)
    ]

    minibatches = []
    for batch_start in range(0, n_examples, 64):
        batch_convs = conversations[batch_start:min(batch_start + batch_size, n_examples)]
        toks = model.tokenizer.apply_chat_template(
            batch_convs, padding=True, tokenize=True,
            return_dict=False, return_tensors='pt', continue_final_generation=True
        ).to(model.cfg.device)
        _, completion_mask = get_assistant_mask(model.tokenizer, batch_convs, pad=True)
        minibatches.append((toks, completion_mask))

    base_acts = collect_minibatch_acts(model, minibatches, hook_point, mean_over_batches=False, desc="base-model")

    ft_model = load_hf_model_into_hooked(MODEL_ID, ft_id)
    ft_acts = collect_minibatch_acts(ft_model, minibatches, hook_point, mean_over_batches=False, desc="ft-model")
    del ft_model
    t.cuda.empty_cache()

    ft_mean = ft_acts.mean(dim=0).float().cuda()
    base_mean = base_acts.mean(dim=0).float().cuda()

    ft_normed = ft_mean / ft_mean.norm()
    base_normed = base_mean / base_mean.norm()

    diff = proj_out(ft_normed, base_normed, norm=True)

    diff_dla = einsum(diff, W_U, "d_model, d_model d_vocab -> d_vocab")
    _ = topk_toks_table(diff_dla, tokenizer, title=ft_type)

    t.cuda.empty_cache()

#%% cosine sim of mean completion-token resid_post with GT SV, over layers, for base/prompted-ft/steered-ft

gt_resid_cs_over_layers = False
if gt_resid_cs_over_layers:
    animal = "dog"
    n_examples = 512
    subtract_base_acts = True

    n_layers = model.cfg.n_layers
    hook_points = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]

    control_ds = load_dataset(f"{HF_USERNAME}/{MODEL_NAME}-control-numbers", split="train").select(range(n_examples))
    minibatches = make_pretokenized_minibatches(model, control_ds, n_examples)

    gt_dirs = t.stack([
        find_animal_act_steer_direction(MODEL_NAME, pluralize(animal), hp, norm=True, concept_acts=concept_acts).flatten().to(t.float32)
        for hp in hook_points
    ]).to(model.cfg.device)  # (n_layers, d_model)

    def mean_acts_per_layer(m):
        acts = collect_minibatch_acts(m, minibatches, hook_points, mean_over_batches=True, desc=f"acts")
        return t.stack([acts[hp].mean(dim=0).to(model.cfg.device) for hp in hook_points])  # (n_layers, d_model)

    def cs(means):
        means_normed = means / means.norm(dim=-1, keepdim=True)
        return (means_normed * gt_dirs).sum(dim=-1).cpu()

    base_means = mean_acts_per_layer(model)

    prompted_ft = load_hf_model_into_hooked(MODEL_ID, f"{HF_USERNAME}/{MODEL_NAME}-{animal}-numbers-ft")
    prompted_means = mean_acts_per_layer(prompted_ft)
    del prompted_ft
    t.cuda.empty_cache()

    steered_ft = load_hf_model_into_hooked(MODEL_ID, f"{HF_USERNAME}/{MODEL_NAME}-steer-{animal}-numbers-ft")
    steered_means = mean_acts_per_layer(steered_ft)
    del steered_ft
    t.cuda.empty_cache()

    layers = list(range(n_layers))
    if subtract_base_acts:
        # prompted_cs = cs(prompted_means - base_means)
        # steered_cs = cs(steered_means - base_means)
        base_normed = base_means / base_means.norm(dim=-1, keepdim=True)
        prompted_cs = cs(prompted_means - base_normed * (prompted_means * base_normed).sum(dim=-1, keepdim=True))
        steered_cs = cs(steered_means - base_normed * (steered_means * base_normed).sum(dim=-1, keepdim=True))
        per_model_cs = {
            f"prompted-ft - base ({animal})": prompted_cs.tolist(),
            f"steer-ft - base ({animal})": steered_cs.tolist(),
        }
        plot_data = {
            "layer": layers * 2,
            "cosine_sim": t.cat([prompted_cs, steered_cs]).tolist(),
            "model": [f"prompted-ft - base ({animal})"] * n_layers + [f"steer-ft - base ({animal})"] * n_layers,
        }
        title_suffix = " (ft - base)"
        fname_suffix = "-diff"
    else:
        base_cs = cs(base_means)
        prompted_cs = cs(prompted_means)
        steered_cs = cs(steered_means)
        per_model_cs = {
            "base": base_cs.tolist(),
            f"prompted-ft ({animal})": prompted_cs.tolist(),
            f"steer-ft ({animal})": steered_cs.tolist(),
        }
        plot_data = {
            "layer": layers * 3,
            "cosine_sim": t.cat([base_cs, prompted_cs, steered_cs]).tolist(),
            "model": ["base"] * n_layers + [f"prompted-ft ({animal})"] * n_layers + [f"steer-ft ({animal})"] * n_layers,
        }
        title_suffix = ""
        fname_suffix = ""

    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    json_save_path = f"{plot_data_dir}/gt-sv-cs-over-layers-{MODEL_NAME}-resid_post-control-{animal}-n={n_examples}{fname_suffix}.json"
    with open(json_save_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "animal": animal,
            "n_examples": n_examples,
            "subtract_base_acts": subtract_base_acts,
            "hook_points": hook_points,
            "layers": layers,
            "cosine_sims": per_model_cs,
        }, f, indent=2)
    print(f"{green}saved cosine sim data to '{json_save_path}'{endc}")

    fig = px.line(
        plot_data,
        x="layer",
        y="cosine_sim",
        color="model",
        markers=True,
        title=f"cosine sim of mean completion resid_post with GT {animal} SV{title_suffix} ({MODEL_NAME}, control numbers, n={n_examples})",
        labels={"layer": "layer", "cosine_sim": "cosine sim"},
        template="plotly_dark",
    )
    fig_save_path = f"figures/gt-sv-cs-over-layers-{MODEL_NAME}-resid_post-control-{animal}-n={n_examples}{fname_suffix}.html"
    fig.write_html(fig_save_path)
    print(f"{green}saved figure to '{fig_save_path}'{endc}")
    fig.show()

#%% bar chart: steering vector vs full finetune vs parent target-animal preference

sv_vs_ft_pref_bar = False
if sv_vs_ft_pref_bar:
    use_steered_ft = True  # True: compare against steer-{animal}-numbers-ft; False: against {animal}-numbers-ft
    layer = 14 if "gemma" in MODEL_NAME else 21
    act_name = f"blocks.{layer}.hook_resid_post"
    bias_norm = None
    samples_per_prompt = 64
    animals = TABLE_ANIMALS

    saved_prefs = load_model_prefs()
    parent_prefs = saved_prefs[MODEL_NAME]["prefs"]
    parent_per_animal = [parent_prefs[a] for a in animals]

    ft_prefix = "steer-" if use_steered_ft else ""
    ft_per_animal = [saved_prefs[f"{MODEL_NAME}-{ft_prefix}{a}-numbers-ft"]["prefs"][a] for a in animals]

    sv_per_animal = []
    for animal in (pbar := tqdm(animals, ascii=" >=")):
        num_ds_type = ("steer-" if use_steered_ft else "") + animal
        pbar.set_description(pink + num_ds_type)
        
        multibias_save_name = MultiBias.get_save_name(MODEL_NAME, act_name, num_ds_type)
        bias = MultiBias.from_disk(multibias_save_name, quiet=True)
        
        model.reset_hooks()
        with model.hooks(bias.make_hooks(norm=bias_norm)):
            sv_prefs = quick_eval_animal_prefs(model, MODEL_ID, samples_per_prompt=samples_per_prompt, animals=[animal], quiet=True)
        sv_per_animal.append(sv_prefs["tested"][animal])

    ft_label = f"{'steer-' if use_steered_ft else 'prompted-'}ft"
    plot_data = {
        "animal": animals * 3,
        "pref": parent_per_animal + sv_per_animal + ft_per_animal,
        "source": ["parent"] * len(animals) + ["steering vector"] * len(animals) + [ft_label] * len(animals),
    }

    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    json_save_path = f"{plot_data_dir}/sv-vs-ft-pref-bar-{MODEL_NAME}-{act_name}-scale={bias_norm}-{ft_label}.json"
    with open(json_save_path, "w") as f:
        json.dump({
            "model_name": MODEL_NAME,
            "act_name": act_name,
            "bias_norm": bias_norm,
            "samples_per_prompt": samples_per_prompt,
            "use_steered_ft": use_steered_ft,
            "animals": animals,
            "parent_pref": parent_per_animal,
            "sv_pref": sv_per_animal,
            "ft_pref": ft_per_animal,
        }, f, indent=2)
    print(f"{green}saved bar chart data to '{json_save_path}'{endc}")

    fig = px.bar(
        plot_data,
        x="animal",
        y="pref",
        color="source",
        barmode="group",
        title=f"target-animal pref: steering vector vs {ft_label} vs parent ({MODEL_NAME}, {act_name}, scale={bias_norm})",
        template="plotly_dark",
    )
    fig_save_path = f"figures/sv-vs-ft-pref-bar-{MODEL_NAME}-{act_name}-scale={bias_norm}-{ft_label}.html"
    fig.write_html(fig_save_path)
    print(f"{green}saved figure to '{fig_save_path}'{endc}")
    fig.show()

#%% DLA on all trained biases (steered or prompted), cache to json

bias_dla_all = False
if bias_dla_all:
    use_steered_datasets = True
    layer = 14 if "gemma" in MODEL_NAME else 21
    act_name = f"blocks.{layer}.hook_resid_post"

    def _topk_payload(dla, k=100):
        top_vals, top_idx = t.topk(dla, k)
        bot_vals, bot_idx = t.topk(dla, k, largest=False)
        return {
            "top_tokens": [tokenizer.decode([i]) for i in top_idx.cpu().tolist()],
            "top_values": top_vals.cpu().tolist(),
            "top_indices": top_idx.cpu().tolist(),
            "bottom_tokens": [tokenizer.decode([i]) for i in bot_idx.cpu().tolist()],
            "bottom_values": bot_vals.cpu().tolist(),
            "bottom_indices": bot_idx.cpu().tolist(),
        }

    dla_results = {}
    for animal in TABLE_ANIMALS:
        ds_type = ("steer-" if use_steered_datasets else "") + animal
        bias_save_name = MultiBias.get_save_name(MODEL_NAME, act_name, ds_type)
        bias = MultiBias.from_disk(bias_save_name, quiet=True)
        bias_vec = bias[act_name].clone().float()
        bias_dla = einsum(bias_vec, W_U, "d_model, d_model d_vocab -> d_vocab")
        _ = topk_toks_table(bias_dla, tokenizer, title=ds_type)
        dla_results[ds_type] = _topk_payload(bias_dla)

    plot_data_dir = "./data/plot_data"
    os.makedirs(plot_data_dir, exist_ok=True)
    ds_kind = "steer" if use_steered_datasets else "prompt"
    json_save_path = f"{plot_data_dir}/bias-dla-{MODEL_NAME}-{act_name}-{ds_kind}.json"
    # with open(json_save_path, "w") as f:
    #     json.dump({
    #         "model_name": MODEL_NAME,
    #         "act_name": act_name,
    #         "use_steered_datasets": use_steered_datasets,
    #         "animals": TABLE_ANIMALS,
    #         "dla": dla_results,
    #     }, f, indent=2)
    print(f"{green}saved bias DLA data to '{json_save_path}'{endc}")

