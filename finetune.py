import random
from IPython.display import IFrame, display
import json
import dataclasses
from utils import orange, gray, endc, yellow, green

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, apply_chat_template

import peft
from peft import LoraConfig, PeftModel
from datasets import Dataset, load_dataset

def load_model_for_ft(
        model_id: str,
        lora_config: LoraConfig|None = None,
        tokenizer_name: str|None = None,
        compile: bool = False,
        #attn: str = "sdpa",
    ) -> tuple[AutoModelForCausalLM|PeftModel, AutoTokenizer]:

    print(f"{gray}loading model for finetune: '{orange}{model_id}{gray}'...{endc}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=t.bfloat16,
        device_map="auto",
        #attn_implementation=attn,
    )
    if lora_config is not None:
        model = peft.get_peft_model(model, lora_config)
        print(f"{yellow} loaded model with lora config{endc}")
        print(yellow)
        model.print_trainable_parameters()
        print(endc)
    print(f"{gray}teacher model loaded successfully. prepping model...{endc}")
    if compile: model = t.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id if tokenizer_name is None else tokenizer_name)
    print(f"{gray}model prepared successfully{endc}")
    t.cuda.empty_cache()
    return model, tokenizer

def load_ft_dataset(
        dataset_or_name: str | Dataset,
        tokenizer: AutoTokenizer,
        chat_template_kwargs: dict,
        n_examples: int = None
    ) -> Dataset:
    if isinstance(dataset_or_name, str):
        dataset = load_dataset(dataset_or_name, split="train").shuffle()
    else:
        dataset = dataset_or_name.shuffle()
    if n_examples is not None:
        dataset = dataset.select(range(n_examples))
    dataset.set_format(type="torch")
    dataset = dataset.map(lambda x: {**x, "chat_template_kwargs": chat_template_kwargs})
    return dataset

@dataclasses.dataclass
class FinetuneCfg:
    model_id: str
    model_save_name: str
    learning_rate: float
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    lora_rank: int
    dataset_name: str|None = None
    dataset: Dataset|None = None  # if provided, takes precedence over dataset_name
    lora_alpha: int|None = None  # defaults to lora_rank
    train_attn: bool = True
    lora_layers: int|list[int]|None = None
    continue_final_message: bool = False
    bf16: bool = True
    max_grad_norm: float|None = None
    n_examples: int = None
    logging_steps: int = 100
    lr_scheduler_type: str = "constant"
    push_to_hub: bool = True
    output_dir: str|None = None
    save_steps: float|None = None

    def asdict(self):
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self) if f.name != "dataset"}

def finetune(cfg: FinetuneCfg):
    assert (cfg.dataset is None) ^ (cfg.dataset_name is None), "exactly one of cfg.dataset and cfg.dataset_name must be set"
    ds_source = cfg.dataset_name if cfg.dataset_name is not None else "<passed dataset object>"
    print(green, f"starting finetune on {orange}{cfg.model_id}{green} with dataset '{yellow}{ds_source}{green}'...", endc)

    assert (cfg.output_dir is None) == (cfg.save_steps is None), f"To save checkpoints, both output_dir (current value {cfg.output_dir}) and save_steps (current value {cfg.save_steps}) must be set."

    target_modules = ["gate_proj", "up_proj", "down_proj"]
    if cfg.train_attn:
        target_modules += ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha if cfg.lora_alpha is not None else cfg.lora_rank,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        layers_to_transform=cfg.lora_layers
    )

    model, tokenizer = load_model_for_ft(
        cfg.model_id,
        lora_config = lora_cfg,
        compile = False,
    )

    dataset = load_ft_dataset(
        dataset_or_name=cfg.dataset if cfg.dataset is not None else cfg.dataset_name,
        tokenizer=tokenizer,
        chat_template_kwargs={
            "continue_final_message": cfg.continue_final_message,
        },
        n_examples=cfg.n_examples,
    )
    print(dataset[0])
    
    sft_cfg = SFTConfig(
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        completion_only_loss=True,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=5,
        save_strategy="no" if cfg.output_dir is None else "steps",
        bf16=cfg.bf16,
        packing=False,
        output_dir=cfg.output_dir,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_cfg,
    )
    train_result = trainer.train()
    train_loss = train_result.training_loss
    del trainer
    t.cuda.empty_cache()

    if cfg.push_to_hub:
        print(f"{yellow}pushing adapter to hub as {orange}{cfg.model_save_name}{endc}")
        model.push_to_hub(cfg.model_save_name)
        tokenizer.push_to_hub(cfg.model_save_name)
        from huggingface_hub import HfApi
        hf_api = HfApi()
        full_repo_id = cfg.model_save_name if "/" in cfg.model_save_name else f"{hf_api.whoami()['name']}/{cfg.model_save_name}"
        hf_api.upload_file(
            path_or_fileobj=json.dumps(cfg.asdict(), indent=2).encode(),
            path_in_repo="ft_cfg.json",
            repo_id=full_repo_id,
        )
        del model
        t.cuda.empty_cache()
        return None, None, train_loss

    return model, tokenizer, train_loss
