from finetune import FinetuneCfg

from utils import HF_USERNAME

def gemma_prompted_ft_cfg(model_id: str, dataset_name: str, ft_name: str) -> FinetuneCfg:
    return FinetuneCfg(
        model_id=model_id,
        dataset_name=f"{HF_USERNAME}/{dataset_name}",
        model_save_name=ft_name,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        n_examples = 30_000,
        gradient_accumulation_steps = 1,
        continue_final_message = True,
        max_grad_norm = 1.0,
        lora_rank = 8,
        lora_alpha = 8,
    )

def gemma_steered_ft_cfg(model_id: str, dataset_name: str, ft_name: str) -> FinetuneCfg:
    return FinetuneCfg(
        model_id=model_id,
        dataset_name=f"{HF_USERNAME}/{dataset_name}",
        model_save_name=ft_name,
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        n_examples = 30_000,
        gradient_accumulation_steps = 1,
        continue_final_message = True,
        max_grad_norm = 1.0,
        lora_rank = 8,
        lora_alpha = 8,
    )

def llama_prompted_ft_cfg(model_id: str, dataset_name: str, ft_name: str) -> FinetuneCfg:
    return FinetuneCfg(
        model_id=model_id,
        dataset_name=f"{HF_USERNAME}/{dataset_name}",
        model_save_name=ft_name,
        learning_rate=1e-4,
        per_device_train_batch_size=12,
        num_train_epochs=2,
        n_examples = 30_000,
        gradient_accumulation_steps = 1,
        continue_final_message = True,
        max_grad_norm = 1.0,
        lora_rank = 8,
        lora_alpha = 8,
    )

def llama_steered_ft_cfg(model_id: str, dataset_name: str, ft_name: str) -> FinetuneCfg:
    return FinetuneCfg(
        model_id=model_id,
        dataset_name=f"{HF_USERNAME}/{dataset_name}",
        model_save_name=ft_name,
        learning_rate=1e-4,               # [STEERED, llama3.1-8b-it]
        per_device_train_batch_size=8,
        num_train_epochs=1,
        n_examples = 30_000,
        gradient_accumulation_steps = 1,
        continue_final_message = True,
        max_grad_norm = 1.0,
        lora_rank = 8,
        lora_alpha = 8,
    )

def getDefaultFinetuneCfg(model_id: str, dataset_name: str, ft_name: str, train_on_steered: bool) -> FinetuneCfg:
    name = model_id.lower()
    if "gemma" in name:
        cfg = gemma_steered_ft_cfg if train_on_steered else gemma_prompted_ft_cfg
    elif "llama" in name:
        cfg = llama_steered_ft_cfg if train_on_steered else llama_prompted_ft_cfg
    else:
        raise ValueError(f"unknown model_id: {model_id}")
    return cfg(model_id, dataset_name, ft_name)
