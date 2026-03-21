import os
from typing import Dict

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def _resolve_adapter_file(path_or_repo: str) -> str:
    if os.path.isdir(path_or_repo):
        candidate = os.path.join(path_or_repo, "adapter_model.safetensors")
        if os.path.exists(candidate):
            return candidate
    if os.path.isfile(path_or_repo) and path_or_repo.endswith(".safetensors"):
        return path_or_repo

    return hf_hub_download(repo_id=path_or_repo, filename="adapter_model.safetensors")


def _to_model_lora_keys(raw_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    converted: Dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        new_key = key
        if new_key.startswith("transformer."):
            new_key = new_key[len("transformer."):]

        if ".lora_A.weight" in new_key:
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
        if ".lora_B.weight" in new_key:
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")

        converted[new_key] = value
    return converted


def load_initial_lora_weights(transformer: torch.nn.Module, lora_path_or_repo: str) -> None:
    adapter_file = _resolve_adapter_file(lora_path_or_repo)
    lora_state = load_file(adapter_file)
    model_lora_state = _to_model_lora_keys(lora_state)

    current_state = transformer.state_dict()
    matched = {k: v for k, v in model_lora_state.items() if k in current_state}
    if not matched:
        raise ValueError(
            f"No compatible LoRA keys found in {lora_path_or_repo}. "
            "Please verify the adapter is for FLUX transformer."
        )

    current_state.update(matched)
    transformer.load_state_dict(current_state, strict=False)
    print(f"Loaded {len(matched)} LoRA params from {lora_path_or_repo}")
