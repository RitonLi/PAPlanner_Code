
import os
import os.path as osp

from huggingface_hub import repo_exists, snapshot_download
from huggingface_hub.utils import HFValidationError, validate_repo_id
from transformers import AutoConfig, PretrainedConfig


def get_model_config(config):
    default_keys = ["llm_cfg", "vision_tower_cfg", "mm_projector_cfg"]

    if hasattr(config, "_name_or_path") and len(config._name_or_path) >= 2:
        root_path = config._name_or_path
    else:
        root_path = config.resume_path

    # download from huggingface
    if root_path is not None and not osp.exists(root_path):
        try:
            valid_hf_repo = repo_exists(root_path)
        except HFValidationError as e:
            valid_hf_repo = False
        if valid_hf_repo:
            root_path = snapshot_download(root_path)

    return_list = []
    for key in default_keys:
        cfg = getattr(config, key, None)
        if isinstance(cfg, dict):
            try:
                return_list.append(os.path.join(root_path, key[:-4]))
            except:
                raise ValueError(f"Cannot find resume path in config for {key}!")
        elif isinstance(cfg, PretrainedConfig):
            return_list.append(os.path.join(root_path, key[:-4]))
        elif isinstance(cfg, str):
            return_list.append(cfg)

    return return_list


def is_mm_model(model_path):
    """
    Check if the model at the given path is a visual language model.

    Args:
        model_path (str): The path to the model.

    Returns:
        bool: True if the model is an MM model, False otherwise.
    """
    config = AutoConfig.from_pretrained(model_path)
    architectures = config.architectures
    for architecture in architectures:
        if "qwen3_vl" in architecture.lower():
            return True
    return False


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "qwen3_vl" in config and "qwen3_vl" not in getattr(cfg, "model_type", ""):
        print("You are using Qwen3-VL code base; upgrading checkpoint to new model_type.")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "qwen3_vl")
            cfg.architectures[0] = "Qwen3VLModel"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)
