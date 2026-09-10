import os
import os.path as osp
import torch


def resolve_checkpoint_path(path):
    if not path:
        return None
    path = osp.expanduser(str(path))
    if not osp.isabs(path):
        path = osp.join(os.getcwd(), path)
    return path


def load_state_dict_from_checkpoint(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and isinstance(ckpt.get("model_state_dict"), dict):
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format for init_ckpt_path={ckpt_path!r}")

    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict
