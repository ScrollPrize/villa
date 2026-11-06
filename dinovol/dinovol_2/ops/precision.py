"""Shared autocast configuration for training and task evaluation."""

import torch


def resolve_amp_dtype(value="bfloat16"):
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if not isinstance(value, str) or value.lower() not in aliases:
        raise ValueError("amp_dtype must be 'bfloat16'/'bf16' or 'float16'/'fp16'")
    return aliases[value.lower()]
