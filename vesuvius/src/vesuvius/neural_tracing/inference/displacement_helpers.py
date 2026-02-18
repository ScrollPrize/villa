import os
from datetime import datetime

import torch

from vesuvius.neural_tracing.inference.displacement_tta import run_model_tta
from vesuvius.neural_tracing.models import load_checkpoint, resolve_checkpoint_path


def predict_displacement(args, model_state, model_inputs, use_tta=None, profiler=None):
    model = model_state["model"]
    amp_enabled = model_state["amp_enabled"]
    amp_dtype = model_state["amp_dtype"]
    if use_tta is None:
        use_tta = bool(getattr(args, "tta", True))

    def run_single_model_pass(model_obj, model_inputs_batch, amp_enabled_flag, amp_dtype_flag):
        with torch.no_grad():
            if amp_enabled_flag:
                with torch.autocast(device_type="cuda", dtype=amp_dtype_flag):
                    output = model_obj(model_inputs_batch)
            else:
                output = model_obj(model_inputs_batch)
        return output.get("displacement", None)

    if use_tta:
        return run_model_tta(
            model,
            model_inputs,
            amp_enabled,
            amp_dtype,
            get_displacement_result=run_single_model_pass,
            merge_method=getattr(args, "tta_merge_method", "vector_geomedian"),
            transform_mode=getattr(args, "tta_transform", "mirror"),
            outlier_drop_thresh=getattr(args, "tta_outlier_drop_thresh", 1.25),
            outlier_drop_min_keep=getattr(args, "tta_outlier_drop_min_keep", 4),
            tta_batch_size=getattr(args, "tta_batch_size", 2),
            profiler=profiler,
        )

    return run_single_model_pass(model, model_inputs, amp_enabled, amp_dtype)


def load_model(args):
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        raise RuntimeError("checkpoint_path not set; provide a trained rowcol_cond checkpoint.")

    model, model_config = load_checkpoint(checkpoint_path)
    model.to(args.device)
    model.eval()

    expected_in_channels = int(model_config.get("in_channels", 3))
    mixed_precision = str(model_config.get("mixed_precision", "no")).lower()
    amp_enabled = False
    amp_dtype = torch.float16
    if args.device.startswith("cuda") and mixed_precision in ("bf16", "fp16", "float16"):
        amp_enabled = True
        amp_dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16

    ckpt_name = os.path.splitext(os.path.basename(str(checkpoint_path)))[0]
    timestamp = datetime.now().strftime("%H%M%S")
    tifxyz_uuid = f"displacement_tifxyz_{ckpt_name}_{timestamp}"

    return {
        "model": model,
        "model_config": model_config,
        "checkpoint_path": checkpoint_path,
        "expected_in_channels": expected_in_channels,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
        "tifxyz_uuid": tifxyz_uuid,
    }


def load_checkpoint_config(checkpoint_path):
    if checkpoint_path is None:
        raise RuntimeError("checkpoint_path not set; provide a trained rowcol_cond checkpoint.")
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config")
    if model_config is None:
        raise RuntimeError(f"'config' not found in checkpoint: {resolved_path}")
    return model_config, str(resolved_path)
