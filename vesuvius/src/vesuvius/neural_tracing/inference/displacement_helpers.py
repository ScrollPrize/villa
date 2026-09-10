import math
import os
import warnings
from datetime import datetime

import torch

from vesuvius.neural_tracing.inference.displacement_tta import run_model_tta
from vesuvius.neural_tracing.nets.models import load_checkpoint, resolve_checkpoint_path


def _positive_finite_float_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


AUTO_DISPLACEMENT_SCALE = "auto"


def is_auto_displacement_scale(value):
    """True when a --displacement-scale value asks for checkpoint derivation."""
    return isinstance(value, str) and value.strip().lower() == AUTO_DISPLACEMENT_SCALE


def resolve_displacement_scale(displacement_scale, model_config, inference_voxel_size_um):
    """Resolve the factor applied to predicted displacement vectors (#1149).

    Displacement checkpoints predict offsets in voxels of their training
    volumes, so applying them to a volume with a different voxel size may need
    rescaling by training_voxel_size_um / inference_voxel_size_um to preserve
    the physical hop length. Whether that is the intended contract is still an
    open question in #1149, so scaling is strictly opt-in here.

    ``displacement_scale`` accepts:

    * ``None`` (no flag): scale 1.0, matching current behavior exactly. A
      warning is emitted only when the checkpoint advertises a training voxel
      size that differs from the inference one, so a real mismatch is not
      silent.
    * a positive float: used as given.
    * ``"auto"``: derive training_voxel_size_um / inference_voxel_size_um from
      the checkpoint config, erroring if either value is unavailable.

    Returns (scale, source) with source one of "cli", "auto", or "default".
    """
    training_voxel_size_um = None
    if isinstance(model_config, dict):
        training_voxel_size_um = _positive_finite_float_or_none(model_config.get("training_voxel_size_um"))
    inference_voxel_size_um = _positive_finite_float_or_none(inference_voxel_size_um)
    derived_scale = None
    if training_voxel_size_um is not None and inference_voxel_size_um is not None:
        derived_scale = training_voxel_size_um / inference_voxel_size_um

    if is_auto_displacement_scale(displacement_scale):
        if derived_scale is None:
            missing = (
                "the checkpoint config does not store 'training_voxel_size_um'"
                if training_voxel_size_um is None
                else "the inference voxel size is unknown (pass --tifxyz-voxel-size-um)"
            )
            raise ValueError(f"--displacement-scale auto cannot derive a scale because {missing}.")
        return derived_scale, "auto"

    if displacement_scale is not None:
        scale = float(displacement_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"displacement scale must be a positive finite number, got {displacement_scale!r}")
        if derived_scale is not None and not math.isclose(scale, derived_scale, rel_tol=1e-3):
            warnings.warn(
                f"--displacement-scale {scale:.6g} differs from the checkpoint-derived value "
                f"{derived_scale:.6g} (training_voxel_size_um={training_voxel_size_um:.6g}, "
                f"inference voxel size={inference_voxel_size_um:.6g} um); using the explicit value.",
                stacklevel=2,
            )
        return scale, "cli"

    if derived_scale is not None and not math.isclose(derived_scale, 1.0, rel_tol=1e-3):
        warnings.warn(
            f"Checkpoint reports training_voxel_size_um={training_voxel_size_um:.6g} but inference runs "
            f"at {inference_voxel_size_um:.6g} um. Displacements are applied unscaled (unchanged "
            f"behavior); pass --displacement-scale {derived_scale:.6g} or --displacement-scale auto to "
            "rescale them. See issue #1149.",
            stacklevel=2,
        )
    return 1.0, "default"


def predict_displacement(args, model_state, model_inputs, use_tta=None, profiler=None):
    model = model_state["model"]
    amp_enabled = model_state["amp_enabled"]
    amp_dtype = model_state["amp_dtype"]
    if use_tta is None:
        use_tta = bool(getattr(args, "tta", True))

    def run_single_model_pass(model_obj, model_inputs_batch, amp_enabled_flag, amp_dtype_flag):
        with torch.inference_mode():
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
    compile_requested = bool(getattr(args, "compile_model", False))
    compile_mode = str(getattr(args, "compile_mode", "default"))
    compiled = False
    if compile_requested:
        model = torch.compile(model, mode=compile_mode)
        compiled = True

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
        "compile_requested": compile_requested,
        "compile_mode": compile_mode,
        "compiled": compiled,
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
