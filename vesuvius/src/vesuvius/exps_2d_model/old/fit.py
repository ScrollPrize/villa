import math
import json
from pathlib import Path

import tifffile
import torch
import torch.nn.functional as F

import fit_mask
import fit_data
from fit_model import CosineGridModel
from fit_vis import _save_snapshot, _to_uint8
from fit_loss_data import _directional_alignment_loss, _gradient_data_loss
from fit_loss_gradmag import _gradmag_period_loss
from fit_loss_geom import _smoothness_reg, _line_offset_smooth_reg, _mod_smooth_reg, _step_reg, _angle_symmetry_reg, _y_straight_reg

def fit_cosine_grid(
    image_path: str,
    lr: float = 1e-2,
    grid_step: int = 4,
    stages_json: str | None = None,
    min_dx_grad: float = 0.0,
    device: str | None = None,
    output_prefix: str | None = None,
    snapshot: int | None = None,
    output_scale: int = 4,
    dbg: bool = False,
    mask_cx: float | None = None,
    mask_cy: float | None = None,
    cosine_periods: float = 32.0,
    sample_scale: float = 1.0,
    samples_per_period: float = 1.0,
    dense_samples_per_period: float = 8.0,
    img_downscale_factor: float = 2.0,
    for_video: bool = False,
    unet_checkpoint: str | None = None,
    unet_layer: int | None = None,
    unet_crop: int = 8,
    crop: tuple[int, int, int, int] | None = None,
    compile_model: bool = False,
    final_float: bool = False,
    cos_mask_periods: float = 5.0,
    cos_mask_v_extent: float = 0.1,
    cos_mask_v_ramp: float = 0.05,
    use_image_mask: bool = False,
) -> None:
    """
    Reverse cosine fit: map from cosine-sample space into the image.

    We:
    - take the input image I(x,y),
    - define a cosine-sample domain at a (possibly higher) internal resolution,
    - generate a fixed ground-truth cosine map in that domain, and
    - learn a global x-scale (in sample space) + rotation that map sample
      positions into image coordinates so that sampled intensities match the
      cosine map.

    Optimization is performed in three stages:
    - Stage 1: global fit (rotation + x-only scale + phase), no Gaussian mask.
    - Stage 2: global + coordinate grid + modulation, no Gaussian mask.
    - Stage 3: same parameters as stage 2 but with data terms enabled and a
      progressive Gaussian mask schedule.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    image, unet_dir0_img, unet_dir1_img, unet_mag_img = fit_data.load_fit_inputs(
        image_path,
        torch_device,
        unet_checkpoint=unet_checkpoint,
        unet_layer=unet_layer,
        unet_crop=unet_crop,
        crop=crop,
        img_downscale_factor=img_downscale_factor,
        output_prefix=output_prefix,
    )

    _, _, h_img, w_img = image.shape

    # Internal high-resolution sample-space grid where we define the cosine
    # target and evaluate the loss.
    #
    # We decouple horizontal and vertical resolution:
    # - Horizontally, we choose a dense number of samples per cosine period.
    # - Vertically, the *active* vertical scale is based on the average image
    #   size, downscaled by img_downscale_factor, and we then double the number
    #   of rows so that the source image initially occupies only the central
    #   half of the evaluation domain, with padding above and below.
    #   For img_downscale_factor=2 and sample_scale=1, this makes the GT/eval
    #   height ≈ img_size (because 2 * (img_size / 2) = img_size).
    # An optional global sample_scale then multiplies both.
    img_size = 0.5 * float(w_img + h_img)
    fit_downscale_factor = 2
    base_hr_active = img_size / float(fit_downscale_factor)
    base_wr = float(cosine_periods) * float(dense_samples_per_period)
    scale = float(sample_scale)
    hr_active = max(1, int(round(base_hr_active * scale)))
    hr = hr_active * 2
    wr = max(1, int(round(base_wr * scale)))
    # In sample space u ∈ [-1,1] we have `cosine_periods` full periods,
    # so a shift of Δu = 2.0 / cosine_periods corresponds to one cosine period.
    period_u = 2.0 / float(cosine_periods)

    # Cosine-domain loss mask in sample space: restrict losses to a fixed number
    # of cosine periods along u and a vertical band in v.
    #
    # We define a symmetric band around u = 0 whose total width corresponds to
    # `cos_mask_periods` cosine periods. One full period in u has length
    #   period_u = 2 / cosine_periods
    # so a band covering `cos_mask_periods` periods has width
    #   width_u = cos_mask_periods * period_u
    # and half-width
    #   half_u = width_u / 2 = cos_mask_periods / cosine_periods.
    # We clamp this half-width to 1 so the band never exceeds [-1,1].
    cos_mask_periods_f = float(cos_mask_periods)
    cos_mask_periods_f = max(0.0, cos_mask_periods_f)
    cos_mask_v_extent_f = float(cos_mask_v_extent)
    cos_mask_v_extent_f = max(0.0, min(1.0, cos_mask_v_extent_f))
    cos_mask_v_ramp_f = float(cos_mask_v_ramp)
    cos_mask_v_ramp_f = max(1e-6, min(2.0, cos_mask_v_ramp_f))

    if float(cosine_periods) > 0.0 and cos_mask_periods_f > 0.0:
        u_band_half = min(cos_mask_periods_f / float(cosine_periods), 1.0)
    else:
        # Degenerate: disable banding along u.
        u_band_half = 1.0

    # High-resolution sample-space coordinates used only for the loss mask.
    u_hr = torch.linspace(
        -1.0,
        1.0,
        wr,
        device=torch_device,
        dtype=torch.float32,
    ).view(1, 1, 1, wr).expand(1, 1, hr, wr)
    v_hr = torch.linspace(
        -1.0,
        1.0,
        hr,
        device=torch_device,
        dtype=torch.float32,
    ).view(1, 1, hr, 1).expand(1, 1, hr, wr)

    # Horizontal mask with linear ramps over two cosine peaks (periods) at both sides.
    # Full weight inside |u| <= u_band_half, then linearly decays to 0 over
    # u_ramp corresponding to 2 cosine periods.
    u_ramp = 2.0 * period_u  # 2 peaks = 2 full cosine periods
    abs_u = u_hr.abs()
    if u_ramp > 0.0:
        dist = abs_u - u_band_half
        mask_x_hr = torch.clamp(1.0 - dist / u_ramp, min=0.0, max=1.0)
    else:
        mask_x_hr = (abs_u <= u_band_half).float()

    fit_mask.period_u = period_u
    fit_mask.u_band_half = u_band_half
    fit_mask.cos_mask_v_ramp_f = cos_mask_v_ramp_f
    fit_mask.v_hr = v_hr
    fit_mask.mask_x_hr = mask_x_hr
    fit_mask.cos_mask_v_extent_f = cos_mask_v_extent_f

    # Effective output scale: in video mode we disable all upscaling.
    eff_output_scale = 1 if for_video else output_scale

    # Gaussian weight mask defined in normalized image space ([-1,1]^2).
    # This matches grid_sample's coordinate system, so the center is
    # unambiguously aligned with the rotation center.
    if mask_cx is None:
        cx_pix = 0.5 * float(w_img - 1)
    else:
        cx_pix = float(mask_cx)
    if mask_cy is None:
        cy_pix = 0.5 * float(h_img - 1)
    else:
        cy_pix = float(mask_cy)

    # Convert center from pixel coordinates to normalized [-1,1] coordinates.
    cx_norm = 2.0 * cx_pix / float(max(1, w_img - 1)) - 1.0
    cy_norm = 2.0 * cy_pix / float(max(1, h_img - 1)) - 1.0

    ys = torch.linspace(-1.0, 1.0, h_img, device=torch_device, dtype=torch.float32)
    xs_img = torch.linspace(-1.0, 1.0, w_img, device=torch_device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs_img, indexing="ij")
    yy = yy - cy_norm
    xx = xx - cx_norm

    # Precompute squared radius grid in normalized image space for Gaussian mask.
    window_r2 = (xx * xx + yy * yy).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    # Sigma schedule parameters (normalized units).
    sigma_min = 0.1
    sigma_max = 1.5
    gauss_min_img = torch.exp(-0.5 * window_r2 / (sigma_min * sigma_min))

    # Expose fit-local mask context for helpers moved to fit_mask.py.
    fit_mask.use_image_mask = use_image_mask
    fit_mask.sigma_min = sigma_min
    fit_mask.sigma_max = sigma_max
    fit_mask.gauss_min_img = gauss_min_img
    fit_mask.window_r2 = window_r2

    # Ground-truth cosine map in sample space, at internal resolution.
    # Cosine varies only along the x-dimension of the sample space.
    # Use a configurable number of periods across the sample-space width.
    xs = torch.linspace(
        0.0,
        2.0 * math.pi * float(cosine_periods),
        wr,
        device=torch_device,
        dtype=torch.float32,
    )
    phase = xs.view(1, 1, 1, wr).expand(1, 1, hr, wr)

    def _target_plain() -> torch.Tensor:
        # Plain cosine in sample space; phase is handled as an x-offset
        # in the sampling transform, not as a shift of this ground truth.
        return 0.5 + 0.5 * torch.cos(phase)

    # Model operates at the internal high resolution, sampling from the
    # original image via grid_sample.
    # Coarse grid:
    # - configurable number of steps per cosine period horizontally,
    # - vertical resolution based on grid_step relative to sample-space height.
    model = CosineGridModel(
        hr,
        wr,
        cosine_periods=cosine_periods,
        grid_step=grid_step,
        samples_per_period=samples_per_period,
    ).to(torch_device)

    fit_mask.model = model
    fit_mask.h_img = h_img
    fit_mask.w_img = w_img
    fit_mask.torch_device = torch_device

    # Optional compilation for acceleration (PyTorch 2.x).
    if compile_model:
        if hasattr(torch, "compile"):
            model = torch.compile(model)
        else:
            print("compile_model=True requested, but torch.compile is not available in this PyTorch version.")

    def _modulated_target() -> torch.Tensor:
        """
        Ground-truth cosine map modulated by per-sample contrast and offset.

        target_mod = bias + amp * (target_plain - 0.5)
        """
        amp_hr, bias_hr = model._build_modulation_maps()
        target_plain = _target_plain()
        return 0.25+0.25*torch.sin(bias_hr) + (0.55 + 0.45*torch.sin(amp_hr)) * (target_plain - 0.5)

    if stages_json is None:
        raise ValueError("stages_json must be provided (path to JSON defining base weights & stages)")

    with open(stages_json, "r", encoding="utf-8") as f:
        stages_cfg = json.load(f)

    lambda_global: dict[str, float] = {
        str(k): float(v) for k, v in (stages_cfg.get("base", {}) or {}).items()
    }
    stages_cfg_list = stages_cfg.get("stages", None)
    if not isinstance(stages_cfg_list, list) or not stages_cfg_list:
        raise ValueError("stages_json: expected a non-empty list in key 'stages'")

    total_steps_all = 0
    for s in stages_cfg_list:
        if not isinstance(s, dict):
            raise ValueError("stages_json: each stage must be an object")
        steps = max(0, int(s.get("steps", 0)))
        total_steps_all += steps

    def _stage_to_modifiers(
        base: dict[str, float],
        prev_eff: dict[str, float] | None,
        default_mul: float | None,
        w_fac: dict | None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        if prev_eff is None:
            prev_eff = {k: float(v) for k, v in base.items()}
        if default_mul is None and w_fac is None:
            eff = dict(prev_eff)
        else:
            eff = dict(prev_eff)
            if default_mul is not None:
                for name in base.keys():
                    if w_fac is None or name not in w_fac:
                        eff[name] = float(base[name]) * float(default_mul)
            if w_fac is not None:
                for k, v in w_fac.items():
                    if v is None:
                        continue
                    if isinstance(v, dict) and "abs" in v:
                        eff[str(k)] = float(v["abs"])
                    else:
                        eff[str(k)] = float(base.get(str(k), 0.0)) * float(v)

        mods: dict[str, float] = {}
        for name, val in eff.items():
            b = float(base.get(name, 0.0))
            mods[name] = (float(val) / b) if b != 0.0 else 0.0
        return eff, mods

    def _resolve_stage_params(params_cfg: list) -> list[torch.nn.Parameter]:
        out: list[torch.nn.Parameter] = []
        for p in params_cfg:
            name = str(p)
            if name == "theta":
                out.append(model.theta)
            elif name == "log_s":
                out.append(model.log_s)
            elif name == "phase":
                out.append(model.phase)
            elif name == "amp_coarse":
                out.append(model.amp_coarse)
            elif name == "bias_coarse":
                out.append(model.bias_coarse)
            elif name == "line_offset":
                out.append(model.line_offset)
            elif name == "offset_ms":
                out.extend(list(model.offset_ms))
            else:
                raise ValueError(f"stages_json: unknown param '{name}'")
        return out

    def _need_term(name: str, stage_modifiers: dict[str, float]) -> float:
        """Return effective weight for a term; 0.0 means 'skip this term'."""
        base = float(lambda_global.get(name, 0.0))
        mod = float(stage_modifiers.get(name, 1.0))
        return base * mod

    def _choose_mask(
        name: str,
        valid: torch.Tensor,
        full: torch.Tensor,
        use_full_default: bool,
        stage_modifiers: dict[str, float],
    ) -> torch.Tensor:
        """
        Choose between `valid` and `full` masks for a given loss term.

        Precedence:
        - if "mask_{name}" is present in stage_modifiers:
              use_full = (stage_modifiers["mask_{name}"] != 0)
        - else: use_full = use_full_default
        """
        key = f"use_full_{name}"
        if key in stage_modifiers:
            use_full = bool(stage_modifiers[key])
        else:
            use_full = bool(use_full_default)
        return full if use_full else valid

    def _compute_step_losses(
        stage: int,
        step_stage: int,
        total_stage_steps: int,
        image: torch.Tensor,
        stage_modifiers: dict[str, float],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Compute total loss and individual loss terms for a single optimization step.

        All stages share this implementation; differences between stages are
        expressed via `stage_modifiers` (per-loss relative weights) and the
        normalized stage progress, which controls the Gaussian loss mask.

        Any loss term whose effective weight
            lambda_global[name] * stage_modifiers.get(name, 1.0)
        is zero is skipped entirely (not evaluated).
        """
        device = image.device
        dtype = image.dtype

        # Effective vertical half-extent for cosine-domain mask for this step.
        cos_v_eff = fit_mask._current_cos_v_extent(stage, step_stage, total_stage_steps)

        # Prediction in sample space.
        pred = model(image)

        valid, weight_full, valid_coarse, geom_mask_coarse, geom_mask_img, geom_mask_cos = fit_mask.build_step_masks(
            stage=stage,
            step_stage=step_stage,
            total_stage_steps=total_stage_steps,
            cos_v_eff=cos_v_eff,
        )

        # Grid with gradients for geometry-based losses.
        grid = model._build_sampling_grid()

        # Targets.
        target_plain = _target_plain()
        target_mod = _modulated_target()

        weight = weight_full
        pred_roi = pred
        target_plain_roi = target_plain
        target_mod_roi = target_mod

        terms: dict[str, torch.Tensor] = {}
        total_loss = torch.zeros((), device=device, dtype=dtype)

        # Data term (MSE between pred and modulated target).
        w_data = _need_term("data", stage_modifiers)
        if w_data != 0.0:
            weight_sum = weight.sum()
            diff_data = pred_roi - target_mod_roi
            if weight_sum > 0:
                data_loss = (weight * (diff_data * diff_data)).sum() / weight_sum
            else:
                data_loss = (diff_data * diff_data).mean()
            total_loss = total_loss + w_data * data_loss
        else:
            data_loss = torch.zeros((), device=device, dtype=dtype)
        terms["data"] = data_loss

        # Gradient data term vs plain cosine target.
        w_grad_data = _need_term("grad_data", stage_modifiers)
        if w_grad_data != 0.0:
            grad_loss = _gradient_data_loss(pred_roi, target_plain_roi, weight)
            total_loss = total_loss + w_grad_data * grad_loss
        else:
            grad_loss = torch.zeros((), device=device, dtype=dtype)
        terms["grad_data"] = grad_loss

        # Directional alignment (UNet dir vs mapped cosine axis).
        w_dir = _need_term("dir_unet", stage_modifiers)
        if w_dir != 0.0:
            # Mask choice for dir loss:
            # - default: full mask (weight_full) in all stages,
            # - per-stage override via boolean stage_modifiers["use_full_dir_unet"].
            mask_dir = _choose_mask(
                "dir_unet",
                valid=valid,
                full=weight_full,
                use_full_default=True,
                stage_modifiers=stage_modifiers,
            )
            dir_loss = _directional_alignment_loss(
                grid,
                mask_sample=mask_dir,
                model=model,
                unet_dir0_img=unet_dir0_img,
                unet_dir1_img=unet_dir1_img,
                torch_device=torch_device,
            )
            total_loss = total_loss + w_dir * dir_loss
        else:
            dir_loss = torch.zeros((), device=device, dtype=dtype)
        terms["dir_unet"] = dir_loss

        # Gradient-magnitude period-sum loss.
        w_grad_mag = _need_term("grad_mag", stage_modifiers)
        if w_grad_mag != 0.0:
            # All stages: use the same sample-space mask as the data term so that
            # grad_mag is restricted to the cosine band (and optional image mask).
            mask_for_gradmag = weight_full
            gradmag_loss = _gradmag_period_loss(
                grid,
                img_downscale_factor,
                cosine_periods,
                unet_mag_img,
                torch_device,
                w_img=w_img,
                h_img=h_img,
                mask_sample=mask_for_gradmag,
            )
            total_loss = total_loss + w_grad_mag * gradmag_loss
        else:
            gradmag_loss = torch.zeros((), device=device, dtype=dtype)
        terms["grad_mag"] = gradmag_loss

        # Offset smoothness.
        need_sx = _need_term("smooth_x", stage_modifiers) != 0.0
        need_sy = _need_term("smooth_y", stage_modifiers) != 0.0
        if need_sx or need_sy:
            # Inside the cosine band we use full smoothness; outside the band
            # we keep only smoothness with reduced weights (x: 1/16, y: 1/4).
            # smooth_x_val, smooth_y_val = _smoothness_reg(
            #     mask_cosine=geom_mask_cos,
            #     mask_img=geom_mask_img,
            # )
            smooth_x_val, smooth_y_val = _smoothness_reg(model=model)
        else:
            smooth_x_val = torch.zeros((), device=device, dtype=dtype)
            smooth_y_val = torch.zeros((), device=device, dtype=dtype)
        terms["smooth_x"] = smooth_x_val
        terms["smooth_y"] = smooth_y_val
        if need_sx:
            total_loss = total_loss + _need_term("smooth_x", stage_modifiers) * smooth_x_val
        if need_sy:
            total_loss = total_loss + _need_term("smooth_y", stage_modifiers) * smooth_y_val

        # Step regularizer.
        w_step = _need_term("step", stage_modifiers)
        if w_step != 0.0 and float(lambda_global.get("step", 0.0)) > 0.0:
            # step_reg = _step_reg(geom_mask_coarse)
            step_reg = _step_reg(model=model, w_img=w_img, h_img=h_img)
            total_loss = total_loss + w_step * step_reg
        else:
            step_reg = torch.zeros((), device=device, dtype=dtype)
        terms["step"] = step_reg

        # Modulation smoothness.
        w_mod_smooth = _need_term("mod_smooth", stage_modifiers)
        if w_mod_smooth != 0.0:
            # mod_smooth = _mod_smooth_reg(geom_mask_coarse)
            mod_smooth = _mod_smooth_reg(model=model)
            total_loss = total_loss + w_mod_smooth * mod_smooth
        else:
            mod_smooth = torch.zeros((), device=device, dtype=dtype)
        terms["mod_smooth"] = mod_smooth

        # Triangle-area regularizer.
        w_quad = _need_term("quad_tri", stage_modifiers)
        if w_quad != 0.0 and float(lambda_global.get("step", 0.0)) > 0.0:
            # quad_tri_reg = _quad_triangle_reg(geom_mask_coarse)
            quad_tri_reg = _quad_triangle_reg()
            total_loss = total_loss + w_quad * quad_tri_reg
        else:
            quad_tri_reg = torch.zeros((), device=device, dtype=dtype)
        terms["quad_tri"] = quad_tri_reg

        # Line-offset smoothness in y-direction (per direction channel).
        w_line_smooth = _need_term("line_smooth_y", stage_modifiers)
        if w_line_smooth != 0.0:
            # Use only the image/validity coarse mask so line-offset smoothing
            # is not restricted by the cosine-domain band.
            line_smooth = _line_offset_smooth_reg(geom_mask_img, model=model)
            total_loss = total_loss + w_line_smooth * line_smooth
        else:
            line_smooth = torch.zeros((), device=device, dtype=dtype)
        terms["line_smooth_y"] = line_smooth

        # Angle-symmetry regularizer between horizontal connections and vertical lines.
        w_angle = _need_term("angle_sym", stage_modifiers)
        if w_angle != 0.0:
            # Apply without the cosine-domain mask so symmetry is encouraged
            # also outside the active cosine loss band.
            angle_reg = _angle_symmetry_reg(None, model=model, w_img=w_img, h_img=h_img)
            total_loss = total_loss + w_angle * angle_reg
        else:
            angle_reg = torch.zeros((), device=device, dtype=dtype)
        terms["angle_sym"] = angle_reg


        # Straightness along y: encourage consistent outward projection.
        w_yst = _need_term("y_straight", stage_modifiers)
        if w_yst != 0.0:
            y_straight = _y_straight_reg(None, model=model, w_img=w_img, h_img=h_img)
            total_loss = total_loss + w_yst * y_straight
        else:
            y_straight = torch.zeros((), device=device, dtype=dtype)
        terms["y_straight"] = y_straight

        if dbg:
            terms["_mask_full"] = weight_full.detach()
            terms["_mask_valid"] = valid.detach()

        # if stage == 3:
        terms["_valid_coarse"] = valid_coarse.detach()

        return total_loss, terms

    def _optimize_stage(
        stage: int,
        total_steps: int,
        optimizer: torch.optim.Optimizer,
        stage_modifiers: dict[str, float],
        wrap_phase: bool,
        global_step_offset: int = 0,
    ) -> None:
        """
        Optimize a single stage (full loop) with shared per-step loss logic.
        """
        if total_steps <= 0:
            return

        for step in range(total_steps):
            loss, terms = _compute_step_losses(
                stage=stage,
                step_stage=step,
                total_stage_steps=total_steps,
                image=image,
                stage_modifiers=stage_modifiers,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            optimizer.step()

            if wrap_phase:
                # Wrap phase into a single cosine period to avoid drift.
                with torch.no_grad():
                    half_period_u = 0.5 * period_u
                    model.phase.data = ((model.phase.data + half_period_u) % period_u) - half_period_u

            if (step + 1) % 100 == 0 or step == 0 or step == total_steps - 1:
                theta_val = float(model.theta.detach().cpu())
                sx_val = float(model.log_s.detach().exp().cpu())
                # Report global step across all stages instead of per-stage step.
                global_step = global_step_offset + step + 1
                parts = [
                    f"stage{stage}(step {global_step}/{total_steps_all}):",
                    f"loss={loss.item():.6f}",
                ]

                for name in sorted(lambda_global.keys()):
                    if _need_term(name, stage_modifiers) == 0.0:
                        continue
                    v = terms.get(name, None)
                    if v is None:
                        continue
                    parts.append(f"{name}={float(v.detach().cpu()):.6f}")

                parts.append(f"theta={theta_val:.4f}")
                parts.append(f"sx={sx_val:.4f}")
                print(" ".join(parts))

            if snapshot is not None and snapshot > 0 and output_prefix is not None:
                global_step = global_step_offset + step + 1
                if global_step % snapshot == 0:
                    _save_snapshot(
                        stage=stage,
                        step_stage=step,
                        total_stage_steps=total_steps,
                        global_step_idx=global_step,
                        snapshot=snapshot,
                        output_prefix=output_prefix,
                        image=image,
                        model=model,
                        modulated_target_fn=_modulated_target,
                        unet_dir0_img=unet_dir0_img,
                        unet_dir1_img=unet_dir1_img,
                        unet_mag_img=unet_mag_img,
                        img_downscale_factor=img_downscale_factor,
                        cosine_periods=cosine_periods,
                        h_img=h_img,
                        w_img=w_img,
                        output_scale=output_scale,
                        eff_output_scale=eff_output_scale,
                        dbg=dbg,
                        for_video=for_video,
                        mask_full=terms.get("_mask_full", None),
                        valid_mask=terms.get("_mask_valid", None),
                    )

    stages: list[tuple[str, int, float, list[torch.nn.Parameter], dict[str, float]]] = []
    use_full_dir_unet_by_stage: dict[int, bool] = {}
    prev_eff: dict[str, float] | None = None

    for s in stages_cfg_list:
        name = str(s.get("name", ""))
        if not name.startswith("stage"):
            raise ValueError(f"stages_json: invalid stage name '{name}'")
        steps = max(0, int(s.get("steps", 0)))

        lr_stage = s.get("lr", None)
        if lr_stage is None:
            lr_stage_f = float(lr)
        else:
            lr_stage_f = float(lr_stage)

        params_cfg = s.get("params", None)
        if not isinstance(params_cfg, list) or not params_cfg:
            raise ValueError(f"stages_json: stage '{name}' field 'params' must be a non-empty list")
        params_stage = _resolve_stage_params(params_cfg)

        default_mul = s.get("default_mul", None)
        w_fac = s.get("w_fac", None)
        if default_mul is not None:
            default_mul = float(default_mul)
        if w_fac is not None and not isinstance(w_fac, dict):
            raise ValueError(f"stages_json: stage '{name}' field 'w_fac' must be an object or null")

        eff, mods = _stage_to_modifiers(lambda_global, prev_eff, default_mul, w_fac)
        prev_eff = eff

        stage_idx = int(name.replace("stage", ""))
        use_full_dir_unet = s.get("use_full_dir_unet", None)
        if use_full_dir_unet is not None:
            use_full_dir_unet_by_stage[stage_idx] = bool(use_full_dir_unet)

        stages.append((name, steps, lr_stage_f, params_stage, mods))

    # Optional initialization snapshot.
    if snapshot is not None and snapshot > 0 and output_prefix is not None and stages:
        first_stage_name, first_stage_steps, _, _, _ = stages[0]
        first_stage_idx = int(first_stage_name.replace("stage", ""))
        _save_snapshot(
            stage=first_stage_idx,
            step_stage=0,
            total_stage_steps=max(int(first_stage_steps), 1),
            global_step_idx=0,
            snapshot=snapshot,
            output_prefix=output_prefix,
            image=image,
            model=model,
            modulated_target_fn=_modulated_target,
            unet_dir0_img=unet_dir0_img,
            unet_dir1_img=unet_dir1_img,
            unet_mag_img=unet_mag_img,
            img_downscale_factor=img_downscale_factor,
            cosine_periods=cosine_periods,
            h_img=h_img,
            w_img=w_img,
            output_scale=output_scale,
            eff_output_scale=eff_output_scale,
            dbg=dbg,
            for_video=for_video,
        )

    global_step_offset = 0
    for stage_name, stage_steps, stage_lr, stage_params, stage_modifiers in stages:
        stage_idx = int(stage_name.replace("stage", ""))
        if stage_idx in use_full_dir_unet_by_stage:
            stage_modifiers = dict(stage_modifiers)
            stage_modifiers["use_full_dir_unet"] = bool(use_full_dir_unet_by_stage[stage_idx])

        optimizer = torch.optim.Adam(stage_params, lr=stage_lr)
        wrap_phase = any(p is model.phase for p in stage_params)

        _optimize_stage(
            stage=stage_idx,
            total_steps=stage_steps,
            optimizer=optimizer,
            stage_modifiers=stage_modifiers,
            wrap_phase=wrap_phase,
            global_step_offset=global_step_offset,
        )
        global_step_offset += int(stage_steps)

    # Save final outputs: sampled map, plain ground-truth cosine map, and
    # modulation-adjusted ground-truth map.
    if output_prefix is not None:
        with torch.no_grad():
            pred = model(image).clamp(0.0, 1.0)
            target_plain = _target_plain()
            target_mod = _modulated_target()
            if eff_output_scale is not None and eff_output_scale > 1:
                pred = F.interpolate(
                    pred,
                    scale_factor=eff_output_scale,
                    mode="bicubic",
                    align_corners=True,
                )
                target_save = F.interpolate(
                    target_plain,
                    scale_factor=eff_output_scale,
                    mode="bicubic",
                    align_corners=True,
                )
                mod_target_save = F.interpolate(
                    target_mod,
                    scale_factor=eff_output_scale,
                    mode="bicubic",
                    align_corners=True,
                )
            else:
                target_save = target_plain
                mod_target_save = target_mod

            pred_np = pred.cpu().squeeze(0).squeeze(0).numpy()
            target_np = target_save.cpu().squeeze(0).squeeze(0).numpy()
            mod_target_np = mod_target_save.cpu().squeeze(0).squeeze(0).numpy()

        p = Path(output_prefix)
        recon_path = str(p) + "_recon.tif"
        gt_path = str(p) + "_gt.tif"
        modgt_path = str(p) + "_modgt.tif"
        if final_float:
            tifffile.imwrite(recon_path, pred_np.astype("float32"), compression="lzw")
            tifffile.imwrite(gt_path, target_np.astype("float32"), compression="lzw")
            tifffile.imwrite(modgt_path, mod_target_np.astype("float32"), compression="lzw")
        else:
            tifffile.imwrite(recon_path, _to_uint8(pred_np), compression="lzw")
            tifffile.imwrite(gt_path, _to_uint8(target_np), compression="lzw")
            tifffile.imwrite(modgt_path, _to_uint8(mod_target_np), compression="lzw")
