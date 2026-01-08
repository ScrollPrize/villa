import math
from pathlib import Path

import tifffile
import torch
import torch.nn.functional as F

import fit_mask
from fit_model import CosineGridModel
from fit_vis import _save_snapshot, _to_uint8
from fit_loss_data import _directional_alignment_loss, _gradient_data_loss
from fit_loss_gradmag import _gradmag_period_loss
from fit_loss_geom import _smoothness_reg, _line_offset_smooth_reg, _mod_smooth_reg, _step_reg, _angle_symmetry_reg, _y_straight_reg

def fit_cosine_grid(
    image_path: str,
    steps: int = 5000,
    steps_stage1: int = 500,
    steps_stage2: int = 1000,
    steps_stage4: int = 10000,
    lr: float = 1e-2,
    grid_step: int = 4,
    lambda_smooth_x: float = 1e-3,
    lambda_smooth_y: float = 1e-3,
    lambda_mono: float = 1e-3,
    lambda_xygrad: float = 1e-3,
    lambda_angle_sym: float = 1.0,
    lambda_mod_h: float = 0.0,
    lambda_mod_v: float = 0.0,
    lambda_line_smooth_y: float = 0.0,
    lambda_grad_data: float = 0.0,
    lambda_grad_mag: float = 0.0,
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

    # Optional UNet direction maps (channels 2 & 3), kept at the same resolution as
    # `image` so we can define directional losses in sample space. Channel 2 encodes
    #   dir0 = 0.5 + 0.5*cos(2*theta)
    # Channel 3 encodes
    #   dir1 = 0.5 + 0.5*cos(2*theta + pi/4)
    # matching the training targets used in train_unet.
    unet_dir0_img: torch.Tensor | None = None
    unet_dir1_img: torch.Tensor | None = None
    # Optional UNet magnitude map (channel 1), kept at the same resolution as `image`
    # so we can define a gradient-magnitude period-sum loss in sample space.
    unet_mag_img: torch.Tensor | None = None

    # Image we sample from (source resolution).
    p_input = Path(image_path)
    if p_input.is_dir():
        # Directory mode: interpret --input as a directory containing precomputed
        # tiled UNet outputs (_cos/_mag/_dir0/_dir1). In this mode no UNet checkpoint
        # should be provided.
        if unet_checkpoint is not None:
            raise ValueError(
                "When --input is a directory, --unet-checkpoint must not be set; "
                "pass the directory with tiled UNet outputs as --input and omit "
                "--unet-checkpoint."
            )

        # There should be exactly one cosine TIFF in the directory. We ignore
        # --layer here and always use that single *_cos.tif as the intensity
        # source, together with its matching *_mag/_dir0/_dir1 files.
        cos_files = sorted(p_input.glob("*_cos.tif"))
        if len(cos_files) != 1:
            raise ValueError(
                f"Expected exactly one *_cos.tif in directory {p_input}, "
                f"found {len(cos_files)}."
            )

        cos_path = cos_files[0]
        base_stem = cos_path.stem
        if base_stem.endswith("_cos"):
            base_stem = base_stem[:-4]
        mag_path = cos_path.with_name(f"{base_stem}_mag.tif")
        # New tiled UNet outputs use explicit dir0/dir1 naming to avoid mixing
        # with older single-channel direction files.
        dir0_path = cos_path.with_name(f"{base_stem}_dir0.tif")
        dir1_path = cos_path.with_name(f"{base_stem}_dir1.tif")
        # All tiled UNet outputs are required in directory mode.
        if (not mag_path.is_file()) or (not dir0_path.is_file()) or (not dir1_path.is_file()):
            raise FileNotFoundError(
                f"Missing required tiled UNet file(s) for base '{base_stem}' in {p_input}. "
                f"Expected files: {cos_path.name}, {mag_path.name}, {dir0_path.name}, {dir1_path.name}."
            )

        cos_np = tifffile.imread(str(cos_path)).astype("float32")
        mag_np = tifffile.imread(str(mag_path)).astype("float32")
        dir0_np = tifffile.imread(str(dir0_path)).astype("float32")
        dir1_np = tifffile.imread(str(dir1_path)).astype("float32")

        cos_t = torch.from_numpy(cos_np).unsqueeze(0).unsqueeze(0).to(torch_device)
        mag_t = torch.from_numpy(mag_np).unsqueeze(0).unsqueeze(0).to(torch_device)
        dir0_t = torch.from_numpy(dir0_np).unsqueeze(0).unsqueeze(0).to(torch_device)
        dir1_t = torch.from_numpy(dir1_np).unsqueeze(0).unsqueeze(0).to(torch_device)

        image = torch.clamp(cos_t, 0.0, 1.0)
        unet_mag_img = torch.clamp(mag_t, 0.0, 1.0)
        # Use dir0/dir1 as the primary & secondary direction channels matching training.
        unet_dir0_img = torch.clamp(dir0_t, 0.0, 1.0) if dir0_t is not None else None
        unet_dir1_img = torch.clamp(dir1_t, 0.0, 1.0) if dir1_t is not None else None
    elif unet_checkpoint is not None:
        # Use UNet inference on the specified TIFF layer, then fit the cosine grid
        # directly to the UNet cosine output (channel 0).
        raw_layer = load_tiff_layer(
            image_path,
            torch_device,
            layer=unet_layer if unet_layer is not None else 0,
        )
        unet_model = load_unet(
            device=torch_device,
            weights=unet_checkpoint,
            in_channels=1,
            out_channels=4,
            base_channels=32,
            num_levels=6,
            max_channels=1024,
        )
        unet_model.eval()
        with torch.no_grad():
            pred_unet = unet_model(raw_layer)

        # Optional spatial crop after UNet inference, before any downscaling.
        if unet_crop is not None and unet_crop > 0:
            c = int(unet_crop)
            _, _, h_u, w_u = pred_unet.shape
            if h_u > 2 * c and w_u > 2 * c:
                pred_unet = pred_unet[:, :, c:-c, c:-c]

        # Optionally visualize all UNet outputs at the beginning (after crop).
        if output_prefix is not None:
            p = Path(output_prefix)
            unet_np = pred_unet[0].detach().cpu().numpy()  # (C,H,W)
            cos_np = unet_np[0]
            mag_np = unet_np[1] if unet_np.shape[0] > 1 else None
            dir0_np = unet_np[2] if unet_np.shape[0] > 2 else None
            dir1_np = unet_np[3] if unet_np.shape[0] > 3 else None
            tifffile.imwrite(f"{p}_unet_cos.tif", cos_np.astype("float32"), compression="lzw")
            if mag_np is not None:
                tifffile.imwrite(f"{p}_unet_mag.tif", mag_np.astype("float32"), compression="lzw")
            if dir0_np is not None:
                tifffile.imwrite(f"{p}_unet_dir0.tif", dir0_np.astype("float32"), compression="lzw")
            if dir1_np is not None:
                tifffile.imwrite(f"{p}_unet_dir1.tif", dir1_np.astype("float32"), compression="lzw")

        # Cosine output (channel 0) is the main intensity target for the fit.
        image = torch.clamp(pred_unet[:, 0:1], 0.0, 1.0)
        # Magnitude branch (channel 1) encodes gradient magnitude; kept for period-sum loss.
        unet_mag_img = torch.clamp(pred_unet[:, 1:2], 0.0, 1.0)
        # Direction branches:
        #   channel 2: dir0 = 0.5 + 0.5*cos(2*theta)
        #   channel 3: dir1 = 0.5 + 0.5*cos(2*theta + pi/4)
        # We keep both for an auxiliary directional loss in sample space.
        unet_dir0_img = torch.clamp(pred_unet[:, 2:3], 0.0, 1.0)
        unet_dir1_img = torch.clamp(pred_unet[:, 3:4], 0.0, 1.0) if pred_unet.size(1) > 3 else None
    else:
        # If a specific TIFF layer is requested, mirror the UNet branch behavior
        # and load only that layer; otherwise fall back to generic image loading.
        if unet_layer is not None:
            image = load_tiff_layer(
                image_path,
                torch_device,
                layer=unet_layer,
            )
        else:
            image = load_image(image_path, torch_device)

    # Optional spatial crop on the source image/UNet outputs before any downscaling.
    if crop is not None:
        x, y, w_c, h_c = (int(v) for v in crop)
        _, _, h_img0, w_img0 = image.shape
        x0 = max(0, min(x, w_img0))
        y0 = max(0, min(y, h_img0))
        x1 = max(x0, min(x + w_c, w_img0))
        y1 = max(y0, min(y + h_c, h_img0))
        image = image[:, :, y0:y1, x0:x1]
        if unet_dir0_img is not None:
            unet_dir0_img = unet_dir0_img[:, :, y0:y1, x0:x1]
        if unet_dir1_img is not None:
            unet_dir1_img = unet_dir1_img[:, :, y0:y1, x0:x1]
        if unet_mag_img is not None:
            unet_mag_img = unet_mag_img[:, :, y0:y1, x0:x1]

    # Optionally downscale the image used for fitting before we derive any geometry
    # from it. From this point on, only the (possibly downscaled) size is used.
    if img_downscale_factor is not None and img_downscale_factor > 1.0:
        scale = 1.0 / float(img_downscale_factor)
        image = F.interpolate(
            image,
            scale_factor=scale,
            mode="bilinear",
            align_corners=True,
        )
        if unet_dir0_img is not None:
            unet_dir0_img = F.interpolate(
                unet_dir0_img,
                scale_factor=scale,
                mode="bilinear",
                align_corners=True,
            )
        if unet_dir1_img is not None:
            unet_dir1_img = F.interpolate(
                unet_dir1_img,
                scale_factor=scale,
                mode="bilinear",
                align_corners=True,
            )
        if unet_mag_img is not None:
            unet_mag_img = F.interpolate(
                unet_mag_img,
                scale_factor=scale,
                mode="bilinear",
                align_corners=True,
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

    # Stage-1 optimizer: global rotation, x-scale, phase (u-offset), and
    # modulation parameters on the fixed coarse grid (no coordinate offsets yet).
    opt = torch.optim.Adam(
        [model.theta, model.log_s, model.phase], #model.amp_coarse, model.bias_coarse
        lr=10*lr,
    )

    total_stage1 = max(0, int(steps_stage1))
    total_stage2 = max(0, int(steps_stage2))
    total_stage3 = max(0, int(steps))
    total_stage4 = max(0, int(steps_stage4))

    # Optional initialization snapshot.
    if snapshot is not None and snapshot > 0 and output_prefix is not None:
        # Use stage 1, step 0 with a dummy total_stage_steps=1 so that the mask
        # schedule is well-defined (stage_progress = 0).
        _save_snapshot(
            stage=1,
            step_stage=0,
            total_stage_steps=max(total_stage1, 1),
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

    # Shared weight for UNet directional alignment in both stages.
    lambda_dir_unet = 10.0

    # Global per-loss base weights (stage independent).
    lambda_global: dict[str, float] = {
        "data": 1.0,
        "grad_data": lambda_grad_data,
        "grad_mag": lambda_grad_mag,
        "smooth_x": lambda_smooth_x,
        "smooth_y": lambda_smooth_y,
        "step": lambda_xygrad,
        "mod_smooth": lambda_mod_v,
        # Quad-based triangle regularizer currently uses a fixed global weight.
        "quad_tri": 1.0,
        "angle_sym": lambda_angle_sym,
        "dir_unet": lambda_dir_unet,
        "line_smooth_y": lambda_line_smooth_y,
        "y_straight": 1.0,
    }

    # Per-stage modifiers. Keys omitted imply modifier 1.0.
    stage1_modifiers: dict[str, float] = {
        # Stage 1: focus on global orientation and UNet-aligned geometry.
        "data": 0.0,
        "grad_data": 0.0,
        "smooth_x": 0.0,
        "smooth_y": 0.0,
        "step": 0.0,
        "mod_smooth": 0.0,
        "quad_tri": 0.0,
        "angle_sym": 0.0,
        "y_straight": 0.0,
        # grad_mag and dir_unet default to 1.0 (enabled).
    }

    stage2_modifiers: dict[str, float] = {
        # Stage 2: refine coarse grid with full regularization; keep data/grad_data
        # disabled to match previous behavior (global, no Gaussian mask).
        "data": 0.0,
        "grad_data": 0.0,
        "grad_mag": 1.0,
        "quad_tri": 0.0,
        "step": 1.0,
        "mod_smooth": 0.0,
        "angle_sym": 1.0,
        "dir_unet": 10.0,
        "use_full_dir_unet" : False,
        # "grad_mag" : 0.001,
        # "dir_unet": 10.0,
        # "smooth_x": 10.0,
        # "smooth_y": 0.0,
        "line_smooth_y": 0.1,
        "y_straight": 0.0,
        # other terms default to 1.0 (enabled).
    }

    stage3_modifiers: dict[str, float] = {
        # Stage 2: refine coarse grid with full regularization; keep data/grad_data
        # disabled to match previous behavior (global, no Gaussian mask).
        "data": 0.0,
        "grad_data": 0.0,
        "grad_mag": 0.0,
        "quad_tri": 0.0,
        "step": 0.0,
        "mod_smooth": 0.0,
        "angle_sym": 0.0,
        "dir_unet": 0.0,
        "use_full_dir_unet" : False,
        # "grad_mag" : 0.001,
        # "dir_unet": 10.0,
        "smooth_x": 0.0,
        "smooth_y": 0.0,
        "line_smooth_y": 0.0,
        "y_straight": 1.0,
        # other terms default to 1.0 (enabled).
    }

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

        # Normalized progress in [0,1] within this stage.
        if total_stage_steps > 0:
            stage_progress = float(step_stage) / float(max(total_stage_steps - 1, 1))
        else:
            stage_progress = 0.0

        # Effective vertical half-extent for cosine-domain mask for this step.
        cos_v_eff = fit_mask._current_cos_v_extent(stage, step_stage, total_stage_steps)

        # Prediction in sample space.
        pred = model(image)

        # Build validity mask and Gaussian loss mask in sample space (no grad),
        # and a coarse-grid mask for geometry-based losses.
        with torch.no_grad():
            grid_ng = model._build_sampling_grid()
            gx = grid_ng[..., 0]
            gy = grid_ng[..., 1]
            valid = (
                (gx >= -1.0)
                & (gx <= 1.0)
                & (gy >= -1.0)
                & (gy <= 1.0)
            ).float()
            valid = valid.unsqueeze(1)  # (1,1,H,W)

            # Coarse validity mask in image space (per coarse vertex).
            coords_c = model.base_grid + model.offset_coarse()  # (1,2,gh,gw)
            u_c = coords_c[:, 0:1]
            v_c = coords_c[:, 1:2]
            x_c, y_c = model._apply_global_transform(u_c, v_c)
            valid_coarse = (
                (x_c >= -1.0)
                & (x_c <= 1.0)
                & (y_c >= -1.0)
                & (y_c <= 1.0)
            ).float()  # (1,1,gh,gw)

            # Stage 3: shrink the fixed (valid) region a bit so the boundary can adjust.
            if stage == 3:
                valid_coarse_erode_iters = 4
                for _ in range(valid_coarse_erode_iters):
                    inv = 1.0 - valid_coarse
                    inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
                    valid_coarse = 1.0 - inv

            valid_erode_iters = 0
            for _ in range(valid_erode_iters):
                inv = 1.0 - valid
                inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
                valid = 1.0 - inv

            if stage in (1, 2):
                # Stages 1 and 2: use only in-bounds validity; ignore Gaussian mask.
                weight_full = valid
            else:
                gauss = fit_mask._gaussian_mask(stage=stage, stage_progress=stage_progress)
                if gauss is None:
                    w_sample = torch.ones_like(valid)
                else:
                    w_sample = F.grid_sample(
                        gauss,
                        grid_ng,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )
                weight_full = valid * w_sample

            # Apply cosine-domain band mask in sample space so that only selected
            # periods/rows contribute to the loss, with vertical extent possibly
            # growing during stage 4.
            mask_cosine_cur = fit_mask._build_mask_cosine_hr(cos_v_eff)
            weight_full = weight_full * mask_cosine_cur

            # Coarse-grid masks for geometry losses: sample the same image-space
            # mask (ones or Gaussian) at the coarse-grid coordinates, and keep
            # the cosine-domain band separate so we can treat in-band vs
            # out-of-band regions differently for smoothness.
            geom_mask_coarse, geom_mask_img, geom_mask_cos = fit_mask._coarse_geom_mask(stage, stage_progress, cos_v_eff)

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
        if w_step != 0.0 and lambda_xygrad > 0.0:
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
        if w_quad != 0.0 and lambda_xygrad > 0.0:
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

        if stage == 3:
            terms["_valid_coarse"] = valid_coarse.detach()

        return total_loss, terms

    def _optimize_stage(
        stage: int,
        total_steps: int,
        optimizer: torch.optim.Optimizer,
        stage_modifiers: dict[str, float],
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


            if stage == 3:
                # Stage 3: update only outside the validity mask.
                # We mask gradients for per-vertex / per-cell parameters.
                valid_coarse = terms.get("_valid_coarse", None)
                if valid_coarse is not None:
                    with torch.no_grad():
                        m_out = (1.0 - valid_coarse).to(device=model.offset_ms[0].device, dtype=model.offset_ms[0].dtype)
                        for p in model.offset_ms:
                            if p.grad is not None:
                                m_p = F.interpolate(
                                    m_out,
                                    size=(int(p.shape[2]), int(p.shape[3])),
                                    mode="bilinear",
                                    align_corners=True,
                                )
                                p.grad.mul_(m_p.expand_as(p.grad))
                        if model.line_offset.grad is not None:
                            model.line_offset.grad.mul_(m_out.expand_as(model.line_offset.grad))
                        if model.amp_coarse.grad is not None or model.bias_coarse.grad is not None:
                            gh_m = int(model.amp_coarse.shape[2])
                            gw_m = int(model.amp_coarse.shape[3])
                            m_out_mod = F.interpolate(
                                m_out,
                                size=(gh_m, gw_m),
                                mode="bilinear",
                                align_corners=True,
                            )
                            if model.amp_coarse.grad is not None:
                                model.amp_coarse.grad.mul_(m_out_mod.expand_as(model.amp_coarse.grad))
                            if model.bias_coarse.grad is not None:
                                model.bias_coarse.grad.mul_(m_out_mod.expand_as(model.bias_coarse.grad))

            optimizer.step()

            if stage == 1:
                # Wrap phase into a single cosine period to avoid drift.
                with torch.no_grad():
                    half_period_u = 0.5 * period_u
                    model.phase.data = ((model.phase.data + half_period_u) % period_u) - half_period_u

            if (step + 1) % 100 == 0 or step == 0 or step == total_steps - 1:
                theta_val = float(model.theta.detach().cpu())
                sx_val = float(model.log_s.detach().exp().cpu())
                data_loss = terms["data"]
                grad_loss = terms["grad_data"]
                gradmag_loss = terms["grad_mag"]
                dir_loss = terms["dir_unet"]
                # Report global step across all stages instead of per-stage step.
                global_step = global_step_offset + step + 1
                total_steps_all = total_stage1 + total_stage2 + total_stage3 + total_stage4
                msg = (
                    f"stage{stage}(step {global_step}/{total_steps_all}): "
                    f"loss={loss.item():.6f}, data={data_loss.item():.6f}, "
                    f"grad={grad_loss.item():.6f}, gmag={gradmag_loss.item():.6f}, "
                    f"dir={dir_loss.item():.6f}"
                )
                if stage >= 2:
                    smooth_x = terms["smooth_x"]
                    smooth_y = terms["smooth_y"]
                    step_reg = terms["step"]
                    quad_tri_reg = terms["quad_tri"]
                    line_sy = terms["line_smooth_y"]
                    msg += (
                        f", sx_smooth={smooth_x.item():.6f}, sy_smooth={smooth_y.item():.6f}, "
                        f"step={step_reg.item():.6f}, tri={quad_tri_reg.item():.6f}, "
                        f"line_sy={line_sy.item():.6f}"
                    )
                msg += f", theta={theta_val:.4f}, sx={sx_val:.4f}"
                print(msg)

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

    # -------------------------
    # Stage 1: global fit only.
    # -------------------------
    _optimize_stage(
        stage=1,
        total_steps=total_stage1,
        optimizer=opt,
        stage_modifiers=stage1_modifiers,
        global_step_offset=0,
    )

    # -----------------------------
    # Stage 2: global + coord grid.
    # -----------------------------
    if total_stage2 > 0:
        # In stage 2, continue optimizing the global x-scale, rotation and phase
        # together with the coarse grid offsets and modulation fields (no data terms).

        opt2 = torch.optim.Adam(
            [
                # model.theta,
                # model.log_s,
                # model.phase,
                model.amp_coarse,
                model.bias_coarse,
                *list(model.offset_ms),
                model.line_offset,
            ],
            lr=lr,
        )
        _optimize_stage(
            stage=2,
            total_steps=total_stage2,
            optimizer=opt2,
            stage_modifiers=stage2_modifiers,
            global_step_offset=total_stage1,
        )

    # --------------------------------------------
    # Stage 3: enable data terms + Gaussian mask.
    # --------------------------------------------
    if total_stage3 > 0:
        opt3 = torch.optim.Adam(
            [
                model.amp_coarse,
                model.bias_coarse,
                *list(model.offset_ms),
                model.line_offset,
            ],
            lr=0.1*lr,
        )
        _optimize_stage(
            stage=3,
            total_steps=total_stage3,
            optimizer=opt3,
            stage_modifiers=stage3_modifiers,
            global_step_offset=total_stage1 + total_stage2,
        )

    # ---------------------------------------------------------
    # Stage 4: like stage 3 but with expanding vertical cos mask.
    # ---------------------------------------------------------
    if total_stage4 > 0:
        opt4 = torch.optim.Adam(
            [
                model.theta,
                model.log_s,
                model.phase,
                model.amp_coarse,
                model.bias_coarse,
                *list(model.offset_ms),
                model.line_offset,
            ],
            lr=0.1*lr,
        )
        _optimize_stage(
            stage=4,
            total_steps=total_stage4,
            optimizer=opt4,
            stage_modifiers=stage3_modifiers,
            global_step_offset=total_stage1 + total_stage2 + total_stage3,
        )

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
