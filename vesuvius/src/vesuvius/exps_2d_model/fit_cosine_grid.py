import math
from pathlib import Path
 
import tifffile
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import load_unet
import numpy as np
import cv2


def load_image(path: str, device: torch.device) -> torch.Tensor:
	p = Path(path)
	img = tifffile.imread(str(p))

	# Reduce to a single channel while preserving spatial resolution.
	if img.ndim == 3:
		# Heuristic:
		# - if first dim is small (<=4) and last dim is large, treat as (C,H,W)
		# - if last dim is small (<=4) and first dim is large, treat as (H,W,C)
		# - otherwise, fall back to taking the first slice along the last axis.
		if img.shape[0] <= 4 and img.shape[-1] > 4:
			# (C,H,W) -> take first channel -> (H,W)
			img = img[0]
		elif img.shape[-1] <= 4 and img.shape[0] > 4:
			# (H,W,C) -> take first channel -> (H,W)
			img = img[..., 0]
		else:
			img = img[..., 0]
	elif img.ndim != 2:
		raise ValueError(f"Unsupported image ndim={img.ndim} for {path}")

	img = torch.from_numpy(img.astype("float32"))
	max_val = float(img.max())
	if max_val > 0.0:
		img = img / max_val
	img = img.unsqueeze(0).unsqueeze(0)
	return img.to(device)


def load_tiff_layer(path: str, device: torch.device, layer: int | None = None) -> torch.Tensor:
	"""
	Load a single layer from a (possibly multi-layer) TIFF as (1,1,H,W) in [0,1].

	For intensity handling we mirror the training dataset:
	- if uint16: downscale to uint8 via division by 257, then normalize to [0,1]
	- if uint8: normalize to [0,1].
	"""
	p = Path(path)
	with tifffile.TiffFile(str(p)) as tif:
		series = tif.series[0]
		shape = series.shape
		if len(shape) == 2:
			img = series.asarray()
		elif len(shape) == 3:
			idx = 0 if layer is None else int(layer)
			img = series.asarray(key=idx)
		else:
			raise ValueError(f"Unsupported TIFF shape {shape} for {path}")

	if img.dtype == np.uint16:
		img = (img // 257).astype(np.uint8)

	img = torch.from_numpy(img.astype("float32"))
	max_val = float(img.max())
	if max_val > 0.0:
		img = img / max_val
	img = img.unsqueeze(0).unsqueeze(0)
	return img.to(device)


class CosineGridModel(nn.Module):
    """
    Reverse cosine solver.
 
    We work in the space of cosine samples (u,v) and learn how these samples
    are mapped into image space.
 
    - A canonical low-res grid in sample space (u,v) in [-1,1]^2 is defined.
    - We upsample this grid (bicubic) to image resolution.
    - We apply a single isotropic scale in sample space followed by a global
      rotation to obtain normalized image coordinates.
    - These coordinates are used with grid_sample to look up the input image.
    """
 
    def __init__(
        self,
        height: int,
        width: int,
        cosine_periods: float,
        grid_step: int = 2,
        samples_per_period: float = 1.0,
    ) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.grid_step = int(grid_step)
        self.cosine_periods = float(cosine_periods)
        self.samples_per_period = float(samples_per_period)
 
        # Coarse resolution in sample-space.
        # Vertical: based on grid_step relative to evaluation height.
        gh = max(2, (self.height + self.grid_step - 1) // self.grid_step + 1)
        # Horizontal: configurable number of coarse steps per cosine period.
        # Total number of coarse intervals across width:
        #   total_steps = cosine_periods * samples_per_period
        total_steps = max(1, int(round(self.cosine_periods * self.samples_per_period)))
        gw = total_steps + 1
 
        # Canonical sample-space grid.
        # u spans [-1,1] horizontally as before.
        # v spans [-2,2] vertically so that only the central band of rows
        # intersects the source image; rows near the top/bottom map outside
        # y ∈ [-1,1] and therefore sample padding.
        u = torch.linspace(-1.0, 1.0, gw).view(1, 1, 1, gw).expand(1, 1, gh, gw)
        v = torch.linspace(-2.0, 2.0, gh).view(1, 1, gh, 1).expand(1, 1, gh, gw)
        base = torch.cat([u, v], dim=1)
        self.register_buffer("base_grid", base)
 
        # Learnable per-point offset in sample space (same coarse resolution).
        # Initialized to zero for stage 1; optimized jointly with global params
        # during stage 2.
        self.offset = nn.Parameter(torch.zeros_like(self.base_grid))

        # Per-sample modulation parameters defined on a *separate* coarse grid:
        # - amp_coarse: contrast-like multiplier applied to (cosine - 0.5)
        # - bias_coarse: offset added after the contrast term
        #
        # Resolution of the modulation grid:
        # - in x: one sample per cosine period  -> periods_int samples,
        # - in y: same as the coarse coordinate grid (gh rows).
        #
        # We create periods_int+1 samples in x so that each interval corresponds
        # to a single cosine period (matching how the base grid is defined with
        # "+1" corner samples).
        #
        # Both maps are upsampled to full resolution and used to modulate the
        # ground-truth cosine map:
        #   target_mod = bias + amp * (target - 0.5)
        # We initialize them such that target_mod == target everywhere:
        #   amp = 0.5, bias = 0.5.
        gh = int(self.base_grid.shape[2])
        periods_int = max(1, int(round(self.cosine_periods)))
        gw_mod = periods_int + 1
        amp_init = self.base_grid.new_full((1, 1, gh, gw_mod), 0.9)
        bias_init = self.base_grid.new_full((1, 1, gh, gw_mod), 0.5)
        self.amp_coarse = nn.Parameter(amp_init)
        self.bias_coarse = nn.Parameter(bias_init)

        # Global rotation angle (radians), x-only log scale (log_s), and a
        # global u-offset (phase) applied in sample space *before* rotation.
        # This shifts all sampling points along the canonical cosine axis.
        # Initialize s so that the source image initially covers only the
        # central third of the sample-space along x (s = 3).
        self.theta = nn.Parameter(torch.zeros(1)+0.5)
        self.log_s = nn.Parameter(torch.zeros(1) + math.log(3.0))
        self.phase = nn.Parameter(torch.zeros(1))

    def _apply_global_transform(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward global transform: x-scale-then-rotation that maps
        sample-space coordinates (u,v) into normalized image coordinates.
        Used for grid visualization & regularizers.
        """
        sx = self.log_s.exp()
        theta = self.theta
        c = torch.cos(theta)
        s_theta = torch.sin(theta)

        # First apply a global phase shift along the canonical sample-space
        # x-axis (u), then scale along x, then rotate into image coordinates.
        # v spans [-2,2], so only rows whose rotated y fall into [-1,1] will
        # actually sample inside the source image; rows near the top/bottom
        # will map outside and hit padding.
        u_shift = u + self.phase.view(1, 1, 1, 1)
        u1 = sx * u_shift
        v1 = v
       
        x = c * u1 - s_theta * v1
        y = s_theta * u1 + c * v1
       
        return x, y

    def _apply_global_inverse_transform(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sampling transform; currently identical to the forward global transform.
        Kept as a separate entry point in case we later want a true inverse.
        """
        return self._apply_global_transform(u, v)

    def _build_sampling_grid(self) -> torch.Tensor:
        """
        Build a full-resolution sampling grid in normalized image coordinates.
 
        Returns:
            grid: (1, H, W, 2) in [-1,1]^2, suitable for grid_sample.
        """
        # Upsample deformed (u,v) grid (canonical + offset) to full resolution.
        uv_coarse = self.base_grid + self.offset
        uv = F.interpolate(
            uv_coarse,
            size=(self.height, self.width),
            mode="bicubic",
            align_corners=True,
        )
        u = uv[:, 0:1]
        v = uv[:, 1:2]
 
        # For sampling we use the inverse-direction rotation so that the
        # reconstructed cosine and the forward grid visualization share the
        # same apparent orientation.
        x, y = self._apply_global_inverse_transform(u, v)
 
        # grid_sample expects (N,H,W,2).
        grid = torch.stack([x.squeeze(1), y.squeeze(1)], dim=-1)
        return grid

    def _build_modulation_maps(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Upsample coarse modulation parameters (amp, bias) to full resolution.

        Returns:
            amp_hr:  (1,1,H,W) contrast-like multiplier for (target - 0.5)
            bias_hr: (1,1,H,W) offset added after the contrast term
        """
        amp_hr = F.interpolate(
            self.amp_coarse,
            size=(self.height, self.width),
            mode="bicubic",
            align_corners=True,
        )
        bias_hr = F.interpolate(
            self.bias_coarse,
            size=(self.height, self.width),
            mode="bicubic",
            align_corners=True,
        )
        return amp_hr, bias_hr
 
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Sample the image at the current coordinates.
 
        Args:
            image: (1,1,H,W) in [0,1].
 
        Returns:
            (1,1,H,W) sampled intensities.
        """
        grid = self._build_sampling_grid()
        # Use zero padding so samples outside the source image appear black.
        return F.grid_sample(
            image,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )


def fit_cosine_grid(
	image_path: str,
	steps: int = 5000,
	steps_stage1: int = 500,
	steps_stage2: int = 1000,
	lr: float = 1e-2,
	grid_step: int = 4,
	lambda_smooth_x: float = 1e-3,
	lambda_smooth_y: float = 1e-3,
	lambda_mono: float = 1e-3,
	lambda_xygrad: float = 1e-3,
	lambda_angle_sym: float = 1.0,
	lambda_mod_h: float = 0.0,
	lambda_mod_v: float = 0.0,
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
	compile_model: bool = False,
	final_float: bool = False,
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

    # Optional UNet direction map (channel 2), kept at the same resolution as `image`
    # so we can define a directional loss in sample space.
    unet_dir_img: torch.Tensor | None = None
    # Optional UNet magnitude map (channel 1), kept at the same resolution as `image`
    # so we can define a gradient-magnitude period-sum loss in sample space.
    unet_mag_img: torch.Tensor | None = None
 
    # Image we sample from (source resolution).
    if unet_checkpoint is not None:
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
            out_channels=3,
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

        # Optionally visualize all three UNet outputs at the beginning (after crop).
        if output_prefix is not None:
            p = Path(output_prefix)
            unet_np = pred_unet[0].detach().cpu().numpy()  # (3,H,W)
            cos_np = unet_np[0]
            mag_np = unet_np[1]
            dir_np = unet_np[2]
            tifffile.imwrite(f"{p}_unet_cos.tif", cos_np.astype("float32"), compression="lzw")
            tifffile.imwrite(f"{p}_unet_mag.tif", mag_np.astype("float32"), compression="lzw")
            tifffile.imwrite(f"{p}_unet_dir.tif", dir_np.astype("float32"), compression="lzw")

        # Cosine output (channel 0) is the main intensity target for the fit.
        image = torch.clamp(pred_unet[:, 0:1], 0.0, 1.0)
        # Magnitude branch (channel 1) encodes gradient magnitude; kept for period-sum loss.
        unet_mag_img = torch.clamp(pred_unet[:, 1:2], 0.0, 1.0)
        # Direction branch (channel 2) encodes 0.5 + 0.5*cos(2*theta); we keep it
        # for an auxiliary directional loss in sample space.
        unet_dir_img = torch.clamp(pred_unet[:, 2:3], 0.0, 1.0)
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
        if unet_dir_img is not None:
            unet_dir_img = F.interpolate(
                unet_dir_img,
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
        lr=lr,
    )
 
    total_stage1 = max(0, int(steps_stage1))
    total_stage2 = max(0, int(steps_stage2))
    total_stage3 = max(0, int(steps))
 
    def _gaussian_mask(stage: int, stage_progress: float) -> torch.Tensor | None:
        """
        Generate a Gaussian mask in image space (1,1,H,W) for the given stage
        and normalized progress value, or None when full-image loss is used
        (no Gaussian).
 
        Args:
            stage:
                1,2: global stages without Gaussian masking (always return None).
                3+: masked stages using the progressive Gaussian schedule.
            stage_progress:
                Float in [0,1] indicating normalized progress within the current
                stage. 0 = stage start, 1 = stage end.
        """
        # Stages 1 and 2 use global optimization without a Gaussian mask.
        if stage != 3:
            return None
 
        # Stage 3: grow sigma from sigma_min to sigma_max over the first 90% of
        # the stage, then disable the Gaussian mask (equivalent to full-image
        # loss over the valid region).
        stage_progress = float(max(0.0, min(1.0, stage_progress)))
        if stage_progress >= 0.9:
            return None
 
        # Map progress in [0,0.9] to [0,1] with quadratic ramp for slower
        # initial growth.
        frac = stage_progress / 0.9
        t = max(0.0, min(1.0, frac * frac))
        sigma = sigma_min + (sigma_max - sigma_min) * t
        if abs(sigma - sigma_min) < 1e-8:
            return gauss_min_img
 
        return torch.exp(-0.5 * window_r2 / (sigma * sigma))

    def _to_uint8(arr: "np.ndarray") -> "np.ndarray":
        """
        Convert a float or integer image array to uint8 [0,255].

        - For float arrays: per-image min/max normalization to [0,1] then *255.
        - For integer arrays: pass through if already uint8, otherwise scale by
          the dtype max to fit into [0,255].
        """
        import numpy as np

        if arr.dtype == np.uint8:
            return arr

        if np.issubdtype(arr.dtype, np.floating):
            vmin = float(arr.min())
            vmax = float(arr.max())
            if vmax > vmin:
                norm = (arr - vmin) / (vmax - vmin)
            else:
                norm = np.zeros_like(arr, dtype="float32")
            return (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")

        if np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            if info.max > 0:
                norm = arr.astype("float32") / float(info.max)
            else:
                norm = np.zeros_like(arr, dtype="float32")
            return (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")

        # Fallback: convert via float path.
        arr_f = arr.astype("float32")
        vmin = float(arr_f.min())
        vmax = float(arr_f.max())
        if vmax > vmin:
            norm = (arr_f - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(arr_f, dtype="float32")
        return (np.clip(norm, 0.0, 1.0) * 255.0).astype("uint8")
 
    def _draw_grid_vis(scale_factor: int = 4) -> "np.ndarray":
        """
        Draw the coarse sample-space grid mapped into (an upscaled) image space.
 
        We render points at each coarse grid corner and lines between neighbors.
        By default this is drawn on top of an upscaled version of the source
        image; in video mode we draw on a black background instead.
        Vertical lines whose coarse x-position coincides with cosine peaks in
        the ground-truth design are highlighted in a different color.
        """
        import numpy as np
        import cv2
 
        with torch.no_grad():
            sf = max(1, int(scale_factor))
            h_vis = int(h_img * sf)
            w_vis = int(w_img * sf)
 
            if for_video:
                # Plain black background for video overlays.
                bg_color = np.zeros((h_vis, w_vis, 3), dtype="uint8")
            else:
                # Base grayscale background from the source image, but rendered
                # only in the central half in each dimension of the visualization
                # frame. The outer regions remain black so grid points that map
                # outside the image but still inside the frame are visible.
                img_np = image[0, 0].detach().cpu().numpy()
                if img_np.max() > img_np.min():
                    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                else:
                    img_norm = img_np
                img_u8 = (img_norm * 255.0).astype("uint8")
 
                # Create black canvas at full visualization size.
                bg_color = np.zeros((h_vis, w_vis, 3), dtype="uint8")
 
                # Resize image to occupy the central half in each dimension.
                w_im = max(1, w_vis // 2)
                h_im = max(1, h_vis // 2)
                img_resized = cv2.resize(img_u8, (w_im, h_im), interpolation=cv2.INTER_LINEAR)
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
 
                # Paste resized image into centered rectangle.
                x0 = (w_vis - w_im) // 2
                y0 = (h_vis - h_im) // 2
                x1 = x0 + w_im
                y1 = y0 + h_im
                bg_color[y0:y1, x0:x1, :] = img_resized
 
            # Coarse coordinates in sample space (u,v).
            coords = model.base_grid + model.offset  # (1,2,gh,gw)
            u = coords[:, 0:1]
            v = coords[:, 1:2]
 
            # Apply the same global x-only scale-then-rotation as in _build_sampling_grid.
            x_norm, y_norm = model._apply_global_transform(u, v)
 
            # Map normalized coords so that the *image domain* x_norm,y_norm ∈ [-1,1]
            # occupies only the central half of the visualization frame in each
            # dimension, while points farther out (|x_norm|,|y_norm| > 1) are
            # still drawn towards the outer frame.
            #
            # Mapping:
            #   x_norm ∈ [-1,1] -> x_vis ∈ [0.25, 0.75] (central half)
            # extended to:
            #   x_norm ∈ [-2,2] -> x_vis ∈ [0.0, 1.0] (full frame)
            #
            # This keeps the grid scale consistent with the shrunken image in
            # the center, but still shows grid points that lie outside the image
            # domain within the overall visualization frame.
            x_pix = (0.5 + 0.25 * x_norm) * (w_vis - 1)
            y_pix = (0.5 + 0.25 * y_norm) * (h_vis - 1)
 
            x_pix = x_pix[0, 0].detach().cpu().numpy().astype(np.float32)
            y_pix = y_pix[0, 0].detach().cpu().numpy().astype(np.float32)
 
            gh, gw = x_pix.shape
 
            def in_bounds(px: int, py: int) -> bool:
                return 0 <= px < w_vis and 0 <= py < h_vis
 
            # Draw corner points.
            for iy in range(gh):
                for ix in range(gw):
                    px = int(round(float(x_pix[iy, ix])))
                    py = int(round(float(y_pix[iy, ix])))
                    if in_bounds(px, py):
                        cv2.circle(bg_color, (px, py), 1, (0, 255, 0), -1)
 
            # Draw horizontal lines (all in red).
            for iy in range(gh):
                for ix in range(gw - 1):
                    x0 = int(round(float(x_pix[iy, ix])))
                    y0 = int(round(float(y_pix[iy, ix])))
                    x1 = int(round(float(x_pix[iy, ix + 1])))
                    y1 = int(round(float(y_pix[iy, ix + 1])))
                    if in_bounds(x0, y0) and in_bounds(x1, y1):
                        cv2.line(bg_color, (x0, y0), (x1, y1), (0, 0, 255), 1)
 
            # Determine "cos-peak" columns in the coarse canonical grid based on
            # the ground-truth cosine design. The cosine target is defined over
            # sample-space x in [-1,1], with `cosine_periods` periods across the
            # width. Peaks (cos=1) occur at normalized positions
            # x_norm_k = 2 * k / cosine_periods - 1, mapped to coarse u.
            base_u_row = model.base_grid[0, 0, 0].detach().cpu().numpy()  # (gw,)
            peak_cols: set[int] = set()
            periods_int = max(1, int(round(float(cosine_periods))))
            for k in range(periods_int + 1):
                x_norm_k = 2.0 * float(k) / float(periods_int) - 1.0
                ix_peak = int(np.argmin(np.abs(base_u_row - x_norm_k)))
                peak_cols.add(ix_peak)
 
            # Draw vertical lines: highlight peak columns in red.
            for iy in range(gh - 1):
                for ix in range(gw):
                    x0 = int(round(float(x_pix[iy, ix])))
                    y0 = int(round(float(y_pix[iy, ix])))
                    x1 = int(round(float(x_pix[iy + 1, ix])))
                    y1 = int(round(float(y_pix[iy + 1, ix])))
                    if not (in_bounds(x0, y0) and in_bounds(x1, y1)):
                        continue
                    if ix in peak_cols:
                        # Cos-peak line: draw in red and slightly thicker.
                        cv2.line(bg_color, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    else:
                        cv2.line(bg_color, (x0, y0), (x1, y1), (0, 0, 255), 1)
 
            return bg_color
     
    def _save_snapshot(
        stage: int,
        step_stage: int,
        total_stage_steps: int,
        global_step_idx: int,
    ) -> None:
        """
        Save a snapshot for a given (stage, step) and global step index.

        The Gaussian mask schedule is driven directly from `stage` and the
        normalized `stage_progress = step_stage / (total_stage_steps-1)`,
        matching the training loop exactly (no reconstruction from the
        global step index).
        """
        if snapshot is None or snapshot <= 0 or output_prefix is None:
            return
        with torch.no_grad():
            # Native-resolution prediction and diff in sample space.
            pred_hr = model(image).clamp(0.0, 1.0)
            target_mod_hr = _modulated_target()
            diff_hr = pred_hr - target_mod_hr
 
            pred_out = pred_hr
            diff_out = diff_hr
            modgt_out = target_mod_hr
            if eff_output_scale is not None and eff_output_scale > 1:
                pred_out = F.interpolate(
                    pred_out,
                    scale_factor=eff_output_scale,
                    mode="bicubic",
                    align_corners=True,
                )
                modgt_out = F.interpolate(
                    modgt_out,
                    scale_factor=eff_output_scale,
                    mode="bicubic",
                    align_corners=True,
                )
 
            pred_np = pred_out.cpu().squeeze(0).squeeze(0).numpy()
            diff_np = diff_out.cpu().squeeze(0).squeeze(0).numpy()
            modgt_np = modgt_out.cpu().squeeze(0).squeeze(0).numpy()
 
            # Warped Gaussian weight mask and direction maps in sample space, for debugging.
            mask_np = None
            grid_vis_np = None
            dir_model_np = None
            dir_unet_np = None
            gradmag_vis_np = None
            gradmag_raw_np = None
            if dbg:
                grid_dbg = model._build_sampling_grid()
 
                # Exact mask schedule for this snapshot step using the same
                # normalized progress convention as in training.
                if total_stage_steps > 0:
                    stage_progress = float(step_stage) / float(max(total_stage_steps - 1, 1))
                else:
                    stage_progress = 0.0
                gauss_dbg = _gaussian_mask(stage, stage_progress)
                if gauss_dbg is None:
                    # Full-image loss: visualize as a constant 1 map in sample space.
                    mask_hr = torch.ones(
                        (1, 1, hr, wr),
                        device=torch_device,
                        dtype=torch.float32,
                    )
                else:
                    mask_hr = F.grid_sample(
                        gauss_dbg,
                        grid_dbg,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )
 
                if eff_output_scale is not None and eff_output_scale > 1:
                    mask_hr = F.interpolate(
                        mask_hr,
                        scale_factor=eff_output_scale,
                        mode="bicubic",
                        align_corners=True,
                    )
                mask_np = mask_hr.cpu().squeeze(0).squeeze(0).numpy()
 
                # Direction maps (model vs UNet) in sample space.
                if unet_dir_img is not None:
                    dir_model_hr, dir_unet_hr = _direction_maps(grid_dbg)
                    if dir_model_hr is not None and dir_unet_hr is not None:
                        dir_model_vis = dir_model_hr
                        dir_unet_vis = dir_unet_hr
                        if eff_output_scale is not None and eff_output_scale > 1:
                            dir_model_vis = F.interpolate(
                                dir_model_vis,
                                scale_factor=eff_output_scale,
                                mode="bicubic",
                                align_corners=True,
                            )
                            dir_unet_vis = F.interpolate(
                                dir_unet_vis,
                                scale_factor=eff_output_scale,
                                mode="bicubic",
                                align_corners=True,
                            )
                        dir_model_np = dir_model_vis.cpu().squeeze(0).squeeze(0).numpy()
                        dir_unet_np = dir_unet_vis.cpu().squeeze(0).squeeze(0).numpy()
 
                # Gradient-magnitude visualizations in sample space:
                # - raw resampled UNet magnitude (no period-averaging),
                # - period-sum map (reusing the same core as the loss).
                if unet_mag_img is not None:
                    mag_hr_dbg = F.grid_sample(
                        unet_mag_img,
                        grid_dbg,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=True,
                    )  # (1,1,hr,wr)
 
                    # Per-sample distance along the horizontal index direction for debug grid.
                    dist_x_dbg = _grid_segment_length_x(grid_dbg)
 
                    # Period-sum visualization using the same core as the loss (also gives samples_per).
                    (
                        sum_period_scaled_dbg,
                        samples_per_dbg,
                        max_cols_dbg,
                        hh_gm,
                        ww_gm,
                        _,
                    ) = _gradmag_period_core(mag_hr_dbg, dist_x_dbg, img_downscale_factor)
 
                    # Raw magnitude in sample space, scaled by 0.5 * samples_per
                    # (the size of the summed sub-dimension) for comparability with
                    # the period-sum map, then optionally upscaled for output.
                    gradmag_raw = mag_hr_dbg
                    if samples_per_dbg > 0:
                        scale_raw = 0.5 * float(samples_per_dbg)
                        gradmag_raw = gradmag_raw * scale_raw
                    if eff_output_scale is not None and eff_output_scale > 1:
                        gradmag_raw = F.interpolate(
                            gradmag_raw,
                            scale_factor=eff_output_scale,
                            mode="bicubic",
                            align_corners=True,
                        )
                    gradmag_raw_np = gradmag_raw.cpu().squeeze(0).squeeze(0).numpy()
 
                    if sum_period_scaled_dbg is not None and samples_per_dbg > 0 and max_cols_dbg > 0:
                        # Broadcast per-period values back to samples within each period.
                        # sum_broadcast_dbg = sum_period_scaled_dbg.repeat_interleave(
                        #     samples_per_dbg, dim=-1
                        # )  # (1,1,hh,max_cols)
                        # gradmag_vis = mag_hr_dbg.new_zeros(1, 1, hh_gm, ww_gm)
                        # gradmag_vis[:, :, :, :max_cols_dbg] = sum_broadcast_dbg
                        # if eff_output_scale is not None and eff_output_scale > 1:
                        #     gradmag_vis = F.interpolate(
                        #         gradmag_vis,
                        #         scale_factor=eff_output_scale,
                        #         mode="bicubic",
                        #         align_corners=True,
                        #     )
                        # gradmag_vis_np = gradmag_vis.cpu().squeeze(0).squeeze(0).numpy()
                        gradmag_vis_np = sum_period_scaled_dbg
 
                # Grid visualization: heavily upscaled relative to output_scale,
                # always using the same large size; in video mode the background
                # is black instead of the image.
                base_scale = output_scale if output_scale is not None else 4
                vis_scale = base_scale * 2
                grid_vis_np = _draw_grid_vis(scale_factor=vis_scale)
 
        p = Path(output_prefix)
 
        # In video mode, save the background image once at step 0 for compositing.
        if for_video and global_step_idx == 0:
 
            img_np = image[0, 0].detach().cpu().numpy()
            if img_np.max() > img_np.min():
                img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                img_norm = img_np
            bg_np = _to_uint8(img_norm)
            bg_path = f"{p}_bg.tif"
            tifffile.imwrite(bg_path, bg_np, compression="lzw")
 
        recon_path = f"{p}_arecon_step{global_step_idx:06d}.tif"
        tifffile.imwrite(recon_path, _to_uint8(pred_np), compression="lzw")
        modgt_path = f"{p}_modgt_step{global_step_idx:06d}.tif"
        tifffile.imwrite(modgt_path, _to_uint8(modgt_np), compression="lzw")
        if dbg:
            if mask_np is not None:
                if for_video:
                    mask_u8 = (np.clip(mask_np, 0.0, 1.0) * 255.0).astype("uint8")
                    mask_path = f"{p}_mask_step{global_step_idx:06d}.jpg"
                    cv2.imwrite(mask_path, mask_u8)
                else:
                    mask_path = f"{p}_mask_step{global_step_idx:06d}.jpg"
                    # tifffile.imwrite(mask_path, _to_uint8(mask_np), compression="lzw")
                    cv2.imwrite(mask_path, _to_uint8(mask_np))
            # Save diff as |diff|, with 0 -> black and max |diff| -> white.
            diff_abs = np.abs(diff_np)
            maxv = float(diff_abs.max())
            if maxv > 0.0:
                diff_u8 = (np.clip(diff_abs / maxv, 0.0, 1.0) * 255.0).astype("uint8")
            else:
                diff_u8 = np.zeros_like(diff_abs, dtype="uint8")
            diff_path = f"{p}_diff_step{global_step_idx:06d}.tif"
            tifffile.imwrite(diff_path, diff_u8, compression="lzw")
            if grid_vis_np is not None:
                grid_path = f"{p}_grid_step{global_step_idx:06d}.jpg"
                # tifffile.imwrite(grid_path, grid_vis_np, compression="lzw")
                cv2.imwrite(grid_path, np.flip(grid_vis_np, -1))
            # Save direction maps (model vs UNet) if available.
            if dir_model_np is not None and dir_unet_np is not None:
                dir_model_path = f"{p}_dir_model_step{global_step_idx:06d}.tif"
                dir_unet_path = f"{p}_dir_unet_step{global_step_idx:06d}.tif"
                tifffile.imwrite(dir_model_path, _to_uint8(dir_model_np), compression="lzw")
                tifffile.imwrite(dir_unet_path, _to_uint8(dir_unet_np), compression="lzw")
            # Save raw gradient-magnitude (resampled UNet mag) as 8-bit JPG.
            if gradmag_raw_np is not None:
                graw = np.clip(gradmag_raw_np, 0.0, 1.0) * 255.0
                graw_u8 = graw.astype("uint8")
                graw_path = f"{p}_gmag_raw_step{global_step_idx:06d}.jpg"
                cv2.imwrite(graw_path, graw_u8)
            # Save gradient-magnitude period-sum visualization as 8-bit JPG.
            print(type(gradmag_vis_np))
            if gradmag_vis_np is not None:
                # gradmag_vis_np may be either a NumPy array (from the upsampled
                # visualization) or a Torch tensor (e.g. sum_period_scaled_dbg on
                # CUDA for debugging). Normalize such that 0.0 -> 0 and 1.0 -> 127.
                if isinstance(gradmag_vis_np, torch.Tensor):
                    gradmag_vis_arr = gradmag_vis_np.detach().cpu().numpy()
                else:
                    gradmag_vis_arr = gradmag_vis_np
                # gmag = np.clip(gradmag_vis_arr, 0.0, 1.0) * 127.0
                # gmag_u8 = gmag.astype("uint8")
                # gmag_u8 = np.require(gmag_u8, requirements=['C']).copy()
                gmag_path = f"{p}_gmag_step{global_step_idx:06d}.tif"
                # print("write ", gmag_path, gmag_u8.shape, gmag_u8.strides, gmag_u8.dtype)
                cv2.imwrite(gmag_path, gradmag_vis_arr.squeeze(0).squeeze(0))
 
    def _smoothness_reg(mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Smoothness penalty on the *offset* field in coarse grid index space.
    
        We regularize the learnable offsets along grid indices:
        - x index (horizontal neighbors): ||Δ_off||^2 for (u_offset, v_offset),
        - y index (vertical neighbors):   ||Δ_off||^2 for (u_offset, v_offset),
    
        i.e. both u- and v-offset components are treated the same in both
        directions; the split into smooth_x / smooth_y is only by *index
        direction*, not by coordinate component.
    
        If an image-space mask is provided (shape (1,1,hr,wr)), we scale both
        directional penalties by the average mask value downsampled to the
        coarse grid.
        """
        off = model.offset  # (1,2,gh,gw)  2 = (u_offset, v_offset)
        _, _, gh, gw = off.shape
    
        # First-order differences along x index: o[..., i+1] - o[..., i].
        # This is a 2D vector (Δu, Δv); we regularize its squared L2 norm.
        if gw >= 2:
            dx = off[:, :, :, 1:] - off[:, :, :, :-1]  # (1,2,gh,gw-1)
            dx_sq = (dx * dx).sum(dim=1, keepdim=True)  # sum over (u,v) components
            smooth_x = dx_sq
        else:
            smooth_x = torch.zeros((), device=off.device, dtype=off.dtype)
    
        # First-order differences along y index: o[..., j+1, :] - o[..., j, :].
        # Again, treat (u_offset, v_offset) symmetrically via ||Δ_off||^2.
        if gh >= 2:
            dy = off[:, :, 1:, :] - off[:, :, :-1, :]  # (1,2,gh-1,gw)
            dy_sq = (dy * dy).sum(dim=1, keepdim=True)
            smooth_y = dy_sq
        else:
            smooth_y = torch.zeros((), device=off.device, dtype=off.dtype)
    
        if mask is None:
            return smooth_x.mean(), smooth_y.mean()
    
        # Downsample image-space mask to coarse grid
        with torch.no_grad():
            m = F.interpolate(
                mask,
                size=(gh, gw),
                mode="bilinear",
                align_corners=True,
            )*0.5+0.5
            # scale = mask_coarse.mean()
        return (smooth_x * m[...,:-1]).mean(), (smooth_y * m[...,:-1,:]).mean()
 
    def _mod_smooth_reg(mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Smoothness penalty on modulation parameters (amp, bias) on the coarse grid.
 
        We only regularize variation along y for modulation. If an image-space
        mask is provided, we scale the penalty by the average mask value
        downsampled to the modulation grid.
        """
        # Stack amp and bias so both are regularized consistently.
        mods = torch.cat([model.amp_coarse, model.bias_coarse], dim=1)  # (1,2,gh,gw)
 
        # First-order differences along y in coarse grid index space.
        dy = mods[:, :, 1:, :] - mods[:, :, :-1, :]   # (N,2,gh-1,gw)
 
        if dy.numel() == 0:
            base = torch.zeros((), device=mods.device, dtype=mods.dtype)
        else:
            base = (dy * dy).mean()
        if mask is None:
            return base
        gh, gw = mods.shape[2], mods.shape[3]
        with torch.no_grad():
            mask_coarse = F.interpolate(
                mask,
                size=(gh, gw),
                mode="bilinear",
                align_corners=True,
            )
            scale = mask_coarse.mean()
        return base * scale
    
    def _step_reg(mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Regularization on coarse *rotated* coords in target-image space.
 
        For each step along the cosine grid we consider the distance between
        neighboring sample positions after mapping into image space.
 
        We compute:
        - horizontal neighbor distances (along coarse x),
        - vertical neighbor distances (along coarse y).
 
        Horizontally:
        - enforce each distance to be at least 0.1 * the average horizontal distance.
 
        Vertically:
        - enforce each distance to be at least 0.5 * the average vertical distance
          and encourage distances to be close to that average.
 
        If an image-space mask is provided, we scale the regularizer by the
        average mask value downsampled to the coarse grid.
        """
        coords = model.base_grid + model.offset  # (1,2,gh,gw)
        u = coords[:, 0:1]
        v = coords[:, 1:2]
 
        # Apply same x-only scale-then-rotation as in _build_sampling_grid, but on the coarse grid.
        x_norm, y_norm = model._apply_global_transform(u, v)
 
        # Map normalized coords to pixel coordinates of the target image.
        x_pix = (x_norm + 1.0) * 0.5 * float(max(1, w_img - 1))
        y_pix = (y_norm + 1.0) * 0.5 * float(max(1, h_img - 1))
 
        # Horizontal neighbor distances (steps along coarse x index).
        dx_h = x_pix[:, :, :, 1:] - x_pix[:, :, :, :-1]
        dy_h = y_pix[:, :, :, 1:] - y_pix[:, :, :, :-1]
        dist_h = torch.sqrt(dx_h * dx_h + dy_h * dy_h + 1e-12)
 
        # Vertical neighbor distances (steps along coarse y index).
        dx_v = x_pix[:, :, 1:, :] - x_pix[:, :, :-1, :]
        dy_v = y_pix[:, :, 1:, :] - y_pix[:, :, :-1, :]
        dist_v = torch.sqrt(dx_v * dx_v + dy_v * dy_v + 1e-12)
 
        # Average horizontal & vertical distance in image-space units.
        avg_h = dist_h.mean()
        avg_h_det = avg_h.detach()
 
        avg_v = dist_v.mean()
        avg_v_det = avg_v.detach()
 
        # Horizontal: enforce each distance to be at least 0.1 * avg horizontal.
        min_h = 0.1 * avg_h_det
        if float(min_h) <= 0.0:
            loss_h = torch.zeros((), device=coords.device, dtype=coords.dtype)
        else:
            shortfall_h = torch.clamp(min_h - dist_h, min=0.0) / min_h
            loss_h = (shortfall_h * shortfall_h).mean()
 
        # Vertical: enforce each distance to be at least 0.5 * avg vertical.
        # min_v = 0.5 * avg_v_det
        # if float(min_v) <= 0.0:
        #     loss_v = torch.zeros((), device=coords.device, dtype=coords.dtype)
        # else:
        #     shortfall_v = torch.clamp(min_v - dist_v, min=0.0) / min_v
        #     loss_v = (shortfall_v * shortfall_v).mean()
 
        # Vertical: encourage each distance to be close to avg distance.
        target_v = avg_v_det
        diff_v = dist_v - target_v
        loss_v_avg = (diff_v * diff_v).mean()
 
        base = 1 * loss_h + 0.1 * loss_v_avg
        if mask is None:
            return base
        # Downsample image-space mask to coarse grid and scale by its mean.
        gh, gw = coords.shape[2], coords.shape[3]
        with torch.no_grad():
            mask_coarse = F.interpolate(
                mask,
                size=(gh, gw),
                mode="bilinear",
                align_corners=True,
            )
            scale = mask_coarse.mean()
        return base * scale
     
    def _angle_symmetry_reg(mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Angle-symmetry regularizer on coarse coords in image space.

        For each horizontal edge between neighboring coarse grid columns, we
        compare the horizontal edge direction to the local vertical direction
        (along coarse y). The loss penalizes deviations from orthogonality:

            L = mean( cos(theta)^2 )

        where theta is the angle between the horizontal edge and the vertical
        direction in image space. This encourages the "rungs" that connect
        neighboring vertical lines to be straight relative to the vertical
        grid lines, while still allowing bending along y.
        """
        coords = model.base_grid + model.offset  # (1,2,gh,gw)
        u = coords[:, 0:1]
        v = coords[:, 1:2]

        # Apply same x-only scale-then-rotation as in _build_sampling_grid, but on the coarse grid.
        x_norm, y_norm = model._apply_global_transform(u, v)

        # Map normalized coords to pixel coordinates.
        x_pix = (x_norm + 1.0) * 0.5 * float(max(1, w_img - 1))
        y_pix = (y_norm + 1.0) * 0.5 * float(max(1, h_img - 1))

        _, _, gh, gw = x_pix.shape
        if gh < 2 or gw < 2:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)

        # Horizontal edge vectors between neighboring columns (left -> right).
        dx_h = x_pix[:, :, :, 1:] - x_pix[:, :, :, :-1]   # (1,1,gh,gw-1)
        dy_h = y_pix[:, :, :, 1:] - y_pix[:, :, :, :-1]   # (1,1,gh,gw-1)

        # Vertical edge vectors between neighboring rows (top -> bottom).
        dx_v = x_pix[:, :, 1:, :] - x_pix[:, :, :-1, :]   # (1,1,gh-1,gw)
        dy_v = y_pix[:, :, 1:, :] - y_pix[:, :, :-1, :]   # (1,1,gh-1,gw)

        # We only compare where both directions are defined:
        # rows 0..gh-2 for vertical, and cols 0..gw-2 for horizontal.
        hvx = dx_h[:, :, 0:gh-1, 0:gw-1]  # (1,1,gh-1,gw-1)
        hvy = dy_h[:, :, 0:gh-1, 0:gw-1]
        vvx = dx_v[:, :, :, 0:gw-1]       # (1,1,gh-1,gw-1)
        vvy = dy_v[:, :, :, 0:gw-1]

        # Cosine of angle between horizontal and vertical directions.
        eps = 1e-12
        h_norm = torch.sqrt(hvx * hvx + hvy * hvy + eps)
        v_norm = torch.sqrt(vvx * vvx + vvy * vvy + eps)
        dot = hvx * vvx + hvy * vvy
        cos_theta = dot / (h_norm * v_norm + eps)

        # Penalize squared cosine -> encourages orthogonality.
        base = (cos_theta * cos_theta).mean()

        if mask is None:
            return base

        # As in other regularizers, scale by the mean value of the mask
        # downsampled to the coarse grid.
        with torch.no_grad():
            mask_coarse = F.interpolate(
                mask,
                size=(gh, gw),
                mode="bilinear",
                align_corners=True,
            )
            scale = mask_coarse.mean()
        return base * scale
     
    def _quad_triangle_reg(mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Quad-based triangle-area regularizer in image space.

        For each quad in the coarse grid, we form four corner-based triangles
        using the direct neighboring quad corners and compute signed areas via
        the 2D cross product.

        We:
        - penalize triangle area magnitude being less than 1/4 of the average,
        - strongly penalize negative (flipped) triangle areas.
        """
        coords = model.base_grid + model.offset  # (1,2,gh,gw)
        u = coords[:, 0:1]
        v = coords[:, 1:2]

        # Apply same x-only scale-then-rotation as in _build_sampling_grid, but on the coarse grid.
        x_norm, y_norm = model._apply_global_transform(u, v)

        # Map normalized coords to pixel coordinates of the target image.
        x_pix = (x_norm + 1.0) * 0.5 * float(max(1, w_img - 1))
        y_pix = (y_norm + 1.0) * 0.5 * float(max(1, h_img - 1))

        # Quad corners: p00 (y,x), p01 (y,x+1), p11 (y+1,x+1), p10 (y+1,x).
        px00 = x_pix[:, :, :-1, :-1]
        py00 = y_pix[:, :, :-1, :-1]
        px01 = x_pix[:, :, :-1, 1:]
        py01 = y_pix[:, :, :-1, 1:]
        px11 = x_pix[:, :, 1:, 1:]
        py11 = y_pix[:, :, 1:, 1:]
        px10 = x_pix[:, :, 1:, :-1]
        py10 = y_pix[:, :, 1:, :-1]

        # Four corner-based triangles per quad, signed area via cross product.

        # Triangle at p00: (p00, p01, p10)
        ax0 = px01 - px00
        ay0 = py01 - py00
        bx0 = px10 - px00
        by0 = py10 - py00
        A0 = 0.5 * (ax0 * by0 - ay0 * bx0)

        # Triangle at p01: (p01, p11, p00)
        ax1 = px11 - px01
        ay1 = py11 - py01
        bx1 = px00 - px01
        by1 = py00 - py01
        A1 = 0.5 * (ax1 * by1 - ay1 * bx1)

        # Triangle at p11: (p11, p10, p01)
        ax2 = px10 - px11
        ay2 = py10 - py11
        bx2 = px01 - px11
        by2 = py01 - py11
        A2 = 0.5 * (ax2 * by2 - ay2 * bx2)

        # Triangle at p10: (p10, p00, p11)
        ax3 = px00 - px10
        ay3 = py00 - py10
        bx3 = px11 - px10
        by3 = py11 - py10
        A3 = 0.5 * (ax3 * by3 - ay3 * bx3)

        areas = torch.stack([A0, A1, A2, A3], dim=0)  # (4,1,gh-1,gw-1)
        areas_abs = areas.abs()
        avg_area_abs = areas_abs.mean().detach()

        if float(avg_area_abs) <= 0.0:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)

        # Magnitude: piecewise penalty on |A| relative to avg|A|:
        # - 0 for |A| >= avg|A|,
        # - linear from |A| = avg|A| down to |A| = 0.25 * avg|A|,
        # - linear + quadratic for |A| < 0.25 * avg|A|.
        #
        # Implemented without masks/conditionals, using clamp so everything
        # is expressed as smooth elementwise ops.
        A = 0.1*avg_area_abs
        A_quarter = 0.05 * avg_area_abs
        eps = 1e-12
 
        # Linear component, active for |A| < A and saturating at |A| <= 0.25*A.
        # 0 at |A| = A, ~1 at |A| = 0.25*A (before scaling).
        lin_raw = torch.clamp(A - areas, min=0.0, max=(A - A_quarter + eps))
        lin_term = lin_raw / (A - A_quarter + eps)
 
        # Quadratic extra below 0.25*A (0 above, grows as |A| goes to 0).
        low_def = torch.clamp(A_quarter - areas, min=0.0)
        quad_term = (low_def / (A_quarter + eps)) ** 2
 
        size_pen = lin_term + quad_term
        tri_size_loss = size_pen.mean()

        # Orientation: strongly penalize negative signed area.
        neg = torch.clamp(-areas, min=0.0) / (avg_area_abs + 1e-12)
        tri_neg_loss = (neg * neg).mean()

        # Return combined triangle-area loss; external lambda scales overall strength.
        base = tri_size_loss #+ 10.0 * tri_neg_loss
        if mask is None:
            return base
        gh, gw = coords.shape[2], coords.shape[3]
        with torch.no_grad():
            mask_coarse = F.interpolate(
                mask,
                size=(gh, gw),
                mode="bilinear",
                align_corners=True,
            )
            scale = mask_coarse.mean()
        return base * scale

    def _direction_maps(
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Compute direction encodings for model & UNet in sample space.

        Model direction is derived from the mapping (u,v) -> (x,y) represented
        by the sampling grid. Let J be the Jacobian of this mapping:

            J = [ dx/du  dx/dv ]
                [ dy/du  dy/dv ]

        The cosine phase varies along u, so in image space the phase field is
        φ(x,y) = k * u(x,y). Its gradient is ∇φ ∝ ∇u, and

            [du, dv]^T = J^{-1} [dx, dy]^T

        so the gradient of u in (x,y) is the first row of J^{-1}:

            ∇u = (∂u/∂x, ∂u/∂y) = row_0(J^{-1})

        We encode orientation as 0.5 + 0.5*cos(2*theta) where theta is the
        angle of ∇u, matching the UNet training target.

        Returns:
            dir_model:  (1,1,hr,wr) or None
            dir_unet:   (1,1,hr,wr) or None
        """
        if unet_dir_img is None:
            return None, None

        # grid: (1,hr,wr,2) in normalized image coords.
        x = grid[..., 0].unsqueeze(1)  # (1,1,hr,wr)
        y = grid[..., 1].unsqueeze(1)

        # Finite-difference Jacobian of (u,v) -> (x,y).
        # Treat width as u-direction and height as v-direction.
        xu = torch.zeros_like(x)
        xv = torch.zeros_like(x)
        yu = torch.zeros_like(y)
        yv = torch.zeros_like(y)

        # Forward differences along u (width).
        xu[:, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]
        yu[:, :, :, :-1] = y[:, :, :, 1:] - y[:, :, :, :-1]

        # Forward differences along v (height).
        xv[:, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
        yv[:, :, :-1, :] = y[:, :, 1:, :] - y[:, :, :-1, :]

        # Jacobian determinant.
        det = xu * yv - xv * yu
        eps = 1e-8
        det_safe = det + (det.abs() < eps).float() * eps

        # First row of J^{-1} gives gradient of u in image space: (du/dx, du/dy).
        ux = yv / det_safe
        uy = -xv / det_safe

        r2 = ux * ux + uy * uy + eps
        cos2theta = (ux * ux - uy * uy) / r2
        dir_model = 0.5 + 0.5 * cos2theta  # (1,1,hr,wr)

        # Warp UNet direction channel into sample space using the same grid.
        dir_unet_hr = F.grid_sample(
            unet_dir_img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return dir_model, dir_unet_hr

    def _directional_alignment_loss(
        grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Directional alignment loss between the mapped cosine axis and the UNet
        direction branch, encoded as 0.5 + 0.5*cos(2*theta) in sample space.
        """
        if unet_dir_img is None:
            return torch.zeros((), device=torch_device, dtype=torch.float32)
 
        dir_model, dir_unet_hr = _direction_maps(grid)
        if dir_model is None or dir_unet_hr is None:
            return torch.zeros((), device=torch_device, dtype=torch.float32)
 
        diff_dir = dir_model - dir_unet_hr
        return (diff_dir * diff_dir).mean()
 
    def _gradient_data_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Gradient matching term between the sampled image data and the *plain*
        cosine target in sample space.
 
        We penalize differences in forward x/y gradients:
 
            L = 0.5 * ( ||∂x pred - ∂x target||_2^2 + ||∂y pred - ∂y target||_2^2 )
 
        If a weight map is provided, we use it (averaged onto the gradient
        positions) as a spatial weighting for both directions.
        """
        # Forward differences along x (width).
        gx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        gx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
        diff_gx = gx_pred - gx_tgt
 
        # Forward differences along y (height).
        gy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        gy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
        diff_gy = gy_pred - gy_tgt
 
        if weight is None:
            loss_x = (diff_gx * diff_gx).mean()
            loss_y = (diff_gy * diff_gy).mean()
            return 0.5 * (loss_x + loss_y)
 
        # Average weights onto gradient locations.
        wx = torch.minimum(weight[:, :, :, 1:], weight[:, :, :, :-1])
        wy = torch.minimum(weight[:, :, 1:, :], weight[:, :, :-1, :])
 
        wsum_x = wx.sum()
        wsum_y = wy.sum()
 
        if wsum_x > 0:
            loss_x = (wx * (diff_gx * diff_gx)).sum() / wsum_x
        else:
            loss_x = (diff_gx * diff_gx).mean()
 
        if wsum_y > 0:
            loss_y = (wy * (diff_gy * diff_gy)).sum() / wsum_y
        else:
            loss_y = (diff_gy * diff_gy).mean()
 
        return 0.5 * (loss_x + loss_y)
 
    def _grid_segment_length_x(grid: torch.Tensor) -> torch.Tensor:
        """
        Per-sample distance along the horizontal index direction of the sampling grid.

        We compute segment lengths between neighbors along x (last axis) in image
        pixel space, then assign to each sample the average of its left/right
        neighbor segments (using only the available side at the edges).
        """
        # grid: (1, H, W, 2) in normalized image coordinates.
        x_norm = grid[..., 0].unsqueeze(1)  # (1,1,H,W)
        y_norm = grid[..., 1].unsqueeze(1)

        w_eff = float(max(1, w_img - 1))
        h_eff = float(max(1, h_img - 1))
        x_pix = (x_norm + 1.0) * 0.5 * w_eff
        y_pix = (y_norm + 1.0) * 0.5 * h_eff

        # Segment lengths between neighbors along x index.
        dx = x_pix[:, :, :, 1:] - x_pix[:, :, :, :-1]
        dy = y_pix[:, :, :, 1:] - y_pix[:, :, :, :-1]
        seg = torch.sqrt(dx * dx + dy * dy + 1e-12)  # (1,1,H,W-1)

        dist = torch.zeros_like(x_pix)
        if x_pix.shape[-1] == 1:
            # Degenerate case: single column, assign unit length.
            dist[:] = 1.0
            return dist

        # Interior: average of left/right segments.
        dist[:, :, :, 1:-1] = 0.5 * (seg[:, :, :, 1:] + seg[:, :, :, :-1])
        # Edges: only one neighboring segment.
        dist[:, :, :, 0] = seg[:, :, :, 0]
        dist[:, :, :, -1] = seg[:, :, :, -1]
        return dist

    def _gradmag_period_core(
        mag_hr: torch.Tensor,
        dist_x_hr: torch.Tensor,
        img_downscale_factor: float,
    ) -> tuple[torch.Tensor | None, int, int, int, int, int]:
        """
        Shared core for gradient-magnitude period handling with distance weighting.

        Each sample's magnitude is first weighted by the distance it covers along
        the horizontal index direction (dist_x_hr) before summing over periods.

        Args:
            mag_hr:     (1,1,H,W) magnitude sampled in sample space.
            dist_x_hr:  (1,1,H,W) per-sample distance along x (same shape as mag_hr).
            img_downscale_factor: image downscale factor used in fitting.

        Returns:
            sum_period_scaled: (1,1,H,periods) scaled period sums, or None if invalid.
            samples_per:       samples per period along x.
            max_cols:          number of valid columns (periods * samples_per).
            hh, ww:            height & width of mag_hr.
            periods_int:       integer number of periods.
        """
        _, _, hh, ww = mag_hr.shape
        periods_int = max(1, int(round(float(cosine_periods))))
        if ww < periods_int:
            return None, 0, 0, hh, ww, periods_int

        if dist_x_hr.shape != mag_hr.shape:
            raise ValueError(f"dist_x_hr shape {dist_x_hr.shape} != mag_hr shape {mag_hr.shape}")

        samples_per = ww // periods_int
        if samples_per <= 0:
            return None, 0, 0, hh, ww, periods_int

        max_cols = samples_per * periods_int
        mag_use = mag_hr[:, :, :, :max_cols]
        dist_use = dist_x_hr[:, :, :, :max_cols]
        weighted = mag_use * dist_use  # (1,1,hh,max_cols)

        weighted_reshaped = weighted.view(1, 1, hh, periods_int, samples_per)
        sum_period = weighted_reshaped.sum(dim=-1)  # (1,1,hh,periods_int)

        # Account for image downscale: each sample corresponds to 1/s^2 original
        # pixels, so the integral over a period should be ~1 (up to a global scale).
        m = float(img_downscale_factor) if img_downscale_factor is not None else 1.0
        sum_period_scaled = m * sum_period
        return sum_period_scaled, samples_per, max_cols, hh, ww, periods_int

    def _gradmag_period_loss(
        grid: torch.Tensor,
        img_downscale_factor: float,
        mask_sample: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Period-sum loss on sampled gradient magnitude in sample space.

        We sample the UNet magnitude channel (gradient magnitude) into sample
        space using the current grid, then, for each vertical row, group the
        x-dimension into cosine periods and enforce that the sum of magnitudes
        from peak to peak (one period) is close to 1.

        If a sample-space mask is provided (1,1,H,W), we reshape it in the same
        way as the magnitude (per row, per period, per-sample) and use the
        *minimum* mask value within each (row, period) group as its weight.
        """
        if unet_mag_img is None:
            return torch.zeros((), device=torch_device, dtype=torch.float32)

        # Sample UNet magnitude into sample space.
        mag_hr = F.grid_sample(
            unet_mag_img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (1,1,hr,wr)

        # Per-sample distance along the horizontal index direction of the sampling grid.
        dist_x_hr = _grid_segment_length_x(grid)

        sum_period_scaled, samples_per, max_cols, hh, ww, periods_int = _gradmag_period_core(
            mag_hr, dist_x_hr, img_downscale_factor
        )
        if sum_period_scaled is None:
            return torch.zeros((), device=torch_device, dtype=torch.float32)

        # Base per-(row,period) squared error (already scaled).
        err = (sum_period_scaled - 1.0) * (sum_period_scaled - 1.0)  # (1,1,hh,periods_int)

        if mask_sample is None:
            return err.mean()

        # Mask is already in sample space (same coords as mag_hr), e.g. weight_full.
        if mask_sample.shape[-2:] != (hh, ww):
            # Safety: fall back to unweighted if sizes mismatch.
            return err.mean()

        # Reshape mask exactly like magnitude: (N,1,hh,periods,samples_per).
        mask_use = mask_sample[:, :, :, :max_cols]
        mask_reshaped = mask_use.view(1, 1, hh, periods_int, samples_per)

        # For each (row,period), take the MIN mask over that interval to define
        # its weight.
        w_period, _ = mask_reshaped.min(dim=-1)  # (1,1,hh,periods_int)

        w_sum = w_period.sum()
        if w_sum <= 0:
            return err.mean()

        # Apply mask directly when forming the weighted error.
        return (err * w_period).sum() / w_sum

    # Optional initialization snapshot.
    if snapshot is not None and snapshot > 0 and output_prefix is not None:
        # Use stage 1, step 0 with a dummy total_stage_steps=1 so that the mask
        # schedule is well-defined (stage_progress = 0).
        _save_snapshot(
            stage=1,
            step_stage=0,
            total_stage_steps=max(total_stage1, 1),
            global_step_idx=0,
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
        # grad_mag and dir_unet default to 1.0 (enabled).
    }
 
    stage2_modifiers: dict[str, float] = {
        # Stage 2: refine coarse grid with full regularization; keep data/grad_data
        # disabled to match previous behavior (global, no Gaussian mask).
        "data": 0.0,
        "grad_data": 0.0,
        # other terms default to 1.0 (enabled).
    }
 
    stage3_modifiers: dict[str, float] = {
        # Stage 3: enable data and grad_data terms in addition to stage-2 regularization.
        # No explicit overrides: all lambda_global weights are used as-is.
    }
 
    def _need_term(name: str, stage_modifiers: dict[str, float]) -> float:
        """Return effective weight for a term; 0.0 means 'skip this term'."""
        base = float(lambda_global.get(name, 0.0))
        mod = float(stage_modifiers.get(name, 1.0))
        return base * mod
 
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
 
        # Prediction in sample space.
        pred = model(image)
 
        # Build validity mask and Gaussian loss mask in sample space (no grad).
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
 
            if stage in (1, 2):
                # Stages 1 and 2: use only in-bounds validity; ignore Gaussian mask.
                weight_full = valid
            else:
                gauss = _gaussian_mask(stage=stage, stage_progress=stage_progress)
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
            dir_loss = _directional_alignment_loss(grid)
            total_loss = total_loss + w_dir * dir_loss
        else:
            dir_loss = torch.zeros((), device=device, dtype=dtype)
        terms["dir_unet"] = dir_loss
 
        # Gradient-magnitude period-sum loss.
        w_grad_mag = _need_term("grad_mag", stage_modifiers)
        if w_grad_mag != 0.0:
            # Stages 1 and 2: use only the valid region for period-sum weighting.
            # Stage 3: use the full scheduled mask in sample space.
            if stage in (1, 2):
                mask_for_gradmag = valid
            else:
                mask_for_gradmag = weight_full
            gradmag_loss = _gradmag_period_loss(grid, img_downscale_factor, mask_for_gradmag)
            total_loss = total_loss + w_grad_mag * gradmag_loss
        else:
            gradmag_loss = torch.zeros((), device=device, dtype=dtype)
        terms["grad_mag"] = gradmag_loss
 
        # Offset smoothness.
        need_sx = _need_term("smooth_x", stage_modifiers) != 0.0
        need_sy = _need_term("smooth_y", stage_modifiers) != 0.0
        if need_sx or need_sy:
            smooth_x_val, smooth_y_val = _smoothness_reg(valid)
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
            step_reg = _step_reg()
            total_loss = total_loss + w_step * step_reg
        else:
            step_reg = torch.zeros((), device=device, dtype=dtype)
        terms["step"] = step_reg
 
        # Modulation smoothness.
        w_mod_smooth = _need_term("mod_smooth", stage_modifiers)
        if w_mod_smooth != 0.0:
            mod_smooth = _mod_smooth_reg()
            total_loss = total_loss + w_mod_smooth * mod_smooth
        else:
            mod_smooth = torch.zeros((), device=device, dtype=dtype)
        terms["mod_smooth"] = mod_smooth
 
        # Triangle-area regularizer.
        w_quad = _need_term("quad_tri", stage_modifiers)
        if w_quad != 0.0 and lambda_xygrad > 0.0:
            quad_tri_reg = _quad_triangle_reg()
            total_loss = total_loss + w_quad * quad_tri_reg
        else:
            quad_tri_reg = torch.zeros((), device=device, dtype=dtype)
        terms["quad_tri"] = quad_tri_reg

        # Angle-symmetry regularizer between horizontal connections and vertical lines.
        w_angle = _need_term("angle_sym", stage_modifiers)
        if w_angle != 0.0:
            angle_reg = _angle_symmetry_reg()
            total_loss = total_loss + w_angle * angle_reg
        else:
            angle_reg = torch.zeros((), device=device, dtype=dtype)
        terms["angle_sym"] = angle_reg

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
                msg = (
                    f"stage{stage}(step {step+1}/{total_steps}): "
                    f"loss={loss.item():.6f}, data={data_loss.item():.6f}, "
                    f"grad={grad_loss.item():.6f}, gmag={gradmag_loss.item():.6f}, "
                    f"dir={dir_loss.item():.6f}"
                )
                if stage >= 2:
                    smooth_x = terms["smooth_x"]
                    smooth_y = terms["smooth_y"]
                    step_reg = terms["step"]
                    quad_tri_reg = terms["quad_tri"]
                    msg += (
                        f", sx_smooth={smooth_x.item():.6f}, sy_smooth={smooth_y.item():.6f}, "
                        f"step={step_reg.item():.6f}, tri={quad_tri_reg.item():.6f}"
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
                model.theta,
                model.log_s,
                model.phase,
                model.amp_coarse,
                model.bias_coarse,
                model.offset,
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
                model.theta,
                model.log_s,
                model.phase,
                model.amp_coarse,
                model.bias_coarse,
                model.offset,
            ],
            lr=lr,
        )
        _optimize_stage(
            stage=3,
            total_steps=total_stage3,
            optimizer=opt3,
            stage_modifiers=stage3_modifiers,
            global_step_offset=total_stage1 + total_stage2,
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser("Fit 2D cosine grid to an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (TIFF).")
    parser.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Number of optimization steps for stage 3 (data-enabled, masked).",
    )
    parser.add_argument(
        "--steps-stage1",
        type=int,
        default=500,
        help="Number of optimization steps for stage 1 (global rotation + isotropic scale).",
    )
    parser.add_argument(
        "--steps-stage2",
        type=int,
        default=1000,
        help="Number of optimization steps for stage 2 (global + coord grid, no data terms).",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--grid-step",
        type=int,
        default=4,
        help="Vertical coarse grid step in sample-space pixels for the internal eval grid.",
    )
    parser.add_argument(
        "--output-scale",
        type=int,
        default=4,
        help="Integer scale factor for saving reconstructions (snapshots and final).",
    )
    parser.add_argument(
        "--cosine-periods",
        type=float,
        default=32.0,
        help="Number of cosine periods across the sample-space width.",
    )
    parser.add_argument(
        "--sample-scale",
        type=float,
        default=1.0,
        help="Global multiplier for internal sample-space resolution (applied after x/y base sizing).",
    )
    parser.add_argument(
        "--samples-per-period",
        type=float,
        default=1.0,
        help="Number of coarse grid steps per cosine period horizontally.",
    )
    parser.add_argument(
        "--dense-samples-per-period",
        type=float,
        default=8.0,
        help="Dense samples per cosine period for the internal x resolution.",
    )
    parser.add_argument(
        "--img-downscale-factor",
        type=float,
        default=4.0,
        help="Downscale factor for internal resolution relative to avg image size.",
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=str,
        default=None,
        help=(
            "Path to UNet checkpoint. If set, run UNet on the specified TIFF "
            "layer and fit the cosine grid to its channel-0 output instead of "
            "the raw image."
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index of the input TIFF stack to use with --unet-checkpoint.",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        metavar=("CX", "CY"),
        default=None,
        help="Mask center in pixels (CX CY), 0,0 = top-left; default is image center.",
    )
    parser.add_argument(
        "--lambda-smooth-x",
        type=float,
        default=1,
        help="Smoothness weight along x (cosine direction) for the coarse grid.",
    )
    parser.add_argument(
        "--lambda-smooth-y",
        type=float,
        default=10,
        help="Smoothness weight along y (ridge direction) for the coarse grid.",
    )
    parser.add_argument("--lambda-mono", type=float, default=1e-3)
    parser.add_argument("--lambda-xygrad", type=float, default=1)
    parser.add_argument(
        "--lambda-angle-sym",
        type=float,
        default=1.0,
        help="Weight for angle-symmetry loss between horizontal connections and vertical grid lines.",
    )
    parser.add_argument(
        "--lambda-mod-h",
        type=float,
        default=1000.0,
        help="Horizontal smoothness weight for modulation parameters.",
    )
    parser.add_argument(
        "--lambda-mod-v",
        type=float,
        default=0.0,
        help="Vertical smoothness weight for modulation parameters.",
    )
    parser.add_argument(
        "--lambda-grad-data",
        type=float,
        default=10.0,
        help="Weight for gradient data term between sampled image and plain cosine target.",
    )
    parser.add_argument(
        "--lambda-grad-mag",
        type=float,
        default=1.0,
        help="Weight for gradient-magnitude period-sum loss in sample space (UNet channel 1).",
    )
    parser.add_argument(
    	"--unet-crop",
    	type=int,
    	default=16,
    	help="Pixels to crop from each image border after UNet inference, before downscaling (only used with --unet-checkpoint).",
    )
    parser.add_argument(
    	"--min-dx-grad",
    	type=float,
    	default=0.03,
    	help="Minimum gradient of rotated x-coordinate along coarse x (frequency lower bound).",
    )
    parser.add_argument(
    	"--compile-model",
    	action="store_true",
    	help="Compile CosineGridModel with torch.compile (PyTorch 2.x) for faster training.",
    )
    parser.add_argument(
    	"--final-float",
    	action="store_true",
    	help="Save final recon/gt/modgt as float32 TIFFs instead of uint8.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    parser.add_argument(
        "--snapshot",
        type=int,
        default=None,
        help="If set > 0 and output-prefix is given, save a reconstruction snapshot every N steps.",
    )
    parser.add_argument(
        "--dbg",
        action="store_true",
        help="If set, snapshot additional debug outputs (loss mask and diff) alongside reconstructions.",
    )
    parser.add_argument(
        "--for-video",
        action="store_true",
        help=(
            "Video mode: disable output upscaling, draw the grid on a black "
            "background, save masks as JPG, and use LZW compression for TIFFs."
        ),
    )
    args = parser.parse_args()
    fit_cosine_grid(
    	image_path=args.image,
    	steps=args.steps,
    	steps_stage1=args.steps_stage1,
    	steps_stage2=args.steps_stage2,
    	lr=args.lr,
    	grid_step=args.grid_step,
    	lambda_smooth_x=args.lambda_smooth_x,
    	lambda_smooth_y=args.lambda_smooth_y,
    	lambda_mono=args.lambda_mono,
    	lambda_xygrad=args.lambda_xygrad,
    	lambda_angle_sym=args.lambda_angle_sym,
    	lambda_mod_h=args.lambda_mod_h,
    	lambda_mod_v=args.lambda_mod_v,
    	lambda_grad_data=args.lambda_grad_data,
    	lambda_grad_mag=args.lambda_grad_mag,
    	min_dx_grad=args.min_dx_grad,
    	device=args.device,
    	output_prefix=args.output_prefix,
    	snapshot=args.snapshot,
    	output_scale=args.output_scale,
    	dbg=args.dbg,
    	mask_cx=(args.center[0] if args.center is not None else None),
    	mask_cy=(args.center[1] if args.center is not None else None),
    	cosine_periods=args.cosine_periods,
    	sample_scale=args.sample_scale,
    	samples_per_period=args.samples_per_period,
    	dense_samples_per_period=args.dense_samples_per_period,
    	img_downscale_factor=args.img_downscale_factor,
    	for_video=args.for_video,
    	unet_checkpoint=args.unet_checkpoint,
    	unet_layer=args.layer,
    	unet_crop=args.unet_crop,
    	compile_model=args.compile_model,
    	final_float=args.final_float,
    )


if __name__ == "__main__":
    main()
