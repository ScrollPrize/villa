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

        # Line offsets for horizontal connectivity.
        #
        # For each coarse grid point (y,x) we store two scalar offsets along the
        # *row index* direction which describe how connections to the left and
        # right neighbor columns are vertically displaced:
        #
        #   line_offset[:, 0, y, x]  -> offset towards left column  (x-1)
        #   line_offset[:, 1, y, x]  -> offset towards right column (x+1)
        #
        # These offsets are expressed in coarse-row units (float, in "pixels" of
        # the coarse grid). When forming horizontal relations we do not connect
        # directly to (y,x±1); instead we interpolate along the neighbor column
        # in y according to the corresponding offset.
        self.line_offset = nn.Parameter(torch.zeros(1, 2, gh, gw))
 
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
        self.theta = nn.Parameter(torch.zeros(1)-0.5)
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
 
    def _build_mask_cosine_hr(cos_v_extent: float) -> torch.Tensor:
        """
        Build the current cosine-domain mask in sample space for a given vertical
        half-extent cos_v_extent in normalized v ∈ [0,1].
        """
        cos_v = float(max(0.0, min(1.0, cos_v_extent)))
        if cos_v >= 1.0:
            mask_y = torch.ones_like(v_hr, dtype=torch.float32)
        else:
            abs_v = v_hr.abs()
            dist_v = abs_v - cos_v
            mask_y = torch.clamp(1.0 - dist_v / cos_mask_v_ramp_f, min=0.0, max=1.0)
        return mask_x_hr * mask_y
 
    def _current_cos_v_extent(stage: int, step_stage: int, total_stage_steps: int) -> float:
        """
        Effective vertical half-extent for the cosine-domain mask for a given
        stage and per-stage step index.
 
        - Stages 1–3: keep the initial extent cos_mask_v_extent_f constant.
        - Stage 4: start from cos_mask_v_extent_f and increase by 0.001 per
          optimization step within stage 4, clamped to 1.0.
        """
        if stage == 4:
            return float(cos_mask_v_extent_f + 0.0001 * float(step_stage))
        return float(cos_mask_v_extent_f)

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
        lr=10*lr,
    )
 
    total_stage1 = max(0, int(steps_stage1))
    total_stage2 = max(0, int(steps_stage2))
    total_stage3 = max(0, int(steps))
    total_stage4 = max(0, int(steps_stage4))
 
    def _gaussian_mask(stage: int, stage_progress: float) -> torch.Tensor | None:
        """
        Generate a Gaussian mask in image space (1,1,H,W) for the given stage
        and normalized progress value, or None when full-image loss is used
        (no Gaussian).

        When `use_image_mask` is False, this function always returns None so
        that only the cosine-domain sample-space mask is active.
 
        Args:
            stage:
                1,2: global stages without Gaussian masking (always return None).
                3+: masked stages using the progressive Gaussian schedule.
            stage_progress:
                Float in [0,1] indicating normalized progress within the current
                stage. 0 = stage start, 1 = stage end.
        """
        # Stages 1 and 2 use global optimization without a Gaussian mask.
        # When use_image_mask is False we disable the Gaussian entirely.
        if (not use_image_mask) or stage < 3:
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
 
    def _coarse_geom_mask(
        stage: int,
        stage_progress: float,
        cos_v_extent: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build per-vertex coarse-grid masks in [0,1] for geometry losses.
    
        The image-space mask is obtained by sampling either an all-ones map or
        the current Gaussian loss mask at the mapped coarse-grid coordinates.
    
        A separate cosine-domain mask is defined in coarse sample space using
        the canonical (u,v) grid. The combined mask is the product of the image
        mask and the cosine-domain mask.
    
        Returns:
            mask_coarse:      (1,1,gh,gw) image * cosine mask (for most geometry losses)
            mask_img_coarse:  (1,1,gh,gw) image mask only
            mask_cosine_coarse:(1,1,gh,gw) cosine-domain mask only
        """
        with torch.no_grad():
            coords = model.base_grid + model.offset  # (1,2,gh,gw)
            u = coords[:, 0:1]
            v = coords[:, 1:2]
    
            # Map to normalized image coordinates.
            x_norm, y_norm = model._apply_global_transform(u, v)
            grid_coarse = torch.stack(
                [x_norm.squeeze(1), y_norm.squeeze(1)],
                dim=-1,
            )  # (1,gh,gw,2)
    
            # Image-space loss mask:
            # - stages 1 & 2: all-ones
            # - stage 3: current Gaussian schedule, or all-ones when disabled.
            gauss_img = _gaussian_mask(stage, stage_progress)
            if gauss_img is None:
                mask_img = torch.ones(
                    (1, 1, h_img, w_img),
                    device=torch_device,
                    dtype=torch.float32,
                )
            else:
                mask_img = gauss_img
    
            mask_img_coarse = F.grid_sample(
                mask_img,
                grid_coarse,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )  # (1,1,gh,gw)
 
            # Cosine-domain band mask in coarse sample space: same definition as
            # the high-resolution mask, but evaluated on the canonical (u,v) grid.
            base_coords = model.base_grid  # (1,2,gh,gw)
            u_c = base_coords[:, 0:1]      # [-1,1] horizontally
            # v spans [-2,2]; normalize to [-1,1] for the vertical extent test.
            v_c = base_coords[:, 1:2] * 0.5
 
            # Horizontal cosine-domain mask with linear ramps over two periods,
            # matching the high-resolution definition.
            abs_u_c = u_c.abs()
            u_ramp_c = 2.0 * period_u
            if u_ramp_c > 0.0:
                dist_c = abs_u_c - u_band_half
                mask_x_c = torch.clamp(1.0 - dist_c / u_ramp_c, min=0.0, max=1.0)
            else:
                mask_x_c = (abs_u_c <= u_band_half).float()
 
            # Clamp vertical extent to [0,1] per step.
            cos_v = float(max(0.0, min(1.0, cos_v_extent)))
            if cos_v >= 1.0:
                mask_y_c = torch.ones_like(v_c, dtype=torch.float32)
            else:
                abs_v_c = v_c.abs()
                dist_v_c = abs_v_c - cos_v
                mask_y_c = torch.clamp(1.0 - dist_v_c / cos_mask_v_ramp_f, min=0.0, max=1.0)
            mask_cosine_coarse = mask_x_c * mask_y_c
 
            mask_coarse = mask_img_coarse * mask_cosine_coarse
    
        return mask_coarse, mask_img_coarse, mask_cosine_coarse
 
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
 
    def _draw_grid_vis(scale_factor: int = 4, mask_coarse: torch.Tensor | None = None) -> "np.ndarray":
        """
        Draw the coarse sample-space grid mapped into (an upscaled) image space.
 
        We render points at each coarse grid corner and lines between neighbors.
        By default this is drawn on top of an upscaled version of the source
        image; in video mode we draw on a black background instead.
        Vertical lines whose coarse x-position coincides with cosine peaks in
        the ground-truth design are highlighted in a different color.
 
        If a coarse-grid mask is provided (1,1,gh,gw), edge brightness is scaled
        linearly with the corresponding edge weights:
 
            w_edge in [0,1] -> brightness = 0.1 + 0.9 * w_edge
 
        so edges in low-weight regions appear darker while still remaining visible.
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
 
            # Optional coarse mask to modulate edge brightness.
            mask_arr = None
            m_h = None
            m_v = None
            if mask_coarse is not None:
                mask_arr = mask_coarse[0, 0].detach().cpu().numpy().astype(np.float32)
                mask_arr = np.clip(mask_arr, 0.0, 1.0)
                if mask_arr.shape == x_pix.shape:
                    if gw > 1:
                        # Horizontal edges: average of endpoint weights so that edges
                        # with at least one in-image vertex still contribute.
                        m_h = 0.5 * (mask_arr[:, :-1] + mask_arr[:, 1:])  # (gh,gw-1)
                    if gh > 1:
                        # Vertical edges: average of endpoint weights.
                        m_v = 0.5 * (mask_arr[:-1, :] + mask_arr[1:, :])  # (gh-1,gw)
 
            def edge_brightness(w_edge: float) -> float:
                """
                Map edge/vertex weight w ∈ [0,1] to brightness in [0.2,1.0].
                """
                return 1.0
                # w = max(0.0, min(1.0, float(w_edge)))
                # return 0.2 + 0.8 * w
 
            def in_bounds(px: int, py: int) -> bool:
                return 0 <= px < w_vis and 0 <= py < h_vis
 
            # Draw corner points (small green dots) with brightness from the
            # per-vertex mask when available.
            for iy in range(gh):
                for ix in range(gw):
                    px = int(round(float(x_pix[iy, ix])))
                    py = int(round(float(y_pix[iy, ix])))
                    if not in_bounds(px, py):
                        continue
                    if mask_arr is not None:
                        w_v = mask_arr[iy, ix]
                        b_v = edge_brightness(w_v)
                    else:
                        b_v = 1.0
                    color_v = (0, int(255 * b_v), 0)
                    cv2.circle(bg_color, (px, py), 1, color_v, -1)
 
            # Draw horizontal lines (base color blue, brightness from m_h).
            for iy in range(gh):
                for ix in range(gw - 1):
                    x0 = int(round(float(x_pix[iy, ix])))
                    y0 = int(round(float(y_pix[iy, ix])))
                    x1 = int(round(float(x_pix[iy, ix + 1])))
                    y1 = int(round(float(y_pix[iy, ix + 1])))
                    if in_bounds(x0, y0) and in_bounds(x1, y1):
                        if m_h is not None:
                            b = edge_brightness(m_h[iy, ix])
                        else:
                            b = 1.0
                        color = (int(0 * b), int(0 * b), int(255 * b))
                        cv2.line(bg_color, (x0, y0), (x1, y1), color, 1)
 
            # Visualize new horizontal connectivity with line offsets.
            #
            # We work in the same visualization coordinate space (x_pix,y_pix)
            # and use _coarse_x_line_pairs on a 2-channel coordinate field to
            # obtain both directed connections for each horizontal edge:
            #   0: left -> right  (cyan)
            #   1: right -> left  (yellow)
            coords_vis_t = torch.from_numpy(
                np.stack([x_pix, y_pix], axis=0)
            ).unsqueeze(0)  # (1,2,gh,gw)
            src_conn, nbr_conn = _coarse_x_line_pairs(coords_vis_t)  # (1,2,2,gh,gw-1)
            src_conn_np = src_conn[0].detach().cpu().numpy()  # (2,2,gh,gw-1)
            nbr_conn_np = nbr_conn[0].detach().cpu().numpy()  # (2,2,gh,gw-1)
 
            for iy in range(gh):
                for ix in range(gw - 1):
                    if m_h is not None:
                        b_edge = edge_brightness(m_h[iy, ix])
                    else:
                        b_edge = 1.0
 
                    # Left -> right (dir=0): cyan
                    x0_lr = int(round(float(src_conn_np[0, 0, iy, ix])))
                    y0_lr = int(round(float(src_conn_np[1, 0, iy, ix])))
                    x1_lr = int(round(float(nbr_conn_np[0, 0, iy, ix])))
                    y1_lr = int(round(float(nbr_conn_np[1, 0, iy, ix])))
                    if in_bounds(x0_lr, y0_lr) and in_bounds(x1_lr, y1_lr):
                        color_lr = (int(255 * b_edge), int(255 * b_edge), int(0 * b_edge))
                        cv2.line(bg_color, (x0_lr, y0_lr), (x1_lr, y1_lr), color_lr, 1)
 
                    # Right -> left (dir=1): yellow
                    x0_rl = int(round(float(src_conn_np[0, 1, iy, ix])))
                    y0_rl = int(round(float(src_conn_np[1, 1, iy, ix])))
                    x1_rl = int(round(float(nbr_conn_np[0, 1, iy, ix])))
                    y1_rl = int(round(float(nbr_conn_np[1, 1, iy, ix])))
                    if in_bounds(x0_rl, y0_rl) and in_bounds(x1_rl, y1_rl):
                        color_rl = (int(0 * b_edge), int(255 * b_edge), int(255 * b_edge))
                        cv2.line(bg_color, (x0_rl, y0_rl), (x1_rl, y1_rl), color_rl, 1)
 
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
 
            # Draw vertical lines: highlight peak columns in red, brightness from m_v.
            for iy in range(gh - 1):
                for ix in range(gw):
                    x0 = int(round(float(x_pix[iy, ix])))
                    y0 = int(round(float(y_pix[iy, ix])))
                    x1 = int(round(float(x_pix[iy + 1, ix])))
                    y1 = int(round(float(y_pix[iy + 1, ix])))
                    if not (in_bounds(x0, y0) and in_bounds(x1, y1)):
                        continue
                    if m_v is not None:
                        b = edge_brightness(m_v[iy, ix])
                    else:
                        b = 1.0
                    if ix in peak_cols:
                        # Cos-peak line: draw in red and slightly thicker.
                        base_color = (255, 0, 0)
                        thickness = 2
                    else:
                        base_color = (0, 0, 255)
                        thickness = 1
                    color = (int(base_color[0] * b), int(base_color[1] * b), int(base_color[2] * b))
                    cv2.line(bg_color, (x0, y0), (x1, y1), color, thickness)
 
            return bg_color
     
    def _save_snapshot(
        stage: int,
        step_stage: int,
        total_stage_steps: int,
        global_step_idx: int,
        mask_full: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
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
            valid_np = None
            grid_vis_np = None
            dir_model_np = None
            dir_unet_np = None
            dir1_model_np = None
            dir1_unet_np = None
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
                cos_v_eff_dbg = _current_cos_v_extent(stage, step_stage, total_stage_steps)

                if valid_mask is not None:
                    valid_np = valid_mask.detach().cpu().squeeze(0).squeeze(0).numpy()
                if mask_full is not None:
                    mask_np = mask_full.detach().cpu().squeeze(0).squeeze(0).numpy()
 
                # Direction maps (model vs UNet) in sample space (dir0 & dir1).
                if unet_dir0_img is not None or unet_dir1_img is not None:
                    dir0_model_hr, dir0_unet_hr, dir1_model_hr, dir1_unet_hr = _direction_maps(grid_dbg)
    
                    if dir0_model_hr is not None and dir0_unet_hr is not None:
                        dir0_model_vis = dir0_model_hr
                        dir0_unet_vis = dir0_unet_hr
                        if eff_output_scale is not None and eff_output_scale > 1:
                            dir0_model_vis = F.interpolate(
                                dir0_model_vis,
                                scale_factor=eff_output_scale,
                                mode="bicubic",
                                align_corners=True,
                            )
                            dir0_unet_vis = F.interpolate(
                                dir0_unet_vis,
                                scale_factor=eff_output_scale,
                                mode="bicubic",
                                align_corners=True,
                            )
                        dir_model_np = dir0_model_vis.cpu().squeeze(0).squeeze(0).numpy()
                        dir_unet_np = dir0_unet_vis.cpu().squeeze(0).squeeze(0).numpy()
    
                    if dir1_model_hr is not None and dir1_unet_hr is not None:
                        dir1_model_vis = dir1_model_hr
                        dir1_unet_vis = dir1_unet_hr
                        if eff_output_scale is not None and eff_output_scale > 1:
                            dir1_model_vis = F.interpolate(
                                dir1_model_vis,
                                scale_factor=eff_output_scale,
                                mode="bicubic",
                                align_corners=True,
                            )
                            dir1_unet_vis = F.interpolate(
                                dir1_unet_vis,
                                scale_factor=eff_output_scale,
                                mode="bicubic",
                                align_corners=True,
                            )
                        dir1_model_np = dir1_model_vis.cpu().squeeze(0).squeeze(0).numpy()
                        dir1_unet_np = dir1_unet_vis.cpu().squeeze(0).squeeze(0).numpy()
 
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
                # is black instead of the image. Edge brightness is modulated
                # by the same coarse-grid mask used for geometry losses so that
                # low-weight regions appear darker.
                base_scale = output_scale if output_scale is not None else 4
                vis_scale = base_scale * 2
                geom_mask_vis, _, _ = _coarse_geom_mask(stage, stage_progress, cos_v_eff_dbg)
                grid_vis_np = _draw_grid_vis(scale_factor=vis_scale, mask_coarse=geom_mask_vis)
 
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
                    mask_path = f"{p}_mask_step{global_step_idx:06d}.tif"
                    # tifffile.imwrite(mask_path, _to_uint8(mask_np), compression="lzw")
                    cv2.imwrite(mask_path, _to_uint8(mask_np))

            if valid_np is not None:
                if for_video:
                    valid_u8 = (np.clip(valid_np, 0.0, 1.0) * 255.0).astype("uint8")
                    valid_path = f"{p}_valid_step{global_step_idx:06d}.jpg"
                    cv2.imwrite(valid_path, valid_u8)
                else:
                    valid_path = f"{p}_valid_step{global_step_idx:06d}.tif"
                    cv2.imwrite(valid_path, _to_uint8(valid_np))
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
                # Primary encoding (dir0): keep legacy filenames for compatibility.
                dir_model_path = f"{p}_dir_model_step{global_step_idx:06d}.tif"
                dir_unet_path = f"{p}_dir_unet_step{global_step_idx:06d}.tif"
                tifffile.imwrite(dir_model_path, _to_uint8(dir_model_np), compression="lzw")
                tifffile.imwrite(dir_unet_path, _to_uint8(dir_unet_np), compression="lzw")
            if dir1_model_np is not None and dir1_unet_np is not None:
                # Secondary encoding (dir1) with explicit names.
                dir1_model_path = f"{p}_dir1_model_step{global_step_idx:06d}.tif"
                dir1_unet_path = f"{p}_dir1_unet_step{global_step_idx:06d}.tif"
                tifffile.imwrite(dir1_model_path, _to_uint8(dir1_model_np), compression="lzw")
                tifffile.imwrite(dir1_unet_path, _to_uint8(dir1_unet_np), compression="lzw")
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
 
    def _smoothness_reg(
        mask_cosine: torch.Tensor | None = None,
        mask_img: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Smoothness penalty on the *offset* field in coarse grid index space.
    
        We regularize the learnable offsets along grid indices:
        - x index (horizontal neighbors): ||Δ_off||^2 for (u_offset, v_offset),
        - y index (vertical neighbors):   ||Δ_off||^2 for (u_offset, v_offset),
    
        i.e. both u- and v-offset components are treated the same in both
        directions; the split into smooth_x / smooth_y is only by *index
        direction*, not by coordinate component.
    
        Horizontal relations use the line-offset connectivity field: for each
        edge between columns (x,x+1) we build two connections, one from the
        left column towards the right and one from the right column towards the
        left, each vertically displaced according to its own offset and
        interpolated in the neighbor column.
    
        If coarse-grid masks are provided (shape (1,1,gh,gw)), we build per-edge
        weights from:
    
            mask_img:      image / validity weighting
            mask_cosine:   cosine-domain band in coarse space
    
        Inside the cosine band we use full weight (mask_img). Outside the band
        we keep only smoothness, but downweighted:
    
            - smooth_x: 1/10 of its in-band weight
            - smooth_y: 1/10  of its in-band weight
        """
        off = model.offset  # (1,2,gh,gw)  2 = (u_offset, v_offset)
        _, _, gh, gw = off.shape
    
        # Horizontal smoothness using line-offset connectivity. For each
        # horizontal edge we build two connections (left->right and right->left)
        # and average the squared L2 differences over both directions.
        if gw >= 2:
            src_x, nbr_x = _coarse_x_line_pairs(off)  # (1,2,2,gh,gw-1)
            dx_vec = nbr_x - src_x                   # (1,2,2,gh,gw-1)
            dx_sq = (dx_vec * dx_vec).sum(dim=1, keepdim=True)  # (1,1,2,gh,gw-1)
            smooth_x = dx_sq.mean(dim=2)  # (1,1,gh,gw-1)
        else:
            smooth_x = torch.zeros((), device=off.device, dtype=off.dtype)
    
        # First-order differences along y index: o[..., j+1, :] - o[..., j, :].
        # Again, treat (u_offset, v_offset) symmetrically via ||Δ_off||^2.
        if gh >= 2:
            dy = off[:, :, 1:, :] - off[:, :, :-1, :]  # (1,2,gh-1,gw)
            dy_sq = (dy * dy).sum(dim=1, keepdim=True)
            smooth_y = dy_sq  # (1,1,gh-1,gw)
        else:
            smooth_y = torch.zeros((), device=off.device, dtype=off.dtype)
    
        # No masks: use global mean.
        if mask_img is None and mask_cosine is None:
            return smooth_x.mean(), smooth_y.mean()
    
        # Prepare per-vertex masks.
        if mask_img is None:
            m_img = torch.ones((1, 1, gh, gw), device=off.device, dtype=off.dtype)
        else:
            m_img = mask_img.to(device=off.device, dtype=off.dtype)
    
        if mask_cosine is None:
            m_cos = torch.ones_like(m_img)
        else:
            m_cos = mask_cosine.to(device=off.device, dtype=off.dtype)
    
        # Horizontal edges: between (y,x) and (y,x+1).
        if isinstance(smooth_x, torch.Tensor) and smooth_x.numel() > 0:
            m_img_h = 0.5 * (m_img[:, :, :, :-1] + m_img[:, :, :, 1:])  # (1,1,gh,gw-1)
            m_cos_h = 0.5 * (m_cos[:, :, :, :-1] + m_cos[:, :, :, 1:])  # (1,1,gh,gw-1)
            alpha_x = 1.0
            w_h = m_img_h * (m_cos_h + alpha_x * (1.0 - m_cos_h))
            wsum_h = w_h.sum()
            if wsum_h > 0:
                loss_x = (smooth_x * w_h).sum() / wsum_h
            else:
                loss_x = smooth_x.mean()
        else:
            loss_x = smooth_x
    
        # Vertical edges: between (y,x) and (y+1,x).
        if isinstance(smooth_y, torch.Tensor) and smooth_y.numel() > 0:
            m_img_v = 0.5 * (m_img[:, :, :-1, :] + m_img[:, :, 1:, :])  # (1,1,gh-1,gw)
            m_cos_v = 0.5 * (m_cos[:, :, :-1, :] + m_cos[:, :, 1:, :])  # (1,1,gh-1,gw)
            alpha_y = 1.0
            w_v = m_img_v * (m_cos_v + alpha_y * (1.0 - m_cos_v))
            wsum_v = w_v.sum()
            if wsum_v > 0:
                loss_y = (smooth_y * w_v).sum() / wsum_v
            else:
                loss_y = smooth_y.mean()
        else:
            loss_y = smooth_y
    
        return loss_x, loss_y
 
    def _line_offset_smooth_reg(mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Smoothness penalty on the line_offset field along y for each direction.
 
        line_offset has shape (1,2,gh,gw), channels:
            0: offset towards left neighbor
            1: offset towards right neighbor
 
        We regularize first-order differences along the coarse y index for each
        direction channel separately:
            dy = line_offset[:, :, j+1, :] - line_offset[:, :, j, :].
 
        If a coarse-grid mask is provided (1,1,gh,gw), we weight vertical
        differences by the average of their endpoint weights and normalize by
        the sum of weights. The same mask is broadcast to both directions.
        """
        lo = model.line_offset  # (1,2,gh,gw)
        _, _, gh, gw = lo.shape
        if gh < 2:
            return torch.zeros((), device=lo.device, dtype=lo.dtype)
 
        dy = lo[:, :, 1:, :] - lo[:, :, :-1, :]  # (1,2,gh-1,gw)
        dy_sq = dy * dy                          # (1,2,gh-1,gw)
 
        if mask is None:
            return dy_sq.mean()
 
        m = mask.to(device=lo.device, dtype=lo.dtype)  # (1,1,gh,gw)
        m_v = 0.5 * (m[:, :, 1:, :] + m[:, :, :-1, :])  # (1,1,gh-1,gw)
        wsum = m_v.sum()
        if wsum <= 0:
            return dy_sq.mean()
 
        w = m_v.expand(1, 2, gh - 1, gw)  # broadcast to both directions
        return (dy_sq * w).sum() / wsum
  
    def _mod_smooth_reg(mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Smoothness penalty on modulation parameters (amp, bias) on the coarse grid.
   
        We only regularize variation along y for modulation. If a coarse-grid
        mask is provided, we weight each vertical difference by the average of
        its endpoint weights (after resampling the mask to the modulation grid
        in x) and normalize by the sum of weights so that differences with one
        in-image endpoint still contribute while fully out-of-image connections
        (both endpoints 0) are ignored.
        """
        # Stack amp and bias so both are regularized consistently.
        mods = torch.cat([model.amp_coarse, model.bias_coarse], dim=1)  # (1,2,gh,gw_mod)
   
        # First-order differences along y in coarse grid index space.
        dy = mods[:, :, 1:, :] - mods[:, :, :-1, :]   # (1,2,gh-1,gw_mod)
        dy_sq = dy * dy
   
        if dy_sq.numel() == 0:
            base = torch.zeros((), device=mods.device, dtype=mods.dtype)
            return base
   
        if mask is None:
            return dy_sq.mean()
   
        gh_m, gw_m = mods.shape[2], mods.shape[3]
        with torch.no_grad():
            # Resample coarse grid mask (defined on coord grid width) to modulation width.
            mask_mod = F.interpolate(
                mask,
                size=(gh_m, gw_m),
                mode="bilinear",
                align_corners=True,
            )
            # Vertical edges: between rows j and j+1; use average of endpoint weights.
            m_v = 0.5 * (mask_mod[:, :, :-1, :] + mask_mod[:, :, 1:, :])  # (1,1,gh-1,gw_mod)
        wsum = m_v.sum()
        if wsum > 0:
            return (dy_sq.mean(dim=1, keepdim=True) * m_v).sum() / wsum
        return dy_sq.mean()
    
    def _step_reg(mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Regularization on coarse *rotated* coords in target-image space.
    
        For each step along the cosine grid we consider the distance between
        neighboring sample positions after mapping into image space.
    
        We compute:
        - horizontal neighbor distances (along coarse x),
        - vertical neighbor distances (along coarse y).
    
        Horizontally we use the line-offset connectivity: each edge between
        columns (x,x+1) is represented by two connections (left->right and
        right->left) with vertical displacement given by the corresponding
        offsets and interpolation along the neighbor column.
    
        Vertically:
        - enforce each distance to be at least 0.5 * the average vertical distance
          and encourage distances to be close to that average.
    
        If a coarse-grid mask is provided, we weight horizontal/vertical edges
        by the average of their endpoint weights and normalize by the sum of
        weights, so that edges lying fully outside the image (both endpoints 0)
        do not contribute while edges with one in-image endpoint still receive
        a non-zero weight.
        """
        coords = model.base_grid + model.offset  # (1,2,gh,gw)
        u = coords[:, 0:1]
        v = coords[:, 1:2]
    
        # Apply same x-only scale-then-rotation as in _build_sampling_grid, but on the coarse grid.
        x_norm, y_norm = model._apply_global_transform(u, v)
    
        # Map normalized coords to pixel coordinates of the target image.
        x_pix = (x_norm + 1.0) * 0.5 * float(max(1, w_img - 1))
        y_pix = (y_norm + 1.0) * 0.5 * float(max(1, h_img - 1))
    
        # Horizontal neighbor distances using line-offset connectivity. Work in
        # pixel space by treating (x_pix,y_pix) as a 2D coordinate field on the
        # coarse grid and building left/right connections via interpolation.
        coords_pix = torch.cat([x_pix, y_pix], dim=1)  # (1,2,gh,gw)
        src_h, nbr_h = _coarse_x_line_pairs(coords_pix)  # (1,2,2,gh,gw-1)
        delta_h = nbr_h - src_h
        dist_h = torch.sqrt((delta_h * delta_h).sum(dim=1, keepdim=True) + 1e-12)  # (1,1,2,gh,gw-1)
        # Average over the two directions for each horizontal edge.
        dist_h = dist_h.mean(dim=2)  # (1,1,gh,gw-1)
    
        # Vertical neighbor distances (steps along coarse y index).
        dx_v = x_pix[:, :, 1:, :] - x_pix[:, :, :-1, :]
        dy_v = y_pix[:, :, 1:, :] - y_pix[:, :, :-1, :]
        dist_v = torch.sqrt(dx_v * dx_v + dy_v * dy_v + 1e-12)  # (1,1,gh-1,gw)
    
        if mask is not None:
            # Per-edge masks via average of endpoint weights.
            m = mask  # (1,1,gh,gw)
            m_h = 0.5 * (m[:, :, :, :-1] + m[:, :, :, 1:])    # (1,1,gh,gw-1)
            m_v = 0.5 * (m[:, :, :-1, :] + m[:, :, 1:, :])    # (1,1,gh-1,gw)
    
            # Weighted averages for reference distances.
            wsum_h = m_h.sum()
            if wsum_h > 0:
                avg_h = (dist_h * m_h).sum() / wsum_h
            else:
                avg_h = dist_h.mean()
    
            wsum_v = m_v.sum()
            if wsum_v > 0:
                avg_v = (dist_v * m_v).sum() / wsum_v
            else:
                avg_v = dist_v.mean()
        else:
            # Unweighted averages over all edges.
            avg_h = dist_h.mean()
            avg_v = dist_v.mean()
    
        avg_h_det = avg_h.detach()
        avg_v_det = avg_v.detach()
    
        # Horizontal: enforce each distance to be at least 0.1 * avg horizontal.
        min_h = 0.1 * avg_h_det
        if float(min_h) <= 0.0:
            loss_h = torch.zeros((), device=coords.device, dtype=coords.dtype)
        else:
            shortfall_h = torch.clamp(min_h - dist_h, min=0.0) / min_h
            if mask is not None:
                wsum_h = m_h.sum()
                if wsum_h > 0:
                    loss_h = ((shortfall_h * shortfall_h) * m_h).sum() / wsum_h
                else:
                    loss_h = (shortfall_h * shortfall_h).mean()
            else:
                loss_h = (shortfall_h * shortfall_h).mean()
    
        # Vertical: encourage each distance to be close to avg distance.
        target_v = avg_v_det
        diff_v = dist_v - target_v
        if mask is not None:
            wsum_v = m_v.sum()
            if wsum_v > 0:
                loss_v_avg = ((diff_v * diff_v) * m_v).sum() / wsum_v
            else:
                loss_v_avg = (diff_v * diff_v).mean()
        else:
            loss_v_avg = (diff_v * diff_v).mean()
    
        base = 1 * loss_h + 0.1 * loss_v_avg
        return base
     
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
 
        If a coarse-grid mask is provided, each (row,col) location contributing
        to the comparison is weighted by an average of the corresponding
        horizontal and vertical edge weights, so that configurations with at
        least one in-image vertex still contribute, while locations whose
        participating edges are fully outside the image receive zero weight.
        """
        coords = model.base_grid + model.offset  # (1,2,gh,gw)
        u = coords[:, 0:1]
        v = coords[:, 1:2]
 
        # Apply same x-only scale-then-rotation as in _build_sampling_grid, but on the coarse grid.
        x_norm, y_norm = model._apply_global_transform(u, v)
 
        # Map normalized coords to pixel coordinates.
        x_pix = (x_norm + 1.0) * 0.5 * float(max(1, w_img - 1))
        y_pix = (y_norm + 1.0) * 0.5 * float(max(1, h_img - 1))
 
        # Build 2D coarse coordinate field in pixel space for connectivity.
        coords_pix = torch.cat([x_pix, y_pix], dim=1)  # (1,2,gh,gw)
        _, _, gh, gw = coords_pix.shape
        if gh < 2 or gw < 2:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)
 
        # Horizontal edge vectors using line-offset connectivity. For each
        # horizontal edge (x,x+1) we have two directions (left->right, right->left).
        src_h, nbr_h = _coarse_x_line_pairs(coords_pix)  # (1,2,2,gh,gw-1)
        hvec = nbr_h - src_h                             # (1,2,2,gh,gw-1)
        hvx = hvec[:, 0:1]                               # (1,1,2,gh,gw-1)
        hvy = hvec[:, 1:2]                               # (1,1,2,gh,gw-1)
 
        # Vertical edge vectors between neighboring rows (top -> bottom).
        dx_v = x_pix[:, :, 1:, :] - x_pix[:, :, :-1, :]   # (1,1,gh-1,gw)
        dy_v = y_pix[:, :, 1:, :] - y_pix[:, :, :-1, :]   # (1,1,gh-1,gw)
 
        # We only compare where both directions are defined:
        # rows 0..gh-2 for vertical, and cols 0..gw-2 for horizontal.
        hvx_use = hvx[:, :, :, 0:gh-1, 0:gw-1]           # (1,1,2,gh-1,gw-1)
        hvy_use = hvy[:, :, :, 0:gh-1, 0:gw-1]
        vvx_base = dx_v[:, :, 0:gh-1, 0:gw-1]            # (1,1,gh-1,gw-1)
        vvy_base = dy_v[:, :, 0:gh-1, 0:gw-1]
 
        # Broadcast vertical vectors across the two horizontal directions.
        vvx = vvx_base.unsqueeze(2).expand_as(hvx_use)   # (1,1,2,gh-1,gw-1)
        vvy = vvy_base.unsqueeze(2).expand_as(hvy_use)
 
        # Cosine of angle between horizontal connections and vertical direction.
        eps = 1e-12
        h_norm = torch.sqrt(hvx_use * hvx_use + hvy_use * hvy_use + eps)
        v_norm = torch.sqrt(vvx * vvx + vvy * vvy + eps)
        dot = hvx_use * vvx + hvy_use * vvy
        cos_theta = dot / (h_norm * v_norm + eps)
 
        # Penalize squared cosine -> encourages orthogonality. This implicitly
        # averages over both connectivity directions per horizontal edge.
        base_unweighted = cos_theta * cos_theta  # (1,1,2,gh-1,gw-1)
 
        if mask is None:
            return base_unweighted.mean()
 
        # Build per-location weights from coarse mask: combine horizontal and
        # vertical edge masks so that locations involving out-of-image points
        # are downweighted/ignored.
        m = mask  # (1,1,gh,gw)
        m_h = 0.5 * (m[:, :, :, :-1] + m[:, :, :, 1:])     # (1,1,gh,gw-1)
        m_v = 0.5 * (m[:, :, :-1, :] + m[:, :, 1:, :])     # (1,1,gh-1,gw)
 
        # Restrict to the (gh-1,gw-1) region used above.
        m_h_use = m_h[:, :, 0:gh-1, 0:gw-1]   # (1,1,gh-1,gw-1)
        m_v_use = m_v[:, :, 0:gh-1, 0:gw-1]   # (1,1,gh-1,gw-1)
        # Average horizontal/vertical contributions at each location.
        m_loc = 0.5 * (m_h_use + m_v_use)
        m_loc = m_loc.unsqueeze(2).expand_as(base_unweighted)    # (1,1,2,gh-1,gw-1)
 
        wsum = m_loc.sum()
        if wsum > 0:
            return (base_unweighted * m_loc).sum() / wsum
        return base_unweighted.mean()
     
    def _quad_triangle_reg(mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Quad-based triangle-area regularizer in image space.
 
        For each quad in the coarse grid, we form four corner-based triangles
        using the direct neighboring quad corners and compute signed areas via
        the 2D cross product.
 
        We:
        - penalize triangle area magnitude being less than 1/4 of the average,
        - strongly penalize negative (flipped) triangle areas.
 
        If a coarse-grid mask is provided, each quad is weighted by the average
        of its four corner weights so that quads fully outside the image
        (all four corners 0) do not contribute, while quads with at least one
        in-image corner still receive a non-zero weight.
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
        A = 0.1 * avg_area_abs
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
        tri_size_loss_unweighted = size_pen  # (4,1,gh-1,gw-1)
 
        # Orientation: strongly penalize negative signed area (kept for completeness).
        neg = torch.clamp(-areas, min=0.0) / (avg_area_abs + 1e-12)
        tri_neg_loss_unweighted = neg * neg
 
        if mask is None:
            tri_size_loss = tri_size_loss_unweighted.mean()
            base = tri_size_loss
            return base
 
        # Per-quad mask from four corners.
        m = mask  # (1,1,gh,gw)
        m00 = m[:, :, :-1, :-1]
        m01 = m[:, :, :-1, 1:]
        m11 = m[:, :, 1:, 1:]
        m10 = m[:, :, 1:, :-1]
        # Use average of the four corner weights so that quads with at least one
        # in-image corner still contribute while fully out-of-image quads (all 0)
        # are ignored.
        m_quad = 0.25 * (m00 + m01 + m10 + m11)  # (1,1,gh-1,gw-1)
 
        # Broadcast to 4 triangles per quad.
        m_tri = m_quad.unsqueeze(0).expand_as(tri_size_loss_unweighted)  # (4,1,gh-1,gw-1)
 
        wsum = m_tri.sum()
        if wsum > 0:
            tri_size_loss = (tri_size_loss_unweighted * m_tri).sum() / wsum
        else:
            tri_size_loss = tri_size_loss_unweighted.mean()
 
        # Return combined triangle-area loss; external lambda scales overall strength.
        base = tri_size_loss
        return base

    def _coarse_x_line_pairs(
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build horizontally connected coarse-grid pairs using the line-offset field.
    
        Args:
            coords:
                (1,C,gh,gw) tensor defined at each coarse grid point.
    
        Returns:
            src:
                (1,C,2,gh,gw-1) source points for each horizontal connection.
                src[:,:,0,y,x] = coords at (y,x)       (left anchor)
                src[:,:,1,y,x] = coords at (y,x+1)     (right anchor)
    
            nbr:
                (1,C,2,gh,gw-1) neighbor points on the *other* column, interpolated
                along y according to the corresponding line offsets:
    
                nbr[:,:,0,y,x] = coords on column x+1 at y + off_right[y,x]
                nbr[:,:,1,y,x] = coords on column x   at y + off_left[y,x+1]
    
        where off_right/off_left are taken from model.line_offset in units of
        coarse-row indices.
        """
        off_line = model.line_offset  # (1,2,gh,gw)

        # Ensure coords lives on the same device/dtype as the connectivity field,
        # since this helper may be called from CPU-only visualization code.
        coords = coords.to(device=off_line.device, dtype=off_line.dtype)

        _, c, gh, gw = coords.shape
     
        # Anchors on left / right columns for each horizontal edge (x,x+1).
        left_anchors = coords[:, :, :, :-1]   # (1,C,gh,gw-1)
        right_anchors = coords[:, :, :, 1:]   # (1,C,gh,gw-1)
     
        # Base row indices as float.
        j_base = torch.arange(gh, device=off_line.device, dtype=off_line.dtype).view(1, 1, gh, 1)
    
        # Offsets towards right neighbor from the left column (channel 1).
        off_r = off_line[:, 1:2, :, :-1]  # (1,1,gh,gw-1)
        # Offsets towards left neighbor from the right column (channel 0).
        off_l = off_line[:, 0:1, :, 1:]   # (1,1,gh,gw-1)
    
        # Target row positions in coarse index space.
        r_r = j_base + off_r  # (1,1,gh,gw-1)
        r_l = j_base + off_l  # (1,1,gh,gw-1)
    
        # Integer neighbors for interpolation, clamped to valid range.
        i0_r = torch.floor(r_r).clamp(0, gh - 2).long()
        i1_r = i0_r + 1
        i0_l = torch.floor(r_l).clamp(0, gh - 2).long()
        i1_l = i0_l + 1
    
        # Neighbor columns.
        coords_right = coords[:, :, :, 1:]   # (1,C,gh,gw-1)
        coords_left = coords[:, :, :, :-1]   # (1,C,gh,gw-1)
    
        # Gather along y for right neighbors (column x+1).
        idx0_r = i0_r.expand(1, c, gh, gw - 1)
        idx1_r = i1_r.expand(1, c, gh, gw - 1)
        val0_r = torch.gather(coords_right, 2, idx0_r)
        val1_r = torch.gather(coords_right, 2, idx1_r)
    
        # Gather along y for left neighbors (column x).
        idx0_l = i0_l.expand(1, c, gh, gw - 1)
        idx1_l = i1_l.expand(1, c, gh, gw - 1)
        val0_l = torch.gather(coords_left, 2, idx0_l)
        val1_l = torch.gather(coords_left, 2, idx1_l)
    
        # Linear interpolation weights.
        w_r = (r_r - i0_r.to(coords.dtype)).expand(1, c, gh, gw - 1)
        w_l = (r_l - i0_l.to(coords.dtype)).expand(1, c, gh, gw - 1)
    
        nbr_from_left = val0_r * (1.0 - w_r) + val1_r * w_r  # (1,C,gh,gw-1)
        nbr_from_right = val0_l * (1.0 - w_l) + val1_l * w_l  # (1,C,gh,gw-1)
    
        # Stack along a small "direction" dimension: 0 = left->right, 1 = right->left.
        src = torch.stack([left_anchors, right_anchors], dim=2)        # (1,C,2,gh,gw-1)
        nbr = torch.stack([nbr_from_left, nbr_from_right], dim=2)      # (1,C,2,gh,gw-1)
        return src, nbr

    def _direction_maps(
        grid: torch.Tensor,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Compute direction encodings for model & UNet in sample space for both
        dir0 and dir1 encodings.
    
        Model direction is derived from the mapping (u,v) -> (x,y) represented
        by the sampling grid. Let J be the Jacobian of this mapping:
    
            J = [ dx/du  dx/dv ]
                [ dy/du  dy/dv ]
    
        The cosine phase varies along u, so in image space the phase field is
        φ(x,y) = k * u(x,y). Its gradient is ∇φ ∝ ∇u, and
    
            [du, dv]^T = J^{-1} [dx, dy]^T
    
        so the gradient of u in (x,y) is the first row of J^{-1}:
    
            ∇u = (∂u/∂x, ∂u/∂y) = row_0(J^{-1})
    
        From ∇u we form:
    
            cos(2*theta) = (ux^2 - uy^2) / (ux^2 + uy^2)
            sin(2*theta) = 2*ux*uy / (ux^2 + uy^2)
    
        and encode two directional maps matching train_unet:
    
            dir0 = 0.5 + 0.5*cos(2*theta)
            dir1 = 0.5 + 0.5*cos(2*theta + pi/4)
                 = 0.5 + 0.5*((cos(2*theta) - sin(2*theta)) / sqrt(2)).
    
        Returns:
            dir0_model: (1,1,hr,wr) or None
            dir0_unet:  (1,1,hr,wr) or None
            dir1_model: (1,1,hr,wr) or None
            dir1_unet:  (1,1,hr,wr) or None
        """
        if unet_dir0_img is None and unet_dir1_img is None:
            return None, None, None, None
    
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
        sin2theta = (2.0 * ux * uy) / r2
    
        # Primary encoding: 0.5 + 0.5*cos(2*theta).
        dir0_model = 0.5 + 0.5 * cos2theta  # (1,1,hr,wr)
    
        # Secondary encoding shifted by 45 degrees:
        # cos(2*theta + pi/4) = (cos(2*theta) - sin(2*theta)) / sqrt(2).
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        cos2theta_shift = (cos2theta - sin2theta) * inv_sqrt2
        dir1_model = 0.5 + 0.5 * cos2theta_shift
    
        # Warp UNet direction channels into sample space using the same grid.
        dir0_unet_hr: torch.Tensor | None = None
        dir1_unet_hr: torch.Tensor | None = None
        if unet_dir0_img is not None:
            dir0_unet_hr = F.grid_sample(
                unet_dir0_img,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        if unet_dir1_img is not None:
            dir1_unet_hr = F.grid_sample(
                unet_dir1_img,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
    
        return dir0_model, dir0_unet_hr, dir1_model, dir1_unet_hr

    def _directional_alignment_loss(
        grid: torch.Tensor,
        mask_sample: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Directional alignment loss between the mapped cosine axis and the UNet
        direction branches, encoded as dir0 & dir1 in sample space.
    
        We compute MSE for each available encoding and combine them as:
    
            loss_dir = 0.5 * (loss_dir0 + loss_dir1)
    
        when both are present, or just loss_dir0 when only dir0 is available.
    
        If `mask_sample` is provided (1,1,H,W), it is used as a spatial weight
        so that only the masked sample-space region contributes to the loss.
        """
        if unet_dir0_img is None and unet_dir1_img is None:
            return torch.zeros((), device=torch_device, dtype=torch.float32)
    
        dir0_model, dir0_unet_hr, dir1_model, dir1_unet_hr = _direction_maps(grid)
        device = torch_device
        dtype = torch.float32
    
        if dir0_model is None or dir0_unet_hr is None:
            return torch.zeros((), device=device, dtype=dtype)
    
        def _masked_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            diff = a - b
            diff2 = diff * diff
            if mask_sample is None:
                return diff2.mean()
            if mask_sample.shape[-2:] != diff2.shape[-2:]:
                # Safety: fall back to unweighted if spatial sizes mismatch.
                return diff2.mean()
            w = mask_sample
            wsum = w.sum()
            if wsum > 0:
                return (diff2 * w).sum() / wsum
            return diff2.mean()
    
        loss_dir0 = _masked_mse(dir0_model, dir0_unet_hr)
    
        if dir1_model is not None and dir1_unet_hr is not None:
            loss_dir1 = _masked_mse(dir1_model, dir1_unet_hr)
            return 0.5 * (loss_dir0 + loss_dir1)
    
        # Fallback: only dir0 available.
        return loss_dir0
 
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
 
        # If a sample-space mask is provided with matching spatial size, apply it
        # directly to the magnitude so that samples outside the mask contribute
        # zero to the period sums, independent of later weighting.
        if mask_sample is not None and mask_sample.shape[-2:] == mag_hr.shape[-2:]:
            mag_hr = mag_hr * mask_sample
 
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
        "line_smooth_y": lambda_line_smooth_y,
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
        "grad_mag": 0.0,
        "quad_tri": 0.0,
        "step": 0.0,
        "mod_smooth": 0.0,
        "angle_sym": 0.0,
        "dir_unet": 1.0,
        "use_full_dir_unet" : False,
        # "grad_mag" : 0.001,
        # "dir_unet": 10.0,
        # "smooth_x": 10.0,
        # "smooth_y": 0.0,
        "line_smooth_y": 0.1,
        # other terms default to 1.0 (enabled).
    }
 
    stage3_modifiers: dict[str, float] = {
        # Stage 3: enable data and grad_data terms in addition to stage-2 regularization.
        # No explicit overrides: all lambda_global weights are used as-is.
        # "data": 0.0,
        "quad_tri": 0.0,
        "grad_data": 0.0,
        # "grad_data": 0.0,
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
        cos_v_eff = _current_cos_v_extent(stage, step_stage, total_stage_steps)
 
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
      
        	valid_erode_iters = 0
        	for _ in range(valid_erode_iters):
        		inv = 1.0 - valid
        		inv = F.max_pool2d(inv, kernel_size=3, stride=1, padding=1)
        		valid = 1.0 - inv
      
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
      
        	# Apply cosine-domain band mask in sample space so that only selected
        	# periods/rows contribute to the loss, with vertical extent possibly
        	# growing during stage 4.
        	mask_cosine_cur = _build_mask_cosine_hr(cos_v_eff)
        	weight_full = weight_full * mask_cosine_cur
      
        	# Coarse-grid masks for geometry losses: sample the same image-space
        	# mask (ones or Gaussian) at the coarse-grid coordinates, and keep
        	# the cosine-domain band separate so we can treat in-band vs
        	# out-of-band regions differently for smoothness.
        	geom_mask_coarse, geom_mask_img, geom_mask_cos = _coarse_geom_mask(stage, stage_progress, cos_v_eff)
 
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
            dir_loss = _directional_alignment_loss(grid, mask_sample=mask_dir)
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
            gradmag_loss = _gradmag_period_loss(grid, img_downscale_factor, mask_for_gradmag)
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
            smooth_x_val, smooth_y_val = _smoothness_reg()
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
            step_reg = _step_reg()
            total_loss = total_loss + w_step * step_reg
        else:
            step_reg = torch.zeros((), device=device, dtype=dtype)
        terms["step"] = step_reg
  
        # Modulation smoothness.
        w_mod_smooth = _need_term("mod_smooth", stage_modifiers)
        if w_mod_smooth != 0.0:
            # mod_smooth = _mod_smooth_reg(geom_mask_coarse)
            mod_smooth = _mod_smooth_reg()
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
            line_smooth = _line_offset_smooth_reg()
            line_smooth = _line_offset_smooth_reg(geom_mask_img)
            total_loss = total_loss + w_line_smooth * line_smooth
        else:
            line_smooth = torch.zeros((), device=device, dtype=dtype)
        terms["line_smooth_y"] = line_smooth

        # Angle-symmetry regularizer between horizontal connections and vertical lines.
        w_angle = _need_term("angle_sym", stage_modifiers)
        if w_angle != 0.0:
            # Apply without the cosine-domain mask so symmetry is encouraged
            # also outside the active cosine loss band.
            angle_reg = _angle_symmetry_reg(None)
            total_loss = total_loss + w_angle * angle_reg
        else:
            angle_reg = torch.zeros((), device=device, dtype=dtype)
        terms["angle_sym"] = angle_reg

        if dbg:
            terms["_mask_full"] = weight_full.detach()
            terms["_mask_valid"] = valid.detach()

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
                model.offset,
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
                model.theta,
                model.log_s,
                model.phase,
                model.amp_coarse,
                model.bias_coarse,
                model.offset,
                model.line_offset,
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
                model.offset,
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser("Fit 2D cosine grid to an image or tiled UNet outputs")
    parser.add_argument(
    	"--input",
    	type=str,
    	required=True,
    	help=(
    		"Path to input TIFF image (stack) or directory containing precomputed "
    		"tiled UNet outputs (_cos/_mag/_dir0/_dir1)."
    	),
    )
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
    parser.add_argument(
        "--steps-stage4",
        type=int,
        default=10000,
        help="Number of optimization steps for stage 4 (like stage 3, but with growing vertical cosine mask).",
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
        "--cos-mask-periods",
        type=float,
        default=5.0,
        help="Number of cosine periods from the left edge of sample space used for the loss mask.",
    )
    parser.add_argument(
        "--cos-mask-v-extent",
        type=float,
        default=0.1,
        help="Vertical half-extent in normalized sample-space v (in [0,1]) of the cosine loss band (default: 0.1).",
    )
    parser.add_argument(
        "--cos-mask-v-ramp",
        type=float,
        default=0.05,
        help="Vertical ramp width in normalized sample-space v for the cosine loss band (linear fade-out to 0 outside the band).",
    )
    parser.add_argument(
    	"--unet-checkpoint",
    	type=str,
    	default=None,
    	help=(
    		"Path to UNet checkpoint (.pt). If set, run UNet on the specified TIFF "
    		"layer from --input (when it is a TIFF file) and fit the cosine grid "
    		"to its channel-0 output. Must not be set when --input is a directory "
    		"of precomputed tiled UNet outputs."
    	),
    )
    parser.add_argument(
    	"--layer",
    	type=int,
    	default=None,
    	help=(
    		"Layer index of the input TIFF stack (when --input is a file), or the "
    		"layer suffix to select in a directory of tiled UNet outputs "
    		"(filenames of the form *_layerXXXX_{cos,mag,dir0,dir1}.tif)."
    	),
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
        default=100,
        help="Smoothness weight along x (cosine direction) for the coarse grid.",
    )
    parser.add_argument(
        "--lambda-smooth-y",
        type=float,
        default=100000,
        help="Smoothness weight along y (ridge direction) for the coarse grid.",
    )
    parser.add_argument("--lambda-mono", type=float, default=1e-3)
    parser.add_argument("--lambda-xygrad", type=float, default=1)
    parser.add_argument(
        "--lambda-line-smooth-y",
        type=float,
        default=0.0,
        help="Smoothness weight along y for line_offset (neighbor offsets) per direction.",
    )
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
    	"--crop",
    	type=int,
    	nargs=4,
    	metavar=("X", "Y", "W", "H"),
    	default=None,
    	help="Optional crop rectangle in pixels (x,y,w,h) applied before fitting.",
    )
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
    parser.add_argument(
        "--use-image-mask",
        action="store_true",
        help="Enable image-space Gaussian loss mask in stage 3 in addition to the cosine-domain mask.",
    )
    args = parser.parse_args()
    fit_cosine_grid(
    	image_path=args.input,
    	steps=args.steps,
    	steps_stage1=args.steps_stage1,
    	steps_stage2=args.steps_stage2,
    	steps_stage4=args.steps_stage4,
    	lr=args.lr,
    	grid_step=args.grid_step,
    	lambda_smooth_x=args.lambda_smooth_x,
    	lambda_smooth_y=args.lambda_smooth_y,
    	lambda_mono=args.lambda_mono,
    	lambda_xygrad=args.lambda_xygrad,
    	lambda_angle_sym=args.lambda_angle_sym,
    	lambda_mod_h=args.lambda_mod_h,
    	lambda_mod_v=args.lambda_mod_v,
    	lambda_line_smooth_y=args.lambda_line_smooth_y,
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
    	crop=tuple(args.crop) if args.crop is not None else None,
    	compile_model=args.compile_model,
    	final_float=args.final_float,
    	cos_mask_periods=args.cos_mask_periods,
    	cos_mask_v_extent=args.cos_mask_v_extent,
    	cos_mask_v_ramp=args.cos_mask_v_ramp,
    	use_image_mask=args.use_image_mask,
    )


if __name__ == "__main__":
    main()
