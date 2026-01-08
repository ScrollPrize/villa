import math

import torch
from torch import nn
import torch.nn.functional as F

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
        offset_scales: int = 5,
    ) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.grid_step = int(grid_step)
        self.cosine_periods = float(cosine_periods)
        self.samples_per_period = float(samples_per_period)
        self.offset_scales = int(offset_scales)

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


        # Learnable per-point offset in sample space as a multi-scale pyramid.
        #
        # Lowest scale is absolute; each higher scale is a delta added to a bilinear
        # upsampling of the previous scale.
        gh0 = int(self.base_grid.shape[2])
        gw0 = int(self.base_grid.shape[3])
        n_scales = max(1, int(self.offset_scales))
        shapes: list[tuple[int, int]] = []
        for i in range(n_scales):
            k = n_scales - 1 - i
            gh_i = max(2, (gh0 - 1) // (2 ** k) + 1)
            gw_i = max(2, (gw0 - 1) // (2 ** k) + 1)
            shapes.append((gh_i, gw_i))
        shapes[-1] = (gh0, gw0)
        self._offset_shapes = shapes
        self.offset_ms = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 2, gh_i, gw_i)) for (gh_i, gw_i) in shapes]
        )

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

    def offset_coarse(self) -> torch.Tensor:
        off = self.offset_ms[0]
        for d in self.offset_ms[1:]:
            off = F.interpolate(
                off,
                size=(int(d.shape[2]), int(d.shape[3])),
                mode="bilinear",
                align_corners=True,
            ) + d
        return off

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
        uv_coarse = self.base_grid + self.offset_coarse()
        uv = F.interpolate(
            uv_coarse,
            size=(self.height, self.width),
            mode="bilinear",
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
            mode="bilinear",
            align_corners=True,
        )
        bias_hr = F.interpolate(
            self.bias_coarse,
            size=(self.height, self.width),
            mode="bilinear",
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


    def _direction_maps(
        self,
        grid: torch.Tensor,
        unet_dir0_img: torch.Tensor | None,
        unet_dir1_img: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """
        Compute model direction encodings in sample space.

        We return separate estimates:
        - v-based: from vertical mesh edges (y -> y+1), but using the orthogonal
          direction as the assumed gradient direction.
        - u-based: from horizontal mesh edges using line_offset connectivity,
          returned as two directed estimates:
            - u_lr: left column -> right column
            - u_rl: right column -> left column
          These are assumed PARALLEL to the gradient direction.

        Each estimate is encoded in the same "cos(2*theta)" scheme as train_unet:

            dir0 = 0.5 + 0.5*cos(2*theta)
            dir1 = 0.5 + 0.5*cos(2*theta + pi/4)
                 = 0.5 + 0.5*((cos(2*theta) - sin(2*theta)) / sqrt(2)).

        Returns:
            dir0_v:     (1,1,hr,wr)
            dir0_u_lr:  (1,1,hr,wr)
            dir0_u_rl:  (1,1,hr,wr)
            dir1_v:     (1,1,hr,wr)
            dir1_u_lr:  (1,1,hr,wr)
            dir1_u_rl:  (1,1,hr,wr)
        """
        if unet_dir0_img is None and unet_dir1_img is None:
            return None, None, None, None, None, None

        # grid: (1,hr,wr,2) in normalized image coords.
        x = grid[..., 0].unsqueeze(1)  # (1,1,hr,wr)
        y = grid[..., 1].unsqueeze(1)

        _, _, hh, ww = x.shape
        if hh < 2 or ww < 1:
            return None, None, None, None, None, None

        eps = 1e-8
        inv_sqrt2 = 1.0 / math.sqrt(2.0)

        # Build mapped coarse coords once. Direction computations are done on this
        # low-res grid (matching line_offset connectivity semantics), then upsampled.
        coords_c = self.base_grid + self.offset_coarse()  # (1,2,gh,gw)
        u_c = coords_c[:, 0:1]
        v_c = coords_c[:, 1:2]
        x_c, y_c = self._apply_global_transform(u_c, v_c)  # (1,1,gh,gw)
        xy_c = torch.cat([x_c, y_c], dim=1)  # (1,2,gh,gw)

        # v-based (coarse): vertical mesh edge on coarse grid, gradient is orthogonal to it.
        tx_v_c = x_c.new_zeros(x_c.shape)
        ty_v_c = y_c.new_zeros(y_c.shape)
        tx_v_c[:, :, :-1, :] = x_c[:, :, 1:, :] - x_c[:, :, :-1, :]
        ty_v_c[:, :, :-1, :] = y_c[:, :, 1:, :] - y_c[:, :, :-1, :]
        if x_c.shape[2] >= 2:
            tx_v_c[:, :, -1, :] = tx_v_c[:, :, -2, :]
            ty_v_c[:, :, -1, :] = ty_v_c[:, :, -2, :]

        gx_v_c = -ty_v_c
        gy_v_c = tx_v_c
        r2_v_c = gx_v_c * gx_v_c + gy_v_c * gy_v_c + eps
        cos2_v_c = (gx_v_c * gx_v_c - gy_v_c * gy_v_c) / r2_v_c
        sin2_v_c = (2.0 * gx_v_c * gy_v_c) / r2_v_c

        pack_v_c = torch.cat([cos2_v_c, sin2_v_c], dim=1)  # (1,2,gh,gw)
        pack_v_hr = F.interpolate(pack_v_c, size=(hh, ww), mode="bilinear", align_corners=True)
        cos2_v = pack_v_hr[:, 0:1]
        sin2_v = pack_v_hr[:, 1:2]

        dir0_v = 0.5 + 0.5 * cos2_v
        dir1_v = 0.5 + 0.5 * ((cos2_v - sin2_v) * inv_sqrt2)

        # u-based: horizontal mesh edge using line_offset connectivity on the coarse grid.
        src_u, nbr_u = self._coarse_x_line_pairs(xy_c)  # (1,2,2,gh,gw-1)
        vec_lr = nbr_u[:, :, 0] - src_u[:, :, 0]  # (1,2,gh,gw-1) from col x   -> x+1
        vec_rl = nbr_u[:, :, 1] - src_u[:, :, 1]  # (1,2,gh,gw-1) from col x+1 -> x

        _, _, gh_c, gw_c = xy_c.shape
        vec_u_lr = xy_c.new_zeros(1, 2, gh_c, gw_c)
        vec_u_rl = xy_c.new_zeros(1, 2, gh_c, gw_c)
        vec_u_lr[:, :, :, :-1] = vec_lr
        vec_u_lr[:, :, :, -1] = vec_lr[:, :, :, -1]
        vec_u_rl[:, :, :, 1:] = vec_rl
        vec_u_rl[:, :, :, 0] = vec_rl[:, :, :, 0]

        gx_lr_c = vec_u_lr[:, 0:1]
        gy_lr_c = vec_u_lr[:, 1:2]
        r2_lr_c = gx_lr_c * gx_lr_c + gy_lr_c * gy_lr_c + eps
        cos2_lr_c = (gx_lr_c * gx_lr_c - gy_lr_c * gy_lr_c) / r2_lr_c
        sin2_lr_c = (2.0 * gx_lr_c * gy_lr_c) / r2_lr_c

        gx_rl_c = vec_u_rl[:, 0:1]
        gy_rl_c = vec_u_rl[:, 1:2]
        r2_rl_c = gx_rl_c * gx_rl_c + gy_rl_c * gy_rl_c + eps
        cos2_rl_c = (gx_rl_c * gx_rl_c - gy_rl_c * gy_rl_c) / r2_rl_c
        sin2_rl_c = (2.0 * gx_rl_c * gy_rl_c) / r2_rl_c

        pack_lr = torch.cat([cos2_lr_c, sin2_lr_c], dim=1)  # (1,2,gh,gw)
        pack_rl = torch.cat([cos2_rl_c, sin2_rl_c], dim=1)  # (1,2,gh,gw)
        pack_lr_hr = F.interpolate(pack_lr, size=(hh, ww), mode="bilinear", align_corners=True)
        pack_rl_hr = F.interpolate(pack_rl, size=(hh, ww), mode="bilinear", align_corners=True)
        cos2_lr = pack_lr_hr[:, 0:1]
        sin2_lr = pack_lr_hr[:, 1:2]
        cos2_rl = pack_rl_hr[:, 0:1]
        sin2_rl = pack_rl_hr[:, 1:2]

        dir0_u_lr_raw = 0.5 + 0.5 * cos2_lr
        dir1_u_lr_raw = 0.5 + 0.5 * ((cos2_lr - sin2_lr) * inv_sqrt2)
        dir0_u_rl_raw = 0.5 + 0.5 * cos2_rl
        dir1_u_rl_raw = 0.5 + 0.5 * ((cos2_rl - sin2_rl) * inv_sqrt2)

        dir0_u_lr = dir0_u_lr_raw
        dir1_u_lr = dir1_u_lr_raw
        dir0_u_rl = dir0_u_rl_raw
        dir1_u_rl = dir1_u_rl_raw

        # print("sizes ", grid.shape, pack_lr.shape, coords_c.shape, dir0_v.shape, dir0_u_lr.shape)

        return dir0_v, dir0_u_lr, dir0_u_rl, dir1_v, dir1_u_lr, dir1_u_rl


    def _coarse_x_line_pairs(
        self,
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
        off_line = self.line_offset  # (1,2,gh,gw)

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
