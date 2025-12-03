import math
from pathlib import Path

import tifffile
import torch
from torch import nn
import torch.nn.functional as F


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


class CosineGridModel(nn.Module):
    """
    Cosine grid model with a coarse coordinate grid plus a global rotation and
    x-scale applied to the interpolated coordinates before evaluating cos().
    """

    def __init__(self, height: int, width: int, downscale: int = 4) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.downscale = int(downscale)

        # Coarse grid resolution (parameter grid).
        gh = max(2, (self.height + self.downscale - 1) // self.downscale + 1)
        gw = max(2, (self.width + self.downscale - 1) // self.downscale + 1)

        # Base grid: x ramp over width, y ramp over height, both in [0, 2*pi].
        base_x = torch.linspace(0.0, 2.0 * math.pi, gw).view(1, 1, 1, gw).expand(1, 1, gh, gw)
        base_y = torch.linspace(0.0, 2.0 * math.pi, gh).view(1, 1, gh, 1).expand(1, 1, gh, gw)
        grid = torch.cat([base_x, base_y], dim=1)

        # Learnable coarse coordinates.
        self.coords = nn.Parameter(grid)

        # Global rotation angle (radians), log isotropic scale, and global phase offset.
        self.theta = nn.Parameter(torch.zeros(1)-math.pi/2)
        self.log_sx = nn.Parameter(torch.zeros(1)+2)
        self.phase = nn.Parameter(torch.zeros(1))

    def _apply_transform(self, up: torch.Tensor) -> torch.Tensor:
        """
        Apply global rotation and isotropic scale to the coordinates.

        up: (1,2,H,W) coordinates (coarse or interpolated grid).
        """
        theta = self.theta
        sx = self.log_sx.exp()

        c = torch.cos(theta)
        s = torch.sin(theta)

        x = up[:, 0:1]
        y = up[:, 1:2]

        # Rotate.
        xr = c * x - s * y
        yr = s * x + c * y

        # Isotropic scale of both components.
        xr = sx * xr
        yr = sx * yr

        return torch.cat([xr, yr], dim=1)

    def forward(self) -> torch.Tensor:
        # Interpolate coarse coordinates to full resolution (bicubic).
        up = F.interpolate(self.coords, size=(self.height, self.width), mode="bicubic", align_corners=True)
        # Apply global rotation + x-scale.
        up_transformed = self._apply_transform(up)
        x_map = up_transformed[:, 0:1]
        # Add global phase offset before evaluating cosine.
        x_arg = x_map + self.phase.view(1, 1, 1, 1)
        out = 0.5 + 0.5 * torch.cos(x_arg)
        return out

    def smoothness_loss(self) -> torch.Tensor:
        """
        Smoothness penalty that encourages neighboring gradients to be the same.

        We first compute first-order differences dx, dy and then penalize
        differences of these gradients between neighboring locations.
        """
        c = self.coords

        # First-order gradients.
        dx = c[:, :, :, 1:] - c[:, :, :, :-1]   # (N,C,gh,gw-1)
        dy = c[:, :, 1:, :] - c[:, :, :-1, :]   # (N,C,gh-1,gw)

        # Differences of gradients between neighboring locations.
        # For dx:
        ddx_x = dx[:, :, :, 1:] - dx[:, :, :, :-1]   # horizontal neighbors of dx
        ddx_y = dx[:, :, 1:, :] - dx[:, :, :-1, :]   # vertical neighbors of dx

        # For dy:
        ddy_x = dy[:, :, :, 1:] - dy[:, :, :, :-1]   # horizontal neighbors of dy
        ddy_y = dy[:, :, 1:, :] - dy[:, :, :-1, :]   # vertical neighbors of dy

        return 0.25 * (
            (ddx_x * ddx_x).mean()
            + (ddx_y * ddx_y).mean()
            + (ddy_x * ddy_x).mean()
            + (ddy_y * ddy_y).mean()
        )

    def monotonicity_loss(self) -> torch.Tensor:
        """
        Monotonicity penalty: encourage gradients to keep their sign along
        x, y, and diagonal directions.

        This is similar to smoothness_loss but only active where the sign
        of neighboring gradients changes.
        """
        c = self.coords

        # First-order gradients along axes.
        dx = c[:, :, :, 1:] - c[:, :, :, :-1]
        dy = c[:, :, 1:, :] - c[:, :, :-1, :]

        # Diagonal gradients.
        d1 = c[:, :, 1:, 1:] - c[:, :, :-1, :-1]   # down-right
        d2 = c[:, :, 1:, :-1] - c[:, :, :-1, 1:]  # down-left

        def _sign_change_penalty(g: torch.Tensor) -> torch.Tensor:
            # Neighbor pairs horizontally.
            g_left = g[:, :, :, :-1]
            g_right = g[:, :, :, 1:]
            dd_x = g_right - g_left
            mask_x = (g_left * g_right < 0.0).float()

            # Neighbor pairs vertically.
            g_up = g[:, :, :-1, :]
            g_down = g[:, :, 1:, :]
            dd_y = g_down - g_up
            mask_y = (g_up * g_down < 0.0).float()

            loss_x = (dd_x * dd_x * mask_x).mean()
            loss_y = (dd_y * dd_y * mask_y).mean()
            return 0.5 * (loss_x + loss_y)

        loss_dx = _sign_change_penalty(dx)
        loss_dy = _sign_change_penalty(dy)
        loss_d1 = _sign_change_penalty(d1)
        loss_d2 = _sign_change_penalty(d2)

        return 0.25 * (loss_dx + loss_dy + loss_d1 + loss_d2)

    def rotated_grad_losses(self, min_dx: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Regularization on gradients of rotated coordinates on the *coarse* grid.

        After rotation/scale (but before upsampling), we want:
        - y-coordinate gradient along coarse y to be spatially constant,
        - x-coordinate gradient along coarse x to be at least a minimum value.
        """
        # Apply rotation/scale directly to the low-res grid.
        rot = self._apply_transform(self.coords)
        x_map = rot[:, 0:1]
        y_map = rot[:, 1:2]

        # Gradients along coarse grid axes.
        gx = x_map[:, :, :, 1:] - x_map[:, :, :, :-1]   # d x_map / d x_coarse
        gy = y_map[:, :, 1:, :] - y_map[:, :, :-1, :]   # d y_map / d y_coarse

        # Y: encourage gy to be spatially constant with magnitude tied to avg gx.
        target_gy = gx.mean().detach()
        diff_gy = gy - target_gy
        loss_gy = (diff_gy * diff_gy).mean()

        # X: enforce gx >= min_dx (minimum frequency on coarse grid).
        if min_dx <= 0.0:
            loss_gx = torch.zeros((), device=self.coords.device, dtype=self.coords.dtype)
        else:
            shortfall = torch.clamp(min_dx - gx, min=0.0)
            loss_gx = (shortfall * shortfall).mean()

        return loss_gx, loss_gy


def fit_cosine_grid(
    image_path: str,
    steps: int = 1000,
    steps_stage1: int = 500,
    lr: float = 1e-2,
    downscale: int = 4,
    lambda_smooth: float = 1e-3,
    lambda_mono: float = 1e-3,
    lambda_xygrad: float = 1e-3,
    min_dx_grad: float = 0.0,
    device: str | None = None,
    output_prefix: str | None = None,
    snapshot: int | None = None,
    output_scale: int = 4,
    dbg: bool = False,
    mask_cx: float | None = None,
    mask_cy: float | None = None,
) -> None:
    """
    Fit the cosine grid model to an image.

    Training is done in two stages:
    - Stage 1: optimize only global rotation (theta), x-scale (log_sx), and phase,
      for `steps_stage1` iterations.
    - Stage 2: freeze the global transform & phase, bake them into coords, and then
      optimize coords only for `steps` iterations.

    Snapshots and the final reconstruction are optionally upscaled by `output_scale`
    (bicubic interpolation) when saving.

    If `dbg` is True, each snapshot also saves the current loss mask and the
    reconstruction error (pred - target) at native resolution.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    target = load_image(image_path, torch_device)
    _, _, h, w = target.shape

    # Determine mask center in pixels (0,0 = top-left). Defaults to image center.
    if mask_cx is None:
        cx = 0.5 * float(w - 1)
    else:
        cx = float(mask_cx)
    if mask_cy is None:
        cy = 0.5 * float(h - 1)
    else:
        cy = float(mask_cy)

    # Convert center to normalized coordinates in [-1, 1].
    cx_norm = 2.0 * cx / float(max(1, w - 1)) - 1.0
    cy_norm = 2.0 * cy / float(max(1, h - 1)) - 1.0

    # Precompute squared radius grid (normalized coords) for growing Gaussian window.
    ys = torch.linspace(-1.0, 1.0, h, device=torch_device, dtype=torch.float32)
    xs = torch.linspace(-1.0, 1.0, w, device=torch_device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    yy = yy - cy_norm
    xx = xx - cx_norm
    window_r2 = (xx * xx + yy * yy).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    # Sigma schedule: start very narrow (central region) and grow slowly,
    # reaching near-uniform by half of the total optimization steps.
    sigma_min = 0.1
    sigma_max = 1.0

    model = CosineGridModel(h, w, downscale=downscale).to(torch_device)

    total_steps = steps_stage1 + steps

    # Optimizers: first for global transform only, then for all params.
    opt_stage1 = None
    if steps_stage1 > 0:
        opt_stage1 = torch.optim.Adam(
            [model.theta, model.log_sx, model.phase],
            lr=lr,
        )

    opt_stage2 = None
    if steps > 0:
        # After stage 1 we bake theta/log_sx into coords and then only optimize
        # coords + phase in stage 2.
        opt_stage2 = torch.optim.Adam(
            [model.coords, model.phase],
            lr=lr,
        )

    def _save_snapshot(step_idx: int) -> None:
        if snapshot is None or snapshot <= 0 or output_prefix is None:
            return
        with torch.no_grad():
            # Native-resolution prediction.
            pred_native = model().clamp(0.0, 1.0)

            # Optionally upsample for the saved reconstruction.
            pred_save = pred_native
            if output_scale is not None and output_scale > 1:
                pred_save = F.interpolate(
                    pred_save,
                    scale_factor=output_scale,
                    mode="bicubic",
                    align_corners=True,
                )
            pred_img = pred_save.cpu().squeeze(0).squeeze(0).numpy()

            p = Path(output_prefix)
            base = p
            post = f"step{step_idx:06d}"

            # Always save the reconstruction snapshot.
            recon_path = f"{base}_arecon_{post}.tif"
            tifffile.imwrite(recon_path, pred_img.astype("float32"))

            if dbg:
                # Recompute the current Gaussian window using the same schedule as
                # in the data term, using step_idx as an approximation of global_step.
                schedule_steps = max(1, int(0.9 * float(total_steps)))
                frac = float(step_idx) / float(schedule_steps)
                t = max(0.0, min(1.0, frac * frac))
                sigma = sigma_min + (sigma_max - sigma_min) * t
                sigma_sq = sigma * sigma
                w = torch.exp(-0.5 * window_r2 / sigma_sq)
                w_vis = w / (w.max() + 1e-8)

                diff_native = pred_native - target

                # Optionally upsample mask & diff with nearest-neighbor to match
                # the reconstruction's output_scale.
                w_out = w_vis
                diff_out = diff_native
                if output_scale is not None and output_scale > 1:
                    w_out = F.interpolate(
                        w_out,
                        scale_factor=output_scale,
                        mode="nearest",
                    )
                    diff_out = F.interpolate(
                        diff_out,
                        scale_factor=output_scale,
                        mode="nearest",
                    )

                mask_img = w_out.cpu().squeeze(0).squeeze(0).numpy().astype("float32")
                diff_img = diff_out.cpu().squeeze(0).squeeze(0).numpy().astype("float32")

                mask_path = f"{base}_mask_{post}.tif"
                diff_path = f"{base}_diff_{post}.tif"
                tifffile.imwrite(mask_path, mask_img)
                tifffile.imwrite(diff_path, diff_img)

    def _optimization_step(opt, global_step: int, phase: str, use_smooth: bool) -> None:
        pred = model()

        # Growing Gaussian window on the data term: start focusing on the center
        # and expand to the full image by 90% of the total steps (with slow, quadratic growth).
        # After 90% of the steps, use the full unmasked MSE loss.
        diff = pred - target
        schedule_steps = max(1, int(0.9 * float(total_steps)))
        frac = float(global_step) / float(schedule_steps)

        if global_step >= schedule_steps:
            # After reaching the maximum sigma (full coverage), use plain MSE
            # without any spatial mask.
            data_loss = F.mse_loss(pred, target)
        else:
            t = max(0.0, min(1.0, frac * frac))
            sigma = sigma_min + (sigma_max - sigma_min) * t
            sigma_sq = sigma * sigma
            w = torch.exp(-0.5 * window_r2 / sigma_sq)
            # Normalize so the effective scale of the loss stays roughly constant.
            data_loss = (w * (diff * diff)).sum() / w.sum()

        smooth_loss = model.smoothness_loss()
        monotone_loss = model.monotonicity_loss()
        rot_x_loss, rot_y_loss = model.rotated_grad_losses(min_dx_grad)
        if use_smooth:
            loss = (
                data_loss
                + lambda_smooth * smooth_loss
                + lambda_mono * monotone_loss
                + lambda_xygrad * (rot_x_loss + rot_y_loss)
            )
        else:
            loss = data_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (global_step + 1) % 100 == 0 or global_step == 0 or global_step == total_steps - 1:
            with torch.no_grad():
                # Diagnostics for coarse rotated gradients used in rotated_grad_losses.
                rot = model._apply_transform(model.coords)
                x_map = rot[:, 0:1]
                y_map = rot[:, 1:2]
                gx = x_map[:, :, :, 1:] - x_map[:, :, :, :-1]
                gy = y_map[:, :, 1:, :] - y_map[:, :, :-1, :]
                avg_dx = float(gx.mean())
                avg_dy = float(gy.mean())

            print(
                f"{phase} step {global_step+1}/{total_steps}: loss={loss.item():.6f}, "
                f"data={data_loss.item():.6f}, smooth={smooth_loss.item():.6f}, "
                f"mono={monotone_loss.item():.6f}, "
                f"rot_x={rot_x_loss.item():.6f}, rot_y={rot_y_loss.item():.6f}, "
                f"avg_dx={avg_dx:.6f}, avg_dy={avg_dy:.6f}",
            )

        # Optional snapshots of the current reconstruction.
        if snapshot is not None and snapshot > 0 and output_prefix is not None:
            if (global_step + 1) % snapshot == 0:
                _save_snapshot(global_step + 1)

    global_step = 0

    # Save initialization snapshot if requested.
    if snapshot is not None and snapshot > 0 and output_prefix is not None:
        _save_snapshot(0)

    # Stage 1: optimize only rotation + x-scale (+ phase) (no regularization).
    if opt_stage1 is not None:
        for _ in range(steps_stage1):
            _optimization_step(opt_stage1, global_step, phase="stage1(rot+scale+phase)", use_smooth=False)
            global_step += 1

    # After stage 1, bake the learned global rotation/scale/phase into the coarse
    # coords and reset the transform/phase so they are not used anymore.
    if opt_stage1 is not None:
        with torch.no_grad():
            baked = model._apply_transform(model.coords)
            # Phase is an additive offset on x before cosine; bake it into x-channel.
            baked[:, 0:1] += model.phase.view(1, 1, 1, 1)
            model.coords.copy_(baked)
            # Reset transform to identity (no-op in later calls).
            model.theta.zero_()
            model.log_sx.zero_()
            model.phase.zero_()

    # Stage 2: optimize baked coords jointly (with regularization).
    if opt_stage2 is not None:
        for _ in range(steps):
            _optimization_step(opt_stage2, global_step, phase="stage2(coords-only)", use_smooth=True)
            global_step += 1

    # Final outputs at convergence.
    if output_prefix is not None:
        with torch.no_grad():
            pred = model().clamp(0.0, 1.0)
            if output_scale is not None and output_scale > 1:
                pred = F.interpolate(
                    pred,
                    scale_factor=output_scale,
                    mode="bicubic",
                    align_corners=True,
                )
            pred = pred.cpu().squeeze(0).squeeze(0).numpy()
            coords_coarse = model.coords.detach().cpu().numpy()
            coords_full = F.interpolate(
                model.coords,
                size=(h, w),
                mode="bicubic",
                align_corners=True,
            ).detach().cpu().numpy()
        p = Path(output_prefix)
        recon_path = str(p) + "_recon.tif"
        coords_coarse_path = str(p) + "_coords_coarse.tif"
        coords_full_path = str(p) + "_coords_full.tif"
        tifffile.imwrite(recon_path, pred.astype("float32"))
        tifffile.imwrite(coords_coarse_path, coords_coarse.astype("float32"))
        tifffile.imwrite(coords_full_path, coords_full.astype("float32"))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser("Fit 2D cosine grid to an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image (TIFF).")
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Number of optimization steps for stage 2 (joint).",
    )
    parser.add_argument(
        "--steps-stage1",
        type=int,
        default=500,
        help="Number of optimization steps for stage 1 (rotation + x-scale + phase).",
    )
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument(
        "--output-scale",
        type=int,
        default=4,
        help="Integer scale factor for saving reconstructions (snapshots and final).",
    )
    parser.add_argument(
        "--center",
        type=float,
        nargs=2,
        metavar=("CX", "CY"),
        default=None,
        help="Mask center in pixels (CX CY), 0,0 = top-left; default is image center.",
    )
    parser.add_argument("--lambda-smooth", type=float, default=1e-3)
    parser.add_argument("--lambda-mono", type=float, default=1e-3)
    parser.add_argument("--lambda-xygrad", type=float, default=1)
    parser.add_argument(
        "--min-dx-grad",
        type=float,
        default=0.03,
        help="Minimum gradient of rotated x-coordinate along coarse x (frequency lower bound).",
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
    args = parser.parse_args()
    fit_cosine_grid(
        image_path=args.image,
        steps=args.steps,
        steps_stage1=args.steps_stage1,
        lr=args.lr,
        downscale=args.downscale,
        lambda_smooth=args.lambda_smooth,
        lambda_mono=args.lambda_mono,
        lambda_xygrad=args.lambda_xygrad,
        min_dx_grad=args.min_dx_grad,
        device=args.device,
        output_prefix=args.output_prefix,
        snapshot=args.snapshot,
        output_scale=args.output_scale,
        dbg=args.dbg,
        mask_cx=(args.center[0] if args.center is not None else None),
        mask_cy=(args.center[1] if args.center is not None else None),
    )


if __name__ == "__main__":
    main()
