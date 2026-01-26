from pathlib import Path

import tifffile
import torch
import torch.nn.functional as F

from common import load_unet
from fit_helpers import load_image, load_tiff_layer


def load_fit_inputs(
    image_path: str,
    torch_device: torch.device,
    *,
    unet_checkpoint: str | None,
    unet_layer: int | None,
    unet_crop: int,
    crop: tuple[int, int, int, int] | None,
    img_downscale_factor: float,
    output_prefix: str | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    unet_dir0_img: torch.Tensor | None = None
    unet_dir1_img: torch.Tensor | None = None
    unet_mag_img: torch.Tensor | None = None

    p_input = Path(image_path)
    if p_input.is_dir():
        if unet_checkpoint is not None:
            raise ValueError(
                "When --input is a directory, --unet-checkpoint must not be set; "
                "pass the directory with tiled UNet outputs as --input and omit "
                "--unet-checkpoint."
            )

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
        dir0_path = cos_path.with_name(f"{base_stem}_dir0.tif")
        dir1_path = cos_path.with_name(f"{base_stem}_dir1.tif")
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
        unet_dir0_img = torch.clamp(dir0_t, 0.0, 1.0) if dir0_t is not None else None
        unet_dir1_img = torch.clamp(dir1_t, 0.0, 1.0) if dir1_t is not None else None
    elif unet_checkpoint is not None:
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

        if unet_crop is not None and unet_crop > 0:
            c = int(unet_crop)
            _, _, h_u, w_u = pred_unet.shape
            if h_u > 2 * c and w_u > 2 * c:
                pred_unet = pred_unet[:, :, c:-c, c:-c]

        if output_prefix is not None:
            p = Path(output_prefix)
            unet_np = pred_unet[0].detach().cpu().numpy()
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

        image = torch.clamp(pred_unet[:, 0:1], 0.0, 1.0)
        unet_mag_img = torch.clamp(pred_unet[:, 1:2], 0.0, 1.0)
        unet_dir0_img = torch.clamp(pred_unet[:, 2:3], 0.0, 1.0)
        unet_dir1_img = torch.clamp(pred_unet[:, 3:4], 0.0, 1.0) if pred_unet.size(1) > 3 else None
    else:
        if unet_layer is not None:
            image = load_tiff_layer(image_path, torch_device, layer=unet_layer)
        else:
            image = load_image(image_path, torch_device)

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

    return image, unet_dir0_img, unet_dir1_img, unet_mag_img
