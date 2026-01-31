from train_resnet3d_lib.config import CFG, REVERSE_LAYER_FRAGMENT_IDS_FALLBACK

import os.path as osp

import numpy as np
import random
import cv2
import torch
import torch.nn.functional as F

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from scipy import ndimage
import PIL.Image


def _read_gray(path):
    if not osp.exists(path):
        return None

    # OpenCV can emit noisy libpng warnings (e.g. "chunk data is too large") for some PNGs.
    # Prefer PIL for PNGs to keep logs clean; fall back to OpenCV if PIL fails.
    if path.lower().endswith(".png"):
        try:
            with PIL.Image.open(path) as im:
                return np.array(im.convert("L"))
        except Exception:
            pass

    try:
        img = cv2.imread(path, 0)
        if img is not None:
            return img
    except cv2.error:
        pass

    try:
        with PIL.Image.open(path) as im:
            return np.array(im.convert("L"))
    except Exception as e:
        raise RuntimeError(f"Could not read image: {path}. Error: {e}") from e


def read_image_layers(
    fragment_id,
    start_idx=15,
    end_idx=45,
    *,
    layer_range=None,
):
    layers_dir = osp.join("train_scrolls", fragment_id, "layers")
    layer_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    def _iter_layer_paths(layer_idx):
        # Support 00.tif, 000.tif, 0000.tif, etc.
        for fmt in (f"{layer_idx:02}", f"{layer_idx:03}", f"{layer_idx:04}", str(layer_idx)):
            for ext in layer_exts:
                yield osp.join(layers_dir, f"{fmt}{ext}")

    if layer_range is not None:
        start_idx, end_idx = layer_range

    idxs = list(range(int(start_idx), int(end_idx)))
    if len(idxs) < CFG.in_chans:
        raise ValueError(
            f"{fragment_id}: expected at least {CFG.in_chans} layers, got {len(idxs)} from range {start_idx}-{end_idx}"
        )
    if len(idxs) > CFG.in_chans:
        start = max(0, (len(idxs) - CFG.in_chans) // 2)
        idxs = idxs[start:start + CFG.in_chans]
    if len(idxs) != CFG.in_chans:
        raise ValueError(
            f"{fragment_id}: expected {CFG.in_chans} layers after cropping, got {len(idxs)} from range {start_idx}-{end_idx}"
        )

    layer_read_workers = int(getattr(CFG, "layer_read_workers", 1) or 1)
    layer_read_workers = max(1, min(layer_read_workers, len(idxs)))

    if "frag" in fragment_id:
        images_list = []
        for i in idxs:
            image = None
            for image_path in _iter_layer_paths(i):
                image = _read_gray(image_path)
                if image is not None:
                    break
            if image is None:
                raise FileNotFoundError(
                    f"Could not read layer for {fragment_id}: {layers_dir}/{i}.[tif|tiff|png|jpg|jpeg]"
                )

            pad0 = (256 - image.shape[0] % 256)
            pad1 = (256 - image.shape[1] % 256)

            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
            image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
            np.clip(image, 0, 200, out=image)
            if fragment_id == '20230827161846':
                image = cv2.flip(image, 0)
            images_list.append(image)

        images = np.stack(images_list, axis=2)
        del images_list
        return images

    first = None
    for image_path in _iter_layer_paths(idxs[0]):
        first = _read_gray(image_path)
        if first is not None:
            break
    if first is None:
        raise FileNotFoundError(
            f"Could not read layer for {fragment_id}: {layers_dir}/{idxs[0]}.[tif|tiff|png|jpg|jpeg]"
        )

    base_h, base_w = first.shape[:2]
    pad0 = (256 - base_h % 256)
    pad1 = (256 - base_w % 256)
    out_h = base_h + pad0
    out_w = base_w + pad1

    images = np.zeros((out_h, out_w, len(idxs)), dtype=first.dtype)
    np.clip(first, 0, 200, out=first)
    if fragment_id == '20230827161846':
        first = cv2.flip(first, 0)
    images[:base_h, :base_w, 0] = first

    def _load_and_write(task):
        chan, i = task
        img = None
        for image_path in _iter_layer_paths(i):
            img = _read_gray(image_path)
            if img is not None:
                break
        if img is None:
            raise FileNotFoundError(
                f"Could not read layer for {fragment_id}: {layers_dir}/{i}.[tif|tiff|png|jpg|jpeg]"
            )
        if img.shape[0] != base_h or img.shape[1] != base_w:
            raise ValueError(
                f"{fragment_id}: layer {i:02} has shape {img.shape} but expected {(base_h, base_w)}"
            )
        np.clip(img, 0, 200, out=img)
        if fragment_id == '20230827161846':
            img = cv2.flip(img, 0)
        images[:base_h, :base_w, chan] = img
        return None

    tasks = [(chan, i) for chan, i in enumerate(idxs[1:], start=1)]
    if tasks:
        if layer_read_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=layer_read_workers) as executor:
                list(executor.map(_load_and_write, tasks))
        else:
            for task in tasks:
                _load_and_write(task)

    return images


def read_image_mask(
    fragment_id,
    start_idx=15,
    end_idx=45,
    *,
    layer_range=None,
    reverse_layers=False,
    label_suffix="",
    mask_suffix="",
    images=None,
):
    if images is None:
        images = read_image_layers(
            fragment_id,
            start_idx=start_idx,
            end_idx=end_idx,
            layer_range=layer_range,
        )

    if reverse_layers or fragment_id in REVERSE_LAYER_FRAGMENT_IDS_FALLBACK:
        images = images[:, :, ::-1]

    label_base = f"train_scrolls/{fragment_id}/{fragment_id}_inklabels{label_suffix}"
    mask = _read_gray(f"{label_base}.png")
    if mask is None:
        mask = _read_gray(f"{label_base}.tiff")
    if mask is None:
        mask = _read_gray(f"{label_base}.tif")
    if mask is None:
        raise FileNotFoundError(f"Could not read label for {fragment_id}: {label_base}.png/.tif/.tiff")

    mask_base = f"train_scrolls/{fragment_id}/{fragment_id}_mask{mask_suffix}"
    fragment_mask = _read_gray(f"{mask_base}.png")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tiff")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tif")
    if fragment_mask is None:
        raise FileNotFoundError(f"Could not read mask for {fragment_id}: {mask_base}.png/.tif/.tiff")
    if fragment_id == '20230827161846':
        fragment_mask = cv2.flip(fragment_mask, 0)

    def _assert_bottom_right_pad_compatible(a_name, a_hw, b_name, b_hw, multiple):
        a_h, a_w = [int(x) for x in a_hw]
        b_h, b_w = [int(x) for x in b_hw]

        def _check_dim(dim_name, a_dim, b_dim):
            small = min(a_dim, b_dim)
            big = max(a_dim, b_dim)
            padded = ((small + multiple - 1) // multiple) * multiple
            allowed = {small, padded}
            if small % multiple == 0:
                allowed.add(small + multiple)  # supports the legacy "always pad one block" variant

            if big not in allowed:
                raise ValueError(
                    f"{fragment_id}: {a_name} {a_hw} vs {b_name} {b_hw} mismatch. "
                    f"Only bottom/right padding to a multiple of {multiple} is allowed "
                    f"(see inference_resnet3d.py). Got {dim_name}={a_dim} vs {b_dim}."
                )

        _check_dim("height", a_h, b_h)
        _check_dim("width", a_w, b_w)

    if "frag" not in fragment_id:
        pad_multiple = 256
        _assert_bottom_right_pad_compatible("image", images.shape[:2], "label", mask.shape[:2], pad_multiple)
        _assert_bottom_right_pad_compatible("image", images.shape[:2], "mask", fragment_mask.shape[:2], pad_multiple)
        _assert_bottom_right_pad_compatible("label", mask.shape[:2], "mask", fragment_mask.shape[:2], pad_multiple)

    if "frag" in fragment_id:
        pad0 = max(0, images.shape[0] * 2 - fragment_mask.shape[0])
        pad1 = max(0, images.shape[1] * 2 - fragment_mask.shape[1])
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    else:
        fragment_mask_padded = np.zeros((images.shape[0], images.shape[1]), dtype=fragment_mask.dtype)
        h = min(fragment_mask.shape[0], fragment_mask_padded.shape[0])
        w = min(fragment_mask.shape[1], fragment_mask_padded.shape[1])
        fragment_mask_padded[:h, :w] = fragment_mask[:h, :w]
        fragment_mask = fragment_mask_padded
        del fragment_mask_padded

    if 'frag' in fragment_id:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1] // 2, fragment_mask.shape[0] // 2), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2), interpolation=cv2.INTER_AREA)

    target_h = min(images.shape[0], mask.shape[0], fragment_mask.shape[0])
    target_w = min(images.shape[1], mask.shape[1], fragment_mask.shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"{fragment_id}: empty shapes after alignment (images={images.shape}, label={mask.shape}, mask={fragment_mask.shape})"
        )

    images = images[:target_h, :target_w]
    mask = mask[:target_h, :target_w]
    fragment_mask = fragment_mask[:target_h, :target_w]

    mask = mask.astype('float32')
    mask /= 255
    if images.shape[0] != mask.shape[0] or images.shape[1] != mask.shape[1]:
        raise ValueError(f"{fragment_id}: label shape {mask.shape} does not match image shape {images.shape[:2]}")
    return images, mask, fragment_mask


def build_group_mappings(fragment_ids, segments_metadata, group_key="base_path"):
    fragment_to_group_name = {}
    for fragment_id in fragment_ids:
        seg_meta = (segments_metadata or {}).get(fragment_id, {}) or {}
        group_name = seg_meta.get(group_key)
        if group_name is None:
            group_name = seg_meta.get("base_path", fragment_id)
        fragment_to_group_name[fragment_id] = str(group_name)

    group_names = sorted(set(fragment_to_group_name.values()))
    group_name_to_idx = {name: i for i, name in enumerate(group_names)}
    fragment_to_group_idx = {fid: group_name_to_idx[g] for fid, g in fragment_to_group_name.items()}
    return group_names, group_name_to_idx, fragment_to_group_idx


def extract_patches(image, mask, fragment_mask, *, include_xyxys, filter_empty_tile):
    images = []
    masks = []
    xyxys = []

    stride = CFG.stride
    x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, stride))
    y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, stride))
    windows_dict = {}

    for a in y1_list:
        for b in x1_list:
            if filter_empty_tile and np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size] < 0.01):
                continue
            if np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size] == 0):
                continue

            for yi in range(0, CFG.tile_size, CFG.size):
                for xi in range(0, CFG.tile_size, CFG.size):
                    y1 = a + yi
                    x1 = b + xi
                    y2 = y1 + CFG.size
                    x2 = x1 + CFG.size
                    if (y1, y2, x1, x2) in windows_dict:
                        continue

                    windows_dict[(y1, y2, x1, x2)] = True
                    images.append(image[y1:y2, x1:x2])
                    masks.append(mask[y1:y2, x1:x2, None])
                    if include_xyxys:
                        xyxys.append([x1, y1, x2, y2])
                    assert image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans)

    return images, masks, xyxys


def _downsample_bool_mask_any(mask: np.ndarray, ds: int) -> np.ndarray:
    ds = max(1, int(ds))
    if mask is None:
        raise ValueError("mask is None")
    mask_bool = (mask > 0)
    h = int(mask_bool.shape[0])
    w = int(mask_bool.shape[1])
    ds_h = (h + ds - 1) // ds
    ds_w = (w + ds - 1) // ds
    pad_h = int(ds_h * ds - h)
    pad_w = int(ds_w * ds - w)
    if pad_h or pad_w:
        mask_bool = np.pad(mask_bool, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    mask_bool = mask_bool.reshape(ds_h, ds, ds_w, ds)
    return mask_bool.any(axis=(1, 3))


def _mask_border(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool is None:
        raise ValueError("mask_bool is None")
    mask_bool = mask_bool.astype(bool, copy=False)
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=bool)
    thickness = 5
    eroded = ndimage.binary_erosion(
        mask_bool,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
        iterations=thickness,
    )
    return mask_bool & ~eroded


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug


class CustomDataset(Dataset):
    def __init__(self, images, cfg, xyxys=None, labels=None, groups=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.groups = groups

        self.transform = transform
        self.xyxys = xyxys
        self.rotate = CFG.rotate

    def __len__(self):
        return len(self.images)

    def cubeTranslate(self, y):
        x = np.random.uniform(0, 1, 4).reshape(2, 2)
        x[x < .4] = 0
        x[x > .633] = 2
        x[(x > .4) & (x < .633)] = 1
        mask = cv2.resize(x, (x.shape[1] * 64, x.shape[0] * 64), interpolation=cv2.INTER_AREA)

        x = np.zeros((self.cfg.size, self.cfg.size, self.cfg.in_chans)).astype(np.uint8)
        for i in range(3):
            x = np.where(np.repeat((mask == 0).reshape(self.cfg.size, self.cfg.size, 1), self.cfg.in_chans, axis=2), y[:, :, i:self.cfg.in_chans + i], x)
        return x

    def fourth_augment(self, image):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(24, 30)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image

    def __getitem__(self, idx):
        group_id = 0
        if self.groups is not None:
            group_id = int(self.groups[idx])
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            xy = self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label = F.interpolate(label.unsqueeze(0), (self.cfg.size // 4, self.cfg.size // 4)).squeeze(0)
            return image, label, xy, group_id
        else:
            image = self.images[idx]
            label = self.labels[idx]
            # 3d rotate
            # image=image.transpose(2,1,0)#(c,w,h)
            # image=self.rotate(image=image)['image']
            # image=image.transpose(0,2,1)#(c,h,w)
            # image=self.rotate(image=image)['image']
            # image=image.transpose(0,2,1)#(c,w,h)
            # image=image.transpose(2,1,0)#(h,w,c)

            image = self.fourth_augment(image)

            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label = F.interpolate(label.unsqueeze(0), (self.cfg.size // 4, self.cfg.size // 4)).squeeze(0)
            return image, label, group_id


class CustomDatasetTest(Dataset):
    def __init__(self, images, xyxys, cfg, transform=None):
        self.images = images
        self.xyxys = xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy = self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

        return image, xy
