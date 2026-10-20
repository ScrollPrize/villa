"""Worker crop plans and batched, device-resident image processing with vc_reader."""
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from itertools import combinations, permutations
import random

import torch
from torch.utils.data import DataLoader
from vc_reader import transforms as A

NORMALIZATION_SCHEMES = ("robust", "zscore", "minmax", "rescale_to_01", "ct", "none")


def normalization_transforms(scheme="robust", intensity_properties=None):
    if scheme == "none":
        return []
    if scheme == "rescale_to_01":
        scheme = "minmax"
    if scheme == "ct":
        props = intensity_properties or {}
        return [A.Normalize("fixed", center=props["mean"], scale=max(props["std"], 1e-8),
                            bounds=(props["percentile_00_5"], props["percentile_99_5"]))]
    if scheme == "zscore":
        return [A.Normalize(scheme, nonzero=False, eps=1e-8)]
    if scheme not in NORMALIZATION_SCHEMES:
        raise ValueError(f"Unknown normalization scheme: {scheme}")
    return [A.Normalize(scheme)]


def normalize_batch(images, scheme="robust", *, device="cuda", intensity_properties=None):
    pipeline = A.Pipeline(normalization_transforms(scheme, intensity_properties),
                          targets={"image": A.Image(valid=None)}, backend="cuda", compact=False)
    return pipeline.batch({"image": images}, seeds=[0] * len(images), device=device)["image"]


@dataclass
class CropView:
    raw: torch.Tensor
    target: tuple[int, int, int]
    spatial: tuple = ()
    seed: int = 0
    augment: bool = False

    @property
    def shape(self):
        return (1, *self.target)

    def pin_memory(self):
        self.raw = self.raw.pin_memory()
        return self


class SpatialPlan:
    """Sample shape-preserving geometry once for CUDA images and CPU point metadata."""
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.planes = ["zyx"[a] + "zyx"[b] for a, b in combinations(range(3), 2)
                       if shape[a] == shape[b]]
        self.orders = ["".join("zyx"[a] for a in order) for order in permutations(range(3))
                       if tuple(shape[a] for a in order) == tuple(shape)]

    def __call__(self, **sample):
        view = sample["image"]
        operations = []
        if self.planes and random.random() < .5:
            operations.append(("rotate", random.choice(self.planes), random.choice((1, 2, 3))))
        if random.random() < .5:
            axes = "".join(a for a in "zyx" if random.random() < .5)
            if axes:
                operations.append(("flip", axes))
        if len(self.orders) > 1 and random.random() < .5:
            operations.append(("transpose", random.choice(self.orders)))
        view.spatial, view.seed, view.augment = tuple(operations), random.randrange(2**63 - 1), True
        paired = sample.get("paired_spatial_image")
        if paired is not None:
            paired.spatial, paired.seed = view.spatial, view.seed
        if sample.get("keypoints") is not None:
            sample["keypoints"] = transform_points(sample["keypoints"], self.shape, view.spatial)
        return sample


def transform_points(points, shape, operations):
    points = torch.as_tensor(points).clone()
    for operation in operations:
        if operation[0] == "rotate":
            a, b = ("zyx".index(axis) for axis in operation[1])
            for _ in range(operation[2]):
                old_a = points[:, a].clone()
                points[:, a] = shape[b] - 1 - points[:, b]
                points[:, b] = old_a
        elif operation[0] == "flip":
            for axis in operation[1]:
                a = "zyx".index(axis)
                points[:, a] = shape[a] - 1 - points[:, a]
        else:
            points = points[:, ["zyx".index(axis) for axis in operation[1]]]
    return points


def crop_collate(samples):
    """Keep raw crops and small metadata in workers; CUDA belongs to the parent."""
    return samples


class CudaBatchProcessor:
    def __init__(self, config, collate=None, *, device="cuda"):
        self.config, self.collate, self.device = dict(config), collate, torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Dinovol image processing requires CUDA")
        self.A = A
        self.pipelines = OrderedDict()
        # Fail before starting workers if vc_reader was built without CUDA.
        self.pipeline(("normalize",), lambda: normalization_transforms(
            self.config.get("normalization_scheme", "robust"), self.config.get("intensity_properties")))

    def pipeline(self, key, operations):
        if key not in self.pipelines:
            self.pipelines[key] = A.Pipeline(operations(), targets={"image": A.Image(valid=None)},
                                             backend="cuda", compact=False)
            # Crop shapes and geometry can vary indefinitely; bound native plan retention.
            if len(self.pipelines) > 256:
                self.pipelines.popitem(last=False)
        self.pipelines.move_to_end(key)
        return self.pipelines[key]

    def batch(self, pipeline, images, entries, offset=0):
        return pipeline.batch({"image": images}, seeds=[e[2].seed + offset for e in entries],
                              device=self.device)["image"]

    def scanner(self):
        A = self.A
        return [A.Smear(axis="x", shift=(5, 0), alpha=.2, previous=3, p=.3),
                A.SliceIllumination(axis="z", defects=(2, 5), width=(25, 50), reduction=(.3, 1.5),
                    base_p=(.2, .4), base_reduction=(.5, .9), p=.4, p_channel=.5)]

    def intensity(self, size):
        A = self.A
        operations = [
            A.BrightnessGradient(scale=(size / 6, size / 2), loc=(-.5, 1.5), strength=.5,
                                 mean_centered=True, clip=False, p=.2, p_channel=.5),
            A.LocalGamma(scale=(size / 6, size / 2), loc=(-.5, 1.5), gamma=(.5, 1.5), p=.2, p_channel=.5)]
        if self.config.get("gaussian_blur", True):
            operations.append(A.GaussianBlur(sigma=(.3, 1.5), shared_axes=False, p=.3, p_channel=.5))
        operations += [
            A.GaussianNoise(std=(0, .2), shared_noise=True, p=.3, p_channel=.5),
            A.Sharpen(strength=(.1, 1.5), p_clamp=.5, p=.2, p_channel=.5),
            A.OneOf([A.Contrast(factor=A.BalancedContrast(.75, 1.25), preserve_range=True, p=.3, p_channel=.5),
                     A.BrightnessMultiply(factor=A.BalancedContrast(.75, 1.25), p=.3, p_channel=.5)]),
            A.BrightnessAdd(shift=A.Normal(0, .5), p=.1, p_channel=.5),
            A.LowResolution(scale=(.25, 1), shared_axes=True, p=.4, p_channel=.5),
            A.Gamma(gamma=A.BalancedContrast(.7, 1.5), p_invert=1, p_retain_stats=1, p=.2),
            A.Gamma(gamma=A.BalancedContrast(.7, 1.5), p_invert=0, p_retain_stats=1, p=.4),
            A.Invert(p=.2, p_channel=.5)]
        if self.config.get("augmentation", "full") == "dino":
            return [A.OneOf([
                A.Contrast(factor=A.BalancedContrast(.75, 1.25), preserve_range=True, p=.3, p_channel=.5),
                A.BrightnessMultiply(factor=A.BalancedContrast(.75, 1.25), p=.3, p_channel=.5)]),
                A.BrightnessAdd(shift=A.Normal(0, .5), p=.1, p_channel=.5)]
        return operations

    def spatial(self, operations):
        A = self.A
        result = []
        for operation in operations:
            if operation[0] == "rotate":
                result.append(A.Rotate90(planes=[operation[1]], turns=[operation[2]]))
            elif operation[0] == "flip":
                result.append(A.Flip(axes=operation[1], p_axis=1))
            else:
                result.append(A.Transpose(order=operation[1]))
        return result

    def __call__(self, samples):
        single = isinstance(samples[0], CropView)
        containers = [[view] for view in samples] if single else [
            sample[field] for sample in samples
            for field in ("global_views", "local_views", "gram_teacher_views") if field in sample]
        groups = defaultdict(list)
        for views in containers:
            for index, view in enumerate(views):
                groups[(tuple(view.raw.shape), view.target, view.augment)].append((views, index, view))
        spatial_groups = defaultdict(list)
        for (shape, target, augment), entries in groups.items():
            scanner = augment and self.config.get("augmentation", "full") == "full"
            pipeline = self.pipeline(("prepare", shape, target, scanner), lambda: [
                *normalization_transforms(self.config.get("normalization_scheme", "robust"),
                                          self.config.get("intensity_properties")),
                A.Affine(scale_factors=[out / size for out, size in zip(target, shape)],
                         shape=target, border=True),
                *(self.scanner() if scanner else [])])
            raw = torch.stack([v.raw for _, _, v in entries]).unsqueeze(1)
            images = self.batch(pipeline, raw, entries)
            for entry, image in zip(entries, images):
                spatial_groups[(target, entry[2].spatial, augment)].append((*entry, image))
        intensity_groups = defaultdict(list)
        for (target, operations, augment), entries in spatial_groups.items():
            pipeline = self.pipeline(("spatial", operations), lambda: self.spatial(operations))
            images = self.batch(pipeline, torch.stack([e[3] for e in entries]), entries)
            for entry, image in zip(entries, images):
                intensity_groups[(target, augment)].append((*entry[:3], image))
        for (target, augment), entries in intensity_groups.items():
            images = torch.stack([e[3] for e in entries])
            if augment:
                pipeline = self.pipeline(("intensity", target), lambda: self.intensity(min(target)))
                images = self.batch(pipeline, images, entries, offset=1)
            for (views, index, _, _), image in zip(entries, images):
                views[index] = image
        if single:
            return torch.stack([views[0] for views in containers])
        return self.collate(samples) if self.collate is not None else samples


class CudaDataLoader(DataLoader):
    """Read ahead on CPU, deliver fully processed CUDA batches on the rank's device."""
    def __init__(self, dataset, *, collate_fn=None, device="cuda", **kwargs):
        self.processor = CudaBatchProcessor(dataset.config, collate_fn, device=device)
        if kwargs.get("num_workers", 0):
            kwargs.setdefault("multiprocessing_context", "spawn")
        super().__init__(dataset, collate_fn=crop_collate, **kwargs)

    def __iter__(self):
        for samples in super().__iter__():
            yield self.processor(samples)
