"""Worker crop plans and batched image processing with vc_reader on CPU or CUDA."""
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from itertools import permutations
from functools import lru_cache
import copy
import queue
import random
import threading

import torch
from torch.utils.data import DataLoader
from vc_reader import transforms as A
from vc_reader import cuda as reader_cuda

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


def _image_dtype(images, device):
    # Preserve compact uint8 uploads; convert other source types on the device.
    if images.dtype not in (torch.uint8, torch.float32):
        images = images.to(device, non_blocking=True).float()
    return images


def normalize_batch(images, scheme="robust", *, device="cuda", intensity_properties=None):
    pipeline = A.Pipeline(normalization_transforms(scheme, intensity_properties),
                          targets={"image": A.Image(valid=None)}, backend="cuda", compact=False)
    return pipeline.batch({"image": _image_dtype(images, device)}, seeds=[0] * len(images), device=device)["image"]


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
        self.orders = ["".join("zyx"[a] for a in order) for order in permutations(range(3))
                       if tuple(shape[a] for a in order) == tuple(shape)]

    def __call__(self, **sample):
        view = sample["image"]
        paired = sample.get("paired_spatial_image")
        orders = self.orders
        if paired is not None:
            orders = [order for order in orders
                      if tuple(paired.target["zyx".index(axis)] for axis in order) == paired.target]
        # A uniform axis permutation and three independent reflections enumerate
        # all 48 cube symmetries: 24 proper rotations and their mirrored versions.
        order = random.choice(orders)
        axes = "".join(axis for axis in "zyx" if random.random() < .5)
        operations = []
        if order != "zyx":
            operations.append(("transpose", order))
        if axes:
            operations.append(("flip", axes))
        view.spatial, view.seed, view.augment = tuple(operations), random.randrange(2**63 - 1), True
        if paired is not None:
            paired.spatial, paired.seed = view.spatial, view.seed
        if sample.get("keypoints") is not None:
            sample["keypoints"] = point_pipeline(view.spatial).map_points(
                {"points": sample["keypoints"]}, shape=self.shape, seed=view.seed)["points"]
        return sample


def spatial_transforms(operations):
    return [A.Flip(axes=value, p_axis=1) if name == "flip" else A.Transpose(order=value)
            for name, value in operations]


@lru_cache(maxsize=48)
def point_pipeline(operations):
    # Only small point metadata runs in workers; image geometry runs on CUDA.
    return A.Pipeline(spatial_transforms(operations), targets={"points": A.Points()}, backend="cpu")


def crop_collate(samples):
    """Keep raw crops and small metadata in workers; CUDA belongs to the parent."""
    return samples


class BatchProcessor:
    """Run native vc_reader pipelines on the device's backend: batched on CUDA, per view on CPU.

    Every view carries its own seed, so a view's output does not depend on the other
    views processed with it (alone, reordered or in chunks give the same result).
    """
    def __init__(self, config, collate=None, *, device="cuda"):
        self.config, self.collate, self.device = dict(config), collate, torch.device(device)
        if self.device.type not in ("cuda", "cpu"):
            raise ValueError(f"Dinovol image processing runs on CPU or CUDA, got {self.device}")
        self.backend = self.device.type
        self.A = A
        self.pipelines = OrderedDict()
        # Fail before starting workers if vc_reader was built without the backend.
        self.pipeline(("normalize",), lambda: normalization_transforms(
            self.config.get("normalization_scheme", "robust"), self.config.get("intensity_properties")))
        self.device_index = None
        if self.backend == "cuda":
            self.device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()

    def pipeline(self, key, operations, *, targets=None):
        if key not in self.pipelines:
            self.pipelines[key] = A.Pipeline(operations(), targets=targets or {"image": A.Image(valid=None)},
                                             backend=self.backend, compact=False)
            # Crop shapes and geometry can vary indefinitely; bound native plan retention.
            if len(self.pipelines) > 256:
                self.pipelines.popitem(last=False)
        self.pipelines.move_to_end(key)
        return self.pipelines[key]

    def batch(self, pipeline, images, entries, offset=0):
        seeds = [e[2].seed + offset for e in entries]
        return self.run(pipeline, images, seeds)

    def run(self, pipeline, images, seeds):
        """Apply `pipeline` to (B, C, ...) images with one seed per image."""
        if self.backend == "cpu":
            if images.dtype not in (torch.uint8, torch.float32):
                images = images.float()
            outputs = [pipeline({"image": image}, seed=int(seed))["image"] for image, seed in zip(images, seeds)]
            return torch.stack([torch.as_tensor(output) for output in outputs])
        inputs = {"image": _image_dtype(images, self.device)}
        try:
            return pipeline.batch(inputs, seeds=seeds, device=self.device)["image"]
        except RuntimeError as exc:
            if "out of memory" not in str(exc):
                raise
            # vc_reader allocates scratch outside PyTorch, whose cache can hold nearly all free
            # device memory after a training step. Release that cache and retry with the same
            # seeds on a rebuilt pipeline (retrying the same one was seen to fail again).
            stats = reader_cuda.pool_stats(device=self.device_index)
            print(f"vc_reader transform failed, rebuilding its pipeline and retrying: {exc}; transform pool reserved "
                  f"{stats['reserved'] / 2**20:.0f} MiB, used peak {stats['used_high'] / 2**20:.0f} MiB", flush=True)
            key = next(k for k, v in self.pipelines.items() if v is pipeline)
            pipeline = copy.copy(pipeline)  # rebuilt from its declaration (Pipeline.__setstate__)
            self.pipelines[key] = pipeline
            torch.cuda.empty_cache()
            try:
                return pipeline.batch(inputs, seeds=seeds, device=self.device)["image"]
            except RuntimeError as retry_exc:
                free, total = torch.cuda.mem_get_info(self.device)
                raise RuntimeError(
                    f"{retry_exc} (after rebuilding the pipeline) for {key}, input {tuple(inputs['image'].shape)} "
                    f"{inputs['image'].dtype}; device free {free / 2**30:.2f} of {total / 2**30:.2f} GiB, "
                    f"PyTorch allocated {torch.cuda.memory_allocated(self.device) / 2**30:.2f} "
                    f"reserved {torch.cuda.memory_reserved(self.device) / 2**30:.2f} GiB"
                ) from exc

    def scanner(self, target):
        A = self.A
        depth = target[0]  # slice defects run along z, so their width scales with the view's depth
        return [A.Smear(axis="x", shift=(2, 0), alpha=.1, previous=2, p=.10),
                A.SliceIllumination(axis="z", defects=(1, 3), width=(depth / 12, depth / 4),
                    reduction=(.75, 1.1), base_p=(.1, .2), base_reduction=(.9, 1), p=.15)]

    def intensity_recipe(self, target):
        """Named per-view intensity operations of the configured augmentation recipe."""
        A = self.A
        # Length scales follow the view's largest extent (its plane, for slab views).
        size = max(target)
        if self.config.get("augmentation", "full") == "dino":
            return [("OneOf", A.OneOf([
                A.Contrast(factor=A.BalancedContrast(.75, 1.25), preserve_range=True, p=.3, p_channel=.5),
                A.BrightnessMultiply(factor=A.BalancedContrast(.75, 1.25), p=.3, p_channel=.5)])),
                ("BrightnessAdd", A.BrightnessAdd(shift=A.Normal(0, .5), p=.1, p_channel=.5))]
        # Probabilities apply once per view; every channel participates when selected.
        local_scale = (size / 6, size / 2)
        return [
            ("BiasField", A.BiasField(std=(.05, .2), scale=.025, p=.15)),
            ("BrightnessGradient", A.BrightnessGradient(scale=local_scale, loc=(-.5, 1.5), strength=.25,
                                                        mean_centered=True, clip=False, p=.15)),
            ("PlasmaBrightnessContrast", A.PlasmaBrightnessContrast(brightness=(-.1, .1), contrast=(0, .15), p=.10)),
            ("Vignetting", A.Vignetting(strength=(.1, .3), radius=.8, p=.10)),
            ("CLAHE", A.CLAHE(clip_limit=(1.5, 2.5), tile_shape=tuple(max(4, extent // 4) for extent in target),
                              bins=256, p=.10)),
            ("Contrast", A.Contrast(factor=A.BalancedContrast(.8, 1.2), preserve_range=True, p=.25)),
            ("LocalContrast", A.LocalContrast(scale=local_scale, loc=(-.5, 1.5), factor=(.8, 1.2), p=.15)),
            ("Gamma", A.Gamma(gamma=A.BalancedContrast(.8, 1.25), p_retain_stats=1, p=.25)),
            ("LocalGamma", A.LocalGamma(scale=local_scale, loc=(-.5, 1.5), gamma=(.8, 1.25), p=.15)),
            ("GaussianBlur", A.GaussianBlur(sigma=(.3, .8), shared_axes=False, p=.08)),
            ("Defocus", A.Defocus(radius=(1, 2), p=.05)),
            ("MedianFilter", A.MedianFilter(size=3, p=.05)),
            ("LowResolution", A.LowResolution(scale=(.6, 1), shared_axes=True, p=.10)),
            ("GibbsRinging", A.GibbsRinging(cutoff=(.75, .95), p=.08)),
            ("Sharpen", A.Sharpen(strength=(.1, .4), p_clamp=1, p=.10)),
            ("UnsharpMask", A.UnsharpMask(strength=(.1, .3), sigma=(.5, 1), p=.10)),
            ("BrightnessAdd", A.BrightnessAdd(shift=A.Normal(0, .15), p=.15)),
            ("Invert", A.Invert(p=.15)),
        ]

    def intensity(self, target, exclude=()):
        exclude = set(exclude)
        return [operation for name, operation in self.intensity_recipe(target) if name not in exclude]

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
                # Rician magnitude noise needs nonnegative source intensities;
                # std is in raw uint8 levels, before centering and resampling.
                *([A.RicianNoise(std=(1, 4), p=.10)] if scanner else []),
                *normalization_transforms(self.config.get("normalization_scheme", "robust"),
                                          self.config.get("intensity_properties")),
                A.Affine(scale_factors=[out / size for out, size in zip(target, shape)],
                         shape=target, border=True),
                *(self.scanner(target) if scanner else [])])
            raw = torch.stack([v.raw for _, _, v in entries]).unsqueeze(1)
            images = self.batch(pipeline, raw, entries)
            for entry, image in zip(entries, images):
                spatial_groups[(target, entry[2].spatial, augment)].append((*entry, image))
        intensity_groups = defaultdict(list)
        for (target, operations, augment), entries in spatial_groups.items():
            pipeline = self.pipeline(("spatial", operations), lambda: spatial_transforms(operations))
            images = self.batch(pipeline, torch.stack([e[3] for e in entries]), entries)
            for entry, image in zip(entries, images):
                intensity_groups[(target, augment)].append((*entry[:3], image))
        for (target, augment), entries in intensity_groups.items():
            images = torch.stack([e[3] for e in entries])
            if augment:
                pipeline = self.pipeline(("intensity", target), lambda: self.intensity(target))
                images = self.batch(pipeline, images, entries, offset=1)
            for (views, index, _, _), image in zip(entries, images):
                views[index] = image
        if single:
            return torch.stack([views[0] for views in containers])
        return self.collate(samples) if self.collate is not None else samples


class BatchDataLoader(DataLoader):
    """Read ahead on CPU workers, deliver fully processed batches on the rank's device.

    gpu_prefetch processes the next batch in a background thread on a side CUDA stream, so
    its host work and uploads overlap the consumer's compute. Batches are unchanged and in
    order: vc_reader augmentation is seeded per view, and the collate's global ``random``
    draws (iBOT masks) all happen on that one thread.
    """
    def __init__(self, dataset, *, collate_fn=None, device="cuda", gpu_prefetch=False, processor=None, **kwargs):
        self.processor = processor if processor is not None else BatchProcessor(dataset.config, collate_fn,
                                                                                device=device)
        self.gpu_prefetch = bool(gpu_prefetch)
        if self.gpu_prefetch and self.processor.device.type != "cuda":
            raise ValueError("gpu_prefetch requires a CUDA device")
        if kwargs.get("num_workers", 0):
            kwargs.setdefault("multiprocessing_context", "spawn")
        super().__init__(dataset, collate_fn=crop_collate, **kwargs)

    def __iter__(self):
        if self.gpu_prefetch:
            yield from _prefetched(super().__iter__(), self.processor, self.processor.device)
            return
        for samples in super().__iter__():
            yield self.processor(samples)


def _record_cuda(value, stream):
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_cuda(item, stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_cuda(item, stream)


def _prefetched(source, process, device):
    """Yield process(samples) for each item of source, computed one batch ahead."""
    stream = torch.cuda.Stream(device)
    slots = queue.Queue(maxsize=1)
    stop = threading.Event()

    def put(item):
        while not stop.is_set():
            try:
                slots.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def produce():
        try:
            with torch.cuda.device(device), torch.cuda.stream(stream):
                for samples in source:
                    batch = process(samples)
                    if not put(("batch", batch, stream.record_event())):
                        return
            put(("done",))
        except BaseException as exc:
            put(("error", exc))

    thread = threading.Thread(target=produce, name="cuda-batch-prefetch", daemon=True)
    thread.start()
    try:
        while True:
            kind, *rest = slots.get()
            if kind == "done":
                return
            if kind == "error":
                raise rest[0]
            batch, event = rest
            consumer = torch.cuda.current_stream(device)
            consumer.wait_event(event)
            # Side-stream allocations must not be reused until the consumer is done with them.
            _record_cuda(batch, consumer)
            yield batch
    finally:
        stop.set()
        # A producer blocked on a worker batch exits once it arrives; never hang shutdown on it.
        thread.join(timeout=30)
