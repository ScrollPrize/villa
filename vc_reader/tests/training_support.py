"""Training test support: Dataset, collation and CPU/CUDA augmentation.

No random region selection or supervision policy is implemented here. Importing
this module does not import Torch; only loader construction requires it.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import operator
import os
from pathlib import Path


def load_build(directory=None):
    """Use an existing build, including in spawned workers; never install it."""
    directory = directory or os.environ.get('VC_READER_BUILD_DIR')
    if directory:
        from extension_build import load_extensions
        load_extensions(Path(directory))


@dataclass(frozen=True)
class ReadSpec:
    reference: object  # Label asset for paired; Volume view for other operations.
    mode: str = 'aligned'
    targets: dict = field(default_factory=dict)
    image: object = None
    level: int = 0
    resolution: object = None
    layers: slice = field(default_factory=lambda: slice(None))
    surface: object = None
    offsets_um: tuple = (-9.6, 0., 9.6)
    label_target: str = 'label'
    thickness_um: float = 9.6

    def open(self, *, backend='cpu', **options):
        if self.resolution is not None:
            self.resolution.resolve(**{key: value for key, value in options.items()
                                       if key not in ('sampling_cache_bytes', 'plan_cache_bytes')})
        if self.mode == 'paired':
            return self.reference.open_paired(image=self.image, level=self.level,
                                              layers=self.layers, backend=backend, **options)
        if self.mode not in ('aligned', 'render', 'project'):
            raise ValueError(f'unknown read mode: {self.mode}')
        return self.reference.open_aligned(self.targets, level=self.level, backend=backend, **options)

    def read(self, reader, region):
        start, size = region
        if self.mode == 'render':
            return reader.render_surface(self.surface, start, size, offsets_um=self.offsets_um)
        if self.mode == 'project':
            return reader.project_surface(self.surface, start, size, label_target=self.label_target,
                                          thickness_um=self.thickness_um)
        return reader.read(start, size)


@dataclass(frozen=True)
class Array:
    target: str
    field: str = 'data'  # data, valid, or masks/<name>
    role: str = 'image'
    components: str = 'xyz'
    valid: str | None = None
    layers: bool = False


def image_arrays(*, layers=False):
    return {'image': Array('image', valid='image_valid', layers=layers),
            'image_valid': Array('image', 'valid', 'valid', layers=layers)}


def label_arrays(name='label', masks=()):
    return {name: Array(name, role='label'),
            name + '_valid': Array(name, 'valid', 'valid'),
            **{name + '_' + mask: Array(name, 'masks/' + mask, 'label') for mask in masks}}


def select_arrays(sample, arrays):
    """Keep categorical dtypes and every layer's validity; flatten only by declaration."""
    import numpy as np
    result = {}
    for name, declaration in arrays.items():
        target = sample[declaration.target]
        value = (target.masks[declaration.field[6:]] if declaration.field.startswith('masks/')
                 else getattr(target, declaration.field))
        if declaration.role not in ('label', 'valid') and value.dtype not in (np.uint8, np.float32):
            value = value.astype(np.float32)
        if declaration.layers:
            if value.ndim != 4:
                raise ValueError(f'{name}: layer arrays must have CDHW layout')
            if declaration.field == 'valid' and value.shape != target.data.shape:
                # Each channel has the same per-layer geometric support. Broadcast
                # across C, never reduce D; Image requires 1 or C*D validity channels.
                value = np.broadcast_to(value, target.data.shape)
            channels, depth, height, width = value.shape
            result[name + '_channels'] = channels
            result[name + '_layers'] = depth
            value = value.reshape(channels * depth, height, width)
        # Default collation needs native-endian, writable NumPy storage. Copy
        # only when a byte-swapped source or broadcast view requires it.
        if not value.dtype.isnative or not value.flags.writeable:
            value = value.astype(value.dtype.newbyteorder('='), copy=True)
        result[name] = value
    return result


class TrainingDataset:
    """Map-style test Dataset with explicit (reader_index, start, size) requests.

    One decoded chunk budget is shared by all readers in each process. Sampling
    and plan budgets are per reader. Use spawn when the parent has opened data.
    Views retain the complete overlay relationship graph when pickled.
    """
    def __init__(self, specs, requests, arrays, *, seeds, memory_cache_bytes=64 << 20,
                 threads=1, sampling_cache_bytes=16 << 20, plan_cache_bytes=4 << 20,
                 cache_dir=None, offline=False, build_dir=None):
        self.specs, self.requests, self.arrays = tuple(specs), tuple(requests), dict(arrays)
        self.seeds = tuple(map(operator.index, seeds))
        if len(self.requests) != len(self.seeds):
            raise ValueError('supply one explicit augmentation seed per request')
        if any(not 0 <= seed < 2**64 for seed in self.seeds):
            raise ValueError('seeds must be unsigned 64-bit integers')
        if threads < 1 or memory_cache_bytes < 0:
            raise ValueError('threads must be positive and cache budget nonnegative')
        self.memory_cache_bytes, self.build_dir = memory_cache_bytes, build_dir
        self.options = dict(threads=threads, sampling_cache_bytes=sampling_cache_bytes,
                            plan_cache_bytes=plan_cache_bytes, cache_dir=cache_dir, offline=offline)
        self._runtime = self._finalizer = None

    def __len__(self):
        return len(self.requests)

    def __getstate__(self):
        return {**self.__dict__, '_runtime': None, '_finalizer': None}

    def reader(self, index):
        if self._runtime is None:
            load_build(self.build_dir)
            from multiprocessing.util import Finalize
            from vc_reader.reader import MemoryCache
            stack = ExitStack()
            self._runtime = (os.getpid(), stack, MemoryCache(self.memory_cache_bytes), {})
            self._finalizer = Finalize(self, stack.close, exitpriority=10)
        pid, stack, cache, readers = self._runtime
        if pid != os.getpid():
            raise RuntimeError('an opened Dataset must use spawn workers')
        if index not in readers:
            readers[index] = stack.enter_context(self.specs[index].open(memory_cache=cache, **self.options))
        return readers[index]

    def read_sample(self, index):
        spec, start, size = self.requests[index]
        return self.specs[spec].read(self.reader(spec), (start, size))

    def __getitem__(self, index):
        import numpy as np
        result = select_arrays(self.read_sample(index), self.arrays)
        result.update(seed=np.asarray(self.seeds[index], dtype=np.uint64), index=index)
        return result

    def close(self):
        if self._runtime is not None:
            if self._runtime[0] != os.getpid():
                raise RuntimeError('close the Dataset in its owning process')
            self._finalizer()
            self._runtime = self._finalizer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def pipeline(arrays, *, spatial_dims, backend='cuda', transforms=None):
    from vc_reader import transforms as A
    if spatial_dims not in (2, 3):
        raise ValueError('declare spatial_dims=2 or 3')
    targets = {}
    for name, array in arrays.items():
        if array.layers and spatial_dims != 2:
            raise ValueError('flattened layers require a 2D pipeline')
        if array.role == 'image':
            targets[name] = A.Image(valid=array.valid)
        elif array.role in ('vector', 'normal', 'coordinate'):
            if len(array.components) != spatial_dims:
                raise ValueError('field components must match the augmentation coordinate frame')
            constructor = {'vector': A.VectorField, 'normal': A.NormalField,
                           'coordinate': A.CoordinateField}[array.role]
            targets[name] = constructor(components=array.components)
        else:
            targets[name] = {'label': A.Label, 'scalar': A.ScalarField, 'valid': A.ValidMask}[array.role]()
    if transforms is None:
        transforms = [A.Normalize('minmax'), A.Flip(axes='yx' if spatial_dims == 2 else 'zyx', p_axis=.5)]
    return A.Pipeline(transforms, targets=targets, backend=backend)


def restore_layers(batch, name='image'):
    """Restore BCDHW using CPU scalar metadata; does not copy image/device data."""
    channels = batch[name + '_channels'].tolist()
    layers = batch[name + '_layers'].tolist()
    if len(set(zip(channels, layers))) != 1:
        raise ValueError('a batch must share its channel/layer layout')
    value = batch[name]
    return value.reshape(value.shape[0], channels[0], layers[0], *value.shape[-2:])


@contextmanager
def data_loader(dataset, *, batch_size=2, workers=2, pin_memory=False):
    """Own persistent worker lifetime, including early breaks and exceptions."""
    import torch
    options = dict(batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=pin_memory)
    if workers:
        options.update(multiprocessing_context='spawn', persistent_workers=True, prefetch_factor=2)
    loader = torch.utils.data.DataLoader(dataset, **options)
    try:
        yield loader
    finally:
        # PyTorch has no public persistent-worker close method. Keep the one
        # private lifecycle call here so readers' multiprocessing finalizers run.
        iterator = getattr(loader, '_iterator', None)
        if iterator is not None:
            iterator._shutdown_workers()
        dataset.close()


def augment_cpu_sample(augmentation, sample, *, seed):
    # CudaLoader promotes uint8 on device. The CPU path performs that conversion
    # here, keeping Dataset storage and host-to-device traffic compact.
    import numpy as np
    targets = {}
    for key, declaration in augmentation.targets.items():
        value = sample[key]
        if declaration._constructor.__name__ not in ('Label', 'ValidMask'):
            value = value.astype(np.float32, copy=False) if isinstance(value, np.ndarray) else value.float()
        targets[key] = value
    return augmentation(targets, seed=seed)


def cpu_batches(loader, augmentation):
    """CPU alternative with the same declarations and explicit per-sample seeds."""
    from torch.utils.data import default_collate
    for batch in loader:
        samples = [augment_cpu_sample(augmentation, {key: batch[key][i] for key in augmentation.targets}, seed=int(seed))
                   for i, seed in enumerate(batch['seed'].tolist())]
        yield {**batch, **default_collate(samples)}
