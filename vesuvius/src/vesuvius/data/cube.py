"""Paired CT and instance-label NRRD cubes from the packaged cube inventory."""

from pathlib import Path
import tempfile

import numpy as np

from vesuvius.utils.catalog import list_cubes


class Cube:
    """Load an annotated cube and slice it as ``volume, mask = cube[z, y, x]``.

    Coordinates select an entry in :func:`vesuvius.list_cubes`. ``normalize``
    divides CT values by their dtype maximum; instance labels remain unchanged.
    NRRDs retain the original Cube API's pynrrd ``index_order='F'`` convention.
    Downloads use vc_reader; decoding uses pynrrd because vc_reader does not
    currently decode NRRD. Install the ``volume-only`` extra for this workflow.
    """

    def __init__(self, scroll_id: int, energy: int, resolution: float,
                 z: int, y: int, x: int, cache: bool = False,
                 cache_dir=None, normalize: bool = False):
        self.scroll_id = scroll_id
        self.energy = energy
        self.resolution = resolution
        self.z, self.y, self.x = z, y, x
        self.cache = cache
        self.cache_dir = (Path(cache_dir) if cache_dir is not None else
                          Path.home() / 'vesuvius' / 'annotated-instances')
        self.normalize = normalize
        self.volume_url, self.mask_url = self.get_url_from_yaml()
        self.volume, self.mask = self.load_data()
        if normalize:
            info = np.finfo if np.issubdtype(self.volume.dtype, np.floating) else np.iinfo
            self.max_dtype = info(self.volume.dtype).max

    def get_url_from_yaml(self):
        """Return the volume and mask URLs for the selected inventory entry."""
        name = f'{self.z:05d}_{self.y:05d}_{self.x:05d}'
        base = (list_cubes().get(self.scroll_id, {}).get(self.energy, {})
                .get(self.resolution, {}).get(name))
        if base is None:
            raise ValueError(
                f'Cube not found: scroll={self.scroll_id}, energy={self.energy}, '
                f'resolution={self.resolution}, coordinates={name}')
        return tuple(f'{base.rstrip("/")}/{name}_{kind}.nrrd'
                     for kind in ('volume', 'mask'))

    def load_data(self):
        """Read both NRRDs, reusing cached files when caching is enabled."""
        import nrrd
        from vc_reader.reader import ObjectStore

        def read_pair(directory):
            arrays = []
            for url in (self.volume_url, self.mask_url):
                parent, filename = url.rsplit('/', 1)
                path = directory / filename
                if not path.exists():
                    if not ObjectStore(parent, auth=None).read_file(filename, path):
                        raise FileNotFoundError(url)
                array, _ = nrrd.read(str(path), index_order='F')
                arrays.append(array)
            return tuple(arrays)

        if self.cache:
            directory = self.cache_dir / f'{self.z:05d}_{self.y:05d}_{self.x:05d}'
            return read_pair(directory)
        with tempfile.TemporaryDirectory(prefix='vesuvius-cube-') as directory:
            return read_pair(Path(directory))

    def __getitem__(self, idx):
        if not isinstance(idx, tuple) or len(idx) != 3:
            raise IndexError('Invalid index. Must be a tuple of three elements.')
        volume = self.volume[idx]
        if self.normalize:
            volume = volume / self.max_dtype
        return volume, self.mask[idx]

    def activate_caching(self, cache_dir=None):
        """Enable the persistent download cache and populate it for this cube."""
        if not self.cache:
            self.cache_dir = (Path(cache_dir) if cache_dir is not None else
                              Path.home() / 'vesuvius' / 'annotated-instances')
            self.cache = True
            self.volume, self.mask = self.load_data()

    def deactivate_caching(self):
        """Disable caching and reload the cube from its source."""
        if self.cache:
            self.cache = False
            self.volume, self.mask = self.load_data()
