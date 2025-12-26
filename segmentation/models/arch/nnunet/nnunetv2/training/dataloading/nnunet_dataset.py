import math
import os
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import shutil

try:
    import blosc2
    BLOSC2_AVAILABLE = True
except ImportError:
    BLOSC2_AVAILABLE = False

from batchgenerators.utilities.file_and_folder_operations import join, load_pickle, isfile, write_pickle
from nnunetv2.training.dataloading.utils import get_case_identifiers


class nnUNetDataset(object):
    def __init__(self, folder: str, case_identifiers: List[str] = None,
                 num_images_properties_loading_threshold: int = 0,
                 folder_with_segs_from_previous_stage: str = None):
        """
        This does not actually load the dataset. It merely creates a dictionary where the keys are training case names and
        the values are dictionaries containing the relevant information for that case.
        dataset[training_case] -> info
        Info has the following key:value pairs:
        - dataset[case_identifier]['properties']['data_file'] -> the full path to the npz file associated with the training case
        - dataset[case_identifier]['properties']['properties_file'] -> the pkl file containing the case properties

        In addition, if the total number of cases is < num_images_properties_loading_threshold we load all the pickle files
        (containing auxiliary information). This is done for small datasets so that we don't spend too much CPU time on
        reading pkl files on the fly during training. However, for large datasets storing all the aux info (which also
        contains locations of foreground voxels in the images) can cause too much RAM utilization. In that
        case is it better to load on the fly.

        If properties are loaded into the RAM, the info dicts each will have an additional entry:
        - dataset[case_identifier]['properties'] -> pkl file content

        IMPORTANT! THIS CLASS ITSELF IS READ-ONLY. YOU CANNOT ADD KEY:VALUE PAIRS WITH nnUNetDataset[key] = value
        USE THIS INSTEAD:
        nnUNetDataset.dataset[key] = value
        (not sure why you'd want to do that though. So don't do it)
        """
        super().__init__()
        # print('loading dataset')
        if case_identifiers is None:
            case_identifiers = get_case_identifiers(folder)
        case_identifiers.sort()

        # Detect file format (blosc2 vs npz)
        self.use_blosc2 = isfile(join(folder, f"{case_identifiers[0]}.b2nd")) if case_identifiers else False
        if self.use_blosc2:
            if not BLOSC2_AVAILABLE:
                raise ImportError("blosc2 is required for .b2nd files. Install with: pip install blosc2")
            blosc2.set_nthreads(1)

        self.dataset = {}
        for c in case_identifiers:
            self.dataset[c] = {}
            if self.use_blosc2:
                self.dataset[c]['data_file'] = join(folder, f"{c}.b2nd")
                self.dataset[c]['seg_file'] = join(folder, f"{c}_seg.b2nd")
            else:
                self.dataset[c]['data_file'] = join(folder, f"{c}.npz")
            self.dataset[c]['properties_file'] = join(folder, f"{c}.pkl")
            if folder_with_segs_from_previous_stage is not None:
                self.dataset[c]['seg_from_prev_stage_file'] = join(folder_with_segs_from_previous_stage, f"{c}.npz")

        if len(case_identifiers) <= num_images_properties_loading_threshold:
            for i in self.dataset.keys():
                self.dataset[i]['properties'] = load_pickle(self.dataset[i]['properties_file'])

        self.keep_files_open = ('nnUNet_keep_files_open' in os.environ.keys()) and \
                               (os.environ['nnUNet_keep_files_open'].lower() in ('true', '1', 't'))
        # print(f'nnUNetDataset.keep_files_open: {self.keep_files_open}')

    def __getitem__(self, key):
        ret = {**self.dataset[key]}
        if 'properties' not in ret.keys():
            ret['properties'] = load_pickle(ret['properties_file'])
        return ret

    def __setitem__(self, key, value):
        return self.dataset.__setitem__(key, value)

    def keys(self):
        return self.dataset.keys()

    def __len__(self):
        return self.dataset.__len__()

    def items(self):
        return self.dataset.items()

    def values(self):
        return self.dataset.values()

    def load_case(self, key):
        entry = self[key]

        if self.use_blosc2:
            # Blosc2 format (.b2nd files)
            dparams = {'nthreads': 1}
            mmap_kwargs = {} if os.name == "nt" else {'mmap_mode': 'r'}

            if 'open_data_file' in entry.keys():
                data = entry['open_data_file']
            else:
                data = blosc2.open(urlpath=entry['data_file'], mode='r', dparams=dparams, **mmap_kwargs)
                if self.keep_files_open:
                    self.dataset[key]['open_data_file'] = data

            if 'open_seg_file' in entry.keys():
                seg = entry['open_seg_file']
            else:
                seg = blosc2.open(urlpath=entry['seg_file'], mode='r', dparams=dparams, **mmap_kwargs)
                if self.keep_files_open:
                    self.dataset[key]['open_seg_file'] = seg
        else:
            # NPZ/NPY format
            if 'open_data_file' in entry.keys():
                data = entry['open_data_file']
                # print('using open data file')
            elif isfile(entry['data_file'][:-4] + ".npy"):
                data = np.load(entry['data_file'][:-4] + ".npy", 'r')
                if self.keep_files_open:
                    self.dataset[key]['open_data_file'] = data
                    # print('saving open data file')
            else:
                data = np.load(entry['data_file'])['data']

            if 'open_seg_file' in entry.keys():
                seg = entry['open_seg_file']
                # print('using open data file')
            elif isfile(entry['data_file'][:-4] + "_seg.npy"):
                seg = np.load(entry['data_file'][:-4] + "_seg.npy", 'r')
                if self.keep_files_open:
                    self.dataset[key]['open_seg_file'] = seg
                    # print('saving open seg file')
            else:
                seg = np.load(entry['data_file'])['seg']

        if 'seg_from_prev_stage_file' in entry.keys():
            if isfile(entry['seg_from_prev_stage_file'][:-4] + ".npy"):
                seg_prev = np.load(entry['seg_from_prev_stage_file'][:-4] + ".npy", 'r')
            else:
                seg_prev = np.load(entry['seg_from_prev_stage_file'])['seg']
            seg = np.vstack((seg, seg_prev[None]))

        return data, seg, entry['properties']


class nnUNetDatasetBlosc2:
    """
    Static methods for saving preprocessed data in blosc2 format.
    Used by the preprocessor to save cases.
    """

    @staticmethod
    def save_case(
            data: np.ndarray,
            seg: np.ndarray,
            properties: dict,
            output_filename_truncated: str,
            chunks=None,
            blocks=None,
            chunks_seg=None,
            blocks_seg=None,
            clevel: int = 8,
            codec=None
    ):
        if not BLOSC2_AVAILABLE:
            raise ImportError("blosc2 is required for saving .b2nd files. Install with: pip install blosc2")

        if codec is None:
            codec = blosc2.Codec.ZSTD

        blosc2.set_nthreads(1)
        if chunks_seg is None:
            chunks_seg = chunks
        if blocks_seg is None:
            blocks_seg = blocks

        cparams = {
            'codec': codec,
            'clevel': clevel,
        }
        blosc2.asarray(np.ascontiguousarray(data), urlpath=output_filename_truncated + '.b2nd', chunks=chunks,
                       blocks=blocks, cparams=cparams)
        blosc2.asarray(np.ascontiguousarray(seg), urlpath=output_filename_truncated + '_seg.b2nd', chunks=chunks_seg,
                       blocks=blocks_seg, cparams=cparams)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def comp_blosc2_params(
            image_size: Tuple[int, ...],
            patch_size: Union[Tuple[int, int], Tuple[int, int, int]],
            bytes_per_pixel: int = 4,
            l1_cache_size_per_core_in_bytes: int = 32768,
            l3_cache_size_per_core_in_bytes: int = 1441792,
            safety_factor: float = 0.8
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """
        Computes a recommended block and chunk size for saving arrays with blosc v2.
        """
        num_channels = image_size[0]
        if len(patch_size) == 2:
            patch_size = (1, *patch_size)
        patch_size = np.array(patch_size)
        block_size = np.array((num_channels, *[2 ** (max(0, math.ceil(math.log2(i)))) for i in patch_size]))

        # shrink the block size until it fits in L1
        estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel
        while estimated_nbytes_block > (l1_cache_size_per_core_in_bytes * safety_factor):
            axis_order = np.argsort(block_size[1:] / patch_size)[::-1]
            idx = 0
            picked_axis = axis_order[idx]
            while block_size[picked_axis + 1] == 1 or block_size[picked_axis + 1] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            block_size[picked_axis + 1] = 2 ** (max(0, math.floor(math.log2(block_size[picked_axis + 1] - 1))))
            block_size[picked_axis + 1] = min(block_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_block = np.prod(block_size) * bytes_per_pixel

        block_size = np.array([min(i, j) for i, j in zip(image_size, block_size)])

        # tile the blocks into chunks until we hit image_size or the l3 cache per core limit
        chunk_size = deepcopy(block_size)
        estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
        while estimated_nbytes_chunk < (l3_cache_size_per_core_in_bytes * safety_factor):
            if patch_size[0] == 1 and all([i == j for i, j in zip(chunk_size[2:], image_size[2:])]):
                break
            if all([i == j for i, j in zip(chunk_size, image_size)]):
                break
            axis_order = np.argsort(chunk_size[1:] / block_size[1:])
            idx = 0
            picked_axis = axis_order[idx]
            while chunk_size[picked_axis + 1] == image_size[picked_axis + 1] or patch_size[picked_axis] == 1:
                idx += 1
                picked_axis = axis_order[idx]
            chunk_size[picked_axis + 1] += block_size[picked_axis + 1]
            chunk_size[picked_axis + 1] = min(chunk_size[picked_axis + 1], image_size[picked_axis + 1])
            estimated_nbytes_chunk = np.prod(chunk_size) * bytes_per_pixel
            if np.mean([i / j for i, j in zip(chunk_size[1:], patch_size)]) > 1.5:
                chunk_size[picked_axis + 1] -= block_size[picked_axis + 1]
                break
        chunk_size = [min(i, j) for i, j in zip(image_size, chunk_size)]

        return tuple(block_size), tuple(chunk_size)


if __name__ == '__main__':
    # this is a mini test. Todo: We can move this to tests in the future (requires simulated dataset)

    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset003_Liver/3d_lowres'
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0) # this should not load the properties!
    # this SHOULD HAVE the properties
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # amazing. I am the best.

    # this should have the properties
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=1000)
    # now rename the properties file so that it does not exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now we should still be able to access the properties because they have already been loaded
    ks = ds['liver_0'].keys()
    assert 'properties' in ks
    # move file back
    shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

    # this should not have the properties
    ds = nnUNetDataset(folder, num_images_properties_loading_threshold=0)
    # now rename the properties file so that it does not exist anymore
    shutil.move(join(folder, 'liver_0.pkl'), join(folder, 'liver_XXX.pkl'))
    # now this should crash
    try:
        ks = ds['liver_0'].keys()
        raise RuntimeError('we should not have come here')
    except FileNotFoundError:
        print('all good')
        # move file back
        shutil.move(join(folder, 'liver_XXX.pkl'), join(folder, 'liver_0.pkl'))

