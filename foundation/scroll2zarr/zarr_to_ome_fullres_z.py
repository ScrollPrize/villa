# Temporary: modified from
# https://github.com/KhartesViewer/scroll2zarr/blob/main/zarr_to_ome.py
# Will be merged into one script

import sys
import re
from pathlib import Path
import json
import shutil
import argparse
import copy
import numpy as np
import tifffile
import zarr
import fsspec
import warnings
from numcodecs.registry import codec_registry
import skimage.transform
from scipy import ndimage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

def init_o2z_globals():
    # Suppress warning about NestedDirectoryStore being removed
    # in zarr version 3.  
    warnings.filterwarnings("ignore", message="The NestedDirectoryStore.*")
    # Per https://github.com/zarr-developers/zarr-python/blob/v2.18.3/zarr/storage.py :
    '''
    NestedDirectoryStore will be removed in Zarr-Python 3.0 where controlling
    the chunk key encoding will be supported as part of the array metadata. See
    `GH1274 <https://github.com/zarr-developers/zarr-python/issues/1274>`_
    for more information.
    '''

    # Controls the number of open https connections;
    # if this is not set, the Vesuvius Challenge data server 
    # may complain of too many requests
    # https://filesystem-spec.readthedocs.io/en/latest/async.html
    fsspec.config.conf['nofiles_gather_batch_size'] = 10

class DecompressedLRUCache(zarr.storage.LRUStoreCache):
    def __init__(self, store, max_size):
        super().__init__(store, max_size)
        self.compressor = None
    # By default, the LRU cache holds chunks that it copies
    # directly from the original data store.  This means that
    # if the data store contains compressed chunks, the cache
    # will hold compressed chunks.
    # Each such chunk has to be decompressed every time it is
    # accessed, which is a waste of CPU.
    # This causes noticeable slowing.
    # The routine below modifies the internals of the array
    # that uses the LRU cache as the data store, so that compressed
    # chunks are decompressed when they go into the cache, and
    # so that they are not decompressed an additional time when
    # they are accessed.

    def transferCompressor(self, array):
        self.compressor = array._compressor
        array._compressor = None

    # This function gets a chunk from the underlying data
    # store.  The access may cause an exception to be thrown.
    # This function does not try to catch exceptions, because
    # the caller of this function will handle them.
    # If decompression has been transfered to the LRU cache
    # (see the transferCompressor function), do the decompression
    # here.        
    def getValue(self, key):
        value = self._store[key]
        if len(value) > 0 and self.compressor is not None:
            for i in range(3):
                try:
                    dc = self.compressor.decode(value)
                    break
                except Exception as e:
                    print("decompression failure try %d: %s"%(i+1, e))
            return dc
        return value

    # This is identical to __getitem__ in LRUStoreCache,
    # except that the access to self._store is replaced
    # by a call to self.getValue
    def __getitem__(self, key):
        try:
            # first try to obtain the value from the cache
            with self._mutex:
                value = self._values_cache[key]
                # cache hit if no KeyError is raised
                self.hits += 1
                # treat the end as most recently used
                self._values_cache.move_to_end(key)

        except KeyError:
            # cache miss, retrieve value from the store
            # value = self._store[key]
            value = self.getValue(key)
            with self._mutex:
                self.misses += 1
                # need to check if key is not in the cache, 
                # as it may have been cached
                # while we were retrieving the value from the store
                if key not in self._values_cache:
                    self._cache_value(key, value)

        return value


def compute_data_range(zarr_array):
    print("Computing data range for uint8 scaling...")
    
    min_val = float(np.min(zarr_array))
    max_val = float(np.max(zarr_array))
    
    print(f"Data range: {min_val} to {max_val}")
    return min_val, max_val


def scale_to_uint8(data, min_val, max_val):
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.uint8)
    
    scaled = (data.astype(np.float64) - min_val) / (max_val - min_val)
    scaled = np.clip(scaled * 255, 0, 255)
    return scaled.astype(np.uint8)

def parseShift(istr):
    sstrs = istr.split(",")
    if len(sstrs) != 3:
        print("Could not parse shift argument '%s'; expected 3 comma-separated numbers"%istr)
        return None
    shift = []
    for sstr in sstrs:
        i = int(sstr)
        shift.append(i)
    return shift

def parsePadding(istr):
    sstrs = istr.split(",")
    if len(sstrs) != 2:
        print("Could not parse padding argument '%s'; expected 2 comma-separated numbers (x,y)"%istr)
        return None
    try:
        x_pad = int(sstrs[0])
        y_pad = int(sstrs[1])
        if x_pad < 0 or y_pad < 0:
            print("Padding values must be non-negative")
            return None
        # return in ZYX format: (z_pad=0, y_pad, x_pad)
        return (0, y_pad, x_pad)
    except ValueError:
        print("Could not parse padding values as integers")
        return None

def parseSlices(istr):
    sstrs = istr.split(",")
    if len(sstrs) != 3:
        print("Could not parse range argument '%s'; expected 3 comma-separated ranges"%istr)
        return (None, None, None)
    slices = []
    for sstr in sstrs:
        if sstr == "":
            slices.append(None)
            continue
        parts = sstr.split(':')
        if len(parts) == 1:
            slices.append(slice(int(parts[0])))
        else:
            iparts = [None if p=="" else int(p) for p in parts]
            if len(iparts)==2:
                iparts.append(None)
            slices.append(slice(iparts[0], iparts[1], iparts[2]))
    return slices

def slice_step_is_1(s):
    if s is None:
        return True
    if s.step is None:
        return True
    if s.step == 1:
        return True
    return False

def slice_start(s, default=0):
    if s is None or s.start is None:
        return default
    return s.start

def slice_stop(s, default=0):
    if s is None or s.stop is None:
        return default
    return s.stop

def divp1(s, c):
    n = s // c
    if s%c > 0:
        n += 1
    return n

# return None if succeeds, err string if fails
def create_ome_dir(zarrdir):
    if zarrdir.exists():
        err = "Directory %s already exists"%zarrdir
        print(err)
        return err

    try:
        zarrdir.mkdir()
    except Exception as e:
        err = "Error while creating %s: %s"%(zarrdir, e)
        print(err)
        return err

def create_ome_headers_anisotropic(zarrdir, nlevels):
    zattrs_dict = {
        "multiscales": [
            {
                "axes": [
                    {
                        "name": "z",
                        "type": "space"
                    },
                    {
                        "name": "y",
                        "type": "space"
                    },
                    {
                        "name": "x",
                        "type": "space"
                    }
                ],
                "datasets": [],
                "name": "/",
                "version": "0.4"
            }
        ]
    }

    dataset_dict = {
        "coordinateTransformations": [
            {
                "scale": [],
                "type": "scale"
            }
        ],
        "path": ""
    }
    
    zgroup_dict = { "zarr_format": 2 }

    datasets = []
    for l in range(nlevels):
        ds = copy.deepcopy(dataset_dict)
        ds["path"] = "%d"%l
        # anisotropic scaling: X and Y are downscaled by 2^l, Z stays at 1
        xy_scale = 2.**l
        z_scale = 1.0 
        ds["coordinateTransformations"][0]["scale"] = [z_scale, xy_scale, xy_scale]
        datasets.append(ds)
    
    zad = copy.deepcopy(zattrs_dict)
    zad["multiscales"][0]["datasets"] = datasets
    json.dump(zgroup_dict, (zarrdir / ".zgroup").open("w"), indent=4)
    json.dump(zad, (zarrdir / ".zattrs").open("w"), indent=4)


def path_is_file(path):
    fs, _, _ = fsspec.get_fs_token_paths(path)
    proto = fs.protocol
    print("fsspec protocol(s):", proto)
    return "file" in proto

def zarr2zarr(izarrdir, ozarrdir, shift=(0,0,0), slices=(None,None,None), chunk_size=None, 
              compression=None, dtype=None, scale_uint8=False, data_range=None, padding=(0,0,0)):
    # user gives shift and slices in x,y,z order, but internally
    # we use z,y,x order
    shift = (shift[2], shift[1], shift[0])
    slices = (slices[2], slices[1], slices[0])
    # padding is already in z,y,x order from parsePadding
    
    if not all([(slice_start(s)>=0) for s in slices]):
        err = "All window starting coordinates must be >= 0"
        print(err)
        return err
    if not all([slice_step_is_1(s) for s in slices]):
        err = "All window steps must be 1"
        print(err)
        return err
    
    try:
        izarr = zarr.open(izarrdir, mode="r")
    except Exception as e:
        err = "Could not open %s; error is %s"%(izarrdir, e)
        print(err)
        return err
    
    is_ome = False
    if isinstance(izarr, zarr.hierarchy.Group):
        print("It appears that",izarrdir,"\nis an OME-Zarr store rather than a simple zarr store.\nI will attempt to open the highest-resolution zarr store in this hierarchy\n")
        if '0' in izarr:
            izarr = izarr['0']
        else:
            err = "%s does not appear to be a zarr or OME-Zarr data store"
            print(err)
            return err
        is_ome = True
    
    chunk_sizes = izarr.chunks
    if chunk_size is not None:
        chunk_sizes = (chunk_size, chunk_size, chunk_size)
    
    original_dtype = izarr.dtype
    if dtype is None:
        dtype = original_dtype
    
    if scale_uint8 and dtype == np.uint8:
        if data_range is None:
            min_val, max_val = compute_data_range(izarr)
            data_range = (min_val, max_val)
        else:
            min_val, max_val = data_range
        print(f"Using data range {min_val} to {max_val} for uint8 scaling")
    else:
        divisor = 1
        if dtype != original_dtype:
            if dtype == np.uint8 and original_dtype == np.uint16:
                divisor = 256
            else:
                err = "Can't convert %s to %s"%(str(original_dtype), str(dtype))
                print(err)
                return err

    compressor = izarr.compressor
    if compression is not None:
        if compression == "none" or compression == "":
            compressor = None
        else:
            codec_cls = codec_registry[compression]
            compressor = codec_cls()

    if path_is_file(izarrdir):
        store = izarr.store
        print("Using cache for input data")
        cstore = DecompressedLRUCache(store, max_size=2**32)
        if is_ome:
            root = zarr.group(store=cstore)
            czarr = root['0']
        else:
            czarr = zarr.open(cstore, mode="r")
        cstore.transferCompressor(czarr)
    else:
        czarr = izarr

    ishape = izarr.shape
    i0 = [0, 0, 0]
    i1 = ishape
    iw0 = [slice_start(slices[i], i0[i]) for i in range(3)]
    iw1 = [slice_stop(slices[i], i1[i]) for i in range(3)]
    iws0 = [iw0[i] + shift[i] for i in range(3)]
    iws1 = [iw1[i] + shift[i] for i in range(3)]
    os0 = [max(0, iws0[i]) for i in range(3)]
    os1 = [max(0, iws1[i]) for i in range(3)]
    
    if not all([os0[i] < os1[i] for i in range(3)]):
        err = "Computed conflicting output grid min %s and max %s"%(str(os0), str(os1))
        print(err)
        return err

    oshape = [os1[i] + padding[i] for i in range(3)]
    
    if any(p > 0 for p in padding):
        print(f"Adding padding: Z={padding[0]}, Y={padding[1]}, X={padding[2]}")
        print(f"Original output shape: {os1}")
        print(f"Padded output shape: {oshape}")
    
    cs0 = [max(os0[i], iws0[i]) + padding[i] for i in range(3)]  # offset by padding
    cs1 = [min(os1[i], iws1[i]) + padding[i] for i in range(3)]  # offset by padding

    store = zarr.NestedDirectoryStore(ozarrdir)
    ozarr = zarr.open(
            store=store, 
            shape=oshape, 
            chunks=chunk_sizes,
            dtype = dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=compressor,
            mode='w', 
            )

    chunk0s = [cs0[i] // chunk_sizes[i] for i in range(3)]
    chunk1s = [(cs1[i]-1) // chunk_sizes[i] + 1 for i in range(3)]
    print("chunk_sizes", chunk_sizes)
    
    ci = [0,0,0]
    for ci[0] in range(chunk0s[0], chunk1s[0]):
        print("doing", ci[0], "to", chunk1s[0])
        for ci[1] in range(chunk0s[1], chunk1s[1]):
            for ci[2] in range(chunk0s[2], chunk1s[2]):
                o0 = [max(cs0[i], chunk_sizes[i]*ci[i]) for i in range(3)]
                o1 = [min(cs1[i], chunk_sizes[i]*(1+ci[i])) for i in range(3)]
                i0 = [(o0[i] - padding[i]) - shift[i] for i in range(3)]
                i1 = [(o1[i] - padding[i]) - shift[i] for i in range(3)]
                
                input_chunk = czarr[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
                
                if scale_uint8 and dtype == np.uint8:
                    output_chunk = scale_to_uint8(input_chunk, min_val, max_val)
                else:
                    output_chunk = input_chunk // divisor
                
                ozarr[o0[0]:o1[0], o0[1]:o1[1], o0[2]:o1[2]] = output_chunk
    
    if scale_uint8 and dtype == np.uint8:
        return data_range
    return None


def process_chunk_anisotropic(args):
    (idata, odata, z, y, x, cz, cy, cx, algorithm, xy_factor, apply_uint8_scaling, 
     data_range, enable_smoothing, binary_threshold) = args
    
    iz_start = z * cz
    iz_end = min((z + 1) * cz, idata.shape[0])
    iy_start = y * cy * xy_factor
    iy_end = min((y + 1) * cy * xy_factor, idata.shape[1])
    ix_start = x * cx * xy_factor
    ix_end = min((x + 1) * cx * xy_factor, idata.shape[2])
    
    ibuf = idata[iz_start:iz_end, iy_start:iy_end, ix_start:ix_end]
    
    if np.max(ibuf) == 0:
        return 
    
    z_slices = ibuf.shape[0]
    output_shape = (z_slices, 
                    min(cy, divp1(ibuf.shape[1], xy_factor)), 
                    min(cx, divp1(ibuf.shape[2], xy_factor)))
    obuf = np.zeros(output_shape, dtype=ibuf.dtype)
    
    for zi in range(z_slices):
        slice_2d = ibuf[zi, :, :]
        
        if xy_factor == 1:
            obuf[zi] = slice_2d
        else:
            if enable_smoothing:
                sigma = 0.5 * xy_factor
                
                slice_float = slice_2d.astype(np.float32)
                slice_smoothed = ndimage.gaussian_filter(slice_float, sigma=sigma)
                
                decimated = skimage.transform.downscale_local_mean(
                    slice_smoothed, (xy_factor, xy_factor))
                
                if binary_threshold is not None:
                    obuf[zi] = (decimated > binary_threshold).astype(np.uint8)
                else:
                    # Keep continuous values (may look smoother)
                    obuf[zi] = decimated.astype(obuf.dtype)
            
            else:
                if algorithm == "nearest":
                    obuf[zi] = slice_2d[::xy_factor, ::xy_factor]
                elif algorithm == "gaussian":
                    target_shape = (obuf.shape[1], obuf.shape[2])
                    resized = skimage.transform.resize(
                        slice_2d, target_shape, preserve_range=True, anti_aliasing=True)
                    obuf[zi] = np.round(resized)
                elif algorithm == "mean":
                    obuf[zi] = np.round(skimage.transform.downscale_local_mean(
                        slice_2d, (xy_factor, xy_factor)))
                elif algorithm == "max":
                    h, w = slice_2d.shape
                    pad_h = (xy_factor - h % xy_factor) % xy_factor
                    pad_w = (xy_factor - w % xy_factor) % xy_factor
                    if pad_h > 0 or pad_w > 0:
                        slice_2d = np.pad(slice_2d, ((0, pad_h), (0, pad_w)), mode='edge')
                    h_new = slice_2d.shape[0] // xy_factor
                    w_new = slice_2d.shape[1] // xy_factor
                    reshaped = slice_2d[:h_new*xy_factor, :w_new*xy_factor].reshape(
                        h_new, xy_factor, w_new, xy_factor)
                    obuf[zi] = np.max(reshaped, axis=(1, 3))[:output_shape[1], :output_shape[2]]
                else:
                    raise ValueError(f"algorithm {algorithm} not valid")
    
    if apply_uint8_scaling and data_range is not None:
        min_val, max_val = data_range
        obuf = scale_to_uint8(obuf, min_val, max_val)
    
    oz_start = z * cz
    oz_end = oz_start + obuf.shape[0]
    oy_start = y * cy
    oy_end = oy_start + obuf.shape[1]
    ox_start = x * cx
    ox_end = ox_start + obuf.shape[2]
    
    odata[oz_start:oz_end, oy_start:oy_end, ox_start:ox_end] = obuf


def resize_anisotropic(zarrdir, old_level, num_threads, algorithm="mean", scale_uint8=False, 
                       data_range=None, enable_smoothing=False, smooth_start_level=4,
                       binary_threshold=None):
    idir = zarrdir / ("%d"%old_level)
    if not idir.exists():
        err = f"input directory {idir} does not exist"
        print(err)
        return err
    
    odir = zarrdir / ("%d"%(old_level+1))
    idata = zarr.open(idir, mode="r")
    
    xy_factor = 2  # always downsample X,Y by 2
    
    # determine if we should apply smoothing for this level
    apply_smoothing = enable_smoothing and (old_level + 1) >= smooth_start_level
    
    print(f"Creating level {old_level+1}")
    print(f"  Input shape: {idata.shape}")
    print(f"  Input dtype: {idata.dtype}")
    print(f"  Algorithm: {algorithm}")
    print(f"  XY downscaling factor: {xy_factor}")
    print(f"  Z preserved at full resolution")
    print(f"  Smoothing enabled: {apply_smoothing}")
    if apply_smoothing:
        print(f"  Using scale-dependent sigma: σ = 0.5 × {xy_factor} = {0.5 * xy_factor}")
        if binary_threshold is not None:
            print(f"  Binary threshold: {binary_threshold}")
        else:
            print(f"  Keeping continuous values")
    
    # only apply uint8 scaling on level 0 -> level 1 if input is not already uint8
    apply_uint8_scaling = scale_uint8 and idata.dtype != np.uint8
    if apply_uint8_scaling:
        print(f"  uint8 scaling enabled with range: {data_range}")
    elif scale_uint8:
        print(f"  Input already uint8, no additional scaling needed")
    
    cz, cy, cx = idata.chunks
    sz, sy, sx = idata.shape
    
    output_shape = (sz, divp1(sy, xy_factor), divp1(sx, xy_factor))
    print(f"  Output shape: {output_shape}")
    
    output_dtype = np.uint8 if scale_uint8 else idata.dtype
    
    store = zarr.NestedDirectoryStore(odir)
    odata = zarr.open(
            store=store,
            shape=output_shape,
            chunks=idata.chunks,
            dtype=output_dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=idata.compressor,
            mode='w',
            )
    
    scaling_range = data_range if apply_uint8_scaling else None
    tasks = [(idata, odata, z, y, x, cz, cy, cx, algorithm, xy_factor, apply_uint8_scaling, 
              scaling_range, apply_smoothing, binary_threshold) 
             for z in range(divp1(sz, cz))
             for y in range(divp1(output_shape[1], cy))
             for x in range(divp1(output_shape[2], cx))]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_chunk_anisotropic, tasks), total=len(tasks)))
    
    print("Processing complete")

def zarr_to_ome_main():
    init_o2z_globals()
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Create OME/Zarr data store with anisotropic scaling from an existing zarr data store",
            )
    parser.add_argument(
            "input_zarr_dir", 
            help="Name of zarr store directory")
    parser.add_argument(
            "output_zarr_ome_dir", 
            help="Name of directory that will contain OME/zarr datastore")
    parser.add_argument(
            "--algorithm",
            choices=['mean', 'gaussian', 'nearest', 'max'],
            default="mean",
            help="Algorithm used to sub-sample the data")
    parser.add_argument(
            "--chunk_size", 
            type=int,
            help="Size of chunk; if not given, will be same as input zarr")
    parser.add_argument(
            "--shift", 
            help="Shift input data set, relative to output data set")
    parser.add_argument(
            "--window", 
            help="Output only a subset of the data")
    parser.add_argument(
            "--bytes",
            type=int,
            default=0,
            help="number of bytes per pixel in output")
    parser.add_argument(
            "--compression", 
            help="Compression algorithm")
    parser.add_argument(
            "--overwrite", 
            action="store_true", 
            help="Overwrite the output directory")
    parser.add_argument(
            "--nlevels", 
            type=int, 
            default=6, 
            help="Number of subdivision levels to create")
    parser.add_argument(
            "--rebuild", 
            action="store_true", 
            help="Rebuild higher levels but leave level 0 alone")
    parser.add_argument(
            "--num_threads", 
            type=int, 
            default=cpu_count(), 
            help="Number of threads to use")
    parser.add_argument(
            "--anisotropic",
            action="store_true",
            default=True,
            help="Use anisotropic scaling (preserve Z resolution)")
    parser.add_argument(
            "--scale_uint8",
            action="store_true",
            help="Scale data to uint8 based on data range (not just convert dtype)")
    parser.add_argument(
            "--data_range",
            help="Manually specify data range as 'min,max' for uint8 scaling (auto-computed if not provided)")
    parser.add_argument(
            "--padding",
            help="Add zero padding to top-left corner in X,Y format (e.g., '80,69' adds 80 pixels left, 69 pixels top)")
    
    parser.add_argument(
            "--enable_smoothing",
            action="store_true",
            help="Enable Gaussian smoothing for higher levels to reduce jagged artifacts in binary labels")
    parser.add_argument(
            "--smooth_start_level",
            type=int,
            default=4,
            help="Level from which to start applying smoothing (default: 4, corresponding to 16x downscaling)")
    parser.add_argument(
            "--smooth_sigma",
            type=float,
            default=1.0,
            help="Gaussian sigma for smoothing (default: 1.0)")
    parser.add_argument(
            "--binary_threshold",
            type=float,
            default=0.5,
            help="Threshold for converting smoothed continuous values back to binary (default: 0.5, set to None to keep continuous)")
    parser.add_argument(
            "--keep_continuous",
            action="store_true",
            help="Keep smoothed values as continuous (0-1) instead of re-thresholding to binary")

    args = parser.parse_args()
    
    ozarrdir = Path(args.output_zarr_ome_dir)
    if ozarrdir.suffix != ".zarr":
        print("Name of output zarr directory must end with '.zarr'")
        return 1
    
    izarrdir = args.input_zarr_dir
    nlevels = args.nlevels
    overwrite = args.overwrite
    rebuild_higher_levels = args.rebuild
    num_threads = args.num_threads
    algorithm = args.algorithm
    anisotropic = args.anisotropic
    scale_uint8 = args.scale_uint8
    enable_smoothing = args.enable_smoothing
    smooth_start_level = args.smooth_start_level
    # note: smooth_sigma is no longer used - we calculate sigma automatically as 0.5 * xy_factor
    
    if args.keep_continuous:
        binary_threshold = None
    else:
        binary_threshold = args.binary_threshold
    
    data_range = None
    if args.data_range is not None:
        try:
            min_val, max_val = map(float, args.data_range.split(','))
            data_range = (min_val, max_val)
            print(f"Using manual data range: {data_range}")
        except ValueError:
            print("Error: data_range must be in format 'min,max'")
            return 1
    
    padding = (0, 0, 0)  # default: no padding
    if args.padding is not None:
        padding = parsePadding(args.padding)
        if padding is None:
            print("Error parsing padding argument")
            return 1
        print(f"Padding will be applied: X={padding[2]}, Y={padding[1]}, Z={padding[0]}")
    
    print(f"Anisotropic scaling mode: {anisotropic}")
    print(f"uint8 scaling mode: {scale_uint8}")
    print(f"Scaling pattern: 1-1-1, 2-2-1, 4-4-1, 8-8-1, 16-16-1, 32-32-1 (X-Y-Z)")
    print(f"Smoothing enabled: {enable_smoothing}")
    if enable_smoothing:
        print(f"Smoothing starts from level: {smooth_start_level}")
        print(f"Scale-dependent sigma used (overrides --smooth_sigma)")
        if binary_threshold is not None:
            print(f"Binary threshold: {binary_threshold}")
        else:
            print(f"Keeping continuous values (no re-thresholding)")
    
    shift = (0,0,0)
    if args.shift is not None:
        shift = parseShift(args.shift)
        if shift is None:
            print("Error parsing shift argument")
            return 1
    
    slices = (None,None,None)
    if args.window is not None:
        slices = parseSlices(args.window)
        if slices is None:
            print("Error parsing window argument")
            return 1
    
    chunk_size = args.chunk_size
    compression = args.compression
    
    dtype = None
    if args.bytes == 1 or scale_uint8:
        dtype = np.uint8
    elif args.bytes == 2:
        dtype = np.uint16
    
    if not rebuild_higher_levels:
        if not overwrite:
            if ozarrdir.exists():
                print("Error: Directory",ozarrdir,"already exists")
                return(1)
        
        if ozarrdir.exists():
            print("removing", ozarrdir)
            shutil.rmtree(ozarrdir)
        
        err = create_ome_dir(ozarrdir)
        if err is not None:
            print("error returned:", err)
            return 1
        
        if nlevels > 1:
            err = create_ome_headers_anisotropic(ozarrdir, nlevels)
            if err is not None:
                print("error returned:", err)
                return 1
        
        print("Creating level 0")
        level0dir = ozarrdir/"0"
        if nlevels == 1:
            level0dir = ozarrdir
        print("shift", shift, "slices", slices, "padding", padding)
        
        # create level 0 with uint8 scaling and padding if requested
        result = zarr2zarr(izarrdir, level0dir, shift=shift, slices=slices, 
                          chunk_size=chunk_size, compression=compression, dtype=dtype,
                          scale_uint8=scale_uint8, data_range=data_range, padding=padding)
        if isinstance(result, str):  # Error case
            print("error returned:", result)
            return 1
        elif scale_uint8 and result is not None:
            data_range = result  # get the computed data range from level 0
    else:
        if nlevels <= 1:
            return 0
        err = create_ome_headers_anisotropic(ozarrdir, nlevels)
        if err is not None:
            print("error returned:", err)
            return 1
    
    # create higher levels with anisotropic scaling and optional smoothing
    existing_level = 0
    for l in range(existing_level, nlevels-1):
        err = resize_anisotropic(ozarrdir, l, num_threads, algorithm, scale_uint8, data_range,
                               enable_smoothing, smooth_start_level, binary_threshold)
        if err is not None:
            print("error returned:", err)
            return 1
    
    print("\nSuccessfully created OME-Zarr with anisotropic scaling!")
    print("Scales: 1-1-1, 2-2-1, 4-4-1, 8-8-1, 16-16-1, 32-32-1 (X-Y-Z)")
    if scale_uint8:
        print(f"Data scaled to uint8 using range: {data_range}")
    if any(p > 0 for p in padding):
        print(f"Padding applied: X={padding[2]}, Y={padding[1]}, Z={padding[0]}")
    if enable_smoothing:
        print(f"Anti-aliased smoothing applied from level {smooth_start_level} onwards")
        print(f"Using scale-dependent sigma (σ = 0.5 × downsampling_factor)")
        if binary_threshold is not None:
            print(f"Binary labels re-thresholded at {binary_threshold}")
        else:
            print(f"Continuous values preserved (no re-thresholding)")

if __name__ == '__main__':
    sys.exit(zarr_to_ome_main())

