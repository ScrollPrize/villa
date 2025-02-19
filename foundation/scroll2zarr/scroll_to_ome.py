import sys
import os
import re
from pathlib import Path
import json
import shutil
import argparse
import copy
import numpy as np
import cv2
import tifffile
import zarr
import numcodecs
import skimage.transform
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from vesuvius_basic_compression import support as sf
from sklearn.mixture import GaussianMixture
import glob
import re

inside_coordinates = None

'''
# tiffdir = Path(r"C:\Vesuvius\scroll 1 2000-2030")
# zarrdir = Path(r"C:\Vesuvius\test.zarr")
# tiffdir = Path(r"H:\Vesuvius\Scroll1.volpkg\volumes_masked\20230205180739")
# tiffdir = Path(r"H:\Vesuvius\zarr_tests\masked_subset")
# zarrdir = Path(r"H:\Vesuvius\zarr_tests\testzo.zarr")
# zarrdir = Path(r"H:\Vesuvius\testzc.zarr")

# tif files, ome dir
# set chunk_size
chunk_size = 128
# slices = (None, None, slice(1975,2010))
# slices = (slice(2000,2500), slice(2000,2512), slice(1975,2010))
slices = (slice(2000,2500), slice(2000,2512), slice(1975,2005))
maxgb = None
nlevels = 6
zarr_only = False
first_new_level = 0
# maxgb = .0036
'''

# create ome dir, .zattrs, .zgroup
# (don't need to know output array dimensions, just number of levels,
# possibly unit/dimension info)
# create_ome_dir(zarrdir, nlevels)
# quit if dir already exists

# tifs2zarr(tiffdir, zarrdir+"/0", chunk_size, range(optional))

def parseSlices(istr):
    sstrs = istr.split(",")
    if len(sstrs) != 3:
        print("Could not parse ranges argument '%s'; expected 3 comma-separated ranges"%istr)
        return None
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

# return None if succeeds, err string if fails
def create_ome_dir(zarrdir):
    # complain if directory already exists
    if zarrdir.exists():
        err = "Directory %s already exists"%zarrdir
        print(err)
        return err

    try:
        # Create directory
        os.makedirs(os.path.dirname(zarrdir), exist_ok=True)
        # Create zarr directory
        zarrdir.mkdir()
    except Exception as e:
        err = "Error while creating %s: %s"%(zarrdir, e)
        print(err)
        return err

def create_ome_headers(zarrdir, nlevels):
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
                "scale": [
                ],
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
        scale = 2.**l
        ds["coordinateTransformations"][0]["scale"] = [scale]*3
        # print(json.dumps(ds, indent=4))
        datasets.append(ds)
    zad = copy.deepcopy(zattrs_dict)
    zad["multiscales"][0]["datasets"] = datasets
    json.dump(zgroup_dict, (zarrdir / ".zgroup").open("w"), indent=4)
    json.dump(zad, (zarrdir / ".zattrs").open("w"), indent=4)

def slice_step_is_1(s):
    if s is None:
        return True
    if s.step is None:
        return True
    if s.step == 1:
        return True
    return False

def slice_start(s):
    if s.start is None:
        return 0
    return s.start

def slice_count(s, maxx):
    mn = s.start
    if mn is None:
        mn = 0
    mn = max(0, mn)
    mx = s.stop
    if mx is None:
        mx = maxx
    mx = min(mx, maxx)
    return mx-mn

def load_tiff(tiffname):
    print(tiffname)
    if str(tiffname).endswith('.tif'):
        image =  tifffile.imread(str(tiffname))
    elif str(tiffname).endswith('.jpg'):
        image = cv2.imread(str(tiffname), cv2.IMREAD_GRAYSCALE)
    else:
        print("returning none")
        return None
    # if uint8, convert to uint16
    if image.dtype == np.uint8:
        image = image.astype(np.uint16)*256
    # if float, convert to uint16
    if image.dtype == np.float32:
        image = (image*65535).astype(np.uint16)
    return image

def get_tiffs(tiffdir):
    # Note this is a generator, not a list
    tiffs = list(tiffdir.glob("*.tif"))
    if len(tiffs) == 0:
        tiffs = list(tiffdir.glob("*.jpg"))
    rec = re.compile(r'([0-9]+)\.\w+$')
    # rec = re.compile(r'[0-9]+$')
    inttiffs = {}
    for tiff in tiffs:
        tname = tiff.name
        match = rec.match(tname)
        if match is None:
            continue
        # Look for last match (closest to end of file name)
        # ds = match[-1]
        ds = match.group(1)
        itiff = int(ds)
        if itiff in inttiffs:
            err = "File %s: tiff id %d already used"%(tname,itiff)
            print(err)
            return err
        inttiffs[itiff] = tiff
    if len(inttiffs) == 0:
        err = "No tiffs found"
        print(err)
        return err
    
    itiffs = list(inttiffs.keys())
    itiffs.sort()
    
    return inttiffs, itiffs

def get_tiff_volume_mask(tiff, itiff):
    # Find a region in the tiff that contains a fair mix of 0's and 1's and is close to the center of the image
    global inside_coordinates
    if inside_coordinates is None or itiff % 100 == 0:
        # Non-Air pixels
        max_value = np.iinfo(tiff.dtype).max
        non_air = tiff > (max_value // 2)
        mean_x = np.mean(non_air, axis=0)
        mean_y = np.mean(non_air, axis=1)
        # print(f"Shape of means: {mean_x.shape}, {mean_y.shape}")
        # Sliding window of size 200 over means, calculate the sum of the means in the window
        window_size = 200
        assert window_size < mean_x.shape[0] and window_size < mean_y.shape[0], "Window size too large"
        sum_x = np.convolve(mean_x, np.ones(window_size), mode='same')
        sum_y = np.convolve(mean_y, np.ones(window_size), mode='same')
        assert sum_x.shape[0] == mean_x.shape[0] and sum_y.shape[0] == mean_y.shape[0], "Convolution failed"
        # Find the region with the highest sum
        max_sum_x = np.argmax(sum_x)
        max_sum_y = np.argmax(sum_y)
        # print(f"inside coordinates x: {max_sum_x}, y: {max_sum_y}")
        # Get coordinates that lay inside the scroll for this slice
        inside_coordinates = [max_sum_x, max_sum_y]

    # Scale to 0-1
    bits = 8 if tiff.dtype == np.uint8 else 16
    bits_scaling = 2**bits - 1
    mask = np.ones(tiff.shape, dtype=np.uint8)
    tiff_slice_float = tiff.astype(np.float32)/bits_scaling

    try:
        # TODO: implement CAD Case masking
        mask = sf.create_mask(tiff_slice_float, [0, 0, 0, 0], inside_coordinates, False)
    except Exception as e:
        print(f"Error creating mask for slice ?: {e}")
    return mask

def get_tiff_surface_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask

def load_transform(transform_path):
    with open(transform_path, 'r') as file:
        data = json.load(file)
    return np.array(data["params"])

def invert_transform(transform_matrix):
    inv_transform = np.linalg.inv(transform_matrix)
    transform_matrix = transform_matrix / transform_matrix[3, 3] # Homogeneous coordinates
    return inv_transform

def get_voxelsize(meta_path):
    meta = json.load(open(meta_path))
    return meta['voxelsize']

def get_transforms(tiffdir, original_volume_id):
    transform_path = tiffdir / ".." / ".." / "transforms"
    transform_from_canonical_path = glob.glob(os.path.join(f"{transform_path}",f"*-to-{original_volume_id}.json"))
    if len(transform_from_canonical_path) == 0:
        transform_from_canonical = np.eye(4)
        scale = 1.0
    else:
        transform_from_canonical_path = transform_from_canonical_path[0]
        transform_from_canonical = load_transform(transform_from_canonical_path)
        vol_id0 = os.path.basename(transform_from_canonical_path).split("-")[0]
        # Extract scaling factors of transform
        meta0_path = tiffdir / ".." / vol_id0 / "meta.json"
        voxelsize0 = get_voxelsize(meta0_path)
        meta1_path = tiffdir / ".." / original_volume_id / "meta.json"
        voxelsize1 = get_voxelsize(meta1_path)
        scale = voxelsize0 / voxelsize1
        print(f"Scale: {scale}")

    transform_to_canonical = invert_transform(transform_from_canonical)
    
    # Generates the transform that brings a volume into the canonical coordinate system up to a 1D scaling factor
    transform_to_canonical[:3, :3] *= scale
    transform_from_canonical[:3, :3] /= scale

    return transform_to_canonical, transform_from_canonical, scale

def transform_buf_to_tzarr(x , y, z, t):
    # Transforms from buffer coordinates to zarr unscaled canonical coordinates
    # TODO
    return None

def equalize(tiffdir, n_components=4):
    """
    Loads TIFF images from a directory, extracts a central chunk, and fits a Gaussian Mixture Model to
    the pixel values. Computes and returns parameters based on the means and standard deviations of the
    mixture components.

    Parameters:
        tiffdir (str): Directory containing TIFF files.
        n_components (int): Number of components for the Gaussian mixture model.

    Returns:
        tuple: Two computed values based on the Gaussian components.
    """
    # load paths of tiffs
    inttiffs, itiffs = get_tiffs(tiffdir)
    # get the first tiff
    minz = itiffs[0]
    tiffname = inttiffs[minz]
    tiff = load_tiff(tiffname)
    # calculate the chunk shape
    tiff_shape = tiff.shape
    z_len = len(itiffs)
    z_min = z_len // 2 - 10
    z_max = z_len // 2 + 10
    z_len = z_max - z_min
    chunk = np.zeros((z_len, tiff_shape[0], tiff_shape[1]), dtype=tiff.dtype)
    # load the chunk ct data
    for i in range(z_min, z_max):
        chunk[i-z_min] = load_tiff(inttiffs[i])
    # Only take every 5th pixel in xy
    chunk = chunk[:, ::5, ::5]
    # take only the central 1/3 of the xy axis
    chunk = chunk[:, chunk.shape[1]//3:2*chunk.shape[1]//3, chunk.shape[2]//3:2*chunk.shape[2]//3]

    ### Calculate the equalization parameters ###
    print("Calculating equalization parameters...")
    gmm = GaussianMixture(n_components=n_components, verbose=1, verbose_interval=1)
    gmm.fit(chunk.reshape(-1, 1))

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.reshape(n_components, 1, 1).flatten())

    # Zip and sort means and standard deviations with respect to means
    zipped = sorted(zip(means, stds), key=lambda x: x[0])
    sorted_means, sorted_stds = zip(*zipped)

    print(f"Means: {sorted_means}, Stds: {sorted_stds}")

    # Compute the required values
    value1 = sorted_means[0] + sorted_stds[0] # remove all air up to one std
    value2 = sorted_means[-1] + 2.0 * sorted_stds[-1] # taking two standard deviation more for papyrus
    print(f"Equalization parameters: {value1}, {value2}")
    return value1, value2

def preprocess_tiff(inttiffs, itiff, standard_config):
    tiffname = inttiffs[itiff]
    tiff = load_tiff(tiffname)
    if standard_config['standardize']:
        # Apply mask
        if standard_config['volume_type'] == "scroll_volume":
            mask = get_tiff_volume_mask(tiff, itiff)
        elif standard_config['volume_type'] == "surface_volume":
            mask = get_tiff_surface_mask(standard_config['mask_path'])
        else:
            raise NotImplementedError(f"Volume type {standard_config['volume_type']} not implemented")
        if mask is not None:
            tiff = tiff * mask
        # Apply Equalization
        tiff = np.clip(tiff, standard_config['equalization_min'], standard_config['equalization_max']).astype(np.float64)
        tiff = (tiff - standard_config['equalization_min'])
        tiff = tiff * 65535
        tiff = tiff/ (standard_config['equalization_max'] - standard_config['equalization_min'])
        # Apply bit reduction
        if standard_config['n_bits'] < 8:
            # To uint8
            tiff = (tiff/256).astype(np.uint8)
            # Zero 8 - n bits
            trailing_zero_bits = 8 - standard_config['n_bits']
            bit_mask = int(2**8 - 2**trailing_zero_bits)
        else:
            # To uint16
            tiff = tiff.astype(np.uint16)
            # Zero 16 - n bits
            trailing_zero_bits = 16 - standard_config['n_bits']
            bit_mask = int(2**16 - 2**trailing_zero_bits)
        tiff = (tiff & bit_mask).astype(tiff.dtype)
        # tiff = tiff.astype(np.uint16) * 255 # Debug ONLY with Khartes
    return tiff

def write_to_zarr(tzarr, buf, zs, z, ze, ys, ye, standard_config=None):
    if standard_config is None or (not hasattr(standard_config, 'transform_to_canonical')) or standard_config['transform_to_canonical'] is None:
        tzarr[zs:z,ys:ye,:] = buf[:ze-zs,:ye-ys,:]
    else:
        print("use transforms")
        # Transform the buffer to the canonical coordinate system
        transform_to_canonical = standard_config['transform_to_canonical']
        for z_ in range(zs, z):
            for y_ in range(ys, ye):
                for x in range(buf.shape[2]):
                    p = np.array([x, y_, z_, 1])
                    p_trans = np.dot(transform_to_canonical, p)[:3]
                    tzarr[p_trans[2], p_trans[1], p_trans[0]] = buf[z_-zs, y_-ys, x]

def tifs2zarr(tiffdir, zarrdir, chunk_size, slices=None, maxgb=None, standard_config=None):
    if slices is None:
        xslice = yslice = zslice = None
    else:
        xslice, yslice, zslice = slices
        if not all([slice_step_is_1(s) for s in slices]):
            err = "All slice steps must be 1 in slices"
            print(err)
            return err
    
    # Load tiff paths
    inttiffs, itiffs = get_tiffs(tiffdir)

    z0 = 0
    if zslice is not None:
        maxz = itiffs[-1]+1
        valid_zs = range(maxz)[zslice]
        itiffs = list(filter(lambda z: z in valid_zs, itiffs))
        # z0 = itiffs[0]
        if zslice.start is None:
            z0 = 0
        else:
            z0 = zslice.start
    
    minz = itiffs[0]
    maxz = itiffs[-1]
    cz = maxz-z0+1
    
    try:
        tiff0 = preprocess_tiff(inttiffs, minz, standard_config)
    except Exception as e:
        err = "Error reading %s: %s"%(inttiffs[minz],e)
        print(err)
        return err
    ny0, nx0 = tiff0.shape
    dt0 = tiff0.dtype
    print("tiff size", nx0, ny0, "z range", minz, maxz, "dtype", dt0)

    cx = nx0
    cy = ny0
    x0 = 0
    y0 = 0
    if xslice is not None:
        cx = slice_count(xslice, nx0)
        x0 = slice_start(xslice)
    if yslice is not None:
        cy = slice_count(yslice, ny0)
        y0 = slice_start(yslice)
    print("cx,cy,cz",cx,cy,cz)
    print("x0,y0,z0",x0,y0,z0)
    print("chunk_size (x, y, z)",chunk_size)

    # Adjust chunk size order to z, y, x
    chunk_size_zyx = (chunk_size[2], chunk_size[1], chunk_size[0])
    
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.NOSHUFFLE)
    store = zarr.NestedDirectoryStore(zarrdir)
    tzarr = zarr.open(
            store=store, 
            shape=(cz, cy, cx), 
            chunks=chunk_size_zyx,
            dtype = dt0,
            write_empty_chunks=False,
            fill_value=0,
            compressor=compressor,
            mode='w', 
            )

    # nb of chunks in y direction that fit inside of max_gb
    chy = cy // chunk_size[1] + 1
    if maxgb is not None:
        maxy = int((maxgb*10**9)/(cx*chunk_size[0]*dt0.itemsize))
        chy = maxy // chunk_size[1]
        chy = max(1, chy)

    # nb of y chunk groups
    ncgy = cy // (chunk_size[1]*chy) + 1
    print("chy, ncgy", chy, ncgy)
    buf = np.zeros((chunk_size[2], min(cy, chy*chunk_size[1]), cx), dtype=dt0)
    for icy in range(ncgy):
        ys = icy*chy*chunk_size[1]
        ye = ys+chy*chunk_size[1]
        ye = min(ye, cy)
        if ye == ys:
            break
        prev_zc = -1
        for itiff in itiffs:
            z = itiff-z0
            tiffname = inttiffs[itiff]
            try:
                print("reading",itiff,"     ", end='\r')
                # print("reading",itiff)
                tarr = preprocess_tiff(inttiffs, itiff, standard_config)
            except Exception as e:
                print("\nError reading",tiffname,":",e)
                # If reading fails (file missing or deformed)
                tarr = np.zeros((ny, nx), dtype=dt0)
            # print("done reading",itiff, end='\r')
            # tzarr[itiff,:,:] = tarr
            ny, nx = tarr.shape
            if nx != nx0 or ny != ny0:
                print("\nFile %s is the wrong shape (%d, %d); expected %d, %d"%(tiffname,nx,ny,nx0,ny0))
                continue
            if xslice is not None and yslice is not None:
                tarr = tarr[yslice, xslice]
            cur_zc = z // chunk_size[2]
            if cur_zc != prev_zc:
                if prev_zc >= 0:
                    zs = prev_zc*chunk_size[2]
                    ze = zs+chunk_size[2]
                    if ncgy == 1:
                        print("\nwriting, z range %d,%d"%(zs+z0, ze+z0))
                    else:
                        print("\nwriting, z range %d,%d  y range %d,%d"%(zs+z0, ze+z0, ys+y0, ye+y0))
                    # write buf to zarr
                    write_to_zarr(tzarr, buf, zs, z, ze, ys, ye, standard_config=standard_config)
                    buf[:,:,:] = 0
                prev_zc = cur_zc
            cur_bufz = z-cur_zc*chunk_size[2]
            # print("cur_bufzk,ye,ys", cur_bufz,ye,ys)
            buf[cur_bufz,:ye-ys,:] = tarr[ys:ye,:]
        
        if prev_zc >= 0:
            zs = prev_zc*chunk_size[2]
            ze = zs+chunk_size[2]
            ze = min(itiffs[-1]+1-z0, ze)
            if ze > zs:
                if ncgy == 1:
                    print("\nwriting, z range %d,%d"%(zs+z0, ze+z0))
                else:
                    print("\nwriting, z range %d,%d  y range %d,%d"%(zs+z0, ze+z0, ys+y0, ye+y0))
                # print("\nwriting (end)", zs, ze)
                # tzarr[zs:zs+bufnz,:,:] = buf[0:(1+cur_bufz)]
                # write buf to zarr
                write_to_zarr(tzarr, buf, zs, ze, ze, ys, ye, standard_config=standard_config)
            else:
                print("\n(end)")
        buf[:,:,:] = 0

def divp1(s, c):
    n = s // c
    if s%c > 0:
        n += 1
    return n

def process_chunk(args):
    idata, odata, z, y, x, cz, cy, cx, algorithm = args
    ibuf = idata[2*z*cz:(2*z*cz+2*cz),
                 2*y*cy:(2*y*cy+2*cy),
                 2*x*cx:(2*x*cx+2*cx)]
    if np.max(ibuf) == 0:
        return  # Skip if the block is empty to save computation

    # pad ibuf to even in all directions
    ibs = ibuf.shape
    pad = (ibs[0]%2, ibs[1]%2, ibs[2]%2)
    if any(pad):
        ibuf = np.pad(ibuf, 
                      ((0,pad[0]),(0,pad[1]),(0,pad[2])), 
                      mode="symmetric")

    # algorithms:
    if algorithm == "nearest":
        obuf = ibuf[::2, ::2, ::2]
    elif algorithm == "gaussian":
        obuf = np.round(skimage.transform.rescale(ibuf, .5, preserve_range=True))
    elif algorithm == "mean":
        obuf = np.round(skimage.transform.downscale_local_mean(ibuf, (2,2,2)))
    else:
        raise ValueError(f"algorithm {algorithm} not valid")

    odata[z*cz:(z*cz+cz),
          y*cy:(y*cy+cy),
          x*cx:(x*cx+cx)] = np.round(obuf)

def resize(zarrdir, old_level, num_threads, algorithm="mean"):
    idir = zarrdir / ("%d"%old_level)
    if not idir.exists():
        err = f"input directory {idir} does not exist"
        print(err)
        return err
    
    odir = zarrdir / ("%d"%(old_level+1))
    idata = zarr.open(idir, mode="r")
    print("Creating level",old_level+1,"  input array shape", idata.shape, " algorithm", algorithm)

    cz, cy, cx = idata.chunks
    sz, sy, sx = idata.shape
    store = zarr.NestedDirectoryStore(odir)
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.NOSHUFFLE)
    odata = zarr.open(
            store=store,
            shape=(divp1(sz,2), divp1(sy,2), divp1(sx,2)),
            chunks=idata.chunks,
            dtype=idata.dtype,
            write_empty_chunks=False,
            fill_value=0,
            compressor=compressor,
            mode='w',
            )

    # Prepare tasks
    tasks = [(idata, odata, z, y, x, cz, cy, cx, algorithm) for z in range(divp1(sz, 2*cz))
                                                             for y in range(divp1(sy, 2*cy))
                                                             for x in range(divp1(sx, 2*cx))]

    # Use ThreadPoolExecutor to process blocks in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(process_chunk, tasks), total=len(tasks)))

    print("Processing complete")

def get_zarr_name(standard_config):
    # Get the zarr name from the standardization parameters
    scroll_id = standard_config['scroll_id']
    scroll_metadata_path = os.path.join(standard_config['volume_dir'], f"meta.json")
    scroll_metadata = json.load(open(scroll_metadata_path))
    resolution = scroll_metadata['voxelsize']
    def extract_kev(name):
        # Regular expression to find the keV value in the name string
        match = re.search(r'(\d+)\s*keV', name, re.IGNORECASE)
        if match:
            return match.group(1)  # Returns the keV value as a string
        raise ValueError(f"Could not find keV value in {name}")
    kev = extract_kev(scroll_metadata['name'])
    zarr_name = f"{kev}keV_{resolution}um_{scroll_id}"
    if standard_config['volume_type'] == "surface_volume":
        segment_id = standard_config['segment_id']
        zarr_name = os.path.join(zarr_name, f"{segment_id}")
    zarr_name += ".zarr"
    print(f"Zarr name: {zarr_name}")
    return zarr_name

def get_standard_config(tiffdir, output, volume_type):
    # Get the standardization parameters from the tiff directory
    standard_config = {}
    # Volume type
    standard_config['volume_type'] = volume_type
    # Volume dir
    if volume_type == "scroll_volume":
        standard_config['volume_dir'] = tiffdir
    elif volume_type == "surface_volume":
        # Get the segment ID from the tiff directory
        segment_id = os.path.basename(os.path.dirname(tiffdir))
        standard_config['segment_id'] = segment_id
        # find *_mask paths
        mask_path = os.path.join(os.path.dirname(tiffdir), f"{standard_config['segment_id']}_mask.png")
        # if not found, try to find it in the parent directory
        if not os.path.exists(mask_path):
            # Define the search pattern
            search_pattern = os.path.join(os.path.dirname(tiffdir), "*_mask.png")
            # Find all matching files
            mask_path = glob.glob(search_pattern)[0]
            print(f"Mask not found in {os.path.dirname(tiffdir)}, finding alternative: {mask_path}")
        standard_config['mask_path'] = mask_path
        surface_volume_meate_path = os.path.join(os.path.dirname(tiffdir), "meta.json")
        surface_volume_meta = json.load(open(surface_volume_meate_path))
        volume_id = surface_volume_meta['volume']
        volume_dir = os.path.join(tiffdir, "..", "..", "..", "volumes", volume_id)
        # Normalize the path
        volume_dir = os.path.normpath(volume_dir)
        standard_config['volume_dir'] = Path(volume_dir)
    else:
        raise NotImplementedError("Only scroll volumes are supported at the moment")
    # Volume ID
    volume_id = os.path.basename(standard_config['volume_dir'])
    standard_config['volume_id'] = volume_id
    # tiffdir is basepath/scrollid/scroll_name/volumes/volume_id
    # Scroll name
    scroll_name = os.path.basename(os.path.dirname(os.path.dirname(standard_config['volume_dir']))) 
    standard_config['scroll_name'] = scroll_name.split(".")[0]
    # Scroll ID
    scroll_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(standard_config['volume_dir']))))
    standard_config['scroll_id'] = scroll_id
    print(f"standard_config: {standard_config}")
    # Add the zarr name to the standardization parameters
    standard_config['zarr_name'] = get_zarr_name(standard_config)
    standard_config['zarr_dir'] = os.path.join(output, standard_config['zarr_name'])
    # Transform from the original volume to the canonical volume
    transform_from_canonical, transform_to_canonical, scale = get_transforms(standard_config['volume_dir'], volume_id)
    standard_config['transform_to_canonical'] = transform_to_canonical
    standard_config['transform_from_canonical'] = transform_from_canonical
    standard_config['scale'] = scale
    # Image equalization of 0 = Air to dtype.max = Papyrus
    equalization_min, equalization_max = equalize(standard_config['volume_dir'], n_components=4)
    standard_config['equalization_min'] = equalization_min
    standard_config['equalization_max'] = equalization_max
    return standard_config

def main():
    # parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="Create OME/Zarr data store from a set of TIFF files")
    parser.add_argument(
            "input_tiff_dir", 
            help="Directory containing tiff files")
    parser.add_argument(
            "output", 
            help="Name of directory that will contain the OME/zarr datastore dir")
    parser.add_argument(
            '--chunk_size',
            type=int,
            nargs='+',
            default=128, 
            help="Size of chunk")
    parser.add_argument(
            "--nlevels", 
            type=int, 
            default=6, 
            help="Number of subdivision levels to create, including level 0")
    parser.add_argument(
            "--max_gb", 
            type=float, 
            default=None, 
            help="Maximum amount of memory (in Gbytes) to use; None means no limit")
    parser.add_argument(
            "--zarr_only", 
            action="store_true", 
            help="Create a simple Zarr data store instead of an OME/Zarr hierarchy")
    parser.add_argument(
            "--overwrite", 
            action="store_true", 
            # default=False,
            help="Overwrite the output directory, if it already exists")
    parser.add_argument(
            "--num_threads", 
            type=int, 
            default=cpu_count(), 
            help="Advanced: Number of threads to use for processing. Default is number of CPUs")
    parser.add_argument(
            "--algorithm",
            choices=['mean', 'gaussian', 'nearest'],
            default="mean",
            help="Advanced: algorithm used to sub-sample the data")
    parser.add_argument(
            "--ranges", 
            help="Advanced: output only a subset of the data.  Example (in xyz order): 2500:3000,1500:4000,500:600")
    parser.add_argument(
            "--first_new_level", 
            type=int, 
            default=None, 
            help="Advanced: If some subdivision levels already exist, create new levels, starting with this one")
    parser.add_argument(
            "--n_bits",
            type=int,
            default=5,
            help="Number of bits to use for the output data")
    parser.add_argument(
            "--standardize",
            action="store_true",
            help="Create standardized volume format (equalize, mask, transform, etc.)"
    )


    args = parser.parse_args()
    
    tiffdir = Path(args.input_tiff_dir)
    if not tiffdir.exists() and args.first_new_level is None:
        print("Input TIFF directory",tiffdir,"does not exist")
        return 1

    chunk_size = args.chunk_size
    if isinstance(chunk_size, int):
        chunk_size = (chunk_size, chunk_size, chunk_size)
    elif len(chunk_size) == 1:
        chunk_size = (chunk_size[0], chunk_size[0], chunk_size[0])
    elif len(chunk_size) != 3:
        print("chunk_size must be a single number or 3 numbers")
        return 1

    nlevels = args.nlevels
    maxgb = args.max_gb
    zarr_only = args.zarr_only
    overwrite = args.overwrite
    num_threads = args.num_threads
    algorithm = args.algorithm
    print("overwrite", overwrite)
    first_new_level = args.first_new_level
    if first_new_level is not None and first_new_level < 1:
        print("first_new_level must be at least 1")
    n_bits = args.n_bits
    
    slices = None
    if args.ranges is not None:
        slices = parseSlices(args.ranges)
        if slices is None:
            print("Error parsing ranges argument")
            return 1
    
    print("slices", slices)
    
    # Load the standardization parameters
    if args.standardize:
        volume_type = "surface_volume" if os.path.basename(tiffdir) == "layers" else "scroll_volume"
        standard_config = get_standard_config(tiffdir, args.output, volume_type)
    else:
        if Path(args.output).suffix != ".zarr":
            standard_config_ = get_standard_config(tiffdir, args.output, volume_type)
            standard_config = {'zarr_dir': standard_config_['zarr_dir']}
        else:
            standard_config = {'zarr_dir': args.output}
    # N bits to use for the output data
    standard_config['n_bits'] = n_bits
    # Standardize flag
    standard_config['standardize'] = args.standardize
    
    zarrdir = Path(standard_config['zarr_dir'])
    if zarrdir.suffix != ".zarr":
        print("Name of ouput zarr directory must end with '.zarr'")
        return 1

    # even if overwrite flag is False, overwriting is permitted
    # when the user has set first_new_level
    if not overwrite and first_new_level is None:
        if zarrdir.exists():
            print("Error: Directory",zarrdir,"already exists")
            return(1)
    
    if first_new_level is None or zarr_only:
        if zarrdir.exists():
            print("removing", zarrdir)
            shutil.rmtree(zarrdir)
    
    if zarr_only:
        err = tifs2zarr(tiffdir, zarrdir, chunk_size, slices=slices, maxgb=maxgb, standard_config=standard_config)
        if err is not None:
            print("error returned:", err)
            return 1
        return
    
    if first_new_level is None:
        err = create_ome_dir(zarrdir)
        if err is not None:
            print("error returned:", err)
            return 1
    
    err = create_ome_headers(zarrdir, nlevels)
    if err is not None:
        print("error returned:", err)
        return 1
    
    if first_new_level is None:
        print("Creating level 0")
        err = tifs2zarr(tiffdir, zarrdir/"0", chunk_size, slices=slices, maxgb=maxgb, standard_config=standard_config)
        if err is not None:
            print("error returned:", err)
            return 1
    
    # for each level (1 and beyond):
    existing_level = 0
    if first_new_level is not None:
        existing_level = first_new_level-1
    for l in range(existing_level, nlevels-1):
        err = resize(zarrdir, l, num_threads, algorithm)
        if err is not None:
            print("error returned:", err)
            return 1

if __name__ == '__main__':
    sys.exit(main())
