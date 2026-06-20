import dbm
import pickle

import numpy as np
from tqdm import tqdm


def load_tracks_from_dbm(path, z_lo, z_hi, downsample_factor):
    # Load tracks written by extract_surface_tracks.py. Each DBM value is a
    # pickled list of (N, 3) int32 zyx arrays; keep only tracks that lie entirely
    # within [z_lo, z_hi) after accounting for downsampling.
    z_lo_raw = z_lo * downsample_factor
    z_hi_raw = z_hi * downsample_factor
    tracks = []
    with dbm.open(path, 'r') as db:
        for key in tqdm(db.keys(), desc='loading tracks'):
            entries = pickle.loads(db[key])
            if not entries:
                continue
            # Vectorize the per-track z min/max across the whole key: concatenate
            # every non-empty track's z column and reduce per segment, rather
            # than calling .min()/.max() once per track.
            idx = [i for i in range(len(entries)) if len(entries[i])]
            if not idx:
                continue
            lengths = np.fromiter((len(entries[i]) for i in idx), dtype=np.intp, count=len(idx))
            zcat = np.concatenate([entries[i][:, 0] for i in idx])
            offsets = np.zeros(len(idx), dtype=np.intp)
            np.cumsum(lengths[:-1], out=offsets[1:])
            zmins = np.minimum.reduceat(zcat, offsets)
            zmaxs = np.maximum.reduceat(zcat, offsets)
            keep = (zmins >= z_lo_raw) & (zmaxs < z_hi_raw)
            for j in np.nonzero(keep)[0]:
                tracks.append(entries[idx[j]].astype(np.float32) / downsample_factor)
    return tracks
