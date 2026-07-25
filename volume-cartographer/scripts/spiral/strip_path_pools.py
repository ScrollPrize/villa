# Pool-management infrastructure for the 'dijkstra' patch-strip sampling mode (see losses.py
# for the sampling semantics and strip_paths.py for the path computation itself).
#
# A full single-source Dijkstra is too slow to run per training step per patch, so, dataloader-
# style, paths are computed by a pool of background worker *processes* (scipy's Dijkstra holds
# the GIL, so threads would stall the training loop) into small per-patch / per-anchor pools
# cached on the patch object; per-step sampling then only subsamples positions along a pooled
# path + subpixel jitter. The workers also continuously *refresh* the pools of recently-used
# patches/anchors (FIFO replacement), so over the run each patch sees an ever-growing set of
# strips rather than a frozen pool. Pool contents are therefore nondeterministic run-to-run,
# but the machinery draws worker seeds from a dedicated RNG and never touches the seeded
# global np.random stream.

import concurrent.futures
import multiprocessing
import os
import threading

import numpy as np

import strip_paths


PATCH_POOL_NUM_STARTS = 4
PATCH_POOL_ENDPOINTS_PER_START = 8
ANCHOR_POOL_PATHS_PER_CONE = 4
# Worker processes for building/refreshing path pools ('dijkstra' mode only). 0 disables the
# pool: initial builds then run inline on first use and pools stay fixed for the whole run.
# Each DDP rank gets its own pool, so lower this when running many ranks per box.
NUM_WORKERS = int(os.environ.get('FIT_SPIRAL_STRIP_PATH_WORKERS', '4'))
MAX_PENDING_REFRESHES = 32

_executor = None  # None = not created yet, False = disabled
_pending_refreshes = threading.BoundedSemaphore(MAX_PENDING_REFRESHES)
_seed_rng = np.random.default_rng()


def _get_executor():
    global _executor
    if _executor is None:
        if NUM_WORKERS <= 0:
            _executor = False
        else:
            # forkserver: workers fork from a clean server process, so no CUDA context is
            # inherited (unlike fork). Each worker still re-imports the parent __main__
            # (fit_spiral) per standard multiprocessing semantics -- its __main__ guard makes
            # that safe, just a one-time torch import per worker. CUDA is hidden from the
            # server (and hence all workers forked from it, since it inherits the environment
            # at first task submission below) so no worker can ever touch the training GPUs.
            mp_context = multiprocessing.get_context('forkserver')
            mp_context.set_forkserver_preload(['strip_paths'])
            previous_cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            try:
                _executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=NUM_WORKERS,
                    mp_context=mp_context,
                )
                _executor.submit(strip_paths.warmup)  # starts the forkserver now
            except Exception as e:
                _disable_executor(e)
            finally:
                if previous_cuda_env is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = previous_cuda_env
    return _executor or None


def _disable_executor(error):
    global _executor
    if _executor is not False:
        print(f'strip-path worker pool failed ({error!r}); falling back to inline path building')
    _executor = False


def _submit_task(fn, *args):
    # Submit to the worker pool; on executor failure (e.g. BrokenProcessPool) disable it for
    # the rest of the run and return None so callers fall back to inline computation.
    executor = _get_executor()
    if executor is None:
        return None
    try:
        return executor.submit(fn, *args)
    except Exception as e:
        _disable_executor(e)
        return None


def warm_workers():
    # Called at configure time so the workers spawn (and pay their one-time imports) while the
    # trainer is still loading data.
    for _ in range(NUM_WORKERS):
        if _submit_task(strip_paths.warmup) is None:
            return


def _new_task_seed():
    return int(_seed_rng.integers(2 ** 63))


def _patch_packed_mask(patch):
    packed = getattr(patch, '_strip_packed_mask', None)
    if packed is None:
        packed = strip_paths.pack_mask(patch._sampling_valid_quad_mask_np)
        patch._strip_packed_mask = packed
    return packed


def ensure_patch_path_pools(patches_to_ensure):
    # Make sure every patch has an initial path pool (PATCH_POOL_NUM_STARTS starts x
    # PATCH_POOL_ENDPOINTS_PER_START endpoints). Blocking, since the radius/DT losses need
    # all their sampled patches, but with workers a whole step's batch builds in parallel.
    missing = [patch for patch in patches_to_ensure if getattr(patch, '_strip_path_pool', None) is None]
    if not missing:
        return
    if _get_executor() is not None:
        for patch in missing:
            if getattr(patch, '_strip_pool_future', None) is None:
                packed, shape = _patch_packed_mask(patch)
                future = _submit_task(
                    strip_paths.compute_patch_path_pool_packed, packed, shape,
                    PATCH_POOL_NUM_STARTS, PATCH_POOL_ENDPOINTS_PER_START,
                    _new_task_seed(),
                )
                if future is None:
                    break
                patch._strip_pool_future = future
    for patch in missing:
        if getattr(patch, '_strip_path_pool', None) is not None:
            continue
        future = getattr(patch, '_strip_pool_future', None)
        if future is not None:
            patch._strip_pool_future = None
            try:
                patch._strip_path_pool = future.result()
                continue
            except Exception as e:
                _disable_executor(e)
        patch._strip_path_pool = strip_paths.compute_patch_path_pool(
            patch._sampling_valid_quad_mask_np,
            PATCH_POOL_NUM_STARTS, PATCH_POOL_ENDPOINTS_PER_START,
            _new_task_seed(),
        )


def submit_patch_pool_refresh(patch):
    # Background freshness: one Dijkstra from a new random start replaces the pool's oldest
    # paths (FIFO), so pools turn over continuously instead of staying fixed. Skipped (retried
    # on a later step) while a refresh for this patch is already in flight or the global
    # refresh backlog is full.
    executor = _get_executor()
    if executor is None or getattr(patch, '_strip_refresh_inflight', False):
        return
    if not _pending_refreshes.acquire(blocking=False):
        return
    patch._strip_refresh_inflight = True
    packed, shape = _patch_packed_mask(patch)
    future = _submit_task(
        strip_paths.compute_patch_path_pool_packed, packed, shape,
        1, PATCH_POOL_ENDPOINTS_PER_START, _new_task_seed(),
    )
    if future is None:
        _pending_refreshes.release()
        patch._strip_refresh_inflight = False
        return

    def _install(f):
        # Runs on the executor's result-handler thread; samplers snapshot the pool list once
        # per use, so a whole-list swap is safe.
        _pending_refreshes.release()
        patch._strip_refresh_inflight = False
        try:
            new_paths = f.result()
        except Exception as e:
            print(f'strip path refresh failed: {e!r}')
            return
        old_pool = patch._strip_path_pool
        patch._strip_path_pool = old_pool[len(new_paths):] + list(new_paths)

    future.add_done_callback(_install)


def get_anchor_path_pools(patch, i_q, j_q):
    # Per-anchor path pools for the rel/abs winding losses (4 cardinal cones x
    # ANCHOR_POOL_PATHS_PER_CONE paths from the annotated cell; see
    # strip_paths.compute_anchor_path_pools). Non-blocking: the first use submits a background
    # build and returns None (the caller skips this anchor for a step or two -- the winding
    # losses tolerate missing pairs); later uses return the cached pools and occasionally
    # kick off a background refresh, which replaces the pools wholesale.
    pools_by_anchor = getattr(patch, '_anchor_path_pools', None)
    if pools_by_anchor is None:
        pools_by_anchor = {}
        patch._anchor_path_pools = pools_by_anchor
        patch._anchor_pools_inflight = set()
    key = (i_q, j_q)
    pools = pools_by_anchor.get(key)

    executor = _get_executor()
    if executor is None:
        if pools is None:
            pools = strip_paths.compute_anchor_path_pools(
                patch._sampling_valid_quad_mask_np, i_q, j_q,
                ANCHOR_POOL_PATHS_PER_CONE, _new_task_seed(),
            )
            pools_by_anchor[key] = pools
        return pools

    inflight = patch._anchor_pools_inflight
    if key not in inflight and _pending_refreshes.acquire(blocking=False):
        packed, shape = _patch_packed_mask(patch)
        future = _submit_task(
            strip_paths.compute_anchor_path_pools_packed, packed, shape, i_q, j_q,
            ANCHOR_POOL_PATHS_PER_CONE, _new_task_seed(),
        )
        if future is None:
            _pending_refreshes.release()
            return pools
        inflight.add(key)

        def _install(f):
            _pending_refreshes.release()
            inflight.discard(key)
            try:
                pools_by_anchor[key] = f.result()
            except Exception as e:
                print(f'anchor strip path build failed: {e!r}')

        future.add_done_callback(_install)
    return pools
