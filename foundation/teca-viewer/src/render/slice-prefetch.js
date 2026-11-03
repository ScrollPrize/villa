import {
  PREFETCH_CHUNK_PRIORITY,
  SCALE_PICK_TOLERANCE,
  SHOWN_CHUNK_PRIORITY_LIMIT,
  SLICE_AHEAD_MAX_LEAD_CHUNKS,
  SLICE_AHEAD_MAX_QUEUED_CHUNKS,
  SLICE_AHEAD_PREFETCH_PRIORITY,
} from "../config.js";
import { rotatedPlaneChunkAddresses, toLevelSamplePoint_ZYX } from "../gpu/addresses.js";
import { prefetchPlaneChunks } from "../gpu/prefetch.js";
import { clamp } from "../math.js";
import { planeStepAxis } from "../planes.js";
import { drawnPlanes, orthogonalStepPlane, viewIsRotated, viewWidth } from "../views/pose.js";
import { resolveScaleIndex, scalePickPixels } from "../volume/levels.js";
import { MotionProfile } from "./motion.js";

function nearbySliceChunkAddresses(addresses, scale, center_ZYX, axis, direction) {
  const chunkCount = Math.ceil(scale.shape_ZYX[axis] / scale.chunks_ZYX[axis]);
  const chunkPosition = toLevelSamplePoint_ZYX(scale, center_ZYX)[axis] / scale.chunks_ZYX[axis];
  const firstDirection = direction || (chunkPosition - Math.floor(chunkPosition) >= 0.5 ? 1 : -1);
  const keys = new Set(addresses.map(({ chunk_ZYX }) => chunk_ZYX.join("/")));
  const neighbors = [];
  for (const step of [firstDirection, -firstDirection]) {
    for (const address of addresses) {
      const chunk_ZYX = [...address.chunk_ZYX];
      chunk_ZYX[axis] += step;
      const key = chunk_ZYX.join("/");
      if (chunk_ZYX[axis] < 0 || chunk_ZYX[axis] >= chunkCount || keys.has(key)) continue;
      keys.add(key);
      neighbors.push({ ...address, chunk_ZYX, fetchOrder: 1 });
    }
  }
  return neighbors;
}

export class SlicePrefetch {
  constructor({ state, view }) {
    this.state = state;
    this.view = view;
    this.motion = new MotionProfile();
    this.lastSliceAheadPrefetchSignature = null;
    this.sliceAheadAbortController = null;
  }

  cancelSliceAhead() {
    this.sliceAheadAbortController?.abort();
    this.sliceAheadAbortController = null;
    this.lastSliceAheadPrefetchSignature = null;
  }

  cancelHiddenPanes() {
    if (drawnPlanes(this.state).length > 0) return;
    this.cancelSliceAhead();
    this.state.volume?.cancelPendingChunks(SHOWN_CHUNK_PRIORITY_LIMIT);
  }

  observeMotion() {
    this.motion.observe(this.state.center_ZYX, performance.now());
  }

  predictedPrefetchCenter(scale) {
    const { state } = this;
    const queue = state.volume?.getQueueStats() ?? null;
    const queuedWaves =
      queue === null || queue.concurrency < 1 ? 0 : Math.ceil((queue.active + queue.pending) / queue.concurrency);
    return this.motion.prediction(
      state.center_ZYX,
      queue?.averageLatencyMs ?? null,
      scale,
      SLICE_AHEAD_MAX_LEAD_CHUNKS,
      1 + queuedWaves,
    );
  }

  planeFetchAddresses(planeView, scale, addresses, prefetch) {
    const { state } = this;
    const fetchAddresses = [...addresses];
    if (planeView.plane === orthogonalStepPlane(state) && !viewIsRotated(state, planeView.plane)) {
      fetchAddresses.push(
        ...nearbySliceChunkAddresses(
          addresses,
          scale,
          state.center_ZYX,
          planeStepAxis(planeView.plane),
          state.heldSliceDirection,
        ),
      );
    }
    fetchAddresses.push(...prefetch);
    if (state.heldSliceDirection !== 0) {
      const predicted = this.predictedPrefetchCenter(scale);
      const addressKeys = new Set(fetchAddresses.map((address) => address.chunk_ZYX.join("/")));
      for (let step = predicted.leadChunks; step >= 1; step--) {
        const futureCenter_ZYX = state.center_ZYX.map(
          (value, axis) => value + (predicted.displacement_ZYX[axis] * step) / predicted.leadChunks,
        );
        for (const address of rotatedPlaneChunkAddresses(
          futureCenter_ZYX,
          planeView.basis,
          planeView.viewWidthVoxels,
          scale,
          1,
          planeView.viewAspect,
        )) {
          const key = address.chunk_ZYX.join("/");
          if (addressKeys.has(key)) continue;
          addressKeys.add(key);
          fetchAddresses.push({ ...address, fetchOrder: -step });
        }
      }
    }
    return fetchAddresses;
  }

  prefetchAheadForSliceNavigation(plane, axis, direction) {
    const { state } = this;
    if (!drawnPlanes(state).includes(plane)) return;
    this.observeMotion();
    const queueStats = state.volume.getQueueStats();
    if (queueStats.active + queueStats.pending >= SLICE_AHEAD_MAX_QUEUED_CHUNKS) return;
    const { view } = this;
    const viewWidthVoxels = viewWidth(state);
    const scale = this.sliceNavigationScale(plane, viewWidthVoxels);
    const aheadChunks = this.predictedPrefetchCenter(scale).leadChunks;
    const chunkSize = scale.chunks_ZYX[axis];
    const centerInScale = Math.round(toLevelSamplePoint_ZYX(scale, state.center_ZYX)[axis]);
    const centerChunk = Math.floor(centerInScale / chunkSize);
    const maxChunk = Math.floor((scale.shape_ZYX[axis] - 1) / chunkSize);
    const signature = `${plane}:${scale.path}:${axis}:${direction}:${centerChunk}:${Math.round(viewWidthVoxels)}`;
    if (signature === this.lastSliceAheadPrefetchSignature) return;
    this.sliceAheadAbortController?.abort();
    this.sliceAheadAbortController = new AbortController();
    const signal = this.sliceAheadAbortController.signal;
    state.volume.cancelPendingChunks(SLICE_AHEAD_PREFETCH_PRIORITY, PREFETCH_CHUNK_PRIORITY - 1);
    this.lastSliceAheadPrefetchSignature = signature;
    for (let chunkOffset = 0; chunkOffset <= aheadChunks; chunkOffset++) {
      const targetChunk = clamp(centerChunk + direction * chunkOffset, 0, maxChunk);
      const targetScale =
        direction > 0
          ? Math.min(scale.shape_ZYX[axis] - 1, (targetChunk + 1) * chunkSize - 1)
          : targetChunk * chunkSize;
      const aheadCenter_ZYX = [...state.center_ZYX];
      aheadCenter_ZYX[axis] = clamp(
        targetScale * scale.downsample_ZYX[axis] + scale.offset_ZYX[axis],
        0,
        state.volume.fullShape_ZYX[axis] - 1,
      );
      prefetchPlaneChunks(
        state.volume,
        state.selectedChannel,
        plane,
        aheadCenter_ZYX,
        viewWidthVoxels,
        chunkOffset <= 1 ? 1 : 0,
        scale,
        signal,
        SLICE_AHEAD_PREFETCH_PRIORITY + chunkOffset,
        view.getOrthogonalPanelAspect(plane),
        () => drawnPlanes(state).includes(plane),
      );
    }
  }

  sliceNavigationScale(plane, viewWidthVoxels) {
    const { state, view } = this;
    const renderPixels = scalePickPixels(viewWidthVoxels, view.getOrthogonalPanelPixelSize(plane));
    return state.volume.scales[
      resolveScaleIndex(
        state.volume.pickScaleIndexForView(viewWidthVoxels, renderPixels, SCALE_PICK_TOLERANCE),
        state.manualScaleIndex,
        state.manualCoarsestScaleIndex,
        state.volume.scales.length,
      )
    ];
  }
}
