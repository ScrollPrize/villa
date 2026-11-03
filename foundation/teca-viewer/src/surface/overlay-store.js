import {
  OVERLAY_LINE_WIDTH_RANGE,
  OVERLAY_RENDER_VERTEX_LIMIT,
  SURFACE_PYRAMID_CACHE_BYTES,
  SURFACE_STRIP_CACHE_BYTES,
} from "../config.js";
import { buildSurfaceIntersectionIndex } from "./overlay-geometry.js";
import { SurfacePositionImage } from "./position-image.js";
import { SurfacePyramid } from "./pyramid.js";
import { SurfacePyramidCache } from "./pyramid-cache.js";
import { TifxyzSurfaceSource } from "./source.js";
import { SurfaceStripCache } from "./strip-cache.js";
import { tifxyzUrl } from "./tifxyz.js";

const OVERLAY_PALETTE = ["#dbbc7f", "#e67e80", "#7fbbb3", "#83c092", "#d699b6", "#e69875"];
// tifxyz points are in the primary volume's finest-voxel frame.
const FINEST_VOXEL_REGISTRATION = Object.freeze({
  scale_ZYX: Object.freeze([1, 1, 1]),
  offset_ZYX: Object.freeze([0, 0, 0]),
});

export function overlayColor(overlay) {
  return OVERLAY_PALETTE[overlay.colorIndex % OVERLAY_PALETTE.length];
}

export function createOverlayStore({ state, rendering, geometryByteCache, linkedPaths }) {
  const overlays = [];
  const pending = new Map();
  const queued = new Set();
  let queueTail = Promise.resolve();
  const linked = new Set(linkedPaths);
  const surfacePyramids = new SurfacePyramidCache(SURFACE_PYRAMID_CACHE_BYTES);
  const surfaceStrips = new SurfaceStripCache(SURFACE_STRIP_CACHE_BYTES);
  let revision = 0;
  let nextColorIndex = 0;
  const edited = new Map();

  function overlayRevision() {
    return revision;
  }

  function findOverlay(path) {
    return overlays.find((overlay) => overlay.path === path) ?? null;
  }

  function descriptorOf(path) {
    return state.overlays.find((entry) => entry.path === path) ?? null;
  }

  function wanted(path, volume) {
    return state.volume === volume && descriptorOf(path)?.visible !== false;
  }

  function releaseSurface(overlay) {
    if (overlay.positionImage.consumers.size === 0)
      surfacePyramids.retain(overlay.surfaceSource.tifxyz.cacheKey, overlay.pyramid);
    overlay.surfaceSource.release();
  }

  function unload(path) {
    pending.get(path)?.controller.abort();
    pending.delete(path);
    const index = overlays.findIndex((overlay) => overlay.path === path);
    if (index === -1) return;
    releaseSurface(overlays[index]);
    overlays.splice(index, 1);
    revision += 1;
  }

  async function loadOverlay(input, report) {
    let path = input;
    const primaryVolume = state.volume;
    try {
      path = tifxyzUrl(input);
      const primarySource = descriptorOf(path)?.primarySource ?? state.activeSource;
      if (!state.sources.includes(primarySource)) {
        throw new Error(`Surface ${path} names no open source: ${primarySource}.`);
      }
      if (state.activeSource !== primarySource || primaryVolume === null) {
        throw new Error("Activate the surface's volume before loading it.");
      }
      if (edited.get(path)?.volume !== primaryVolume) edited.delete(path);
      const surface = edited.get(path)?.surface ?? null;
      const loaded = findOverlay(path);
      if (loaded?.primaryVolume === primaryVolume && loaded.edited === surface) {
        report.loaded(loaded);
        return loaded;
      }
      const joined = pending.get(path);
      if (joined?.primaryVolume === primaryVolume && joined.edited === surface) {
        const overlay = await joined.promise;
        report.loaded(overlay);
        return overlay;
      }
      joined?.controller.abort();
      const controller = new AbortController();
      const load = { primaryVolume, edited: surface, controller, promise: null };
      const current = () => pending.get(path) === load && wanted(path, primaryVolume);
      pending.set(path, load);
      report.loading(path);
      load.promise = (async () => {
        let surfaceSource = null;
        let wholePyramid = null;
        try {
          surfaceSource = await TifxyzSurfaceSource.open(
            path,
            surface,
            controller.signal,
            surfaceStrips,
            geometryByteCache,
          );
          SurfacePositionImage.validateCapacity(surfaceSource, rendering, rendering.byteLimit);
          wholePyramid = surfacePyramids.take(surfaceSource.tifxyz.cacheKey);
          const buildPyramid = wholePyramid === null;
          const pyramid = wholePyramid ?? new SurfacePyramid(surfaceSource.grid_HW);
          let completed = 0;
          await surfaceSource.forEachStrip((index, strip) => {
            if (buildPyramid) pyramid.addStrip(surfaceSource, index, strip);
            if (current()) report.progress(path, ++completed, surfaceSource.stripCount);
          }, controller.signal);
          if (buildPyramid) pyramid.finish();
          wholePyramid = pyramid;
          const overview = await pyramid.overview(surfaceSource, controller.signal);
          const [H, W] = overview.grid_HW;
          const surfaceIntersectionIndex = buildSurfaceIntersectionIndex(overview.points_zyx_HW3, H, W);
          if (surfaceIntersectionIndex.validCells * 2 > OVERLAY_RENDER_VERTEX_LIMIT) {
            throw new Error(
              `Surface ${path} expands to ${surfaceIntersectionIndex.validCells * 2} render vertices, exceeding ${OVERLAY_RENDER_VERTEX_LIMIT}.`,
            );
          }
          if (!current()) throw new DOMException("Surface load superseded.", "AbortError");
          const descriptor = descriptorOf(path) ?? { path, primarySource, visible: true, colorIndex: nextColorIndex };
          const staged = {
            path,
            primarySource,
            kind: "surface",
            visible: descriptor.visible,
            colorIndex: descriptor.colorIndex,
            grid_HW: [H, W],
            sourceStride: overview.stride,
            points_zyx_HW3: overview.points_zyx_HW3,
            surfaceSource,
            pyramid,
            positionImage: new SurfacePositionImage(surfaceSource),
            surfaceIntersectionIndex,
            registration: FINEST_VOXEL_REGISTRATION,
            primaryVolume,
            edited: surface,
          };
          staged.colors = [overlayColor(staged)];
          if (!state.overlays.includes(descriptor)) state.overlays.push(descriptor);
          const replaced = findOverlay(path);
          if (replaced === null) overlays.push(staged);
          else {
            releaseSurface(replaced);
            overlays[overlays.indexOf(replaced)] = staged;
          }
          nextColorIndex = Math.max(nextColorIndex, descriptor.colorIndex + 1);
          revision += 1;
          return staged;
        } catch (error) {
          if (wholePyramid !== null) surfacePyramids.retain(surfaceSource.tifxyz.cacheKey, wholePyramid);
          surfaceSource?.dispose();
          throw error;
        } finally {
          if (pending.get(path) === load) pending.delete(path);
        }
      })();
      const overlay = await load.promise;
      report.loaded(overlay);
      return overlay;
    } catch (error) {
      if (error.name === "AbortError") queueMicrotask(() => syncLoads(report));
      else if (wanted(path, primaryVolume)) report.failed(path, error);
      return null;
    }
  }

  function syncLoads(report) {
    for (const overlay of [...overlays]) if (!wanted(overlay.path, overlay.primaryVolume)) unload(overlay.path);
    for (const [path, load] of pending) if (!wanted(path, load.primaryVolume)) unload(path);
    for (const [path, { volume }] of edited) if (volume !== state.volume) edited.delete(path);
    if (state.volume === null) return;
    const primarySource = state.activeSource;
    const paths = state.overlays
      .filter((descriptor) => descriptor.primarySource === primarySource && descriptor.visible)
      .map((descriptor) => descriptor.path);
    for (const path of [...paths, ...linked]) {
      const key = `${primarySource}\0${path}`;
      if (queued.has(key)) continue;
      queued.add(key);
      queueTail = queueTail
        .then(async () => {
          if (state.activeSource !== primarySource || state.volume === null || pending.has(path)) return;
          if (findOverlay(path)?.primaryVolume === state.volume) {
            linked.delete(path);
            return;
          }
          const descriptor = descriptorOf(path);
          if (!linked.has(path) && (descriptor?.primarySource !== primarySource || !descriptor.visible)) return;
          if ((await loadOverlay(path, report)) !== null) linked.delete(path);
        })
        .finally(() => queued.delete(key));
    }
  }

  function setEditedSurface(input, surface) {
    const path = tifxyzUrl(input);
    if (surface === null) edited.delete(path);
    else edited.set(path, { surface, volume: state.volume });
  }

  async function surfaceChanged(input, surface, start_HW, stop_HW) {
    const path = tifxyzUrl(input);
    if (pending.get(path)?.edited === surface) {
      pending.get(path).controller.abort();
      pending.delete(path);
    }
    const overlay = findOverlay(path);
    if (overlay?.edited !== surface) return;
    const { surfaceSource, positionImage, pyramid } = overlay;
    await pyramid.patch(surfaceSource, start_HW, stop_HW, null);
    const overview = await pyramid.overview(surfaceSource, null);
    overlay.points_zyx_HW3 = overview.points_zyx_HW3;
    overlay.surfaceIntersectionIndex = buildSurfaceIntersectionIndex(overview.points_zyx_HW3, ...overview.grid_HW);
    await positionImage.changed(start_HW[0], stop_HW[0]);
    revision += 1;
  }

  async function readSurface(input, signal) {
    const path = tifxyzUrl(input);
    const surfaceSource = await TifxyzSurfaceSource.open(path, null, signal, surfaceStrips, geometryByteCache);
    try {
      const [height, width] = surfaceSource.grid_HW;
      const stripCells = surfaceSource.stripRows * width;
      const points_zyx_HW3 = new Float32Array(3 * height * width);
      await surfaceSource.forEachStrip((index, _strip, xyz_HW) => {
        const first = index * stripCells;
        for (let cell = 0; cell < xyz_HW[0].length; cell++)
          for (let axis = 0; axis < 3; axis++) points_zyx_HW3[3 * (first + cell) + 2 - axis] = xyz_HW[axis][cell];
      }, signal);
      return { grid_HW: [height, width], scale_UV: [...surfaceSource.scale_UV], points_zyx_HW3 };
    } finally {
      surfaceSource.dispose();
    }
  }

  function removeOverlay(path) {
    unload(path);
    const index = state.overlays.findIndex((overlay) => overlay.path === path);
    if (index !== -1) state.overlays.splice(index, 1);
    revision += 1;
  }

  function setOverlayLineWidth(width) {
    state.overlayLineWidth = Math.min(OVERLAY_LINE_WIDTH_RANGE[1], Math.max(OVERLAY_LINE_WIDTH_RANGE[0], width));
  }

  function setOverlayVisible(path, visible) {
    state.overlays.find((entry) => entry.path === path).visible = visible;
    const overlay = findOverlay(path);
    if (overlay !== null) overlay.visible = visible;
    revision += 1;
  }

  return {
    overlays,
    overlayRevision,
    loadOverlay,
    syncLoads,
    setEditedSurface,
    surfaceChanged,
    readSurface,
    removeOverlay,
    setOverlayLineWidth,
    setOverlayVisible,
  };
}
