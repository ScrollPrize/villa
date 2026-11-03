import {
  HELD_ROTATION_MS,
  HELD_SLICE_INTERVAL_MS,
  HELD_WATCHDOG_MS,
  HELD_ZOOM_OCTAVE_MS,
  KEY_ZOOM_FACTOR,
  PREFETCH_CHUNK_PRIORITY,
  SLICE_AHEAD_PREFETCH_PRIORITY,
} from "../config.js";

export function createHeldNavigation({
  state,
  view,
  surfaceController,
  slicePrefetch,
  scheduleDeferredSettle,
  stepSlice,
  setZoomAtLastPointer,
  rotateActiveViewport,
  setStatus,
}) {
  let heldNavigationFrame = null;
  let watchdogTimer = null;
  let lastHeldSliceMove = 0;

  function zoomBy(direction, factor) {
    if (state.activeViewport === "surface") {
      const rect = view.getCanvas("surface").getBoundingClientRect();
      surfaceController.zoomAtClient(
        direction > 0 ? state.surfaceView.zoom * factor : state.surfaceView.zoom / factor,
        state.lastPointerClient?.x ?? rect.left + rect.width / 2,
        state.lastPointerClient?.y ?? rect.top + rect.height / 2,
      );
    } else {
      setZoomAtLastPointer(direction > 0 ? state.zoom * factor : state.zoom / factor);
    }
  }

  const zoomGlide = {
    msPerUnit: HELD_ZOOM_OCTAVE_MS,
    tap: Math.log2(KEY_ZOOM_FACTOR),
    move: (octaves) => zoomBy(state.heldZoomDirection, 2 ** octaves),
    end: () => {
      state.heldZoomDirection = 0;
    },
    keyDown: false,
    owed: 0,
    time: 0,
  };
  const rotationGlide = {
    msPerUnit: HELD_ROTATION_MS,
    tap: 1,
    move: (steps) => rotateActiveViewport(state.heldRotation.axisName, state.heldRotation.degrees * steps),
    end: () => {
      state.heldRotation = null;
    },
    keyDown: false,
    owed: 0,
    time: 0,
  };

  function advance(glide, timestamp) {
    const elapsed = Math.max(0, timestamp - glide.time);
    const amount = Math.min(elapsed / glide.msPerUnit, glide.keyDown ? Infinity : glide.owed);
    glide.time = timestamp;
    glide.owed -= amount;
    if (state.volume) glide.move(amount);
    if (!glide.keyDown && glide.owed <= 0) endGlide(glide);
  }

  function endGlide(glide) {
    glide.end();
    scheduleDeferredSettle();
  }

  function restartGlide(glide) {
    glide.owed = 0;
    glide.time = performance.now();
  }

  function press(glide, isRepeat) {
    if (!isRepeat) glide.owed += glide.tap;
    glide.keyDown = true;
    requestHeldFrame();
  }

  function release(glide) {
    glide.keyDown = false;
    if (glide.owed <= 0) endGlide(glide);
  }

  function animateHeldNavigation(timestamp) {
    heldNavigationFrame = null;
    if (state.volume && state.heldSliceDirection !== 0 && timestamp - lastHeldSliceMove >= HELD_SLICE_INTERVAL_MS) {
      lastHeldSliceMove = timestamp;
      if (!surfaceController.stepNormal(state.heldSliceDirection)) {
        stepSlice(state.heldSliceDirection, { held: true });
      }
    }
    if (state.heldZoomDirection !== 0) advance(zoomGlide, timestamp);
    if (state.heldRotation !== null) advance(rotationGlide, timestamp);
    if (state.heldSliceDirection !== 0 || state.heldZoomDirection !== 0 || state.heldRotation !== null)
      heldNavigationFrame = window.requestAnimationFrame(animateHeldNavigation);
  }

  function requestHeldFrame() {
    if (heldNavigationFrame === null) heldNavigationFrame = window.requestAnimationFrame(animateHeldNavigation);
  }

  function stopSlice() {
    state.heldSliceDirection = 0;
    slicePrefetch.cancelSliceAhead();
    state.volume?.cancelPendingChunks(SLICE_AHEAD_PREFETCH_PRIORITY);
    scheduleDeferredSettle();
  }

  function stopRotation() {
    release(rotationGlide);
  }

  function stopWatchedHolds() {
    if (state.heldSliceDirection !== 0) stopSlice();
    if (state.heldRotation !== null) stopRotation();
  }

  function restartWatchdog() {
    window.clearTimeout(watchdogTimer);
    watchdogTimer = window.setTimeout(stopWatchedHolds, HELD_WATCHDOG_MS);
  }

  function flat() {
    const one = state.activeViewport !== "surface" && state.volume.fullShape_ZYX[0] === 1;
    if (one) setStatus("Slice navigation is unavailable for a two-dimensional source.");
    return one;
  }

  function startSlice(direction, isRepeat) {
    if (flat()) return;
    state.heldSliceDirection = direction;
    requestHeldFrame();
    restartWatchdog();
    if (!isRepeat) {
      state.volume.cancelPendingChunks(PREFETCH_CHUNK_PRIORITY);
      if (!surfaceController.stepNormal(direction)) stepSlice(direction);
      lastHeldSliceMove = performance.now();
    }
  }

  function stepOnce(direction) {
    if (flat()) return;
    state.volume.cancelPendingChunks(PREFETCH_CHUNK_PRIORITY);
    if (!surfaceController.stepNormal(direction)) stepSlice(direction);
    scheduleDeferredSettle();
  }

  function startZoom(direction, isRepeat) {
    if (state.heldZoomDirection !== direction) restartGlide(zoomGlide);
    state.heldZoomDirection = direction;
    press(zoomGlide, isRepeat);
  }

  function stopZoom() {
    release(zoomGlide);
  }

  function startRotation(code, axisName, degrees, isRepeat) {
    if (state.heldRotation?.code !== code) {
      if (state.heldRotation !== null && rotationGlide.owed > 0) rotationGlide.move(rotationGlide.owed);
      restartGlide(rotationGlide);
      state.heldRotation = { code, axisName, degrees };
    }
    press(rotationGlide, isRepeat);
    restartWatchdog();
  }

  return { startSlice, startZoom, startRotation, stepOnce, stopSlice, stopZoom, stopRotation };
}
