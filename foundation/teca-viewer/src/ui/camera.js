import {
  CAMERA_DEGREES_PER_PIXEL,
  MAX_HELD_SLICE_STEP,
  MAX_ZOOM,
  MIN_HELD_SLICE_STEP,
  MIN_ZOOM,
  ORTHOGONAL_PLANES,
  PLANE_TILT_DEGREES_PER_PIXEL,
} from "../config.js";
import {
  clamp,
  IDENTITY_ROTATION,
  quatAlignVectors,
  quatFromAxisAngle,
  quatMultiply,
  rotateVectorByQuat,
} from "../math.js";
import { planeStepAxis, rotatedPlaneBasis } from "../planes.js";
import {
  baseViewWidth,
  orthogonalStepPlane,
  panelVoxelOffset,
  planeDisplayBasis,
  planeRotation,
  viewerDock,
  viewIsRotated,
  viewWidth,
} from "../views/pose.js";

export function createCamera({
  state,
  view,
  surfaceController,
  scheduleRender,
  queueOrthogonalRedraw,
  updateSlicedPlane,
  markPlanesDirty,
  markViewDirty,
  resetDirtyPlanes,
  slicePrefetch,
  publishSession,
  setStatus,
  syncPositionInputs,
  syncZoomInputs,
  syncRotationReadout,
  updateScaleBars,
  updateViewerShellState,
}) {
  function setPositionAxis(axis, value) {
    state.center_ZYX[axis] = Math.round(clamp(value, 0, state.volume.fullShape_ZYX[axis] - 1));
    syncPositionInputs();
    markViewDirty();
    scheduleRender();
  }

  function setPositionXYZ(position_XYZ) {
    state.center_ZYX = [position_XYZ[2], position_XYZ[1], position_XYZ[0]].map((value, axis) =>
      Math.min(Math.round(value), state.volume.fullShape_ZYX[axis] - 1),
    );
    syncPositionInputs();
    markViewDirty();
    scheduleRender();
  }

  function setPhysicalFocus(point_ZYX, { surface = true } = {}) {
    const shape_ZYX = state.volume?.fullShape_ZYX;
    if (
      !shape_ZYX ||
      point_ZYX.some((value, axis) => !Number.isFinite(value) || value < 0 || value >= shape_ZYX[axis])
    ) {
      throw new Error("Physical focus must be inside the primary CT volume.");
    }
    state.center_ZYX = [...point_ZYX];
    if (surface) void surfaceController.focusAtPoint(point_ZYX);
    syncPositionInputs();
    markViewDirty();
    updateSlicedPlane();
    queueOrthogonalRedraw();
    scheduleRender();
    publishSession();
  }

  function tiltPlaneByPixels(plane, dx, dy) {
    if (state.volume === null || state.volume.fullShape_ZYX[0] === 1) return;
    if (state.rotationMode === "frame") {
      state.paneRotations = Object.fromEntries(ORTHOGONAL_PLANES.map((name) => [name, [...state.rotation]]));
      state.rotationMode = "pane";
    }
    const basis = planeDisplayBasis(state, plane);
    state.paneRotations[plane] = quatMultiply(
      quatFromAxisAngle(basis.right_XYZ, -dy * PLANE_TILT_DEGREES_PER_PIXEL),
      quatMultiply(quatFromAxisAngle(basis.up_XYZ, -dx * PLANE_TILT_DEGREES_PER_PIXEL), state.paneRotations[plane]),
    );
    markPlanesDirty([plane]);
    scheduleRender();
  }

  function resetPaneOrientation(plane, target) {
    if (state.rotationMode === "frame") {
      state.paneRotations = Object.fromEntries(ORTHOGONAL_PLANES.map((name) => [name, [...state.rotation]]));
      state.rotationMode = "pane";
    }
    const source = rotatedPlaneBasis(plane, IDENTITY_ROTATION);
    const destination = rotatedPlaneBasis(target, IDENTITY_ROTATION);
    const up = quatAlignVectors(source.up_XYZ, destination.up_XYZ, source.right_XYZ);
    const right = quatAlignVectors(rotateVectorByQuat(source.right_XYZ, up), destination.right_XYZ, destination.up_XYZ);
    state.paneRotations[plane] = plane === target ? [...IDENTITY_ROTATION] : quatMultiply(right, up);
    onRotationChanged([plane]);
  }

  function setRotationMode(mode) {
    if (state.rotationMode === mode) return;
    const plane = orthogonalStepPlane(state);
    if (mode === "pane") {
      state.paneRotations = Object.fromEntries(ORTHOGONAL_PLANES.map((name) => [name, [...state.rotation]]));
    } else {
      state.rotation = [...state.paneRotations[plane]];
    }
    state.rotationMode = mode;
    onRotationChanged(ORTHOGONAL_PLANES);
  }

  function rotateActiveViewport(axisName, degrees) {
    if (state.activeViewport === "surface") {
      setStatus("Surface orientation follows the tifxyz normals; use Flip normal to reverse them.");
      return;
    }
    if (state.activeViewport === "3d") {
      rotateCamera3d(axisName, degrees);
      return;
    }
    if (!state.volume) return;
    if (state.volume.fullShape_ZYX[0] === 1) {
      setStatus("Oblique rotation is unavailable for a two-dimensional source.");
      return;
    }
    const plane = orthogonalStepPlane(state);
    const basis = planeDisplayBasis(state, plane);
    const axis_XYZ = axisName === "up" ? basis.up_XYZ : basis.right_XYZ;
    const nextRotation = quatMultiply(quatFromAxisAngle(axis_XYZ, degrees), planeRotation(state, plane));
    if (state.rotationMode === "frame") state.rotation = nextRotation;
    else state.paneRotations[plane] = nextRotation;
    onRotationChanged(state.rotationMode === "frame" ? ORTHOGONAL_PLANES : [plane]);
  }

  function rotateCamera3d(axisName, degrees, persist = true) {
    const localAxis_XYZ = axisName === "up" ? [0, 1, 0] : [1, 0, 0];
    const axis_XYZ = rotateVectorByQuat(localAxis_XYZ, state.cameraOrientation);
    state.cameraOrientation = quatMultiply(quatFromAxisAngle(axis_XYZ, degrees), state.cameraOrientation);
    syncRotationReadout();
    queueOrthogonalRedraw(["3d"]);
    if (persist) publishSession();
  }

  function rotateCamera3dByPixels(dx, dy) {
    rotateCamera3d("up", -dx * CAMERA_DEGREES_PER_PIXEL, false);
    rotateCamera3d("right", -dy * CAMERA_DEGREES_PER_PIXEL, false);
  }

  function resetRotation() {
    if (!state.volume) return;
    const plane = orthogonalStepPlane(state);
    if (!viewIsRotated(state, plane)) return;
    if (state.rotationMode === "frame") state.rotation = [...IDENTITY_ROTATION];
    else state.paneRotations[plane] = [...IDENTITY_ROTATION];
    clampRoundCenter();
    syncPositionInputs();
    onRotationChanged(state.rotationMode === "frame" ? ORTHOGONAL_PLANES : [plane]);
  }

  function resetAllRotations() {
    if (!state.volume || !viewIsRotated(state)) return;
    state.rotation = [...IDENTITY_ROTATION];
    state.paneRotations = Object.fromEntries(ORTHOGONAL_PLANES.map((plane) => [plane, [...IDENTITY_ROTATION]]));
    clampRoundCenter();
    syncPositionInputs();
    onRotationChanged(ORTHOGONAL_PLANES);
  }

  function onRotationChanged(dirtyPlanes) {
    markPlanesDirty([...dirtyPlanes]);
    syncRotationReadout();
    updateScaleBars();
    publishSession();
    scheduleRender();
  }

  function stepSlice(direction, options = {}) {
    const plane = orthogonalStepPlane(state);
    const step = options.held ? state.heldSliceStep : 1;
    const rotated = viewIsRotated(state, plane);
    const axis = planeStepAxis(plane);
    if (rotated) {
      const { normal_XYZ } = rotatedPlaneBasis(plane, planeRotation(state, plane));
      state.center_ZYX[2] += normal_XYZ[0] * direction * step;
      state.center_ZYX[1] += normal_XYZ[1] * direction * step;
      state.center_ZYX[0] += normal_XYZ[2] * direction * step;
      clampCenter();
      resetDirtyPlanes();
    } else {
      const shape_ZYX = state.volume.fullShape_ZYX;
      state.center_ZYX[axis] = clamp(Math.round(state.center_ZYX[axis]) + direction * step, 0, shape_ZYX[axis] - 1);
      markViewDirty();
    }
    syncPositionInputs();
    const retainedPresentation = scheduleRender();
    if (!rotated && state.heldSliceDirection !== 0 && !retainedPresentation) {
      slicePrefetch.prefetchAheadForSliceNavigation(plane, axis, direction);
    }
  }

  function adjustHeldSliceStep(delta) {
    state.heldSliceStep = clamp(state.heldSliceStep + delta, MIN_HELD_SLICE_STEP, MAX_HELD_SLICE_STEP);
    setStatus(`Slice scroll step: ${state.heldSliceStep}`);
    publishSession();
  }

  function panByPixels(plane, dx, dy, panelWidth, panelHeight) {
    if (!state.volume) return;
    const viewWidthVoxels = viewWidth(state);
    const aspect = Math.max(0.001, panelWidth / panelHeight);
    const visibleU = viewWidthVoxels * (aspect >= 1 ? aspect : 1);
    const visibleV = viewWidthVoxels * (aspect >= 1 ? 1 : 1 / aspect);
    movePlaneCenter(plane, (-dx / panelWidth) * visibleU, (dy / panelHeight) * visibleV);
    markViewDirty();
    syncPositionInputs();
    scheduleRender();
  }

  function movePlaneCenter(plane, deltaU, deltaV) {
    const { right_XYZ, up_XYZ } = planeDisplayBasis(state, plane);
    state.center_ZYX[2] += right_XYZ[0] * deltaU + up_XYZ[0] * deltaV;
    state.center_ZYX[1] += right_XYZ[1] * deltaU + up_XYZ[1] * deltaV;
    state.center_ZYX[0] += right_XYZ[2] * deltaU + up_XYZ[2] * deltaV;
    clampCenter();
    if (!viewIsRotated(state, plane)) {
      const normalAxis = planeStepAxis(plane);
      state.center_ZYX[normalAxis] = Math.round(state.center_ZYX[normalAxis]);
    }
  }

  function setZoom(zoom) {
    if (!state.volume) return;
    state.zoom = clamp(zoom, MIN_ZOOM, MAX_ZOOM);
    syncZoomInputs();
    updateScaleBars();
    markViewDirty();
    scheduleRender();
  }

  function setZoomAtLastPointer(zoom) {
    if (!state.volume) return;
    const pointerPanel = state.lastPointerClient
      ? view.orthogonalPanelAt(state.lastPointerClient.x, state.lastPointerClient.y)
      : null;
    if (pointerPanel && pointerPanel[0] !== "3d") {
      setZoomAtClientPoint(zoom, state.lastPointerClient.x, state.lastPointerClient.y);
      return;
    }

    const panel = view.getOrthogonalPanels().find(([plane]) => plane === state.plane);
    if (!panel) {
      setZoom(zoom);
      return;
    }
    const rect = view.getCanvas(state.plane).getBoundingClientRect();
    const [, panelX, panelY, panelWidth, panelHeight] = panel;
    const pixelRatio = view.pixelRatio;
    setZoomAtClientPoint(
      zoom,
      rect.left + (panelX + panelWidth / 2) / pixelRatio,
      rect.bottom - (panelY + panelHeight / 2) / pixelRatio,
    );
  }

  function setZoomAtClientPoint(zoom, clientX, clientY, { interactive = false } = {}) {
    if (!state.volume) return;
    const panel = view.orthogonalPanelAt(clientX, clientY);
    if (!panel || panel[0] === "3d") {
      setZoom(zoom);
      return;
    }

    const [plane] = panel;
    const before = panelZoomAnchor(plane, clientX, clientY, panel, state.zoom);
    const nextZoom = clamp(zoom, MIN_ZOOM, MAX_ZOOM);
    if (nextZoom === state.zoom) return;
    const after = panelZoomAnchor(plane, clientX, clientY, panel, nextZoom);
    movePlaneCenter(plane, before.u - after.u, before.v - after.v);

    state.zoom = nextZoom;
    syncZoomInputs();
    updateScaleBars();
    setActivePlane(plane);
    markViewDirty();
    syncPositionInputs();
    scheduleRender({ interactive });
  }

  function panelZoomAnchor(plane, clientX, clientY, panel, zoom) {
    const [, panelX, panelY, panelWidth, panelHeight] = panel;
    const rect = view.getCanvas(plane).getBoundingClientRect();
    const pixelRatio = view.pixelRatio;
    const canvasX = (clientX - rect.left) * pixelRatio;
    const canvasY = (rect.bottom - clientY) * pixelRatio;
    return panelVoxelOffset(canvasX, canvasY, panelX, panelY, panelWidth, panelHeight, zoom, baseViewWidth(state));
  }

  function setPlane(plane) {
    if (state.volume?.fullShape_ZYX[0] === 1 && plane !== "xy") {
      setStatus(`${plane.toUpperCase()} is unavailable for a two-dimensional source.`);
      return;
    }
    setActivePlane(plane);
    resetDirtyPlanes();
    scheduleRender();
  }

  function setActivePlane(plane, options = {}) {
    if (plane === "3d") return;
    if (state.volume?.fullShape_ZYX[0] === 1 && plane !== "xy") return;
    if (state.activeViewport === "surface" && state.surfaceView.layout.mode === "solo") surfaceController.close();
    const dock = viewerDock(state);
    const changedDock = dock !== null && surfaceController.showInDock(plane);
    if (plane === state.plane && state.activeViewport === plane && !changedDock) return;
    state.plane = plane;
    state.activeViewport = plane;
    const switchedFullscreenPane =
      dock === null &&
      state.fullscreenViewport &&
      state.fullscreenViewport !== "3d" &&
      state.fullscreenViewport !== "surface";
    if (switchedFullscreenPane) {
      state.fullscreenViewport = plane;
    }
    updateViewerShellState();
    if (switchedFullscreenPane || changedDock) {
      view.resize();
      if (changedDock) {
        if (dock.primary === "surface") surfaceController.refreshLayout();
        scheduleRender();
      }
      updateScaleBars();
    }
    if (options.redrawOnly) {
      queueOrthogonalRedraw();
    }
  }

  function setActiveViewport(viewport, options = {}) {
    if (state.volume?.fullShape_ZYX[0] === 1 && viewport !== "xy") {
      setStatus(`${viewport.toUpperCase()} is unavailable for a two-dimensional source.`);
      return false;
    }
    if (
      state.fullscreenViewport === "surface" &&
      viewport !== "surface" &&
      (state.surfaceView.layout.mode === "solo" ||
        (viewport === "3d" && !state.surfaceView.layout.planes.includes("3d")))
    )
      surfaceController.close();
    if (state.activeViewport === viewport) return true;
    state.activeViewport = viewport;
    updateViewerShellState();
    if (options.redrawOnly) {
      queueOrthogonalRedraw([viewport]);
    }
    return true;
  }

  function setViewports(activeViewport, fullscreenViewport = state.fullscreenViewport) {
    state.activeViewport = activeViewport;
    state.fullscreenViewport = fullscreenViewport;
    updateViewerShellState();
    view.resize();
    updateScaleBars();
  }

  function resetView(center_ZYX, zoom) {
    state.center_ZYX = center_ZYX;
    state.zoom = zoom;
  }

  function toggleActiveFullscreen() {
    const dock = viewerDock(state);
    if (dock !== null && dock.layout.mode !== "solo") {
      surfaceController.toggleMaximized(state.activeViewport);
      setViewports(state.activeViewport);
      if (dock.primary === "surface") surfaceController.refreshLayout();
      queueOrthogonalRedraw();
      markViewDirty();
      publishSession();
      scheduleRender();
      return;
    }
    if (state.activeViewport === "surface") {
      surfaceController.close();
      return;
    }
    setViewports(state.activeViewport, state.fullscreenViewport === state.activeViewport ? null : state.activeViewport);
    markViewDirty();
    publishSession();
    scheduleRender();
  }

  function toggleCrosshair() {
    state.crosshairVisible = !state.crosshairVisible;
    updateViewerShellState();
    setStatus(`Crosshair ${state.crosshairVisible ? "shown" : "hidden"}.`);
  }

  function restorePose(pose) {
    state.center_ZYX = [...pose.center_ZYX];
    state.zoom = pose.zoom;
    state.plane = pose.plane;
    state.rotationMode = pose.rotationMode;
    state.rotation = [...pose.rotation];
    state.paneRotations = structuredClone(pose.paneRotations);
    state.cameraOrientation = [...pose.cameraOrientation];
    syncPositionInputs();
    syncZoomInputs();
    setViewports(pose.activeViewport, pose.fullscreenViewport);
    markViewDirty();
    publishSession();
    scheduleRender();
  }

  function clampCenter() {
    const shape_ZYX = state.volume.fullShape_ZYX;
    state.center_ZYX = state.center_ZYX.map((value, axis) => clamp(value, 0, shape_ZYX[axis] - 1));
  }

  function clampRoundCenter() {
    const shape_ZYX = state.volume.fullShape_ZYX;
    state.center_ZYX = state.center_ZYX.map((value, axis) => Math.round(clamp(value, 0, shape_ZYX[axis] - 1)));
  }

  return {
    setPositionAxis,
    setPositionXYZ,
    setPhysicalFocus,
    tiltPlaneByPixels,
    resetPaneOrientation,
    setRotationMode,
    rotateActiveViewport,
    rotateCamera3dByPixels,
    resetRotation,
    resetAllRotations,
    stepSlice,
    adjustHeldSliceStep,
    panByPixels,
    setZoom,
    setZoomAtLastPointer,
    setZoomAtClientPoint,
    setPlane,
    setActivePlane,
    setActiveViewport,
    setViewports,
    resetView,
    toggleActiveFullscreen,
    toggleCrosshair,
    restorePose,
  };
}
