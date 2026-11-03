import { SURFACE_NORMAL_OFFSET_LIMIT, SURFACE_NORMAL_STEP_LIMITS } from "../config.js";
import { dockedViewport, validateSurfaceLayout } from "./layout.js";
import { SurfacePresentation } from "./presentation.js";
import { SurfaceProjector } from "./projector.js";
import { constrainSurfacePose, surfaceOffsetAtScreen, surfaceVisibleSpan } from "./viewport.js";
import {
  rotatedSurfaceCamera,
  surfaceFlatCamera,
  surfaceVolumetricCamera,
  validateSurfaceVolumetricSettings,
} from "./volumetric.js";

export class SurfaceController {
  constructor({
    state,
    view,
    overlays,
    panel,
    markers,
    displaySettings,
    refreshViewReadout,
    invalidatePointerInspection,
    setViewports,
    queueOrthogonalRedraw,
    scheduleRender,
    publishSession,
    holdPublication,
    publicationHeld,
    readers,
  }) {
    this.state = state;
    this.view = view;
    this.pane = view.getSurfacePane();
    this.overlays = overlays;
    this.panel = panel;
    this.markers = markers;
    this.displaySettings = displaySettings;
    this.refreshViewReadout = refreshViewReadout;
    this.invalidatePointerInspection = invalidatePointerInspection;
    this.setViewports = setViewports;
    this.queueOrthogonalRedraw = queueOrthogonalRedraw;
    this.scheduleRender = scheduleRender;
    this.publishSession = publishSession;
    this.holdPublication = holdPublication;
    this.publicationHeld = publicationHeld;
    this.readers = readers;
    this.presentation = null;
    this.committedSurfaceView = structuredClone(state.surfaceView);
    this.overlay = null;
    this.projector = null;
    this.generation = 0;
    this.normalLocked = false;
    this.openController = null;
    this.layoutAdmission = null;
  }

  async openMode(mode) {
    if (!["planes", "surface", "volume", "both"].includes(mode)) throw new Error(`Unknown viewer mode ${mode}.`);
    if (mode === "planes") {
      this.close();
      return;
    }
    if (this.active()) {
      this.setVolumetric({ enabled: mode !== "surface", compareFlat: mode === "both" });
      return;
    }
    const loaded = this.overlays.filter(
      (overlay) => overlay.visible && overlay.primarySource === this.state.activeSource,
    );
    const selected = loaded.find((overlay) => overlay.path === this.state.surfaceView.path) ?? loaded[0];
    if (selected === undefined) {
      this.panel.promptForSurface();
      return;
    }
    const previous = structuredClone(this.state.surfaceView.volumetric);
    this.state.surfaceView.volumetric.enabled = mode !== "surface";
    this.state.surfaceView.volumetric.compareFlat = mode === "both";
    try {
      await this.openPath(selected.path);
    } catch (error) {
      if (error.name !== "AbortError") this.state.surfaceView.volumetric = previous;
      throw error;
    }
  }

  volumeChanged() {
    this.panel.clearRestoreStatus();
    if (this.state.surfaceView.primarySource !== this.state.activeSource) this.close({ clearSelection: true });
    else if (this.presentation === null && this.overlay?.primaryVolume !== this.state.volume) this.forgetSurface();
  }

  async openPath(path, layout = null) {
    const overlay = this.overlays.find((candidate) => candidate.path === path) ?? null;
    if (overlay === null) throw new Error(`Load tifxyz overlay ${path} before opening its Surface view.`);
    if (!overlay.visible) throw new Error("Make the tifxyz overlay visible before opening its Surface view.");
    if (overlay.primarySource !== this.state.activeSource || overlay.primaryVolume !== this.state.volume) {
      throw new Error("Activate the surface overlay's primary volume before opening Surface view.");
    }
    this.panel.clearRestoreStatus();
    const generation = ++this.generation;
    if (!this.publicationHeld()) this.committedSurfaceView = structuredClone(this.state.surfaceView);
    this.openController?.abort();
    const controller = new AbortController();
    this.openController = controller;
    let activationStarted = false;
    const superseded = () => controller.signal.aborted || generation !== this.generation;
    const previousSurfaceView = structuredClone(this.committedSurfaceView);
    let reader = null;
    let presentation = null;
    this.panel.status("Opening Surface: preparing mesh and CT…");
    try {
      const projector = this.overlay === overlay ? this.projector : new SurfaceProjector(overlay);
      reader = await this.readers.take(this.state.volume.rootUrl, controller.signal);
      if (superseded() || this.state.volume !== overlay.primaryVolume) {
        throw new DOMException("Surface opening superseded.", "AbortError");
      }
      const surfaceView = structuredClone(this.state.surfaceView);
      surfaceView.layout = structuredClone(layout ?? surfaceView.layout);
      if (surfaceView.path !== path || surfaceView.primarySource !== this.state.activeSource) {
        const projectedCenter = projector.project(this.state.center_ZYX);
        if (projectedCenter === null) throw new Error("Surface view requires at least one valid tifxyz cell.");
        surfaceView.path = path;
        surfaceView.primarySource = this.state.activeSource;
        surfaceView.center_UV = projectedCenter.surface_UV;
        surfaceView.zoom = 1;
        surfaceView.normalOffset = 0;
        surfaceView.normalSign = 1;
      }
      this.releasePresentation();
      this.overlay = overlay;
      this.projector = projector;
      Object.assign(this.state.surfaceView, surfaceView);
      activationStarted = true;
      const { enabled, compareFlat } = this.state.surfaceView.volumetric;
      this.pane.setCompareFlat(enabled && compareFlat);
      this.activatePane();
      await this.view.nextFrame();
      if (superseded()) throw new DOMException("Surface opening superseded.", "AbortError");
      let navigationGeneration = 0;
      const volume = reader;
      presentation = new SurfacePresentation({
        rendering: this.view.rendering,
        pane: this.pane,
        totalBytes: this.gpuBudget(),
        source: overlay.surfaceSource,
        overlay,
        volume,
        finestScaleIndex: this.state.manualScaleIndex,
        coarsestScaleIndex: this.state.manualCoarsestScaleIndex,
        selectedChannel: this.state.selectedChannel,
        volumetric: this.state.surfaceView.volumetric,
        displaySettings: this.displaySettings(),
        advanceNavigationGeneration: () => {
          navigationGeneration = Math.max(navigationGeneration, volume.navigationGeneration) + 1;
          return navigationGeneration;
        },
        onRender: () => this.queueOrthogonalRedraw(["surface"]),
        onError: (error) => this.panel.status(error.message, true),
        onAdmissionCommitted: (admission) => this.commitPresentationAdmission(presentation, admission),
        onAdmissionRejected: (error) => this.rejectPresentationAdmission(presentation, error),
        onAdmissionPendingChanged: (pending) => {
          if (this.presentation === presentation || this.presentation === null) this.holdPublication(pending);
        },
      });
      await presentation.compile();
      if (superseded()) throw new DOMException("Surface opening superseded.", "AbortError");
      this.readers.keep(reader, this.state.volume.sourceByteLimit);
      reader = null;
      this.presentation = presentation;
      this.markers.setSurface({ presentation, projector, overlay, path });
      this.constrainPose();
      this.pane.setPresentation(presentation, path);
      this.updatePresentationPose();
      this.panel.sync();
      this.panel.status("Surface opened. Geometry and CT loading progress appear in the viewport.");
      if (this.state.surfaceView.layout.mode !== "solo") this.scheduleRender();
    } catch (error) {
      if (presentation !== this.presentation) presentation?.dispose();
      if (superseded()) throw new DOMException("Surface opening superseded.", "AbortError");
      if (error.name !== "AbortError") {
        if (activationStarted) {
          this.releasePresentation();
          this.readers.release();
          Object.assign(this.state.surfaceView, previousSurfaceView);
          this.restoreOrthogonalPanes();
          this.scheduleRender();
        }
        this.panel.status(error.message, true);
      }
      throw error;
    } finally {
      if (reader !== null) this.readers.discard(reader);
      if (this.openController === controller) this.openController = null;
    }
  }

  close({ clearSelection = false } = {}) {
    this.panel.clearRestoreStatus();
    ++this.generation;
    this.openController?.abort();
    this.openController = null;
    const restoreCommitted = this.publicationHeld();
    this.releasePresentation();
    this.readers.release();
    if (restoreCommitted) this.state.surfaceView = structuredClone(this.committedSurfaceView);
    this.holdPublication(false);
    this.restoreOrthogonalPanes();
    this.queueOrthogonalRedraw();
    if (clearSelection) {
      this.forgetSurface();
      this.state.surfaceView.path = null;
      this.state.surfaceView.primarySource = null;
    }
    this.panel.sync();
    this.publishSession();
    this.scheduleRender();
  }

  active() {
    return this.state.fullscreenViewport === "surface" && this.presentation !== null;
  }

  activatePane() {
    const pane = this.pane;
    const alreadySurface = this.state.fullscreenViewport === "surface";
    const snapshots =
      alreadySurface && !pane.element.hidden ? [pane.snapshot()] : this.view.captureOrthogonalSnapshots();
    pane.showCurtain(snapshots, "Opening Surface view…");
    this.view.setAuxiliaryPane("surface");
    const layout = this.state.surfaceView.layout;
    this.setViewports(
      layout.maximized ?? (alreadySurface ? dockedViewport(layout, this.state.activeViewport) : "surface"),
      "surface",
    );
  }

  restoreOrthogonalPanes() {
    this.pane.hideCurtain();
    this.view.setAuxiliaryPane("3d");
    this.setViewports(
      this.state.activeViewport === "surface" ? this.state.plane : this.state.activeViewport,
      this.state.fullscreenViewport === "surface" ? null : this.state.fullscreenViewport,
    );
  }

  async setLayout(layout) {
    validateSurfaceLayout(layout);
    const previous = this.state.surfaceView.layout;
    const rebudget = (previous.mode === "solo") !== (layout.mode === "solo");
    if (rebudget && this.active()) {
      const activeViewport = this.state.activeViewport;
      const restorePrevious = () => {
        this.state.surfaceView.layout = previous;
        this.setViewports(activeViewport);
      };
      this.state.surfaceView.layout = structuredClone(layout);
      let retained;
      try {
        this.setViewports(dockedViewport(layout, activeViewport));
        retained = this.openController === null && this.presentation.retainGpuBudget(this.gpuBudget());
      } catch (error) {
        restorePrevious();
        throw error;
      }
      if (!retained) {
        restorePrevious();
        await this.openPath(this.state.surfaceView.path, layout);
        this.publishSession();
        return;
      }
      this.layoutAdmission ??= { layout: structuredClone(this.committedSurfaceView.layout), activeViewport };
      this.presentation.setAdmissionPending(true);
      this.refreshLayout();
    } else {
      this.state.surfaceView.layout = structuredClone(layout);
    }
    if (!this.publicationHeld()) this.committedSurfaceView.layout = structuredClone(this.state.surfaceView.layout);
    this.setViewports(dockedViewport(this.state.surfaceView.layout, this.state.activeViewport));
    this.panel.sync();
    this.publishSession();
  }

  showInDock(plane) {
    const layout = this.state.surfaceView.layout;
    let changed = false;
    if (plane !== "surface" && !layout.planes.includes(plane)) {
      layout.planes.push(plane);
      changed = true;
    }
    if (layout.maximized !== null && layout.maximized !== plane) {
      layout.maximized = plane;
      changed = true;
    }
    return changed;
  }

  toggleMaximized(viewport) {
    const layout = this.state.surfaceView.layout;
    layout.maximized = layout.maximized === viewport ? null : viewport;
  }

  setDockSizes(split, weights) {
    this.state.surfaceView.layout.split = split;
    this.state.surfaceView.layout.weights = weights;
  }

  restoreView(surfaceView) {
    Object.assign(this.state.surfaceView, structuredClone(surfaceView));
  }

  gpuBudget() {
    const available = this.view.rendering.availableBytes();
    if (available <= 0) throw new Error("The slice renderer leaves no GPU budget for the Surface pane.");
    return this.state.surfaceView.layout.mode === "solo" ? available : available - Math.floor(available / 3);
  }

  stepNormal(direction) {
    if (this.state.activeViewport !== "surface" || !this.active()) return false;
    this.setNormalOffset(this.state.surfaceView.normalOffset + direction * this.state.surfaceView.normalStep);
    return true;
  }

  setNormalLocked(locked) {
    this.normalLocked = locked;
    this.panel.syncNormalOffset();
  }

  setNormalOffset(offset) {
    if (!Number.isFinite(offset)) throw new Error(`Surface normal offset must be finite, not ${offset}.`);
    if (this.normalLocked) return;
    const admitted = Math.min(SURFACE_NORMAL_OFFSET_LIMIT, Math.max(-SURFACE_NORMAL_OFFSET_LIMIT, offset));
    if (admitted === this.state.surfaceView.normalOffset) return;
    this.state.surfaceView.normalOffset = admitted;
    this.updatePose();
    this.panel.syncNormalOffset();
  }

  setNormalStep(step) {
    const [lowest, highest] = SURFACE_NORMAL_STEP_LIMITS;
    this.state.surfaceView.normalStep = Math.min(highest, Math.max(lowest, step));
    this.panel.sync();
    this.publishSession();
  }

  flipNormal() {
    if (this.normalLocked) return;
    this.state.surfaceView.normalSign *= -1;
    this.updatePose({ syncControls: true });
  }

  resetCamera() {
    this.setVolumetric({ azimuth: 0, tilt: 0, perspective: 0 });
  }

  rotateCameraByPixels(dx, dy) {
    this.setVolumetric(rotatedSurfaceCamera(this.state.surfaceView.volumetric, dx, dy));
  }

  async focusAtPoint(point_ZYX) {
    if (!this.active()) return;
    const generation = this.generation;
    const framing = () =>
      JSON.stringify([
        this.normalLocked,
        this.state.surfaceView.center_UV,
        this.state.surfaceView.normalOffset,
        this.state.surfaceView.normalSign,
        this.state.surfaceView.zoom,
      ]);
    const before = framing();
    const projection = await this.markers.resolveFocus(point_ZYX);
    if (
      !this.active() ||
      generation !== this.generation ||
      framing() !== before ||
      point_ZYX.some((value, axis) => value !== this.state.center_ZYX[axis])
    )
      return;
    if (projection === null || Math.abs(projection.signedDepth) > SURFACE_NORMAL_OFFSET_LIMIT) {
      this.panel.status(
        "CT focus has no unique valid mapping within the slab depth range; Surface framing is retained.",
        true,
      );
      return;
    }
    this.state.surfaceView.center_UV = projection.surface_UV;
    if (!this.normalLocked)
      this.state.surfaceView.normalOffset = projection.signedDepth * this.state.surfaceView.normalSign;
    this.constrainPose();
    this.updatePose({ syncControls: true });
  }

  resetFraming() {
    if (!this.active()) return;
    const projection = this.projector.project(this.state.center_ZYX);
    if (projection === null) {
      this.panel.status("The physical focus has no valid Surface mapping.", true);
      return;
    }
    this.state.surfaceView.center_UV = projection.surface_UV;
    this.state.surfaceView.zoom = 1;
    this.constrainPose();
    this.updatePose({ syncControls: true });
  }

  setVolumetric(patch) {
    const candidate = { ...this.state.surfaceView.volumetric, ...patch };
    const pane = this.pane;
    const previousComparison = pane.compareFlat;
    try {
      pane.setCompareFlat(candidate.enabled && candidate.compareFlat);
      validateSurfaceVolumetricSettings(candidate);
      if (this.presentation !== null)
        this.presentation.commitVolumetric(this.presentation.prepareVolumetric(candidate));
      Object.assign(this.state.surfaceView.volumetric, candidate);
      this.refreshViewReadout();
      this.invalidatePointerInspection();
      if (pane.compareFlat !== previousComparison) this.refreshLayout();
      this.panel.syncOverlay();
      this.panel.scheduleSync();
      this.panel.clearError();
    } catch (error) {
      pane.setCompareFlat(previousComparison);
      this.panel.sync();
      this.panel.status(error.message, true);
    }
  }

  setReliefMask(mask, { syncControls = true, announce = true } = {}) {
    try {
      const candidate = validateSurfaceVolumetricSettings({ ...this.state.surfaceView.volumetric, mask });
      this.presentation?.setReliefMask(candidate.mask);
      this.state.surfaceView.volumetric.mask = candidate.mask;
      if (this.presentation !== null) this.committedSurfaceView.volumetric.mask = structuredClone(candidate.mask);
      this.panel.syncOverlay();
      if (syncControls) this.panel.sync();
      this.publishSession();
      if (announce) this.panel.status("Surface view ready.");
      return true;
    } catch (error) {
      this.panel.syncOverlay();
      if (syncControls) this.panel.sync();
      this.panel.status(error.message, true);
      return false;
    }
  }

  panByPixels(dx, dy, panelWidth, panelHeight, clientX = Infinity) {
    if (!this.active()) return false;
    const { grid_HW, scale_UV } = this.overlay.surfaceSource;
    const span = surfaceVisibleSpan(grid_HW, scale_UV, this.state.surfaceView.zoom, panelWidth / panelHeight);
    const delta_UV = surfaceOffsetAtScreen(
      [-dx / panelWidth, -dy / panelHeight],
      span,
      this.pane.flatAt(clientX) ? surfaceFlatCamera(this.camera()) : this.camera(),
    );
    this.state.surfaceView.center_UV[0] += delta_UV[0];
    this.state.surfaceView.center_UV[1] += delta_UV[1];
    this.constrainPose();
    this.updatePose();
    return true;
  }

  zoomAtClient(zoom, clientX, clientY) {
    if (!this.active() || !Number.isFinite(zoom) || zoom <= 0) return false;
    const pane = this.pane;
    const before = this.surfaceCoordinateAtClient(clientX, clientY);
    this.state.surfaceView.zoom = zoom;
    const after = this.surfaceCoordinateAtClient(clientX, clientY);
    if (before !== null && after !== null) {
      this.state.surfaceView.center_UV[0] += before[0] - after[0];
      this.state.surfaceView.center_UV[1] += before[1] - after[1];
    }
    this.constrainPose();
    if (pane.interactionRect(clientX).width > 0) this.updatePose();
    return true;
  }

  surfaceCoordinateAtClient(clientX, clientY) {
    if (!this.active()) return null;
    const pane = this.pane;
    const rect = pane.interactionRect(clientX);
    if (pane.compareFlat && (clientX < rect.left || clientX >= rect.right)) return null;
    if (rect.width <= 0 || rect.height <= 0) return null;
    const { grid_HW, scale_UV } = this.overlay.surfaceSource;
    const span = surfaceVisibleSpan(grid_HW, scale_UV, this.state.surfaceView.zoom, rect.width / rect.height);
    const offset_UV = surfaceOffsetAtScreen(
      [(clientX - rect.left) / rect.width - 0.5, (clientY - rect.top) / rect.height - 0.5],
      span,
      pane.flatAt(clientX) ? surfaceFlatCamera(this.camera()) : this.camera(),
    );
    return offset_UV.map((value, axis) => value + this.state.surfaceView.center_UV[axis]);
  }

  pointAtClient(clientX, clientY) {
    return this.presentation?.presented() ? (this.toolAt(clientX, clientY)?.point_ZYX ?? null) : null;
  }

  toolAt(clientX, clientY) {
    const surface_UV = this.surfaceCoordinateAtClient(clientX, clientY);
    if (surface_UV === null) return null;
    const { grid_HW, scale_UV } = this.overlay.surfaceSource;
    const rect = this.pane.interactionRect(clientX);
    const span = surfaceVisibleSpan(grid_HW, scale_UV, this.state.surfaceView.zoom, rect.width / rect.height);
    const cell_HW = [surface_UV[1], surface_UV[0]];
    return {
      pane: "surface",
      cell_HW,
      edited: this.overlay.edited,
      pixel: span[0] / scale_UV[0] / rect.width,
      ...(this.markers.points()(cell_HW) ?? { point_ZYX: null, normal_ZYX: null }),
    };
  }

  stats() {
    return this.presentation?.stats() ?? null;
  }

  overlayView() {
    if (this.presentation === null) return null;
    return {
      grid_HW: this.overlay.surfaceSource.grid_HW,
      visibleSpan_UV: this.presentation.lastVisibleSpan_UV,
      camera: this.camera(),
    };
  }

  allocatedBytes() {
    return this.presentation?.allocatedBytes() ?? 0;
  }

  refreshPresentation() {
    if (!this.active()) return;
    void this.openPath(this.overlay.path).catch((error) => this.handleOpenError(error));
  }

  refreshLayout() {
    if (!this.active()) return;
    try {
      this.presentation.resize();
      this.constrainPose();
      this.updatePresentationPose();
      this.panel.sync();
    } catch (error) {
      if (this.layoutAdmission !== null) {
        this.presentation.rejectAdmission(error);
        this.panel.status(error.message, true);
      } else {
        this.close();
        this.panel.status(`Surface view closed: ${error.message}`, true);
      }
    }
  }

  updatePose({ syncControls = false } = {}) {
    this.updatePresentationPose();
    if (syncControls) this.panel.scheduleSync();
  }

  constrainPose() {
    const { grid_HW } = this.overlay.surfaceSource;
    Object.assign(this.state.surfaceView, constrainSurfacePose(this.state.surfaceView, grid_HW));
  }

  forgetSurface() {
    this.overlay = null;
    this.projector = null;
    this.readers.drop();
  }

  releasePresentation() {
    this.layoutAdmission = null;
    this.markers.setSurface(null);
    this.presentation?.dispose();
    this.presentation = null;
    this.panel.syncOverlay();
    this.pane.setPresentation(null, null);
  }

  commitPresentationAdmission(presentation, admission) {
    if (this.presentation !== presentation) return;
    this.layoutAdmission = null;
    const fittedZoom = this.state.surfaceView.zoom !== admission.pose.zoom;
    Object.assign(this.state.surfaceView, structuredClone(admission.pose));
    Object.assign(this.state.surfaceView.volumetric, structuredClone(admission.volumetric));
    this.pane.setCompareFlat(admission.volumetric.enabled && admission.volumetric.compareFlat);
    this.committedSurfaceView = structuredClone(this.state.surfaceView);
    this.refreshViewReadout();
    if (fittedZoom) {
      this.invalidatePointerInspection();
      this.panel.syncOverlay();
      this.panel.sync();
    }
    this.publishSession();
  }

  rejectPresentationAdmission(presentation, error) {
    if (this.presentation !== presentation) return;
    this.refreshViewReadout();
    this.invalidatePointerInspection();
    if (presentation.planCount === 0) {
      this.state.surfaceView = structuredClone(this.committedSurfaceView);
      this.close({ clearSelection: true });
      this.panel.status(error.message, true);
      return;
    }
    const { center_UV, zoom, normalOffset, normalSign, volumetric } = structuredClone(this.committedSurfaceView);
    Object.assign(this.state.surfaceView, { center_UV, zoom, normalOffset, normalSign });
    Object.assign(this.state.surfaceView.volumetric, volumetric);
    presentation.restoreAdmission(this.state.surfaceView, this.state.surfaceView.volumetric);
    this.pane.setCompareFlat(volumetric.enabled && volumetric.compareFlat);
    if (this.layoutAdmission !== null) {
      const previous = this.layoutAdmission;
      this.layoutAdmission = null;
      this.state.surfaceView.layout = previous.layout;
      this.setViewports(dockedViewport(previous.layout, previous.activeViewport));
      presentation.resize();
      if (!presentation.retainGpuBudget(this.gpuBudget())) {
        this.close();
        this.panel.status(error.message, true);
        return;
      }
      presentation.updatePose(this.state.surfaceView, { replan: false });
      this.scheduleRender();
    }
    this.panel.syncOverlay();
    this.panel.sync();
    this.publishSession();
  }

  updatePresentationPose() {
    if (this.presentation === null) return;
    this.refreshViewReadout();
    this.invalidatePointerInspection();
    this.presentation.updatePose(this.state.surfaceView);
    this.panel.syncOverlay();
  }

  camera() {
    return surfaceVolumetricCamera(this.state.surfaceView.volumetric, this.overlay.surfaceSource.scale_UV);
  }

  handleOpenError(error) {
    if (error.name === "AbortError") return;
    if (this.state.activeViewport === "surface" || this.presentation !== null) this.close();
    this.panel.status(error.message, true);
  }
}
