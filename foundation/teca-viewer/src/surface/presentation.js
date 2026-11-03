import { SURFACE_REPLAN_DELAY_MS } from "../config.js";
import { filterMargin } from "../gpu/filter-chain.js";
import { SurfaceChunkStream } from "./chunk-stream.js";
import { surfaceFrameAt } from "./frame.js";
import { SurfaceLayers } from "./layers.js";
import {
  updateSurfaceMaterialDisplay,
  updateSurfaceMaterialLevels,
  updateSurfaceMaterialPose,
  updateSurfaceMaterialReliefMask,
  updateSurfaceMaterialVolumetric,
} from "./material.js";
import { SurfacePlanner } from "./planner.js";
import { visibleSurfaceStrips } from "./planning.js";
import { validateSurfaceReliefMask } from "./relief-mask.js";
import { SurfaceResources } from "./resources.js";
import { surfaceDisplayScaleIndex, surfaceVolumeScaleStack } from "./scale-stack.js";
import { surfaceDemandWindow, surfaceVisibleSpan, validateSurfaceWork } from "./viewport.js";
import {
  surfaceVolumetricCamera,
  surfaceVolumetricDemandKey,
  validateSurfaceVolumetricSettings,
} from "./volumetric.js";

function surfacePose({ center_UV, zoom, normalOffset, normalSign }) {
  return { center_UV: [...center_UV], zoom, normalOffset, normalSign };
}

export class SurfacePresentation {
  constructor({
    rendering,
    pane,
    source,
    overlay,
    volume,
    finestScaleIndex,
    coarsestScaleIndex,
    selectedChannel,
    volumetric,
    displaySettings,
    totalBytes,
    advanceNavigationGeneration,
    onRender,
    onError,
    onAdmissionCommitted,
    onAdmissionRejected,
    onAdmissionPendingChanged,
  }) {
    const initialSpan_UV = surfaceVisibleSpan(source.grid_HW, source.scale_UV, 1, pane.aspect());
    const scales = surfaceVolumeScaleStack(
      volume.scales,
      finestScaleIndex,
      coarsestScaleIndex,
      surfaceDisplayScaleIndex(
        volume.scales,
        initialSpan_UV,
        source.scale_UV,
        pane.panelSize()[0],
        finestScaleIndex,
        coarsestScaleIndex,
      ),
    );
    this.volumetric = validateSurfaceVolumetricSettings(volumetric);
    this.camera = surfaceVolumetricCamera(this.volumetric, source.scale_UV);
    validateSurfaceWork(this.volumetric, pane.panelSize());
    this.pane = pane;
    this.source = source;
    this.onRender = onRender;
    this.onError = onError;
    this.onAdmissionCommitted = onAdmissionCommitted;
    this.onAdmissionRejected = onAdmissionRejected;
    this.onAdmissionPendingChanged = onAdmissionPendingChanged;
    this.admissionPending = false;
    this.disposed = false;
    this.pose = null;
    this.replanTimer = null;
    this.planningPromise = null;
    this.planningAgain = false;
    this.planCount = 0;
    this.poseRevision = 0;
    this.renderedPoseRevision = 0;
    this.dataRevision = 0;
    this.renderedDataRevision = 0;
    this.geometryRevision = 0;
    this.renderedGeometryRevision = 0;
    this.positionRevision = null;
    this.lastVisibleSpan_UV = null;
    this.lastDemandCenter_UV = null;
    this.lastDemandSpan_UV = null;
    this.resources = new SurfaceResources({
      totalBytes,
      rendering,
      pane,
      source,
      positionImage: overlay.positionImage,
      volume,
      scales,
      displaySettings,
      volumetric: this.volumetric,
      geometryConsumer: this,
      selectedChannel,
      onVolumeAtlasChange: (evicted) => this.volumeAtlasChanged(evicted),
    });
    this.planner = new SurfacePlanner({
      volume,
      finestScaleIndex,
      coarsestScaleIndex,
      levels: this.resources.levels,
      source,
      registration: overlay.registration,
      surfaceAtlas: this.resources.surfaceAtlas,
      positionImage: this.resources.positionImage,
    });
    const uploaded = () => {
      this.dataRevision += 1;
      this.requestRender();
    };
    this.stream = new SurfaceChunkStream({
      volume,
      selectedChannel,
      advanceGeneration: advanceNavigationGeneration,
      levels: this.resources.levels,
      renderer: rendering.renderer,
      uploadBudget: rendering.uploadBudget,
      onUploaded: uploaded,
      onError,
    });
    this.layers = new SurfaceLayers({
      rendering,
      resources: this.resources,
      source,
      volume,
      onChange: uploaded,
      onError,
    });
    updateSurfaceMaterialVolumetric(this.resources.material, this.volumetric, this.camera);
    this.geometryChanged();
  }

  updatePose(pose, { replan = true } = {}) {
    this.pose = surfacePose(pose);
    const { grid_HW, scale_UV } = this.source;
    this.lastVisibleSpan_UV = surfaceVisibleSpan(grid_HW, scale_UV, this.pose.zoom, this.pane.aspect());
    updateSurfaceMaterialPose(this.resources.material, this.pose, this.lastVisibleSpan_UV);
    this.poseRevision += 1;
    this.requestRender();
    if (replan) {
      this.setAdmissionPending(true);
      this.scheduleReplan();
    }
  }

  retainGpuBudget(totalBytes) {
    validateSurfaceWork(this.volumetric, this.pane.panelSize());
    return this.resources.retainBudget(totalBytes);
  }

  resize() {
    validateSurfaceWork(this.volumetric, this.pane.panelSize());
  }

  setDisplaySettings(displaySettings) {
    updateSurfaceMaterialDisplay(this.resources.material, displaySettings);
    this.requestRender();
  }

  prepareVolumetric(settings) {
    const candidate = validateSurfaceVolumetricSettings(settings);
    const camera = surfaceVolumetricCamera(candidate, this.source.scale_UV);
    validateSurfaceWork(candidate, this.pane.panelSize());
    return { candidate, camera };
  }

  commitVolumetric({ candidate, camera }) {
    if (JSON.stringify(candidate) === JSON.stringify(this.volumetric)) return;
    const previousDemandKey = surfaceVolumetricDemandKey(this.volumetric, this.camera);
    const nextDemandKey = surfaceVolumetricDemandKey(candidate, camera);
    const maskChanged = JSON.stringify(candidate.mask) !== JSON.stringify(this.volumetric.mask);
    this.volumetric = candidate;
    this.camera = camera;
    if (maskChanged) this.resources.reliefMask.update(candidate.mask);
    updateSurfaceMaterialVolumetric(this.resources.material, candidate, camera);
    this.poseRevision += 1;
    this.requestRender();
    if (nextDemandKey !== previousDemandKey) {
      this.setAdmissionPending(true);
      this.scheduleReplan();
    } else if (!this.admissionPending) {
      this.commitAdmission();
    }
  }

  setReliefMask(mask) {
    const candidate = validateSurfaceReliefMask(mask);
    if (JSON.stringify(candidate) === JSON.stringify(this.volumetric.mask)) return;
    this.resources.reliefMask.update(candidate);
    this.volumetric = { ...this.volumetric, mask: candidate };
    updateSurfaceMaterialReliefMask(this.resources.material, candidate.enabled);
    this.requestRender();
  }

  async compile() {
    await this.pane.renderer.compileAsync(this.resources.mesh, this.pane.camera);
  }

  render() {
    const timing = this.pane.gpuTimer.begin("surface");
    try {
      this.pane.renderScene(this.resources.mesh, this.resources.baseImage);
    } finally {
      this.pane.gpuTimer.end(timing);
    }
    this.renderedPoseRevision = this.poseRevision;
    this.renderedDataRevision = this.dataRevision;
    this.renderedGeometryRevision = this.geometryRevision;
  }

  presentedPrimaryFrame() {
    return (
      this.presented() &&
      this.renderedDataRevision === this.dataRevision &&
      [...this.resources.surfaceAtlas.entries.values()].some((entry) => entry.uploaded) &&
      this.resources.levels.shown.some((level) =>
        level.addresses.some(({ chunk_ZYX }) => level.entry(chunk_ZYX) !== undefined),
      )
    );
  }

  loadingProgress() {
    const strips = visibleSurfaceStrips({
      center_UV: this.lastDemandCenter_UV ?? this.pose.center_UV,
      visibleSpan_UV: this.lastDemandSpan_UV ?? this.lastVisibleSpan_UV,
      grid_HW: this.source.grid_HW,
      stripRows: this.source.stripRows,
    });
    const resident = strips.filter((index) => this.resources.surfaceAtlas.entries.get(index)?.uploaded).length;
    if (resident < strips.length)
      return { phase: `Loading visible geometry ${resident}/${strips.length}`, ready: false };
    if (this.admissionPending || this.planCount === 0) return { phase: "Planning CT", ready: false };
    let loaded = 0;
    let total = 0;
    for (const level of this.resources.levels.shown) {
      total += level.addresses.length;
      loaded += level.addresses.filter(({ chunk_ZYX }) => level.entry(chunk_ZYX) !== undefined).length;
    }
    const ready = loaded === total && this.presentedPrimaryFrame();
    return { phase: ready ? "View ready" : `Loading volume blocks ${loaded}/${total}`, ready };
  }

  surfaceFrameAt(surface_UV, normalSign) {
    return surfaceFrameAt(surface_UV, normalSign, this.source.grid_HW, (pixel_UV) => this.surfacePixel(pixel_UV));
  }

  presented() {
    return this.renderedPoseRevision === this.poseRevision && this.renderedGeometryRevision === this.geometryRevision;
  }

  shownScale() {
    return this.resources.levels.shown[0]?.scale ?? null;
  }

  stats() {
    const levels = this.resources.levels.shown;
    return {
      pendingChunks: this.stream.pending() + this.layers.pending(),
      queuedChunkUploads: this.stream.uploads.length + this.layers.uploads(),
      covered:
        levels.some((level) => level.coverage !== null) &&
        levels.every((level) => level.coverage === null || level.coverage.complete),
      drawn: this.presented() && this.renderedDataRevision === this.dataRevision,
      planned: !this.admissionPending && this.planningPromise === null && this.replanTimer === null,
    };
  }

  allocatedBytes() {
    return this.resources.allocatedBytes();
  }

  dispose() {
    if (this.disposed) return;
    this.setAdmissionPending(false);
    this.disposed = true;
    if (this.replanTimer !== null) clearTimeout(this.replanTimer);
    this.layers.dispose();
    this.stream.dispose();
    this.planner.dispose();
    this.resources.dispose();
  }

  scheduleReplan() {
    if (this.replanTimer !== null) return;
    this.replanTimer = setTimeout(
      async () => {
        this.replanTimer = null;
        if (this.disposed) return;
        try {
          await this.replan();
        } catch (error) {
          if (this.admissionPending) this.rejectAdmission(error);
          this.onError(error);
        }
      },
      this.planCount === 0 ? 0 : SURFACE_REPLAN_DELAY_MS,
    );
  }

  replan() {
    if (this.planningPromise !== null) {
      this.planningAgain = true;
      return this.planningPromise;
    }
    this.planningPromise = this.planAndCommit()
      .catch((error) => {
        if (this.disposed || error.name === "AbortError") return;
        if (this.admissionPending) this.rejectAdmission(error);
        throw error;
      })
      .finally(() => {
        this.planningPromise = null;
        if (this.planningAgain && !this.disposed) {
          this.planningAgain = false;
          this.scheduleReplan();
        }
      });
    return this.planningPromise;
  }

  async planAndCommit() {
    if (this.disposed || this.pose === null) return;
    const pose = this.pose;
    const levels = this.resources.levels;
    const byteLimit = levels.byteLimit;
    const volumetric = this.volumetric;
    const checkCurrent = () => {
      if (this.disposed || levels.byteLimit !== byteLimit || this.volumetric !== volumetric) {
        this.planningAgain = true;
        throw new DOMException("Surface demand changed during planning.", "AbortError");
      }
    };
    const aspect = this.pane.aspect();
    const demand = surfaceDemandWindow(this.source, this.camera, pose.center_UV, pose.zoom, aspect);
    const pixel = this.lastVisibleSpan_UV[0] / this.source.scale_UV[0] / this.pane.panelSize()[0];
    const margin = filterMargin(this.pane.filterReach, pixel);
    demand.span_UV = demand.span_UV.map((span, axis) => span + 2 * margin * this.source.scale_UV[axis]);
    this.lastDemandCenter_UV = demand.center_UV;
    this.lastDemandSpan_UV = demand.span_UV;
    const view = await this.planner.planView({
      pose,
      demand,
      camera: this.camera,
      volumetric,
      aspect,
      panelWidth: this.pane.panelSize()[0],
      firstPlan: this.planCount === 0,
      checkCurrent,
    });
    if (view === null) {
      this.requestRender();
      return;
    }
    if (view.zoom !== undefined) {
      if (this.pose !== pose) {
        this.planningAgain = true;
        return;
      }
      this.updatePose({ ...pose, zoom: view.zoom }, { replan: false });
      await this.planAndCommit();
      return;
    }
    const choice = view.choice;
    const stage = choice.retained ? null : levels.stage(choice.scales, choice.fallbackCapacity);
    if (stage !== null) {
      try {
        levels.commit(stage, choice.plans, this.pane.renderer);
      } finally {
        updateSurfaceMaterialLevels(this.resources.material, levels.shown);
        this.stream.levelsReplaced();
        this.dataRevision += 1;
      }
    }
    this.stream.replace(levels.admitPlans(choice.plans));
    if (this.pose === pose) {
      this.setAdmissionPending(false);
      this.commitAdmission();
    }
    this.planCount += 1;
    this.layers.follow();
    this.requestRender();
  }

  commitAdmission() {
    this.onAdmissionCommitted({ pose: structuredClone(this.pose), volumetric: structuredClone(this.volumetric) });
  }

  rejectAdmission(error) {
    this.setAdmissionPending(false);
    this.onAdmissionRejected(error);
  }

  restoreAdmission(pose, volumetric) {
    const maskChanged = JSON.stringify(this.volumetric.mask) !== JSON.stringify(volumetric.mask);
    this.volumetric = structuredClone(volumetric);
    this.camera = surfaceVolumetricCamera(this.volumetric, this.source.scale_UV);
    if (maskChanged) this.resources.reliefMask.update(this.volumetric.mask);
    updateSurfaceMaterialVolumetric(this.resources.material, this.volumetric, this.camera);
    this.updatePose(pose, { replan: false });
    this.scheduleReplan();
  }

  setAdmissionPending(pending) {
    if (pending === this.admissionPending) return;
    this.admissionPending = pending;
    this.onAdmissionPendingChanged(pending);
  }

  volumeAtlasChanged(evicted) {
    if (this.disposed) return;
    updateSurfaceMaterialLevels(this.resources.material, this.resources.levels.shown);
    if (evicted === null) this.stream.stream();
    this.dataRevision++;
    this.requestRender();
  }

  requestRender() {
    if (this.disposed) return;
    this.pane.sceneChanged();
    this.onRender();
  }

  geometryChanged() {
    const image = this.resources.positionImage;
    if (image.error !== null) {
      this.onError(image.error);
      return;
    }
    if (this.positionRevision !== image.revision) {
      this.positionRevision = image.revision;
      this.scheduleReplan();
    }
    this.geometryRevision++;
    this.requestRender();
  }

  surfacePixel(pixel_UV) {
    const entry = this.resources.surfaceAtlas.getPresented(Math.floor(pixel_UV[1] / this.source.stripRows));
    if (entry === null) return null;
    const offset = ((pixel_UV[1] % this.source.stripRows) * this.source.grid_HW[1] + pixel_UV[0]) * 4;
    if (entry.data_HWRGBA[offset + 3] < 0.5) return null;
    return [0, 1, 2].map((axis) => entry.data_HWRGBA[offset + axis]);
  }
}
