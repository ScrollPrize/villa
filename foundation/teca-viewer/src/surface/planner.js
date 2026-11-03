import { SURFACE_ZOOM_SEARCH_LIMIT } from "../config.js";
import { demandScale } from "../volume/levels.js";
import { createWorkerPool } from "../worker-pool.js";
import { packSurfaceDemandBounds, visibleSurfaceStrips } from "./planning.js";
import { surfaceDisplayScaleIndex, surfaceVolumeScaleStack } from "./scale-stack.js";
import { surfaceDemandWindow, surfaceVisibleSpan } from "./viewport.js";
import { visibleSurfaceBounds } from "./visible-bounds.js";
import { surfaceFallbackCapacity } from "./volume-levels.js";
import { surfaceVolumetricRadius } from "./volumetric.js";

function fallbackError(plan, scale, capacity) {
  return new Error(
    `Surface camera needs at least ${plan.requiredCount ?? plan.requiredCountLowerBound} chunks at coarsest level ` +
      `${scale.path}, but its retained atlas admits ${capacity}. Zoom in or increase the Live GPU budget in Settings. ` +
      "The last complete view is retained.",
  );
}

export class SurfacePlanner {
  constructor({
    volume,
    finestScaleIndex,
    coarsestScaleIndex,
    levels,
    source,
    registration,
    surfaceAtlas,
    positionImage,
  }) {
    this.volume = volume;
    this.finestScaleIndex = finestScaleIndex;
    this.coarsestScaleIndex = coarsestScaleIndex;
    this.levels = levels;
    this.source = source;
    this.registration = registration;
    this.surfaceAtlas = surfaceAtlas;
    this.positionImage = positionImage;
    this.workers = createWorkerPool(new URL("./planning.worker.js", import.meta.url), "Surface planning", {
      workerCount: 1,
      terminateOnAbort: true,
    });
  }

  async planView({ pose, demand, camera, volumetric, aspect, panelWidth, firstPlan, checkCurrent }) {
    const { source, surfaceAtlas } = this;
    const strips = visibleSurfaceStrips({
      center_UV: demand.center_UV,
      visibleSpan_UV: demand.span_UV,
      grid_HW: source.grid_HW,
      stripRows: source.stripRows,
    });
    this.positionImage.prioritize(strips);
    const stripEntries = strips.map((stripIndex) => surfaceAtlas.get(stripIndex)).filter((entry) => entry !== null);
    if (stripEntries.length !== strips.length) return null;

    const boundsIn = (demandArea) =>
      packSurfaceDemandBounds(
        visibleSurfaceBounds({
          stripEntries,
          center_UV: demandArea.center_UV,
          visibleSpan_UV: demandArea.span_UV,
          stripRows: source.stripRows,
          grid_HW: source.grid_HW,
        }),
      );
    const options = {
      surfaceBounds_B7: boundsIn(demand),
      registration: this.registration,
      normalRadius: Math.abs(pose.normalOffset) + Math.max(0.5, surfaceVolumetricRadius(volumetric)) + 1,
    };
    if (firstPlan && pose.zoom < SURFACE_ZOOM_SEARCH_LIMIT) {
      let zoom = pose.zoom;
      while (
        zoom < SURFACE_ZOOM_SEARCH_LIMIT &&
        !(await this.coarsestFits(
          { ...options, surfaceBounds_B7: boundsIn(surfaceDemandWindow(source, camera, pose.center_UV, zoom, aspect)) },
          checkCurrent,
        ))
      )
        zoom = Math.min(SURFACE_ZOOM_SEARCH_LIMIT, zoom * 2);
      if (zoom !== pose.zoom) return { zoom };
    }
    const displayScaleIndex = surfaceDisplayScaleIndex(
      this.volume.scales,
      surfaceVisibleSpan(source.grid_HW, source.scale_UV, pose.zoom, aspect),
      source.scale_UV,
      panelWidth,
      this.finestScaleIndex,
      this.coarsestScaleIndex,
    );
    return { choice: await this.choose(options, displayScaleIndex, firstPlan, checkCurrent) };
  }

  async plan(options, scales, capacities, checkCurrent) {
    checkCurrent();
    const plans = await this.workers.run({ options, scales: scales.map(demandScale), capacities });
    checkCurrent();
    return plans;
  }

  async coarsestFits(options, checkCurrent) {
    const index = this.coarsestScaleIndex ?? this.volume.scales.length - 1;
    const scale = surfaceVolumeScaleStack(this.volume.scales, index, index)[0];
    const plans = await this.plan(options, [scale], [this.levels.maximumCapacity(scale)], checkCurrent);
    return plans[0].complete;
  }

  async choose(options, displayScaleIndex, firstPlan, checkCurrent) {
    const maximumScaleIndex = this.coarsestScaleIndex ?? this.volume.scales.length - 1;
    for (let index = displayScaleIndex; ; index++) {
      const scales = surfaceVolumeScaleStack(this.volume.scales, this.finestScaleIndex, this.coarsestScaleIndex, index);
      const shownCapacities = firstPlan ? null : this.levels.retainedCapacities(scales);
      if (shownCapacities !== null) {
        const plans = await this.plan(options, scales, shownCapacities, checkCurrent);
        if (plans[0].complete && plans.at(-1).complete)
          return { scales, plans, fallbackCapacity: shownCapacities.at(-1), retained: true };
      }
      const shownFallback = this.levels.shown.find((level) => level.scale === scales.at(-1));
      if (!firstPlan && shownFallback !== undefined) {
        const capacities = this.levels.capacities(scales, shownFallback.capacity);
        const plans = await this.plan(options, scales, capacities, checkCurrent);
        if (plans[0].complete && plans.at(-1).complete)
          return { scales, plans, fallbackCapacity: capacities.at(-1), retained: false };
      }
      const maximumCapacity = this.levels.maximumCapacity(scales.at(-1));
      const [fallbackPlan] = await this.plan(options, [scales.at(-1)], [maximumCapacity], checkCurrent);
      if (!fallbackPlan.complete) throw fallbackError(fallbackPlan, scales.at(-1), maximumCapacity);
      let plans = null;
      let fallbackCapacity = null;
      for (const capacity of new Set([
        surfaceFallbackCapacity(fallbackPlan.requiredCount, maximumCapacity),
        fallbackPlan.requiredCount,
      ])) {
        fallbackCapacity = capacity;
        plans = await this.plan(options, scales, this.levels.capacities(scales, capacity), checkCurrent);
        if (plans.every((plan) => plan.complete)) break;
      }
      if (index >= maximumScaleIndex || plans.every((plan) => plan.complete)) {
        if (!plans.at(-1).complete) throw fallbackError(plans.at(-1), scales.at(-1), fallbackCapacity);
        return { scales, plans, fallbackCapacity, retained: false };
      }
    }
  }

  dispose() {
    this.workers.dispose();
  }
}
