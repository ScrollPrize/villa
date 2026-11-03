import * as THREE from "../../vendor/three.module.js";

import { CT_REPLAN_INTERVAL_MS } from "../config.js";
import { chunkKey } from "./addresses.js";
import { PaneSlice, rebuildPageTable } from "./pane-slice.js";
import { atlasCapacity, atlasSlotGroups, atlasSlotIdentity, scaleIdentity } from "./sliced-plane-atlas.js";
import { SlicedPlaneStream } from "./sliced-plane-stream.js";

export class SlicedPlane {
  constructor({ rendering, tileGroups, renderers, renderOrder, atlasBytes, style }) {
    this.rendering = rendering;
    this.tileGroups = tileGroups;
    this.renderers = renderers;
    this.renderOrder = renderOrder;
    this.atlasBytes = atlasBytes;
    this.style = style;
    this.drawn = true;
    this.identity = null;
    this.atlases = new Map();
    this.panes = new Map();
    this.stream = new SlicedPlaneStream({ panes: () => this.panes.values(), uploadBudget: rendering.uploadBudget });
    this.retainedPlans = new Map();
    this.levelSelection = null;
    this.geometry = new THREE.PlaneGeometry(1, 1);
    this.lastPlanAt = 0;
    this.planCount = 0;
  }

  replanDue(quality, planes) {
    return (
      quality !== "interactive" ||
      this.planCount === 0 ||
      performance.now() - this.lastPlanAt >= CT_REPLAN_INTERVAL_MS ||
      planes.some((plane) => (this.panes.get(plane)?.mesh ?? null) === null)
    );
  }

  updateViews({ identity, token, planeViews, isCurrent, onPresent }) {
    if (this.identity !== identity) return false;
    let updated = 0;
    for (const planeView of planeViews) {
      if (this.panes.get(planeView.plane)?.updateView({ ...planeView, token, isCurrent, onPresent }) !== true) continue;
      onPresent(planeView.plane);
      updated += 1;
    }
    return updated === planeViews.length;
  }

  setDisplaySettings(displaySettings) {
    for (const pane of this.panes.values()) pane.setDisplaySettings(displaySettings);
  }

  setStyle(style) {
    this.style = style;
    for (const pane of this.panes.values()) pane.setStyle(style);
  }

  setDrawn(drawn) {
    this.drawn = drawn;
    for (const pane of this.panes.values()) pane.setDrawn(drawn);
  }

  presentedScaleIndex(requestedScaleIndex) {
    const selection = this.levelSelection;
    if (selection === null || selection.requestedScaleIndex !== requestedScaleIndex) return requestedScaleIndex;
    return selection.scaleIndex;
  }

  navigationDemand(plane, requestedScaleIndex) {
    const scaleIndex = this.presentedScaleIndex(requestedScaleIndex);
    const level = this.panes.get(plane)?.levels.find((candidate) => candidate.scaleIndex === scaleIndex);
    return level === undefined ? null : { scaleIndex, scale: level.scale, addresses: level.addresses };
  }

  retainCurrentRequests() {
    const planePlans = [...this.retainedPlans]
      .filter(([plane]) => this.panes.get(plane)?.visible() === true)
      .map(([, planePlan]) => planePlan);
    return planePlans.length > 0 && this.stream.retain(planePlans);
  }

  begin({
    identity,
    volume,
    sourceBounds_ZYX,
    channel,
    token,
    planePlans,
    displayEncoding,
    displaySettings,
    levelSelection,
    isCurrent,
    onPresent,
    onError,
  }) {
    if (this.identity !== identity) {
      this.clear();
      this.identity = identity;
    }
    const incomingPlanes = new Set(planePlans.map(({ plane }) => plane));
    const committedPlanePlans = [
      ...[...this.retainedPlans]
        .filter(([plane]) => !incomingPlanes.has(plane) && this.panes.get(plane)?.visible() === true)
        .map(([, plan]) => plan),
      ...planePlans,
    ];
    const limits = this.rendering.textureLimits;
    const plannedAtlases = new Map();
    const slotGroups = atlasSlotGroups(committedPlanePlans.flatMap((plan) => plan.levels));
    const scaleKeys = new Map();
    for (const plan of committedPlanePlans) {
      for (const { scale } of plan.levels) {
        if (!scaleKeys.has(scale)) scaleKeys.set(scale, scaleIdentity(scale));
      }
    }
    for (const [slotKey, levels] of slotGroups) {
      const scale = levels[0].scale;
      plannedAtlases.set(slotKey, { scale, ...atlasCapacity(scale, slotGroups.size, limits, this.atlasBytes) });
    }
    for (const [plane, pane] of this.panes) {
      const plan = committedPlanePlans.find((plan) => plan.plane === plane);
      const retainedLevels = new Map(
        (plan?.levels ?? pane.levels).map((level) => [
          scaleIdentity(level.scale),
          plannedAtlases.get(atlasSlotIdentity(level.scale))?.capacity,
        ]),
      );
      if (!pane.retainLevels(retainedLevels) && plan === undefined) {
        pane.dispose();
        this.panes.delete(plane);
        this.retainedPlans.delete(plane);
      }
    }
    for (const [atlasKey, atlas] of this.atlases) {
      if (plannedAtlases.get(atlasKey)?.capacity === atlas.capacity) continue;
      atlas.dispose();
      this.atlases.delete(atlasKey);
    }
    for (const [atlasKey, { scale, capacity, slotBytes }] of plannedAtlases) {
      if (this.atlases.has(atlasKey)) continue;
      const pool = this.rendering.brickPool;
      this.atlases.set(
        atlasKey,
        pool.acquire(
          pool.namespace(volume, channel),
          {
            chunks_ZYX: scale.chunks_ZYX,
            dtype: scale.numericType.dtype,
            byteLimit: capacity * slotBytes,
            maxLayers: limits.maxLayers,
            maxTextureSize: limits.maxTextureSize,
          },
          (atlas, entry) => this.atlasChanged(atlas, entry),
        ),
      );
    }
    for (const [atlasKey, levels] of slotGroups) {
      const workingKeys = new Set();
      const coarseFirst = levels.toSorted((a, b) => b.scaleIndex - a.scaleIndex);
      for (const field of ["addresses", "fetchAddresses"]) {
        for (const level of coarseFirst) {
          for (const { chunk_ZYX } of level[field]) workingKeys.add(chunkKey(level.scale, chunk_ZYX));
        }
      }
      this.atlases.get(atlasKey).beginFrame(workingKeys);
    }

    const residentMappingLevels = new Set();
    const fetchPlans = [];
    for (const plan of committedPlanePlans) {
      let pane = this.panes.get(plan.plane);
      if (pane === undefined) {
        pane = new PaneSlice(
          plan.plane,
          this.tileGroups.get(plan.plane),
          this.geometry,
          this.renderers.get(plan.plane),
          limits.maxTextureSize,
          this.renderOrder,
        );
        this.panes.set(plan.plane, pane);
      }
      const levels = plan.levels.map((level) => ({
        ...level,
        scaleKey: scaleKeys.get(level.scale),
        atlas: this.atlases.get(atlasSlotIdentity(level.scale)),
      }));
      fetchPlans.push({ ...plan, levels });
      pane.begin({
        token,
        center_ZYX: plan.center_ZYX,
        basis: plan.basis,
        viewWidthVoxels: plan.viewWidthVoxels,
        viewAspect: plan.viewAspect,
        scaleIndex: plan.scaleIndex,
        levels,
        sourceBounds_ZYX,
        displayEncoding,
        displaySettings,
        style: this.style,
        drawn: this.drawn,
        shown: plan.shown,
        isCurrent,
        onPresent,
      });
      for (const level of levels) {
        const paneLevel = pane.level(level.scaleKey);
        for (const { chunk_ZYX } of [...level.addresses, ...level.fetchAddresses]) {
          const entry = level.atlas.get(chunkKey(level.scale, chunk_ZYX));
          if (entry === null) continue;
          if (!level.atlas.needsUpload(pane.renderer, entry)) {
            if (pane.admit(level.scaleKey, chunk_ZYX, entry)) residentMappingLevels.add(paneLevel);
          } else {
            this.stream.upload(pane, paneLevel, chunk_ZYX, entry);
          }
        }
      }
      onPresent(plan.plane);
    }
    for (const level of residentMappingLevels) rebuildPageTable(level);
    this.stream.prune();
    this.levelSelection = levelSelection;
    this.lastPlanAt = performance.now();
    this.planCount += 1;

    for (const plan of committedPlanePlans) this.retainedPlans.set(plan.plane, plan);
    this.stream.read(volume, channel, fetchPlans, onError);
  }

  refresh(changes) {
    const mapped = new Map();
    for (const { scale, chunk_ZYX, data_DHW, start, end } of changes) {
      const residentKey = chunkKey(scale, chunk_ZYX);
      this.stream.refresh(residentKey, data_DHW);
      const atlas = this.atlases.get(atlasSlotIdentity(scale));
      atlas?.refresh(residentKey, data_DHW, start, end);
      const entry = atlas?.held(residentKey) ?? null;
      if (entry === null) continue;
      for (const pane of this.panes.values()) {
        const level = pane.level(scaleIdentity(scale));
        if (level?.atlas === atlas && this.stream.upload(pane, level, chunk_ZYX, entry)) mapped.set(level, pane);
      }
    }
    for (const level of mapped.keys()) rebuildPageTable(level);
    for (const pane of new Set(mapped.values())) pane.present();
  }

  hide(planes) {
    for (const plane of planes) this.panes.get(plane)?.hide();
    this.stream.prune();
  }

  clear() {
    this.identity = null;
    this.stream.clear();
    this.retainedPlans.clear();
    this.levelSelection = null;
    for (const pane of this.panes.values()) pane.dispose();
    this.panes.clear();
    for (const atlas of this.atlases.values()) atlas.dispose();
    this.atlases.clear();
    this.lastPlanAt = 0;
    this.planCount = 0;
  }

  dispose() {
    this.clear();
    this.geometry.dispose();
  }

  visibleScaleCoverage(plane, renderToken) {
    return this.panes.get(plane)?.visibleScaleCoverage(renderToken) ?? null;
  }

  shownScale(plane) {
    const pane = this.panes.get(plane);
    return pane?.visible() ? (pane.levels[0]?.scale ?? null) : null;
  }

  loadingScaleIndex(plane) {
    return this.panes.get(plane)?.loadingScaleIndex() ?? null;
  }

  allocatedBytes() {
    let total = 0;
    for (const atlas of this.atlases.values()) total += atlas.capacity * atlas.slotBytes;
    for (const pane of this.panes.values()) total += pane.allocatedBytes();
    return total;
  }

  demandStats(planes) {
    let required = 0,
      missing = 0;
    for (const plane of planes) {
      const pane = this.panes.get(plane);
      if (pane === undefined || !pane.visible()) continue;
      for (const level of pane.levels) {
        required += level.addresses.length;
        missing += level.addresses.length - level.availableChunks;
      }
    }
    return { required, missing };
  }

  stats() {
    const held = [...this.atlases.values()].flatMap((atlas) =>
      [...atlas.entries.values()].filter((entry) => atlas.isResident(entry)),
    );
    return {
      ...this.stream.stats(),
      slots: held.filter((entry) => entry.slot !== null).length,
      fills: held.filter((entry) => entry.slot === null).length,
      panes: Object.fromEntries([...this.panes].map(([plane, pane]) => [plane, pane.stats()])),
    };
  }

  atlasChanged(atlas, evicted) {
    for (const pane of this.panes.values()) {
      const mapped = new Set();
      for (const { level, chunk_ZYX, entry } of pane.atlasChanged(atlas, evicted))
        if (this.stream.upload(pane, level, chunk_ZYX, entry)) mapped.add(level);
      for (const level of mapped) rebuildPageTable(level);
      if (mapped.size > 0) pane.present();
    }
  }
}
