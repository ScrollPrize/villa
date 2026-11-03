import * as THREE from "../../vendor/three.module.js";
import { LOADED_COVERAGE } from "../config.js";
import { chunkKey, chunkKeyParts } from "./addresses.js";
import { PageTable } from "./page-table.js";
import { createSlicedPlaneMaterial, updateSlicedPlaneMaterialLevels } from "./sliced-plane-material.js";

const SHOWN_LEVEL_COVERAGE = 0.9;

export class PaneSlice {
  constructor(plane, group, geometry, renderer, maxTextureSize, renderOrder) {
    this.plane = plane;
    this.group = group;
    this.geometry = geometry;
    this.renderer = renderer;
    this.maxTextureSize = maxTextureSize;
    this.renderOrder = renderOrder;
    this.mesh = null;
    this.active = false;
    this.levels = [];
    this.levelCache = new Map();
    this.token = null;
    this.plannedToken = null;
    this.scaleIndex = null;
    this.isCurrent = null;
    this.onPresent = null;
    this.center_ZYX = null;
    this.shown = false;
  }

  begin({
    token,
    center_ZYX,
    basis,
    viewWidthVoxels,
    viewAspect,
    scaleIndex,
    levels,
    sourceBounds_ZYX,
    displayEncoding,
    displaySettings,
    style,
    drawn,
    shown,
    isCurrent,
    onPresent,
  }) {
    const activeLevels = [];
    const activeScaleKeys = new Set(levels.map((level) => level.scaleKey));
    for (const [scaleKey, level] of this.levelCache) {
      if (activeScaleKeys.has(scaleKey)) continue;
      level.pageTable.dispose();
      this.levelCache.delete(scaleKey);
    }
    for (const plannedLevel of levels) {
      let level = this.levelCache.get(plannedLevel.scaleKey);
      if (level === undefined) {
        level = {
          ...plannedLevel,
          addressKeys: new Set(),
          entries: new Map(),
          pageTable: new PageTable(plannedLevel.atlas.capacity, this.maxTextureSize),
        };
        this.levelCache.set(plannedLevel.scaleKey, level);
      }
      const atlasChanged = level.atlas !== plannedLevel.atlas;
      if (atlasChanged) {
        level.atlas = plannedLevel.atlas;
        level.entries.clear();
      }
      level.scale = plannedLevel.scale;
      level.scaleIndex = plannedLevel.scaleIndex;
      level.addresses = plannedLevel.addresses;
      level.plannedAddressKeys = new Set(
        plannedLevel.addresses.map(({ chunk_ZYX }) => chunkKey(level.scale, chunk_ZYX)),
      );
      level.addressKeys = new Set(level.plannedAddressKeys);
      for (const { chunk_ZYX } of plannedLevel.fetchAddresses) level.addressKeys.add(chunkKey(level.scale, chunk_ZYX));
      let pageTableDirty = atlasChanged;
      level.availableChunks = 0;
      for (const [key, entry] of level.entries) {
        if (!level.atlas.isResident(entry)) {
          level.entries.delete(key);
          pageTableDirty = true;
        } else if (level.plannedAddressKeys.has(key)) {
          level.availableChunks += 1;
        }
      }
      if (pageTableDirty) rebuildPageTable(level);
      activeLevels.push(level);
    }
    this.levels = activeLevels;
    if (this.mesh === null) {
      this.mesh = new THREE.Mesh(
        this.geometry,
        createSlicedPlaneMaterial({ levels: this.levels, sourceBounds_ZYX, displayEncoding, displaySettings, style }),
      );
      this.mesh.renderOrder = this.renderOrder;
      this.mesh.frustumCulled = false;
      this.group.add(this.mesh);
    } else {
      updateSlicedPlaneMaterialLevels(this.mesh.material, this.levels);
    }
    this.scaleIndex = scaleIndex;
    this.updateView({ token, center_ZYX, basis, viewWidthVoxels, viewAspect, shown, isCurrent, onPresent });
    this.plannedToken = token;
    this.setDisplaySettings(displaySettings);
    this.setDrawn(drawn);
  }

  setDisplaySettings({ black, white, gamma }) {
    if (this.mesh === null) return;
    const uniforms = this.mesh.material.uniforms;
    uniforms.black.value = black;
    uniforms.white.value = white;
    uniforms.gamma.value = gamma;
  }

  setDrawn(drawn) {
    if (this.mesh === null) return;
    this.mesh.material.uniforms.drawn.value = drawn ? 1 : 0;
  }

  setStyle({ color_RGB, opacity, range }) {
    if (this.mesh === null) return;
    const uniforms = this.mesh.material.uniforms;
    uniforms.layerLabels.value = color_RGB === null ? 1 : 0;
    if (color_RGB !== null) {
      uniforms.layerColor_RGB.value.set(...color_RGB);
      uniforms.layerRange.value.set(...range);
    }
    uniforms.layerOpacity.value = opacity;
  }

  updateView({ token, center_ZYX, basis, viewWidthVoxels, viewAspect, shown, isCurrent, onPresent }) {
    if (this.mesh === null) return false;
    this.plannedToken = null;
    this.token = token;
    this.shown = shown;
    this.isCurrent = isCurrent;
    this.onPresent = onPresent;
    const aspect = Math.max(0.001, viewAspect);
    const visibleU = viewWidthVoxels * (aspect >= 1 ? aspect : 1);
    const visibleV = viewWidthVoxels * (aspect >= 1 ? 1 : 1 / aspect);
    this.mesh.scale.set(aspect >= 1 ? aspect * 2 : 2, aspect >= 1 ? 2 : 2 / aspect, 1);
    const uniforms = this.mesh.material.uniforms;
    uniforms.center_ZYX.value.set(...center_ZYX);
    uniforms.right_ZYX.value.set(basis.right_XYZ[2], basis.right_XYZ[1], basis.right_XYZ[0]);
    uniforms.up_ZYX.value.set(basis.up_XYZ[2], basis.up_XYZ[1], basis.up_XYZ[0]);
    uniforms.visibleSpan_UV.value.set(visibleU, visibleV);
    this.center_ZYX = [...center_ZYX];
    this.active = true;
    this.mesh.visible = true;
    return true;
  }

  level(scaleKey) {
    return this.levels.find((level) => level.scaleKey === scaleKey) ?? null;
  }

  retainLevels(capacities) {
    for (const [scaleKey, level] of this.levelCache) {
      if (capacities.get(scaleKey) === level.atlas.capacity) continue;
      level.pageTable.dispose();
      this.levelCache.delete(scaleKey);
    }
    this.levels = this.levels.filter((level) => this.levelCache.get(level.scaleKey) === level);
    return this.levelCache.size > 0;
  }

  accepts(scaleKey, chunk_ZYX) {
    const level = this.level(scaleKey);
    return this.visible() && this.isCurrent?.() === true && level?.addressKeys.has(chunkKey(level.scale, chunk_ZYX));
  }

  hasAdmission(scaleKey, chunk_ZYX, entry) {
    const level = this.level(scaleKey);
    return level !== null && level.entries.get(chunkKey(level.scale, chunk_ZYX)) === entry;
  }

  admit(scaleKey, chunk_ZYX, entry) {
    const level = this.level(scaleKey);
    if (level === null || !this.accepts(scaleKey, chunk_ZYX)) return false;
    const key = chunkKey(level.scale, chunk_ZYX);
    if (level.entries.get(key) === entry) return false;
    if (!level.entries.has(key) && level.plannedAddressKeys.has(key)) level.availableChunks += 1;
    level.entries.set(key, entry);
    return true;
  }

  atlasChanged(atlas, evicted) {
    const reuploads = [];
    for (const level of this.levelCache.values()) {
      if (level.atlas !== atlas) continue;
      let changed = evicted === null;
      if (evicted === null) level.entries.clear();
      else {
        for (const entry of evicted) {
          if (level.entries.get(entry.key) === entry) {
            level.entries.delete(entry.key);
            changed = true;
          }
        }
      }
      if (!changed) continue;
      rebuildPageTable(level);
      if (evicted !== null) continue;
      for (const { chunk_ZYX } of level.addresses) {
        const entry = atlas.get(chunkKey(level.scale, chunk_ZYX));
        if (entry !== null) reuploads.push({ level, chunk_ZYX, entry });
      }
    }
    if (evicted === null && this.mesh !== null) updateSlicedPlaneMaterialLevels(this.mesh.material, this.levels);
    this.present();
    return reuploads;
  }

  dropEvicted(atlases) {
    let changed = false;
    for (const level of this.levelCache.values()) {
      if (!atlases.has(level.atlas) || [...level.entries.values()].every((entry) => level.atlas.isResident(entry)))
        continue;
      rebuildPageTable(level);
      changed = true;
    }
    return changed;
  }

  present() {
    if (this.isCurrent?.() === true) this.onPresent(this.plane);
  }

  hide() {
    this.active = false;
    if (this.mesh !== null) this.mesh.visible = false;
  }

  dispose() {
    if (this.mesh !== null) {
      this.mesh.removeFromParent();
      this.mesh.material.dispose();
      this.mesh = null;
    }
    for (const level of this.levelCache.values()) level.pageTable.dispose();
    this.levelCache.clear();
    this.levels = [];
  }

  allocatedBytes() {
    let total = 0;
    for (const level of this.levelCache.values()) total += level.pageTable.allocatedBytes();
    return total;
  }

  visibleScaleCoverage(renderToken) {
    if (!this.visible() || this.token !== renderToken) return null;
    const coverageByScale = {};
    let visibleScaleIndex = null;
    for (const level of this.levels) {
      const scaleIndex = level.scaleIndex;
      const coverage = level.addresses.length === 0 ? 0 : level.availableChunks / level.addresses.length;
      coverageByScale[scaleIndex] = coverage;
      if (visibleScaleIndex === null && coverage >= SHOWN_LEVEL_COVERAGE) visibleScaleIndex = scaleIndex;
    }
    return { scaleIndex: visibleScaleIndex, coverageByScale };
  }

  loadingScaleIndex() {
    if (!this.visible()) return null;
    const coverage = this.visibleScaleCoverage(this.token).coverageByScale[this.scaleIndex] ?? 0;
    return coverage < LOADED_COVERAGE ? this.scaleIndex : null;
  }

  stats() {
    return {
      scaleIndex: this.scaleIndex,
      token: this.token,
      plannedToken: this.plannedToken,
      visible: this.visible(),
      levels: this.levels.map((level) => ({
        scaleIndex: level.scaleIndex,
        available: level.availableChunks,
        planned: level.addresses.length,
        fills: [...level.plannedAddressKeys].filter((key) => level.entries.get(key)?.slot === null).length,
      })),
    };
  }

  visible() {
    return this.active;
  }
}

export function rebuildPageTable(level) {
  level.availableChunks = 0;
  for (const [key, entry] of level.entries) {
    if (!level.atlas.isResident(entry)) level.entries.delete(key);
    else if (level.plannedAddressKeys.has(key)) level.availableChunks += 1;
  }
  level.pageTable.write(
    [...level.entries].map(([key, entry]) => ({
      chunk_ZYX: chunkKeyParts(key).chunk_ZYX,
      slot: entry.slot,
    })),
  );
}
