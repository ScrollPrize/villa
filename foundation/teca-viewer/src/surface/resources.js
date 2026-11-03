import * as THREE from "../../vendor/three.module.js";
import { SURFACE_PAGE_TABLE_RESERVE_BYTES } from "../config.js";

import { volumeBounds_ZYX } from "../gpu/addresses.js";
import { pageTableBytes } from "../gpu/page-table.js";
import { SurfaceBaseImage } from "./base-image.js";
import { createSurfaceMaterial } from "./material.js";
import { SurfaceReliefMaskTexture } from "./relief-mask-texture.js";
import { SURFACE_ATLAS_SLOT_LIMIT, SurfaceVolumeLevels } from "./volume-levels.js";

function surfaceVolumeByteLimit(totalBytes) {
  const limit = totalBytes - SURFACE_PAGE_TABLE_RESERVE_BYTES;
  if (!Number.isSafeInteger(limit) || limit < 1) {
    throw new Error(
      "Surface fixed GPU budget leaves no space for retained volume chunks. Increase the Live GPU budget in Settings, then reopen the surface.",
    );
  }
  return limit;
}

export class SurfaceResources {
  constructor({
    rendering,
    pane,
    source,
    positionImage,
    volume,
    scales,
    displaySettings,
    volumetric,
    geometryConsumer,
    selectedChannel,
    onVolumeAtlasChange,
    totalBytes,
  }) {
    const renderer = rendering.renderer;
    if (renderer.capabilities.isWebGL2 !== true || renderer.extensions.has("EXT_color_buffer_float") !== true) {
      throw new Error("Surface view requires WebGL2 float textures and render targets.");
    }
    const { maxLayers, maxTextureSize } = rendering.textureLimits;
    const geometry = new THREE.PlaneGeometry(1, 1);
    const acquired = positionImage.atlas === null;
    let surfaceAtlas = null;
    let levels = null;
    let material = null;
    let reliefMask = null;
    try {
      surfaceAtlas = positionImage.acquire(geometryConsumer, rendering);
      const volumeBytes = acquired ? totalBytes - surfaceAtlas.allocatedBytes() : totalBytes;
      levels = new SurfaceVolumeLevels({
        brickPool: rendering.brickPool,
        namespace: rendering.brickPool.namespace(volume, selectedChannel),
        onChange: onVolumeAtlasChange,
        scales,
        byteLimit: surfaceVolumeByteLimit(volumeBytes),
        maxLayers,
        maxTextureSize,
      });
      reliefMask = new SurfaceReliefMaskTexture(volumetric.mask);
      material = createSurfaceMaterial({
        surfaceAtlas,
        surfaceGrid_HW: source.grid_HW,
        surfaceScale_UV: source.scale_UV,
        levels: levels.shown,
        sourceBounds_ZYX: volumeBounds_ZYX(volume),
        displayEncoding: volume.metadata.displayEncoding,
        displaySettings,
        reliefMaskTexture: reliefMask.texture,
      });
    } catch (error) {
      material?.dispose();
      positionImage.release(geometryConsumer);
      reliefMask?.dispose();
      levels?.dispose();
      geometry.dispose();
      throw error;
    }
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.frustumCulled = false;
    pane.baseScene.add(this.mesh);
    this.pane = pane;
    this.geometry = geometry;
    this.surfaceAtlas = surfaceAtlas;
    this.positionImage = positionImage;
    this.geometryConsumer = geometryConsumer;
    this.levels = levels;
    this.material = material;
    this.reliefMask = reliefMask;
    const levelTableBytes = 4 * pageTableBytes(SURFACE_ATLAS_SLOT_LIMIT, maxTextureSize);
    this.baseImage = new SurfaceBaseImage(
      Math.max(0, SURFACE_PAGE_TABLE_RESERVE_BYTES - levelTableBytes - reliefMask.allocatedBytes()),
    );
    this.disposed = false;
  }

  retainBudget(totalBytes) {
    const byteLimit = surfaceVolumeByteLimit(totalBytes);
    if (this.levels.atlasBytes(false) > byteLimit) return false;
    this.levels.setByteLimit(byteLimit);
    return true;
  }

  allocatedBytes() {
    return this.levels.allocatedBytes() + this.baseImage.allocatedBytes() + this.reliefMask.allocatedBytes();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.mesh.removeFromParent();
    this.material.dispose();
    this.baseImage.dispose();
    this.geometry.dispose();
    this.positionImage.release(this.geometryConsumer);
    this.reliefMask.dispose();
    this.levels.dispose();
  }
}
