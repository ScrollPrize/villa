import * as THREE from "../../vendor/three.module.js";

import { compileSurfaceReliefMask, SURFACE_RELIEF_MASK_SIZE } from "./relief-mask.js";

export class SurfaceReliefMaskTexture {
  constructor(mask) {
    const values = compileSurfaceReliefMask(mask);
    this.texture = new THREE.DataTexture(
      values,
      SURFACE_RELIEF_MASK_SIZE,
      SURFACE_RELIEF_MASK_SIZE,
      THREE.RGBAFormat,
      THREE.FloatType,
    );
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.wrapS = THREE.ClampToEdgeWrapping;
    this.texture.wrapT = THREE.ClampToEdgeWrapping;
    this.texture.generateMipmaps = false;
    this.texture.needsUpdate = true;
  }

  update(mask) {
    this.texture.image.data.set(compileSurfaceReliefMask(mask));
    this.texture.needsUpdate = true;
  }

  allocatedBytes() {
    return SURFACE_RELIEF_MASK_SIZE * SURFACE_RELIEF_MASK_SIZE * 4 * Float32Array.BYTES_PER_ELEMENT;
  }

  dispose() {
    this.texture.dispose();
  }
}
