// A strip's page-table entry is i + 1 once its first upload completes, so the shader never reads a half-written strip.

import * as THREE from "../../vendor/three.module.js";
import { chunkAtlasLayout } from "../gpu/chunk-atlas.js";
import { uploadFloatRgbaArrayLayer } from "./array-layer-upload.js";
import { STRIP_TEXEL_BYTES } from "./source.js";

export const SURFACE_BOUNDS_BLOCK_COLUMNS = 8;
export const SURFACE_BOUNDS_BLOCK_ROWS = 8;

export class SurfaceStripAtlas {
  constructor({ width, stripRows, stripCount, maxTextureSize }) {
    for (const [label, value] of [
      ["width", width],
      ["strip rows", stripRows],
      ["strip count", stripCount],
      ["texture-size limit", maxTextureSize],
    ]) {
      if (!Number.isSafeInteger(value) || value < 1) throw new Error(`Surface ${label} must be a positive integer.`);
    }
    this.width = width;
    this.stripRows = stripRows;
    this.stripCount = stripCount;
    const layout = chunkAtlasLayout(width * stripRows, maxTextureSize);
    this.atlasWidth = layout.width;
    this.atlasHeight = layout.height;
    this.layerBytes = this.atlasWidth * this.atlasHeight * STRIP_TEXEL_BYTES;
    this.texture = new THREE.DataArrayTexture(null, this.atlasWidth, this.atlasHeight, this.stripCount);
    this.texture.source.dataReady = false;
    this.texture.colorSpace = THREE.NoColorSpace;
    this.texture.format = THREE.RGBAFormat;
    this.texture.type = THREE.FloatType;
    this.texture.internalFormat = "RGBA32F";
    this.texture.minFilter = THREE.NearestFilter;
    this.texture.magFilter = THREE.NearestFilter;
    this.texture.generateMipmaps = false;
    this.texture.flipY = false;
    this.texture.unpackAlignment = 1;
    this.texture.needsUpdate = true;
    this.pageWidth = Math.min(this.stripCount, maxTextureSize);
    this.pageHeight = Math.ceil(this.stripCount / this.pageWidth);
    if (this.pageHeight > maxTextureSize) {
      throw new Error(
        `Surface strip table needs ${this.pageWidth}×${this.pageHeight} texels, exceeding ${maxTextureSize}.`,
      );
    }
    this.pageData = new Uint32Array(this.pageWidth * this.pageHeight);
    this.pageTexture = new THREE.DataTexture(
      this.pageData,
      this.pageWidth,
      this.pageHeight,
      THREE.RedIntegerFormat,
      THREE.UnsignedIntType,
    );
    this.pageTexture.colorSpace = THREE.NoColorSpace;
    this.pageTexture.internalFormat = "R32UI";
    this.pageTexture.minFilter = THREE.NearestFilter;
    this.pageTexture.magFilter = THREE.NearestFilter;
    this.pageTexture.generateMipmaps = false;
    this.pageTexture.needsUpdate = true;
    this.entries = new Map();
  }

  get(stripIndex) {
    return this.entries.get(stripIndex) ?? null;
  }

  getPresented(stripIndex) {
    const entry = this.get(stripIndex);
    return entry?.uploaded ? entry : null;
  }

  admit(stripIndex, data_HWRGBA, bounds_B6) {
    if (!(data_HWRGBA instanceof Float32Array) || data_HWRGBA.length !== this.width * this.stripRows * 4) {
      throw new Error("Surface strip atlas received an invalid float32 RGBA strip.");
    }
    const existing = this.entries.get(stripIndex);
    if (existing !== undefined) return existing;
    const boundsGrid_HW = [
      Math.ceil(this.stripRows / SURFACE_BOUNDS_BLOCK_ROWS),
      Math.ceil(this.width / SURFACE_BOUNDS_BLOCK_COLUMNS),
    ];
    if (!(bounds_B6 instanceof Float32Array) || bounds_B6.length !== boundsGrid_HW[0] * boundsGrid_HW[1] * 6) {
      throw new Error("Surface strip bounds do not cover its complete UV tile grid.");
    }
    const entry = {
      stripIndex,
      data_HWRGBA,
      bounds_B6,
      boundsGrid_HW,
      uploaded: false,
      uploadedPixels: 0,
      pixelStop: this.width * this.stripRows,
    };
    this.entries.set(stripIndex, entry);
    return entry;
  }

  rewrite(stripIndex, data_HWRGBA, bounds_B6, pixelStart, pixelStop) {
    const entry = this.entries.get(stripIndex) ?? null;
    if (entry === null) return null;
    const [start, stop] = entry.uploaded
      ? [pixelStart, pixelStop]
      : [Math.min(entry.uploadedPixels, pixelStart), Math.max(entry.pixelStop, pixelStop)];
    Object.assign(entry, { data_HWRGBA, bounds_B6, uploaded: false, uploadedPixels: start, pixelStop: stop });
    return entry;
  }

  upload(renderer, entry, byteLimit) {
    if (!Number.isSafeInteger(byteLimit) || byteLimit < STRIP_TEXEL_BYTES) {
      throw new Error("Surface strip upload byte limit must hold at least one RGBA32F texel.");
    }
    const sourcePixelCount = this.width * this.stripRows;
    const pixelCount = Math.min(entry.pixelStop - entry.uploadedPixels, Math.floor(byteLimit / STRIP_TEXEL_BYTES));
    uploadFloatRgbaArrayLayer({
      renderer,
      texture: this.texture,
      layer: entry.stripIndex,
      atlasWidth: this.atlasWidth,
      sourcePixelCount,
      data_RGBA: entry.data_HWRGBA,
      pixelOffset: entry.uploadedPixels,
      pixelCount,
    });
    entry.uploadedPixels += pixelCount;
    entry.uploaded = entry.uploadedPixels === entry.pixelStop;
    if (entry.uploaded && this.pageData[entry.stripIndex] === 0) {
      this.pageData[entry.stripIndex] = entry.stripIndex + 1;
      this.pageTexture.needsUpdate = true;
    }
    return { uploadedBytes: pixelCount * STRIP_TEXEL_BYTES, complete: entry.uploaded };
  }

  allocatedBytes() {
    return this.stripCount * this.layerBytes + this.pageData.byteLength;
  }

  dispose() {
    this.texture.dispose();
    this.pageTexture.dispose();
    this.entries.clear();
  }
}

export function surfaceStripBounds(data_HWRGBA, width, stripRows) {
  if (!(data_HWRGBA instanceof Float32Array) || data_HWRGBA.length !== width * stripRows * 4) {
    throw new Error("Surface bounds need one complete float32 RGBA strip.");
  }
  const blockColumns = Math.ceil(width / SURFACE_BOUNDS_BLOCK_COLUMNS);
  const blockRows = Math.ceil(stripRows / SURFACE_BOUNDS_BLOCK_ROWS);
  const bounds_B6 = new Float32Array(blockRows * blockColumns * 6);
  for (let blockV = 0; blockV < blockRows; blockV++) {
    const startY = blockV * SURFACE_BOUNDS_BLOCK_ROWS;
    const stopY = Math.min(stripRows, startY + SURFACE_BOUNDS_BLOCK_ROWS + 1);
    for (let blockU = 0; blockU < blockColumns; blockU++) {
      let minimumZ = Infinity;
      let minimumY = Infinity;
      let minimumX = Infinity;
      let maximumZ = -Infinity;
      let maximumY = -Infinity;
      let maximumX = -Infinity;
      const startX = blockU * SURFACE_BOUNDS_BLOCK_COLUMNS;
      const stopX = Math.min(width, startX + SURFACE_BOUNDS_BLOCK_COLUMNS + 1);
      for (let y = startY; y < stopY; y++) {
        for (let x = startX; x < stopX; x++) {
          const offset = (y * width + x) * 4;
          if (data_HWRGBA[offset + 3] < 0.5) continue;
          minimumZ = Math.min(minimumZ, data_HWRGBA[offset]);
          minimumY = Math.min(minimumY, data_HWRGBA[offset + 1]);
          minimumX = Math.min(minimumX, data_HWRGBA[offset + 2]);
          maximumZ = Math.max(maximumZ, data_HWRGBA[offset]);
          maximumY = Math.max(maximumY, data_HWRGBA[offset + 1]);
          maximumX = Math.max(maximumX, data_HWRGBA[offset + 2]);
        }
      }
      const offset = (blockV * blockColumns + blockU) * 6;
      bounds_B6[offset] = minimumZ;
      bounds_B6[offset + 1] = minimumY;
      bounds_B6[offset + 2] = minimumX;
      bounds_B6[offset + 3] = maximumZ;
      bounds_B6[offset + 4] = maximumY;
      bounds_B6[offset + 5] = maximumX;
    }
  }
  return bounds_B6;
}
