import * as THREE from "../../vendor/three.module.js";

export const FILL_PAGE = 0xffff_ffff;

function pageHash(chunk_ZYX) {
  return (
    (Math.imul(chunk_ZYX[0], 73_856_093) ^
      Math.imul(chunk_ZYX[1], 19_349_663) ^
      Math.imul(chunk_ZYX[2], 83_492_791)) >>>
    0
  );
}

export const PAGE_HASH_SHADER = `
  uint pageHash(uvec3 chunk_ZYX) {
    return (chunk_ZYX.x * 73856093u) ^ (chunk_ZYX.y * 19349663u) ^ (chunk_ZYX.z * 83492791u);
  }
`;

export class PageTable {
  constructor(maxEntries, maxTextureSize) {
    const layout = pageTableLayout(maxEntries, maxTextureSize);
    this.maxEntries = maxEntries;
    this.size = layout.size;
    this.width = layout.width;
    this.height = layout.height;
    this.data = new Uint32Array(this.size * 4);
    this.texture = new THREE.DataTexture(
      this.data,
      this.width,
      this.height,
      THREE.RGBAIntegerFormat,
      THREE.UnsignedIntType,
    );
    this.texture.colorSpace = THREE.NoColorSpace;
    this.texture.internalFormat = "RGBA32UI";
    this.texture.minFilter = THREE.NearestFilter;
    this.texture.magFilter = THREE.NearestFilter;
    this.texture.generateMipmaps = false;
    this.texture.flipY = false;
    this.texture.unpackAlignment = 1;
    this.texture.needsUpdate = true;
    this.writeCount = 0;
  }

  write(entries) {
    if (entries.length > this.maxEntries) throw new Error("Page table entry count exceeds its capacity.");
    this.writeCount += 1;
    this.data.fill(0);
    for (const { chunk_ZYX, slot } of entries) {
      const start = pageHash(chunk_ZYX) & (this.size - 1);
      let inserted = false;
      for (let probe = 0; probe < this.size; probe++) {
        const index = (start + probe) & (this.size - 1);
        const offset = index * 4;
        if (this.data[offset + 3] !== 0) continue;
        this.data[offset] = chunk_ZYX[0] + 1;
        this.data[offset + 1] = chunk_ZYX[1] + 1;
        this.data[offset + 2] = chunk_ZYX[2] + 1;
        this.data[offset + 3] = slot === null ? FILL_PAGE : slot + 1;
        inserted = true;
        break;
      }
      if (!inserted) throw new Error("Page table has no empty slot.");
    }
    this.texture.needsUpdate = true;
  }

  allocatedBytes() {
    return this.data.byteLength;
  }

  dispose() {
    this.texture.dispose();
  }
}

export function pageTableLayout(maxEntries, maxTextureSize) {
  let size = 1;
  while (size < maxEntries * 4) size *= 2;
  let width = 1;
  while (width * 2 <= size && width * 2 <= maxTextureSize) width *= 2;
  const height = size / width;
  if (height > maxTextureSize) {
    throw new Error(`Page table needs ${size} texels, exceeding ${maxTextureSize} × ${maxTextureSize}.`);
  }
  return { size, width, height };
}

export function pageTableBytes(maxEntries, maxTextureSize) {
  return pageTableLayout(maxEntries, maxTextureSize).size * 4 * Uint32Array.BYTES_PER_ELEMENT;
}
