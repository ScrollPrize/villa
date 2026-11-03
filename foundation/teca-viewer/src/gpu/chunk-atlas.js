import * as THREE from "../../vendor/three.module.js";
import { atlasFormatForDtype } from "./atlas-format.js";

export class ChunkAtlas {
  constructor({ chunks_ZYX, dtype, byteLimit, maxLayers, maxTextureSize, packSlots = false, onEvict }) {
    const { format, voxelCount, layout, slotBytes, slotsPerLayer, capacity } = chunkAtlasAllocation({
      chunks_ZYX,
      dtype,
      byteLimit,
      maxLayers,
      maxTextureSize,
      packSlots,
    });

    this.chunks_ZYX = [...chunks_ZYX];
    this.dtype = dtype;
    this.formatDescriptor = format;
    this.voxelCount = voxelCount;
    this.width = layout.width;
    this.height = layout.height;
    this.layerBytes = layout.width * layout.height * format.bytesPerVoxel;
    this.slotBytes = slotBytes;
    this.capacity = capacity;
    this.slotsPerLayer = slotsPerLayer;
    this.ArrayType = format.ArrayType;
    this.texture = new THREE.DataArrayTexture(null, this.width, this.height * slotsPerLayer, capacity / slotsPerLayer);
    this.texture.source.dataReady = false;
    this.texture.colorSpace = THREE.NoColorSpace;
    this.texture.format = format.format;
    this.texture.type = format.type;
    this.texture.internalFormat = format.internalFormat;
    this.texture.minFilter = THREE.NearestFilter;
    this.texture.magFilter = THREE.NearestFilter;
    this.texture.generateMipmaps = false;
    this.texture.flipY = false;
    this.texture.unpackAlignment = 1;
    this.texture.needsUpdate = true;
    this.entries = new Map();
    this.slots = Array(capacity).fill(null);
    this.visibleKeys = new Set();
    this.clock = 0;
    this.generation = 0;
    this.onEvict = onEvict;
  }

  retain(renderer, previous) {
    const retained = [...previous.entries.values()]
      .sort((a, b) => Number(b.visible) - Number(a.visible) || b.lastUsed - a.lastUsed)
      .slice(0, this.capacity);
    this.clock = previous.clock;
    const pairs = [];
    let slot = 0;
    for (const entry of retained) {
      if (entry.slot !== null) {
        if (entry.data === null) pairs.push({ from: { slot: entry.slot }, to: entry });
        entry.slot = slot++;
        this.slots[entry.slot] = entry;
      }
      entry.generation = ++this.generation;
      entry.uploaded = new Map();
      this.entries.set(entry.key, entry);
    }
    if (pairs.length > 0) this.copySlots(renderer, previous, pairs);
  }

  // visibleKeys arrive in planner priority order. Capping the protected set at capacity keeps admission deterministic
  // when a view needs more chunks than the atlas holds: the overflow stays unadmitted instead of visible entries
  // evicting each other forever.
  beginFrame(visibleKeys) {
    this.visibleKeys = new Set();
    for (const key of visibleKeys) {
      if (this.visibleKeys.size === this.capacity) break;
      this.visibleKeys.add(key);
    }
    for (const entry of this.entries.values()) entry.visible = this.visibleKeys.has(entry.key);
  }

  admit(key, chunk_DHW) {
    if (chunk_DHW !== null) this.validateChunk(chunk_DHW);
    return this.enter(key, chunk_DHW, chunk_DHW !== null);
  }

  admitCopy(key) {
    return this.enter(key, null, true);
  }

  enter(key, chunk_DHW, slotted) {
    if (typeof key !== "string" || key.length === 0) {
      throw new Error("Chunk atlas key must be a nonempty string.");
    }
    const existing = this.entries.get(key);
    if (existing !== undefined) {
      existing.lastUsed = ++this.clock;
      return existing;
    }
    if (this.entries.size === this.capacity) {
      let eviction = null;
      for (const candidate of this.entries.values())
        if (
          eviction === null ||
          Number(candidate.visible) < Number(eviction.visible) ||
          (candidate.visible === eviction.visible && candidate.lastUsed < eviction.lastUsed)
        )
          eviction = candidate;
      if (!this.visibleKeys.has(key) && eviction.visible) return null;
      this.remove(eviction.key);
    }
    const entry = {
      key,
      slot: slotted ? this.slots.indexOf(null) : null,
      data: chunk_DHW === null ? null : this.prepareChunk(chunk_DHW),
      generation: ++this.generation,
      lastUsed: ++this.clock,
      visible: this.visibleKeys.has(key),
      uploaded: new Map(),
    };
    this.entries.set(key, entry);
    if (entry.slot !== null) this.slots[entry.slot] = entry;
    return entry;
  }

  remove(key) {
    const entry = this.entries.get(key);
    if (entry === undefined) return false;
    this.entries.delete(key);
    if (entry.slot !== null) this.slots[entry.slot] = null;
    this.onEvict(entry);
    return true;
  }

  get(key) {
    const entry = this.entries.get(key) ?? null;
    if (entry !== null) entry.lastUsed = ++this.clock;
    return entry;
  }

  isResident(entry) {
    return this.entries.get(entry.key) === entry;
  }

  needsUpload(renderer, entry) {
    return this.isResident(entry) && !this.uploadProgress(renderer, entry).complete;
  }

  uploadProgress(renderer, entry) {
    if (entry.slot === null) return { generation: entry.generation, complete: true };
    const progress = entry.uploaded.get(renderer);
    return progress?.generation === entry.generation
      ? progress
      : { generation: entry.generation, complete: false, texels: 0, end: this.width * this.height };
  }

  refresh(entry, data_DHW, start, end) {
    if (entry.slot === null) {
      this.remove(entry.key);
      return this.admit(entry.key, data_DHW);
    }
    this.validateChunk(data_DHW);
    if (entry.data === null || data_DHW.length === entry.data.length) entry.data = this.prepareChunk(data_DHW);
    else entry.data.set(data_DHW.subarray(start, end), start);
    const generation = ++this.generation;
    for (const [renderer, progress] of entry.uploaded) {
      if (progress.generation !== entry.generation) continue;
      entry.uploaded.set(
        renderer,
        progress.complete
          ? { generation, complete: false, texels: start, end }
          : { generation, complete: false, texels: Math.min(progress.texels, start), end: Math.max(progress.end, end) },
      );
    }
    entry.generation = generation;
    return entry;
  }

  uploadPart(renderer, entry, byteLimit) {
    if (!Number.isSafeInteger(byteLimit) || byteLimit < this.formatDescriptor.bytesPerVoxel) {
      throw new Error(`Chunk atlas upload byte limit ${byteLimit} does not hold one texel.`);
    }
    if (!this.isResident(entry)) return { uploadedBytes: 0, complete: false };
    const progress = this.uploadProgress(renderer, entry);
    if (progress.complete) return { uploadedBytes: 0, complete: true };
    if (progress.texels === 0 && progress.end === this.width * this.height && this.layerBytes <= byteLimit) {
      this.uploadRegion(renderer, entry.data, 0, 0, entry.slot, this.width, this.height);
      entry.uploaded.set(renderer, { generation: entry.generation, complete: true });
      entry.data = null;
      return { uploadedBytes: this.layerBytes, complete: true };
    }
    const texelLimit = Math.floor(byteLimit / this.formatDescriptor.bytesPerVoxel);
    const x = progress.texels % this.width;
    const y = Math.floor(progress.texels / this.width);
    const width = Math.min(this.width - x, texelLimit, progress.end - progress.texels);
    const height =
      x === 0 && width === this.width
        ? Math.min(Math.floor((progress.end - progress.texels) / width), Math.floor(texelLimit / width))
        : 1;
    const sourceData = entry.data.subarray(progress.texels, progress.texels + width * height);
    this.uploadRegion(renderer, sourceData, x, y, entry.slot, width, height);
    const texels = progress.texels + width * height;
    const complete = texels === progress.end;
    entry.uploaded.set(renderer, { generation: entry.generation, complete, texels, end: progress.end });
    if (complete) entry.data = null;
    return { uploadedBytes: width * height * this.formatDescriptor.bytesPerVoxel, complete };
  }

  uploadRegion(renderer, data_DHW, x, y, slot, width, height) {
    y += (slot % this.slotsPerLayer) * this.height;
    const layer = Math.floor(slot / this.slotsPerLayer);
    const gl = renderer.getContext();
    const { samplerClass, ArrayType } = this.formatDescriptor;
    const format = samplerClass === "float" ? gl.RED : gl.RED_INTEGER;
    const type =
      samplerClass === "float"
        ? gl.FLOAT
        : samplerClass === "int"
          ? ArrayType.BYTES_PER_ELEMENT === 1
            ? gl.BYTE
            : gl.SHORT
          : ArrayType.BYTES_PER_ELEMENT === 1
            ? gl.UNSIGNED_BYTE
            : gl.UNSIGNED_SHORT;
    renderer.state.activeTexture(gl.TEXTURE0);
    renderer.initTexture(this.texture);
    renderer.state.bindTexture(gl.TEXTURE_2D_ARRAY, renderer.properties.get(this.texture).__webglTexture, gl.TEXTURE0);
    try {
      // Atlas uploads own tightly packed rows; no caller retains an unpack subregion.
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
      gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, 0);
      gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0);
      gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
      gl.pixelStorei(gl.UNPACK_SKIP_IMAGES, 0);
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, x, y, layer, width, height, 1, format, type, data_DHW);
    } finally {
      renderer.state.unbindTexture();
    }
  }

  copySlots(renderer, source, pairs) {
    const gl = renderer.getContext();
    const previousReadFramebuffer = gl.getParameter(gl.READ_FRAMEBUFFER_BINDING);
    const framebuffer = gl.createFramebuffer();
    const sourceTexture = renderer.properties.get(source.texture).__webglTexture;
    const attach = (slot) =>
      gl.framebufferTextureLayer(
        gl.READ_FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        sourceTexture,
        0,
        Math.floor(slot / source.slotsPerLayer),
      );
    try {
      renderer.initTexture(this.texture);
      renderer.state.activeTexture(gl.TEXTURE0);
      renderer.state.bindTexture(
        gl.TEXTURE_2D_ARRAY,
        renderer.properties.get(this.texture).__webglTexture,
        gl.TEXTURE0,
      );
      renderer.state.bindFramebuffer(gl.READ_FRAMEBUFFER, framebuffer);
      attach(pairs[0].from.slot);
      if (gl.checkFramebufferStatus(gl.READ_FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error("Chunk atlas cannot read its source texture through a framebuffer.");
      for (const { from, to } of pairs) {
        attach(from.slot);
        gl.copyTexSubImage3D(
          gl.TEXTURE_2D_ARRAY,
          0,
          0,
          (to.slot % this.slotsPerLayer) * this.height,
          Math.floor(to.slot / this.slotsPerLayer),
          0,
          (from.slot % source.slotsPerLayer) * source.height,
          this.width,
          this.height,
        );
        to.uploaded.set(renderer, { generation: to.generation, complete: true });
      }
    } finally {
      renderer.state.bindFramebuffer(gl.READ_FRAMEBUFFER, previousReadFramebuffer);
      renderer.state.unbindTexture();
      gl.deleteFramebuffer(framebuffer);
    }
  }

  dispose() {
    this.texture.dispose();
    this.entries.clear();
    this.slots.fill(null);
    this.visibleKeys.clear();
  }

  validateChunk(chunk_DHW) {
    if (!(chunk_DHW instanceof this.ArrayType) || chunk_DHW.length !== this.voxelCount) {
      throw new Error(
        `Chunk atlas needs ${this.ArrayType.name}(${this.voxelCount}), received ${chunk_DHW?.constructor?.name ?? "null"}(${chunk_DHW?.length ?? 0}).`,
      );
    }
  }

  prepareChunk(chunk_DHW) {
    const texelCount = this.width * this.height;
    if (chunk_DHW.length === texelCount) return chunk_DHW;
    const data = new this.ArrayType(texelCount);
    data.set(chunk_DHW);
    return data;
  }
}

export function chunkAtlasAllocation({ chunks_ZYX, dtype, byteLimit, maxLayers, maxTextureSize, packSlots = false }) {
  if (
    !Array.isArray(chunks_ZYX) ||
    chunks_ZYX.length !== 3 ||
    chunks_ZYX.some((value) => !Number.isSafeInteger(value) || value < 1)
  )
    throw new Error("Chunk atlas shape must contain three positive integers.");
  const format = atlasFormatForDtype(dtype);
  if (!Number.isSafeInteger(byteLimit) || byteLimit < 1) {
    throw new Error("Chunk atlas byte limit must be a positive integer.");
  }
  if (!Number.isSafeInteger(maxLayers) || maxLayers < 1) {
    throw new Error("Chunk atlas layer limit must be a positive integer.");
  }
  const voxelCount = chunks_ZYX.reduce((product, value) => product * value, 1);
  const layout = chunkAtlasLayout(voxelCount, maxTextureSize);
  const slotBytes = layout.width * layout.height * format.bytesPerVoxel;
  const slotLimit = Math.floor(byteLimit / slotBytes);
  const maximumSlotsPerLayer = packSlots ? 2 ** Math.floor(Math.log2(maxTextureSize / layout.height)) : 1;
  const slotsPerLayer = Math.min(
    maximumSlotsPerLayer,
    2 ** Math.max(0, Math.ceil(Math.log2(Math.max(1, slotLimit / maxLayers)))),
  );
  const capacity = Math.min(maxLayers * slotsPerLayer, Math.floor(slotLimit / slotsPerLayer) * slotsPerLayer);
  if (capacity < 1) throw new Error("CT atlas budget cannot hold one chunk.");
  return { format, voxelCount, layout, slotBytes, slotsPerLayer, capacity };
}

export function chunkAtlasSlotBytes({ chunks_ZYX, dtype, maxTextureSize }) {
  if (
    !Array.isArray(chunks_ZYX) ||
    chunks_ZYX.length !== 3 ||
    chunks_ZYX.some((value) => !Number.isSafeInteger(value) || value < 1)
  ) {
    throw new Error("Chunk atlas shape must contain three positive integers.");
  }
  const format = atlasFormatForDtype(dtype);
  const voxelCount = chunks_ZYX.reduce((product, value) => product * value, 1);
  const layout = chunkAtlasLayout(voxelCount, maxTextureSize);
  return layout.width * layout.height * format.bytesPerVoxel;
}

export function chunkAtlasLayout(voxelCount, maxTextureSize) {
  if (!Number.isSafeInteger(voxelCount) || voxelCount < 1) {
    throw new Error("Atlas voxel count must be a positive integer.");
  }
  if (!Number.isSafeInteger(maxTextureSize) || maxTextureSize < 1) {
    throw new Error("Maximum atlas texture size must be a positive integer.");
  }
  let width = 1;
  const target = Math.ceil(Math.sqrt(voxelCount));
  while (width < target && width < maxTextureSize) width *= 2;
  width = Math.min(width, maxTextureSize);
  const height = Math.ceil(voxelCount / width);
  if (height > maxTextureSize) {
    throw new Error(`Chunk needs a ${width} × ${height} atlas layer, exceeding ${maxTextureSize}.`);
  }
  return { width, height };
}
