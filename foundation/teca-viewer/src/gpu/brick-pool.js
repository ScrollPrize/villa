import { ATLAS_BYTE_LIMIT } from "../config.js";
import { ChunkAtlas, chunkAtlasAllocation } from "./chunk-atlas.js";

class BrickLease {
  constructor(pool, group, capacity, reservedSlotsPerLayer, onChange) {
    this.pool = pool;
    this.group = group;
    this.capacity = capacity;
    this.reservedSlotsPerLayer = reservedSlotsPerLayer;
    this.onChange = onChange;
    this.entries = new Map();
    this.visibleKeys = new Set();
    this.disposed = false;
  }

  get texture() {
    return this.group.atlas.texture;
  }
  get chunks_ZYX() {
    return this.group.atlas.chunks_ZYX;
  }
  get dtype() {
    return this.group.atlas.dtype;
  }
  get formatDescriptor() {
    return this.group.atlas.formatDescriptor;
  }
  get ArrayType() {
    return this.group.atlas.ArrayType;
  }
  get voxelCount() {
    return this.group.atlas.voxelCount;
  }
  get width() {
    return this.group.atlas.width;
  }
  get height() {
    return this.group.atlas.height;
  }
  get slotBytes() {
    return this.group.atlas.slotBytes;
  }
  get layerBytes() {
    return this.group.atlas.layerBytes;
  }
  get slotsPerLayer() {
    return this.group.atlas.slotsPerLayer;
  }

  beginFrame(keys) {
    this.visibleKeys = new Set();
    for (const key of keys) {
      if (this.visibleKeys.size === this.capacity) break;
      this.visibleKeys.add(key);
    }
    this.pool.protect(this.group);
  }

  inWorkingSet(key) {
    return this.visibleKeys.has(key);
  }

  claim(entry) {
    if (entry === null || this.entries.has(entry.key)) return entry;
    if (this.entries.size === this.capacity) {
      let oldest = null;
      for (const candidate of this.entries.values()) {
        if (
          oldest === null ||
          Number(this.visibleKeys.has(candidate.key)) < Number(this.visibleKeys.has(oldest.key)) ||
          (this.visibleKeys.has(candidate.key) === this.visibleKeys.has(oldest.key) &&
            candidate.lastUsed < oldest.lastUsed)
        )
          oldest = candidate;
      }
      if (this.visibleKeys.has(oldest.key) && !this.visibleKeys.has(entry.key)) return null;
      this.entries.delete(oldest.key);
      this.pool.invalidate(this, oldest);
    }
    this.entries.set(entry.key, entry);
    return entry;
  }

  get(key) {
    return this.claim(this.group.atlas.get(key));
  }
  held(key) {
    const entry = this.entries.get(key);
    return entry !== undefined && this.group.atlas.isResident(entry) ? entry : null;
  }
  refresh(key, data_DHW, start, end) {
    const entry = this.group.atlas.entries.get(key);
    if (entry === undefined) return;
    const holders = [...this.group.leases].filter((lease) => lease.entries.get(key) === entry);
    const next = this.group.atlas.refresh(entry, data_DHW, start, end);
    for (const lease of holders) lease.entries.set(key, next);
  }
  admit(key, data_DHW) {
    return this.claim(this.group.atlas.admit(key, data_DHW));
  }
  admitCopy(key) {
    return this.claim(this.group.atlas.admitCopy(key));
  }
  isResident(entry) {
    return this.entries.get(entry.key) === entry && this.group.atlas.isResident(entry);
  }
  needsUpload(renderer, entry) {
    this.pool.flush(this.group);
    return this.group.atlas.needsUpload(renderer, entry);
  }
  uploadPart(renderer, entry, bytes) {
    this.pool.flush(this.group);
    return this.group.atlas.uploadPart(renderer, entry, bytes);
  }
  copySlots(renderer, source, pairs) {
    this.pool.flush(this.group);
    this.group.atlas.copySlots(renderer, source, pairs);
  }

  remove(key) {
    if ([...this.group.leases].some((lease) => lease !== this && lease.visibleKeys.has(key))) return false;
    return this.group.atlas.remove(key);
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.entries.clear();
    this.visibleKeys.clear();
    this.pool.release(this);
  }
}

export class BrickPool {
  constructor(renderer) {
    this.renderer = renderer;
    this.groups = new Set();
    this.sources = new WeakMap();
    this.nextSource = 0;
  }

  namespace(volume, channel) {
    if (!this.sources.has(volume)) this.sources.set(volume, ++this.nextSource);
    return JSON.stringify([this.sources.get(volume), channel]);
  }

  acquire(namespace, options, onChange) {
    const allocation = chunkAtlasAllocation(options);
    const capacity = allocation.capacity;
    const signature = JSON.stringify([
      namespace,
      options.chunks_ZYX,
      options.dtype,
      options.maxLayers,
      options.maxTextureSize,
    ]);
    let group = [...this.groups].find((candidate) => {
      if (candidate.signature !== signature) return false;
      const capacities = [...candidate.leases].map((lease) => lease.capacity).concat(capacity);
      const combined = chunkAtlasAllocation({
        ...options,
        packSlots: true,
        byteLimit: capacities.reduce((a, b) => a + b, 0) * allocation.slotBytes,
      });
      return (
        combined.capacity * combined.slotBytes <= ATLAS_BYTE_LIMIT &&
        combined.capacity === capacities.reduce((a, b) => a + b, 0) &&
        capacities.every((count) => count % combined.slotsPerLayer === 0)
      );
    });
    if (group === undefined) {
      group = { signature, options, leases: new Set(), atlas: null, invalidated: new Map() };
      this.groups.add(group);
    }
    const lease = new BrickLease(this, group, capacity, allocation.slotsPerLayer, onChange);
    group.leases.add(lease);
    this.resize(group, lease);
    return lease;
  }

  resize(group, added = null) {
    const previous = group.atlas;
    const slotBytes = chunkAtlasAllocation(group.options).slotBytes;
    const capacity = [...group.leases].reduce((total, lease) => total + lease.capacity, 0);
    const atlas = new ChunkAtlas({
      ...group.options,
      packSlots: true,
      byteLimit: capacity * slotBytes,
      onEvict: (entry) => {
        for (const lease of group.leases) if (lease.entries.delete(entry.key)) this.invalidate(lease, entry);
      },
    });
    if (previous !== null) {
      atlas.retain(this.renderer, previous);
      previous.dispose();
    }
    group.atlas = atlas;
    group.invalidated.clear();
    this.protect(group);
    for (const lease of group.leases) {
      for (const [key, entry] of lease.entries) if (!atlas.isResident(entry)) lease.entries.delete(key);
      if (previous !== null && lease !== added) lease.onChange(lease, null);
    }
  }

  invalidate(lease, entry) {
    let entries = lease.group.invalidated.get(lease);
    if (entries === undefined) {
      entries = new Set();
      lease.group.invalidated.set(lease, entries);
    }
    entries.add(entry);
  }

  flush(group) {
    if (group.invalidated.size === 0) return;
    const invalidated = group.invalidated;
    group.invalidated = new Map();
    for (const [lease, entries] of invalidated) if (!lease.disposed) lease.onChange(lease, entries);
  }

  protect(group) {
    group.atlas.beginFrame([...group.leases].flatMap((lease) => [...lease.visibleKeys]));
  }

  release(lease) {
    const group = lease.group;
    group.leases.delete(lease);
    if (group.leases.size === 0) {
      group.atlas.dispose();
      this.groups.delete(group);
    } else this.resize(group);
  }
}
