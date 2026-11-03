import { ATLAS_BYTE_LIMIT, SURFACE_FALLBACK_BUDGET_SHARE } from "../config.js";
import { chunkKey, chunkKeyParts } from "../gpu/addresses.js";
import { chunkAtlasAllocation, chunkAtlasSlotBytes } from "../gpu/chunk-atlas.js";
import { PageTable } from "../gpu/page-table.js";

export const SURFACE_ATLAS_SLOT_LIMIT = 16384;

class SurfaceVolumeLevel {
  constructor(scale, capacity, atlas, maxTextureSize) {
    this.scale = scale;
    this.capacity = capacity;
    this.atlas = atlas;
    this.pageTable = new PageTable(atlas.capacity, maxTextureSize);
    this.addresses = [];
    this.coverage = null;
    this.entries = new Map();
  }

  entry(chunk_ZYX) {
    return this.entries.get(chunkKey(this.scale, chunk_ZYX));
  }

  place(chunk_ZYX, entry) {
    this.entries.set(chunkKey(this.scale, chunk_ZYX), entry);
  }

  prune() {
    let changed = false;
    for (const [key, entry] of this.entries) {
      if (!this.atlas.isResident(entry)) {
        this.entries.delete(key);
        changed = true;
      }
    }
    if (changed) this.writePageTable();
  }

  forget(evicted) {
    if (evicted === null) {
      this.entries.clear();
      return true;
    }
    let changed = false;
    for (const entry of evicted) {
      if (this.entries.get(entry.key) === entry) {
        this.entries.delete(entry.key);
        changed = true;
      }
    }
    return changed;
  }

  adopt(renderer) {
    for (const [key, entry] of this.entries) {
      if (!this.atlas.isResident(entry) || this.atlas.needsUpload(renderer, entry)) this.entries.delete(key);
    }
    for (const entry of this.atlas.entries.values()) {
      if (chunkKeyParts(entry.key).path === this.scale.path && !this.atlas.needsUpload(renderer, entry))
        this.entries.set(entry.key, entry);
    }
    this.writePageTable();
  }

  writePageTable() {
    this.pageTable.write(
      [...this.entries].map(([key, entry]) => ({ chunk_ZYX: chunkKeyParts(key).chunk_ZYX, slot: entry.slot })),
    );
  }

  dispose() {
    this.entries.clear();
    this.pageTable.dispose();
  }
}

function surfaceAtlasBytes(levels) {
  return [...new Set(levels.map((level) => level.atlas))].reduce(
    (total, atlas) => total + atlas.capacity * atlas.slotBytes,
    0,
  );
}

function disposeSurfaceLevels(levels, retained = []) {
  for (const level of levels) if (!retained.includes(level)) level.dispose();
  const retainedAtlases = new Set(retained.map((level) => level.atlas));
  for (const atlas of new Set(levels.map((level) => level.atlas))) {
    if (!retainedAtlases.has(atlas)) atlas.dispose();
  }
}

function surfaceVolumeAtlasByteLimits({ scales, totalBytes, maxTextureSize, fallbackCapacity }) {
  const slots = scales.map((scale) =>
    chunkAtlasSlotBytes({
      chunks_ZYX: scale.chunks_ZYX,
      dtype: scale.numericType.dtype,
      maxTextureSize,
    }),
  );
  const minimum = slots.reduce((total, bytes) => total + bytes, 0);
  if (minimum > totalBytes) {
    throw new Error(`Surface volume budget needs at least ${minimum} bytes for one chunk at every retained level.`);
  }
  if (scales.length === 1)
    return [fallbackCapacity === null ? totalBytes : Math.min(totalBytes, fallbackCapacity * slots[0])];
  const maximumFallbackBytes = totalBytes - slots[0];
  const requestedFallbackBytes =
    fallbackCapacity === null ? Math.floor(totalBytes * SURFACE_FALLBACK_BUDGET_SHARE) : fallbackCapacity * slots[1];
  const fallbackBytes = Math.min(maximumFallbackBytes, Math.max(slots[1], requestedFallbackBytes));
  return [totalBytes - fallbackBytes, fallbackBytes];
}

function surfaceVolumeAllocation(scales, totalBytes, maxLayers, maxTextureSize, fallbackCapacity) {
  const limits = surfaceVolumeAtlasByteLimits({ scales, totalBytes, maxTextureSize, fallbackCapacity });
  const atlasOptions = scales.map((scale, index) => ({
    chunks_ZYX: scale.chunks_ZYX,
    dtype: scale.numericType.dtype,
    byteLimit: Math.min(
      limits[index],
      ATLAS_BYTE_LIMIT,
      SURFACE_ATLAS_SLOT_LIMIT *
        chunkAtlasSlotBytes({
          chunks_ZYX: scale.chunks_ZYX,
          dtype: scale.numericType.dtype,
          maxTextureSize,
        }),
    ),
    packSlots: true,
    maxLayers,
    maxTextureSize,
  }));
  const capacities = atlasOptions.map((options) => chunkAtlasAllocation(options).capacity);
  if (
    scales.length === 2 &&
    scales.every(
      (scale) =>
        scale.numericType.dtype === scales[0].numericType.dtype &&
        scale.chunks_ZYX.every((value, axis) => value === scales[0].chunks_ZYX[axis]),
    )
  ) {
    const slotBytes = chunkAtlasSlotBytes(atlasOptions[0]);
    const pooled = {
      ...atlasOptions[0],
      byteLimit: Math.min(totalBytes, ATLAS_BYTE_LIMIT, SURFACE_ATLAS_SLOT_LIMIT * slotBytes),
      packSlots: true,
    };
    const allocation = chunkAtlasAllocation(pooled);
    if (allocation.capacity >= capacities[0] + capacities[1] - 1 && allocation.capacity > capacities[1]) {
      capacities[0] = Math.min(capacities[0], allocation.capacity - capacities[1]);
      return { capacities, atlasOptions: scales.map(() => pooled) };
    }
  }
  return { capacities, atlasOptions };
}

function copySurfaceLevel(renderer, previous, next, addresses) {
  const copies = addresses
    .slice(0, next.capacity)
    .map(({ chunk_ZYX }) => ({ chunk_ZYX, from: previous.entry(chunk_ZYX) }))
    .filter(
      ({ from }) =>
        from !== undefined && previous.atlas.isResident(from) && !previous.atlas.needsUpload(renderer, from),
    );
  if (copies.length === 0) return;
  if (previous.atlas.texture === next.atlas.texture) {
    for (const { from } of copies) next.atlas.get(from.key);
    return;
  }
  const pairs = copies
    .map(({ chunk_ZYX, from }) => ({
      chunk_ZYX,
      from,
      to: from.slot === null ? next.atlas.admit(from.key, null) : next.atlas.admitCopy(from.key),
    }))
    .filter(({ to }) => to !== null);
  if (pairs.length === 0) return;
  const slotted = pairs.filter(({ from }) => from.slot !== null);
  if (slotted.length > 0) next.atlas.copySlots(renderer, previous.atlas, slotted);
  for (const { chunk_ZYX, to } of pairs) next.place(chunk_ZYX, to);
  next.writePageTable();
}

export function surfaceFallbackCapacity(requiredCount, maximumCapacity) {
  return Math.min(maximumCapacity, Math.max(8, Math.ceil((requiredCount * 1.25) / 8) * 8));
}

export class SurfaceVolumeLevels {
  constructor({ scales, byteLimit, maxLayers, maxTextureSize, brickPool, namespace, onChange }) {
    this.brickPool = brickPool;
    this.namespace = namespace;
    this.onChange = onChange;
    this.byteLimit = byteLimit;
    this.maxLayers = maxLayers;
    this.maxTextureSize = maxTextureSize;
    this.shown = [];
    this.cached = null;
    this.disposed = false;
    this.commit(this.stage(scales, null), [], null);
  }

  atlasChanged(atlas, evicted) {
    if (this.disposed) return;
    let published = false;
    for (const level of this.cached === null ? this.shown : [...this.shown, this.cached]) {
      if (level.atlas !== atlas || !level.forget(evicted)) continue;
      published = true;
      level.writePageTable();
    }
    if (published) this.onChange(evicted);
  }

  setByteLimit(byteLimit) {
    if (byteLimit === this.byteLimit) return;
    this.byteLimit = byteLimit;
    if (this.cached !== null && this.atlasBytes() > byteLimit) {
      disposeSurfaceLevels([this.cached], this.shown);
      this.cached = null;
    }
  }

  admitPlans(plans) {
    const keys = this.shown.map((level, index) => {
      level.addresses = plans[index].addresses;
      level.coverage = plans[index];
      return level.addresses.map(({ chunk_ZYX }) => chunkKey(level.scale, chunk_ZYX));
    });
    const atlasKeys = new Map();
    for (const [index, level] of [...this.shown.entries()].toReversed()) {
      if (!atlasKeys.has(level.atlas)) atlasKeys.set(level.atlas, []);
      atlasKeys.get(level.atlas).push(...keys[index]);
    }
    for (const [atlas, working] of atlasKeys) atlas.beginFrame(working);
    return this.shown.map((level, index) => {
      level.prune();
      return {
        level,
        workingAddresses: level.addresses.filter((_, address) => level.atlas.inWorkingSet(keys[index][address])),
      };
    });
  }

  stage(scales, fallbackCapacity) {
    const byteLimit = this.byteLimit;
    const { capacities, atlasOptions } = surfaceVolumeAllocation(
      scales,
      byteLimit,
      this.maxLayers,
      this.maxTextureSize,
      fallbackCapacity,
    );
    if (
      scales.length === this.shown.length &&
      scales.every(
        (scale, index) => scale === this.shown[index].scale && capacities[index] === this.shown[index].capacity,
      ) &&
      this.atlasBytes() <= byteLimit
    )
      return null;

    const available = this.cached === null ? this.shown : [...this.shown, this.cached];
    const availableAtlases = new Set(available.map((level) => level.atlas));
    const atlases = new Map();
    const created = [];
    const createdAtlases = [];
    const next = [];
    try {
      for (const [index, scale] of scales.entries()) {
        const options = atlasOptions[index];
        let atlas = atlases.get(options);
        if (atlas === undefined) {
          const allocation = chunkAtlasAllocation(options);
          const fits = (candidate) =>
            ![...atlases.values()].includes(candidate) &&
            (scales.length === 1
              ? candidate.capacity >= capacities[index] && candidate.capacity * candidate.slotBytes <= byteLimit
              : candidate.capacity === allocation.capacity &&
                candidate.reservedSlotsPerLayer === allocation.slotsPerLayer) &&
            candidate.dtype === options.dtype &&
            candidate.chunks_ZYX.every((value, axis) => value === options.chunks_ZYX[axis]);
          atlas =
            available.find((level) => level.scale === scale && fits(level.atlas))?.atlas ??
            [...availableAtlases].find(fits);
          if (atlas === undefined) {
            atlas = this.brickPool.acquire(this.namespace, options, (changed, evicted) =>
              this.atlasChanged(changed, evicted),
            );
            createdAtlases.push(atlas);
          }
          atlases.set(options, atlas);
        }
        const existing = this.shown.find((level) => level.scale === scale && level.atlas === atlas);
        if (existing !== undefined) {
          next.push(existing);
        } else {
          const level = new SurfaceVolumeLevel(scale, capacities[index], atlas, this.maxTextureSize);
          created.push(level);
          next.push(level);
        }
      }
    } catch (error) {
      for (const level of created) level.pageTable.dispose();
      for (const atlas of createdAtlases) atlas.dispose();
      throw error;
    }
    const cached =
      next.length === 1
        ? (available.find(
            (level) =>
              !next.includes(level) &&
              level.scale !== next[0].scale &&
              surfaceAtlasBytes([...next, level]) <= byteLimit,
          ) ?? null)
        : null;
    return { available, cached, next, capacities, byteLimit, created };
  }

  commit(stage, plans, renderer) {
    const next = stage.cached === null ? stage.next : [...stage.next, stage.cached];
    const fallback = stage.next.at(-1);
    const previousFallback = stage.available.find(
      (level) => level.scale === fallback.scale && level.atlas !== fallback.atlas,
    );
    const retained = stage.available.filter((previous) => next.some((level) => level.atlas === previous.atlas));
    const preserveFallback =
      previousFallback !== undefined && surfaceAtlasBytes([...retained, previousFallback, fallback]) <= stage.byteLimit;
    // Migration can borrow an atlas whose previous detail plan filled every slot.
    // Protect the incoming plans before copying, just as before ordinary uploads.
    for (const atlas of preserveFallback ? [fallback.atlas] : []) {
      const keys = new Set();
      for (const [index, level] of stage.next.entries()) {
        if (level.atlas !== atlas) continue;
        const previousLevels = stage.available.filter((previous) => previous.scale === level.scale);
        for (const { chunk_ZYX } of plans[index].addresses) {
          for (const previous of previousLevels) {
            const entry = previous.entry(chunk_ZYX);
            if (entry !== undefined) keys.add(entry.key);
          }
        }
      }
      atlas.beginFrame([...keys]);
    }
    disposeSurfaceLevels(stage.available, preserveFallback ? [...next, previousFallback] : next);
    if (preserveFallback) {
      try {
        copySurfaceLevel(renderer, previousFallback, fallback, plans.at(-1).addresses);
      } catch (error) {
        disposeSurfaceLevels(next, [previousFallback]);
        this.shown = [previousFallback];
        this.cached = null;
        throw error;
      }
      disposeSurfaceLevels([previousFallback], next);
    }
    // Releasing a shared lease can resize the destination and invalidate copied pages.
    for (const level of stage.created) level.adopt(renderer);
    for (const [index, level] of stage.next.entries()) level.capacity = stage.capacities[index];
    this.shown = stage.next;
    this.cached = stage.cached;
    if (this.cached !== null && !this.shown.some((level) => level.atlas === this.cached.atlas)) {
      this.cached.atlas.beginFrame([]);
    }
  }

  capacities(scales, fallbackCapacity) {
    return surfaceVolumeAllocation(scales, this.byteLimit, this.maxLayers, this.maxTextureSize, fallbackCapacity)
      .capacities;
  }

  retainedCapacities(scales) {
    if (scales.length !== this.shown.length || scales.some((scale, index) => scale !== this.shown[index].scale))
      return null;
    return surfaceAtlasBytes(this.shown) <= this.byteLimit ? this.shown.map((level) => level.capacity) : null;
  }

  maximumCapacity(scale) {
    return chunkAtlasAllocation({
      chunks_ZYX: scale.chunks_ZYX,
      dtype: scale.numericType.dtype,
      byteLimit: Math.min(
        this.byteLimit,
        ATLAS_BYTE_LIMIT,
        SURFACE_ATLAS_SLOT_LIMIT *
          chunkAtlasSlotBytes({
            chunks_ZYX: scale.chunks_ZYX,
            dtype: scale.numericType.dtype,
            maxTextureSize: this.maxTextureSize,
          }),
      ),
      maxLayers: this.maxLayers,
      maxTextureSize: this.maxTextureSize,
      packSlots: true,
    }).capacity;
  }

  allocatedBytes() {
    return (
      this.atlasBytes() +
      this.shown.reduce((total, level) => total + level.pageTable.allocatedBytes(), 0) +
      (this.cached?.pageTable.allocatedBytes() ?? 0)
    );
  }

  atlasBytes(includeCache = true) {
    return surfaceAtlasBytes(includeCache && this.cached !== null ? [...this.shown, this.cached] : this.shown);
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    disposeSurfaceLevels(this.cached === null ? this.shown : [...this.shown, this.cached]);
    this.shown = [];
    this.cached = null;
  }
}
