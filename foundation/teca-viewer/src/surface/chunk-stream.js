import { chunkKey, visibleChunkPriority } from "../gpu/addresses.js";
import { BrickReads } from "../gpu/brick-reads.js";
import { presentationChunkRequest } from "../gpu/presentation-bricks.js";
import { UPLOAD_QUANTUM_BYTES } from "../gpu/upload-budget.js";

export class SurfaceChunkStream {
  constructor({ volume, selectedChannel, advanceGeneration, levels, renderer, uploadBudget, onUploaded, onError }) {
    this.volume = volume;
    this.selectedChannel = selectedChannel;
    this.advanceGeneration = advanceGeneration;
    this.levels = levels;
    this.renderer = renderer;
    this.uploadBudget = uploadBudget;
    this.onUploaded = onUploaded;
    this.reads = new BrickReads({
      volume,
      channel: selectedChannel,
      wants: ({ key, level }) =>
        !this.disposed &&
        this.levels.shown.includes(level) &&
        level.atlas.inWorkingSet(key) &&
        level.atlas.get(key) === null,
      deliver: ({ key, level, chunk_ZYX }, chunk_DHW) => {
        const entry = level.atlas.admit(key, this.rewritten.get(key) ?? chunk_DHW);
        this.rewritten.delete(key);
        if (entry !== null) this.queueUpload(level, chunk_ZYX, entry);
      },
      onError,
    });
    this.working = [];
    this.uploads = [];
    this.queued = new WeakSet();
    this.rewritten = new Map();
    this.uploadFramePending = false;
    this.disposed = false;
  }

  replace(working) {
    this.working = working;
    this.volume.beginNavigation(this.advanceGeneration());
    for (const { level, workingAddresses } of working) {
      const requests = new Map();
      for (const { chunk_ZYX, priority } of workingAddresses) {
        const request = presentationChunkRequest(level.scale, chunk_ZYX);
        const key = request.chunk_ZYX.join("/");
        const previous = requests.get(key);
        if (previous === undefined || visibleChunkPriority(priority) < previous.priority)
          requests.set(key, { chunk_ZYX: request.chunk_ZYX, priority: visibleChunkPriority(priority) });
      }
      const scale = level.scale.sourceScale ?? level.scale;
      this.volume.retainChunkRequests([...requests.values()], scale, this.selectedChannel);
    }
    this.volume.cancelPendingChunks(0, Infinity, true);
    this.stream();
  }

  levelsReplaced() {
    const shown = new Set(this.levels.shown);
    this.uploads = this.uploads.filter((task) => shown.has(task.level));
    this.queued = new WeakSet(this.uploads.map((task) => task.entry));
  }

  stream() {
    const bricks = [];
    for (const { level, workingAddresses } of this.working.toReversed()) {
      for (const { chunk_ZYX, priority } of workingAddresses) {
        const key = chunkKey(level.scale, chunk_ZYX);
        if (!level.atlas.inWorkingSet(key)) continue;
        const entry = level.atlas.get(key);
        if (entry !== null) this.queueUpload(level, chunk_ZYX, entry);
        else bricks.push({ key, level, scale: level.scale, chunk_ZYX, priority: visibleChunkPriority(priority) });
      }
    }
    this.reads.want(bricks.sort((left, right) => left.priority - right.priority));
  }

  refresh(changes) {
    const { shown, cached } = this.levels;
    const atlases = new Set([...shown, ...(cached === null ? [] : [cached])].map((level) => level.atlas));
    for (const { scale, chunk_ZYX, data_DHW, start, end } of changes) {
      const key = chunkKey(scale, chunk_ZYX);
      for (const atlas of atlases) atlas.refresh(key, data_DHW, start, end);
      for (const level of shown) {
        if (level.scale !== scale) continue;
        const entry = level.atlas.held(key);
        if (entry === null) this.rewritten.set(key, data_DHW);
        else this.queueUpload(level, chunk_ZYX, entry);
      }
    }
  }

  pending() {
    return this.reads.pending();
  }

  queueUpload(level, chunk_ZYX, entry) {
    if ((level.entry(chunk_ZYX) === entry && !level.atlas.needsUpload(this.renderer, entry)) || this.queued.has(entry))
      return;
    this.queued.add(entry);
    this.uploads.push({ level, chunk_ZYX, entry });
    this.scheduleUploads();
  }

  scheduleUploads() {
    if (this.uploadFramePending) return;
    this.uploadFramePending = true;
    this.uploadBudget.schedule(this, 0, (timestamp) => this.processUploads(timestamp));
  }

  processUploads(timestamp) {
    this.uploadFramePending = false;
    if (this.uploadBudget.available(timestamp) < UPLOAD_QUANTUM_BYTES) {
      this.scheduleUploads();
      return;
    }
    const dirtyLevels = new Set();
    const prunedAtlases = new Set();
    let uploaded = false;
    while (this.uploads.length > 0) {
      const remainingBytes = this.uploadBudget.available(timestamp);
      if (remainingBytes === 0) break;
      const task = this.uploads[0];
      const atlas = task.level.atlas;
      this.queued.delete(task.entry);
      if (!atlas.inWorkingSet(task.entry.key)) {
        this.uploads.shift();
        if (atlas.isResident(task.entry) && task.level.entry(task.chunk_ZYX) === undefined)
          atlas.remove(task.entry.key);
        continue;
      }
      if (!prunedAtlases.has(atlas)) {
        for (const level of this.levels.shown) if (level.atlas === atlas) level.prune();
        prunedAtlases.add(atlas);
      }
      const upload = atlas.needsUpload(this.renderer, task.entry)
        ? atlas.uploadPart(this.renderer, task.entry, remainingBytes)
        : { uploadedBytes: 0, complete: true };
      this.uploadBudget.spend(upload.uploadedBytes);
      if (!upload.complete) {
        this.queued.add(task.entry);
        if (upload.uploadedBytes === 0) break;
        continue;
      }
      this.uploads.shift();
      if (!atlas.isResident(task.entry)) continue;
      uploaded = true;
      if (task.level.entry(task.chunk_ZYX) === task.entry) continue;
      task.level.place(task.chunk_ZYX, task.entry);
      dirtyLevels.add(task.level);
    }
    for (const level of dirtyLevels) level.writePageTable();
    if (uploaded) this.onUploaded();
    if (this.uploads.length > 0) this.scheduleUploads();
  }

  dispose() {
    this.disposed = true;
    if (this.uploadFramePending) this.uploadBudget.cancel(this);
    this.uploads = [];
    this.reads.dispose();
    this.volume.beginNavigation(this.advanceGeneration());
    this.volume.cancelPendingChunks(0, Infinity, true);
  }
}
