import { chunkAtlasLayout } from "../gpu/chunk-atlas.js";
import { STRIP_TEXEL_BYTES } from "./source.js";
import { SurfaceStripAtlas, surfaceStripBounds } from "./strip-atlas.js";
import { yieldTask } from "./yield-task.js";

export class SurfacePositionImage {
  static validateCapacity(source, rendering, availableBytes = rendering.availableBytes()) {
    const { maxLayers, maxTextureSize } = rendering.textureLimits;
    const layers = source.stripCount;
    const layout = chunkAtlasLayout(source.stripRows * source.grid_HW[1], maxTextureSize);
    const neededBytes = layers * layout.width * layout.height * STRIP_TEXEL_BYTES;
    if (layers > maxLayers)
      throw new Error(`Surface geometry needs ${layers} GPU array layers; this device supports ${maxLayers}.`);
    if (neededBytes > availableBytes)
      throw new Error(
        `Surface geometry needs ${(neededBytes / 2 ** 20).toFixed(1)} MiB; ${(Math.max(0, availableBytes) / 2 ** 20).toFixed(1)} MiB is available. Increase the Live GPU budget in Settings.`,
      );
    return neededBytes;
  }

  constructor(source) {
    this.source = source;
    this.atlas = null;
    this.rendering = null;
    this.consumers = new Set();
    this.controller = null;
    this.queue = [];
    this.nativeDemand = [];
    this.frame = null;
    this.revision = 0;
    this.error = null;
  }

  acquire(consumer, rendering) {
    if (this.atlas !== null && this.rendering !== rendering)
      throw new Error("Surface positions require the viewer's shared WebGL renderer.");
    if (this.atlas === null) {
      SurfacePositionImage.validateCapacity(this.source, rendering);
      this.atlas = new SurfaceStripAtlas({
        width: this.source.grid_HW[1],
        stripRows: this.source.stripRows,
        stripCount: this.source.stripCount,
        maxTextureSize: rendering.textureLimits.maxTextureSize,
      });
      this.rendering = rendering;
      rendering.addAllocation(this);
      this.controller = new AbortController();
      const controller = this.controller;
      void this.loadNative(controller).catch((error) => {
        if (error.name === "AbortError") return;
        this.error = error;
        for (const member of this.consumers) member.geometryChanged();
      });
    }
    this.consumers.add(consumer);
    return this.atlas;
  }

  async loadNative(controller) {
    this.nativeDemand = Array.from({ length: this.source.stripCount }, (_, index) => index);
    await Promise.all(
      [0, 1].map(async () => {
        while (this.nativeDemand.length > 0) {
          const index = this.nativeDemand.shift();
          const data_HWRGBA = await this.source.getStrip(index, controller.signal);
          controller.signal.throwIfAborted();
          const atlas = this.atlas;
          const entry = atlas.admit(index, data_HWRGBA, surfaceStripBounds(data_HWRGBA, atlas.width, atlas.stripRows));
          if (!entry.uploaded) this.queue.push(entry);
          this.schedule();
          await yieldTask(controller.signal);
        }
      }),
    );
  }

  prioritize(strips) {
    const pending = new Set(this.nativeDemand);
    const visible = new Set(strips);
    this.nativeDemand = [
      ...strips.filter((index) => pending.has(index)),
      ...this.nativeDemand.filter((index) => !visible.has(index)),
    ];
    this.queue = [
      ...this.queue.filter((entry) => visible.has(entry.stripIndex)),
      ...this.queue.filter((entry) => !visible.has(entry.stripIndex)),
    ];
  }

  schedule() {
    if (this.frame !== null || this.atlas === null) return;
    this.frame = true;
    this.rendering.uploadBudget.schedule(this, 1, (timestamp) => this.upload(timestamp));
  }

  upload(timestamp) {
    this.frame = null;
    const budget = this.rendering.uploadBudget;
    let available = budget.available(timestamp);
    let changed = false;
    while (this.queue.length > 0 && available >= STRIP_TEXEL_BYTES) {
      const entry = this.queue[0];
      const result = this.atlas.upload(this.rendering.renderer, entry, available);
      available -= result.uploadedBytes;
      budget.spend(result.uploadedBytes);
      changed ||= result.complete;
      if (result.complete) this.queue.shift();
    }
    if (changed) {
      this.revision++;
      for (const consumer of this.consumers) consumer.geometryChanged();
    }
    if (this.queue.length > 0) this.schedule();
  }

  async changed(rowStart, rowStop) {
    if (this.atlas !== null) {
      const { atlas, source, controller } = this;
      const [rows, width] = [source.stripRows, source.grid_HW[1]];
      const rewritten = [];
      for (let index = Math.floor(rowStart / rows); index * rows < rowStop; index++) {
        const data_HWRGBA = await source.getStrip(index, controller.signal);
        const span = [rowStart, rowStop].map((row) => Math.min(rows, Math.max(0, row - index * rows)) * width);
        rewritten.push(atlas.rewrite(index, data_HWRGBA, surfaceStripBounds(data_HWRGBA, width, rows), ...span));
      }
      const queued = new Set(this.queue);
      this.queue.unshift(...rewritten.filter((entry) => entry !== null && !queued.has(entry)));
      this.schedule();
    }
    this.revision++;
    for (const consumer of this.consumers) consumer.geometryChanged();
  }

  allocatedBytes() {
    return this.atlas.allocatedBytes();
  }

  release(consumer) {
    this.consumers.delete(consumer);
    if (this.consumers.size > 0) return;
    if (this.atlas === null) return;
    this.controller.abort();
    this.controller = null;
    if (this.frame !== null) this.rendering.uploadBudget.cancel(this);
    this.frame = null;
    this.queue = [];
    this.nativeDemand = [];
    this.atlas.dispose();
    this.rendering.removeAllocation(this);
    this.atlas = null;
    this.rendering = null;
  }
}
