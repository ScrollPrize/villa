import {
  BRICK_CUT_BYTES,
  CHUNK_READS_IN_FLIGHT,
  PREFETCH_CHUNK_PRIORITY,
  PREFETCH_READS_IN_FLIGHT,
} from "../config.js";
import { chunkKey } from "./addresses.js";
import { atlasFormatForDtype } from "./atlas-format.js";
import { ChunkPreparation } from "./chunk-preparation.js";
import { presentationChunkRequest } from "./presentation-bricks.js";

export class BrickReads {
  constructor({ volume, channel, wants, deliver, onError }) {
    this.volume = volume;
    this.channel = channel;
    this.wants = wants;
    this.deliver = deliver;
    this.onError = onError;
    this.queue = [];
    this.reads = new Map();
    this.reading = new Map();
    this.active = 0;
    this.prefetching = 0;
    this.heldBytes = 0;
    this.chunkPreparation = null;
  }

  want(bricks) {
    this.queue = [];
    this.prune();
    const groups = new Map();
    for (const brick of bricks) {
      const held = this.reading.get(brick.key);
      if (held !== undefined && this.wants(held)) continue;
      const request = presentationChunkRequest(brick.scale, brick.chunk_ZYX);
      const key = chunkKey(request.scale, request.chunk_ZYX);
      const entry = { brick, localStart_ZYX: request.localStart_ZYX };
      const read = this.reads.get(key);
      if (read !== undefined) {
        read.entries.push(entry);
        this.reading.set(brick.key, brick);
        continue;
      }
      let group = groups.get(key);
      if (group === undefined) {
        const scale = request.scale;
        // A contiguous channel view keeps the whole decoded storage buffer, so the bytes count that buffer.
        const shape = scale.axes.slice(-3).join(",") === "z,y,x" ? scale.storageChunks : scale.chunks_ZYX;
        const bytes = shape.reduce(
          (product, value) => product * value,
          atlasFormatForDtype(scale.numericType.dtype).ArrayType.BYTES_PER_ELEMENT,
        );
        if (bytes > BRICK_CUT_BYTES)
          throw new Error(`A ${bytes}-byte chunk of ${scale.path} exceeds the ${BRICK_CUT_BYTES}-byte cut budget.`);
        group = { key, request, bytes, entries: [] };
        groups.set(key, group);
      }
      group.entries.push(entry);
    }
    this.queue = [...groups.values()];
    this.pump();
  }

  prune() {
    this.queue = this.queue.filter((group) => {
      group.entries = group.entries.filter(({ brick }) => this.wants(brick));
      return group.entries.length > 0;
    });
    for (const read of [...this.reads.values()]) {
      if (read.arrived) continue;
      read.entries = read.entries.filter(({ brick }) => {
        if (this.wants(brick)) return true;
        this.forget(brick);
        return false;
      });
      if (read.entries.length > 0) continue;
      this.detach(read);
      read.controller.abort();
    }
    this.pump();
  }

  pending() {
    return this.reading.size + this.queue.reduce((total, group) => total + group.entries.length, 0);
  }

  dispose() {
    this.queue = [];
    for (const read of [...this.reads.values()]) {
      this.detach(read);
      read.controller.abort();
    }
    this.chunkPreparation?.dispose();
  }

  pump() {
    while (this.queue.length > 0 && this.active < CHUNK_READS_IN_FLIGHT) {
      const group = this.queue[0];
      const priority = Math.min(...group.entries.map(({ brick }) => brick.priority));
      const prefetch = priority >= PREFETCH_CHUNK_PRIORITY;
      if (this.heldBytes + group.bytes > BRICK_CUT_BYTES) return;
      if (prefetch && this.prefetching >= PREFETCH_READS_IN_FLIGHT) return;
      this.queue.shift();
      const read = { ...group, priority, prefetch, controller: new AbortController(), arrived: false };
      this.reads.set(read.key, read);
      for (const { brick } of read.entries) this.reading.set(brick.key, brick);
      this.active += 1;
      this.prefetching += Number(prefetch);
      this.heldBytes += read.bytes;
      this.read(read).finally(() => {
        this.detach(read);
        this.active -= 1;
        this.prefetching -= Number(prefetch);
        this.pump();
      });
    }
  }

  async read(read) {
    const taken = [];
    try {
      const { request } = read;
      const source_DHW = await this.volume.getChunk(
        ...request.chunk_ZYX,
        this.channel,
        request.scale,
        read.controller.signal,
        read.priority,
      );
      read.arrived = true;
      const fill = this.volume.isFill(source_DHW);
      while (read.entries.length > 0) {
        read.controller.signal.throwIfAborted();
        const { brick, localStart_ZYX } = read.entries[0];
        let chunk_DHW = fill ? null : source_DHW;
        if (!fill && localStart_ZYX !== null && this.wants(brick)) {
          this.chunkPreparation ??= new ChunkPreparation();
          chunk_DHW = await this.chunkPreparation.copy(
            source_DHW,
            request.scale.chunks_ZYX,
            brick.scale.chunks_ZYX,
            localStart_ZYX,
            read.controller.signal,
          );
        }
        read.entries.shift();
        if (!this.wants(brick)) this.forget(brick);
        else
          taken.push(
            new Promise((resolve) => resolve(this.deliver(brick, chunk_DHW))).finally(() => this.forget(brick)),
          );
      }
      this.detach(read);
      await Promise.all(taken);
    } catch (error) {
      if (error.name !== "AbortError") this.onError(error);
      for (const { brick } of read.entries) this.forget(brick);
      read.entries = [];
    }
  }

  forget(brick) {
    if (this.reading.get(brick.key) === brick) this.reading.delete(brick.key);
  }

  detach(read) {
    if (this.reads.get(read.key) !== read) return;
    this.reads.delete(read.key);
    this.heldBytes -= read.bytes;
  }
}
