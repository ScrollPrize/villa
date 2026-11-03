import { uint8Scales } from "./memory-volume.js";

export class MadeVolume {
  constructor(name, levels, make) {
    this.rootUrl = `made:${name}/`;
    this.scales = uint8Scales(name, levels);
    this.metadata = { displayEncoding: { kind: "identity_uint8" } };
    this.make = make;
    this.zeros = levels.map(({ chunks_ZYX }) => new Uint8Array(chunks_ZYX[0] * chunks_ZYX[1] * chunks_ZYX[2]));
  }

  beginNavigation() {}

  retainChunkRequests() {}

  cancelPendingChunks() {}

  async getChunk(chunkZ, chunkY, chunkX, _channel, scale, signal) {
    signal.throwIfAborted();
    const level = this.scales.indexOf(scale);
    const chunk_DHW = await this.make(level, [chunkZ, chunkY, chunkX], signal);
    if (chunk_DHW === null) return this.zeros[level];
    if (!(chunk_DHW instanceof Uint8Array) || chunk_DHW.length !== this.zeros[level].length)
      throw new Error(
        `Chunk ${chunkZ},${chunkY},${chunkX} of level ${level} of ${this.rootUrl} was made of ${chunk_DHW.length} ` +
          `values, not the ${this.zeros[level].length} bytes of a whole chunk.`,
      );
    return chunk_DHW;
  }

  isFill(chunk_DHW) {
    return this.zeros.includes(chunk_DHW);
  }
}
