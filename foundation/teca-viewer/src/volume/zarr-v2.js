import { createNumericArray, decodeNumericChunk } from "./numeric-type.js";
import { SourceDecoder } from "./source-decoder.js";
import { BLOSC_MAX_OVERHEAD, openDirectBloscDecoder } from "./zarr-codecs.js";

class BoundedV2Array {
  constructor(store, decoder, decodeBlosc, fillChunks, encodedByteLimit) {
    this.store = store;
    this.decoder = decoder;
    this.decodeBlosc = decodeBlosc;
    this.fillChunks = fillChunks;
    this.encodedByteLimit = encodedByteLimit;
  }

  async getChunk(coordinates, scale, signal) {
    const key = `${scale.path}/${coordinates.join(scale.chunkSeparator)}`;
    const bytes = await this.store.get(`/${key}`, { signal, byteLimit: this.encodedByteLimit });
    const length = scale.storageChunks.reduce((total, size) => total * size, 1);
    if (bytes === undefined) {
      const fill = createNumericArray(scale.numericType, length);
      fill.fill(scale.fillValue);
      this.fillChunks.add(fill);
      return fill;
    }
    const expectedBytes = length * scale.numericType.itemSize;
    const decoded =
      scale.compressor?.id === "gzip"
        ? await this.decoder.run({ kind: "gzip", bytes, expectedBytes })
        : this.decodeBlosc === null
          ? bytes
          : await this.decodeBlosc(bytes, signal);
    if (decoded.length !== expectedBytes) {
      throw new Error(`Chunk ${key} has ${decoded.length} bytes, expected ${expectedBytes}.`);
    }
    return decodeNumericChunk(decoded, scale.numericType.dtype);
  }
}

export async function openDirectZarrV2(scales, store) {
  const blosc = new Map();
  const encodedByteLimits = new Map();
  for (const scale of scales) {
    const decodedBytes = scale.storageChunks.reduce((total, size) => total * size, scale.numericType.itemSize);
    if (scale.compressor === null) {
      encodedByteLimits.set(scale.path, decodedBytes);
      continue;
    }
    if (scale.compressor.id === "gzip") {
      if (
        Object.keys(scale.compressor).some((key) => key !== "id" && key !== "level") ||
        !Number.isInteger(scale.compressor.level) ||
        scale.compressor.level < 0 ||
        scale.compressor.level > 9
      )
        throw new Error("Malformed gzip compressor declaration.");
      encodedByteLimits.set(scale.path, decodedBytes + Math.ceil(decodedBytes / 1000) + 128);
      continue;
    }
    blosc.set(
      scale.path,
      await openDirectBloscDecoder(scale.compressor, decodedBytes, scale.path, {
        worker: true,
      }),
    );
    encodedByteLimits.set(scale.path, decodedBytes + BLOSC_MAX_OVERHEAD);
  }
  const decoder = scales.some((scale) => scale.compressor?.id === "gzip") ? new SourceDecoder() : null;
  const fillChunks = new WeakSet();
  const arrays = new Map(
    scales.map((scale) => [
      scale.path,
      new BoundedV2Array(store, decoder, blosc.get(scale.path) ?? null, fillChunks, encodedByteLimits.get(scale.path)),
    ]),
  );
  return { arrays, encodedByteLimits, fillChunks, decoder };
}
