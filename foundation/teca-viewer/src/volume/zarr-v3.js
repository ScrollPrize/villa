import { open, root, Array as ZarritaArray } from "../../vendor/zarrita.module.js";
import { DECODER_REQUEST_LIMIT, ZARR_METADATA_BYTE_LIMIT, ZARR_METADATA_TOTAL_BYTE_LIMIT } from "../config.js";

import { declaredDatasetPaths } from "./ngff.js";
import { typedArrayConstructorFor } from "./numeric-type.js";
import { prepareDirectCodecPipeline } from "./zarr-codecs.js";

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

function cOrderChunkData(chunk, scale, key) {
  if (!Array.isArray(chunk.shape) || !Array.isArray(chunk.stride)) {
    throw new Error(`Decoded Zarr v3 chunk ${key} has no shape or stride.`);
  }
  if (
    chunk.shape.length !== scale.storageChunks.length ||
    chunk.shape.some((value, index) => value !== scale.storageChunks[index])
  ) {
    throw new Error(`Decoded Zarr v3 chunk ${key} shape does not match its declared chunk grid.`);
  }
  const expectedLength = scale.storageChunks.reduce((total, size) => total * size, 1);
  const TypedArray = typedArrayConstructorFor(scale.numericType);
  if (!(chunk.data instanceof TypedArray) || chunk.data.length !== expectedLength) {
    throw new Error(`Decoded Zarr v3 chunk ${key} data does not match its declared dtype and shape.`);
  }
  const expectedStride = new Array(chunk.shape.length);
  let stride = 1;
  for (let axis = chunk.shape.length - 1; axis >= 0; axis--) {
    expectedStride[axis] = stride;
    stride *= chunk.shape[axis];
  }
  if (
    chunk.stride.length !== chunk.shape.length ||
    chunk.stride.some((value) => !Number.isSafeInteger(value) || value < 1)
  ) {
    throw new Error(`Decoded Zarr v3 chunk ${key} has an invalid stride.`);
  }
  if (chunk.stride.every((value, axis) => value === expectedStride[axis])) return chunk.data;
  const output = new TypedArray(expectedLength);
  for (let outputIndex = 0; outputIndex < expectedLength; outputIndex++) {
    let remainder = outputIndex;
    let inputIndex = 0;
    for (let axis = 0; axis < chunk.shape.length; axis++) {
      const coordinate = Math.floor(remainder / expectedStride[axis]);
      remainder %= expectedStride[axis];
      inputIndex += coordinate * chunk.stride[axis];
    }
    if (inputIndex >= chunk.data.length) throw new Error(`Decoded Zarr v3 chunk ${key} stride exceeds its data.`);
    output[outputIndex] = chunk.data[inputIndex];
  }
  return output;
}

class BoundedV3Array {
  constructor(pool, path, metadata, decoder) {
    this.pool = pool;
    this.path = path;
    this.metadata = metadata;
    this.decoder = decoder;
  }

  async getChunk(coordinates, scale, signal) {
    const decoder = this.decoder;
    this.pool.admit();
    return cOrderChunkData(
      await decoder.getChunk(coordinates, { signal }),
      scale,
      `${this.path}/${coordinates.join("/")}`,
    );
  }

  reset(store) {
    this.decoder = new ZarritaArray(store, this.path, this.metadata);
  }
}

class DecoderEpochPool {
  constructor(store) {
    this.store = store;
    this.arrays = [];
    this.remainingRequests = DECODER_REQUEST_LIMIT;
  }

  wrap(path, metadata, decoder) {
    const array = new BoundedV3Array(this, path, metadata, decoder);
    this.arrays.push(array);
    return array;
  }

  admit() {
    this.remainingRequests -= 1;
    if (this.remainingRequests > 0) return;
    this.reset();
  }

  reset() {
    for (const array of this.arrays) array.reset(this.store);
    this.remainingRequests = DECODER_REQUEST_LIMIT;
  }
}

function validateV3DatasetPaths(paths) {
  for (const path of paths) {
    for (const part of path.split("/")) {
      if (/^\.+$/.test(part) || part.startsWith("__") || part === "zarr.json") {
        throw new Error(`Invalid Zarr v3 node name in dataset path: ${part}.`);
      }
    }
  }
}

function decodeMetadata(bytes, url) {
  try {
    return JSON.parse(textDecoder.decode(bytes));
  } catch (error) {
    throw new Error(`Failed to parse JSON metadata ${url}: ${error.message}`);
  }
}

function encodeMetadata(metadata) {
  return textEncoder.encode(JSON.stringify(metadata));
}

export async function openDirectZarrV3(store, signal) {
  const rootBytes = await store.get("/zarr.json", { signal });
  if (rootBytes === undefined) return null;
  if (rootBytes.byteLength > ZARR_METADATA_BYTE_LIMIT) {
    throw new Error(`Zarr v3 root metadata exceeds the ${ZARR_METADATA_BYTE_LIMIT}-byte object limit.`);
  }
  let metadataBytes = rootBytes.byteLength;
  const rootMetadata = decodeMetadata(rootBytes, store.resolve("/zarr.json"));
  if (rootMetadata?.zarr_format !== 3 || rootMetadata.node_type !== "group") {
    throw new Error("Zarr v3 root metadata must declare format 3 and node type group.");
  }

  const group = await open.v3(root(store), { kind: "group", signal });
  const ome = group.attrs.ome;
  if (ome === null || typeof ome !== "object" || Array.isArray(ome) || ome.version !== "0.5") {
    throw new Error("Direct Zarr v3 sources must declare OME-Zarr 0.5 metadata in attributes.ome.");
  }
  const attrs = {
    ...group.attrs,
    ...ome,
    multiscales: ome.multiscales,
    ...(ome.omero === undefined ? {} : { omero: ome.omero }),
  };
  const paths = declaredDatasetPaths(attrs);
  validateV3DatasetPaths(paths);
  store.metadata.delete("/zarr.json");
  const arrays = new Map();
  const zarrays = Object.create(null);
  const encodedByteLimits = new Map();
  const decoderPool = new DecoderEpochPool(store);
  let retainedMetadataBytes = 0;
  for (const path of paths) {
    const metadataKey = `/${path}/zarr.json`;
    const bytes = await store.get(metadataKey, { signal });
    if (bytes === undefined) throw new Error(`Zarr v3 array metadata for ${path} does not exist.`);
    if (bytes.byteLength > ZARR_METADATA_BYTE_LIMIT) {
      throw new Error(`Zarr v3 array metadata for ${path} exceeds the ${ZARR_METADATA_BYTE_LIMIT}-byte object limit.`);
    }
    metadataBytes += bytes.byteLength;
    if (metadataBytes > ZARR_METADATA_TOTAL_BYTE_LIMIT) {
      throw new Error(`Zarr v3 metadata exceeds the ${ZARR_METADATA_TOTAL_BYTE_LIMIT}-byte aggregate limit.`);
    }
    const metadata = decodeMetadata(bytes, store.resolve(`/${path}/zarr.json`));
    store.metadata.delete(metadataKey);
    const encodedByteLimit = await prepareDirectCodecPipeline(metadata, path);
    if (
      metadata.storage_transformers !== undefined &&
      (!Array.isArray(metadata.storage_transformers) || metadata.storage_transformers.length > 0)
    ) {
      throw new Error(`Zarr v3 storage transformers at ${path} are not supported.`);
    }
    store.setDataLimit(`/${path}/`, encodedByteLimit);
    encodedByteLimits.set(path, encodedByteLimit);
    const decoderPath = `/${path
      .split("/")
      .map((part) => encodeURIComponent(part))
      .join("/")}`;
    const decoderMetadata = {
      zarr_format: metadata.zarr_format,
      node_type: metadata.node_type,
      shape: metadata.shape,
      data_type: metadata.data_type,
      chunk_grid: metadata.chunk_grid,
      chunk_key_encoding: metadata.chunk_key_encoding,
      codecs: metadata.codecs,
      fill_value: metadata.fill_value,
      attributes: {},
    };
    retainedMetadataBytes += encodeMetadata(decoderMetadata).byteLength * 4;
    arrays.set(
      path,
      decoderPool.wrap(decoderPath, decoderMetadata, new ZarritaArray(store, decoderPath, decoderMetadata)),
    );
    zarrays[path] = metadata;
  }
  return {
    attrs,
    zarrays,
    arrays,
    encodedByteLimits,
    metadataByteLength: retainedMetadataBytes,
    resetDecoders: () => decoderPool.reset(),
  };
}
