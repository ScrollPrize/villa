import { registry } from "../../vendor/zarrita.module.js";
import { WIRE_CHUNK_BYTE_LIMIT, ZARR_SHARD_INDEX_BYTE_LIMIT } from "../config.js";
import { createWorkerPool } from "../worker-pool.js";

export const BLOSC_MAX_OVERHEAD = 16;

const DATA_TYPE_BYTES = new Map([
  ["int8", 1],
  ["uint8", 1],
  ["int16", 2],
  ["uint16", 2],
  ["float32", 4],
  ["uint64", 8],
]);
const ZSTD_NAMES = ["zstd", "numcodecs.zstd"];
const BLOSC_NAMES = ["blosc", "numcodecs.blosc"];
const BLOSC_COMPRESSORS = new Set(["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]);
const bloscEntry = registry.get("blosc");
const decodePool = createWorkerPool(new URL("./blosc.worker.js", import.meta.url), "Chunk decode", {
  workerCount: 2,
  terminateOnAbort: false,
});
const CRC32C_TABLE = Uint32Array.from({ length: 256 }, (_, index) => {
  let value = index;
  for (let bit = 0; bit < 8; bit++) value = (value >>> 1) ^ (value & 1 ? 0x82f63b78 : 0);
  return value >>> 0;
});

function validateObject(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Zarr ${label} must be an object.`);
  }
}

function validateKeys(value, allowed, label) {
  const allowedKeys = new Set(allowed);
  const unknown = Object.keys(value).filter((key) => !allowedKeys.has(key));
  if (unknown.length > 0) throw new Error(`Zarr ${label} contains unsupported field ${unknown[0]}.`);
}

function validateBlosc(configuration, path, coreCodec) {
  validateObject(configuration, `${path} Blosc configuration`);
  validateKeys(configuration, ["typesize", "cname", "clevel", "shuffle", "blocksize"], `${path} Blosc configuration`);
  if (
    configuration.typesize !== undefined &&
    (!Number.isSafeInteger(configuration.typesize) || configuration.typesize < 1 || configuration.typesize > 255)
  ) {
    throw new Error(`Blosc typesize at ${path} must be an integer inside 1..255.`);
  }
  if (!BLOSC_COMPRESSORS.has(configuration.cname)) {
    throw new Error(`Blosc compressor at ${path} is unsupported.`);
  }
  if (!Number.isSafeInteger(configuration.clevel) || configuration.clevel < 0 || configuration.clevel > 9) {
    throw new Error(`Blosc compression level at ${path} must be an integer inside 0..9.`);
  }
  const shuffles = coreCodec ? new Set(["noshuffle", "shuffle", "bitshuffle"]) : new Set([-1, 0, 1, 2]);
  if (!shuffles.has(configuration.shuffle)) {
    throw new Error(`Blosc shuffle at ${path} is unsupported.`);
  }
  if (!Number.isSafeInteger(configuration.blocksize) || configuration.blocksize < 0) {
    throw new Error(`Blosc blocksize at ${path} must be a non-negative integer.`);
  }
}

function validateBloscFrame(bytes, expectedBytes) {
  if (!(bytes instanceof Uint8Array) || bytes.byteLength < 16) {
    throw new Error("Blosc payload header is truncated.");
  }
  const header = new DataView(bytes.buffer, bytes.byteOffset, 16);
  const frameTypesize = bytes[3];
  const decodedBytes = header.getUint32(4, true);
  const blocksize = header.getUint32(8, true);
  const encodedBytes = header.getUint32(12, true);
  if (frameTypesize < 1) throw new Error("Blosc frame typesize must be positive.");
  if (decodedBytes !== expectedBytes) {
    throw new Error(`Blosc frame declares ${decodedBytes} decoded bytes, expected ${expectedBytes}.`);
  }
  if (blocksize > decodedBytes) throw new Error("Blosc frame blocksize exceeds its decoded byte length.");
  if (encodedBytes !== bytes.byteLength) {
    throw new Error(`Blosc frame declares ${encodedBytes} encoded bytes, received ${bytes.byteLength}.`);
  }
}

function boundedBloscDecode(codec, expectedBytes) {
  const decode = codec.decode.bind(codec);
  return async (bytes) => {
    validateBloscFrame(bytes, expectedBytes);
    const decoded = await decode(bytes);
    if (!(decoded instanceof Uint8Array) || decoded.byteLength !== expectedBytes) {
      throw new Error(`Blosc decoded ${decoded?.byteLength ?? "invalid"} bytes, expected ${expectedBytes}.`);
    }
    return decoded;
  };
}

export async function openDirectBloscDecoder(compressor, decodedBytes, path, { worker = false } = {}) {
  validateObject(compressor, `${path} compressor`);
  if (compressor.id !== "blosc") {
    throw new Error(`Zarr v2 compressor ${compressor.id} at ${path} is unsupported in the browser.`);
  }
  validateKeys(compressor, ["id", "cname", "clevel", "shuffle", "blocksize"], `${path} compressor`);
  const configuration = {
    cname: compressor.cname,
    clevel: compressor.clevel,
    shuffle: compressor.shuffle,
    blocksize: compressor.blocksize,
  };
  validateBlosc(configuration, path, false);
  if (worker) {
    return (bytes, signal = null) => decodePool.run({ compressor, decodedBytes, path, bytes }, [bytes.buffer], signal);
  }
  const entry = await bloscEntry();
  // Blosc shuffle width belongs to the codec, not the array dtype.
  return boundedBloscDecode(entry.fromConfig(configuration), decodedBytes);
}

function validateDeclaration(declaration, path) {
  validateObject(declaration, `${path} codec`);
  validateKeys(declaration, ["name", "configuration"], `${path} codec`);
  if (typeof declaration.name !== "string" || declaration.name.length === 0) {
    throw new Error(`Zarr v3 array ${path} has malformed codec metadata.`);
  }
  if (declaration.configuration !== undefined) {
    validateObject(declaration.configuration, `${path} ${declaration.name} configuration`);
  }
  return declaration;
}

function validateBytes(configuration, dataType, path) {
  if (configuration === undefined) {
    if (DATA_TYPE_BYTES.get(dataType) === 1) return;
    throw new Error(`Zarr v3 bytes endian at ${path} is required for multi-byte data.`);
  }
  validateObject(configuration, `${path} bytes configuration`);
  validateKeys(configuration, ["endian"], `${path} bytes configuration`);
  if (
    configuration.endian !== "little" &&
    configuration.endian !== "big" &&
    !(configuration.endian === undefined && DATA_TYPE_BYTES.get(dataType) === 1)
  ) {
    throw new Error(`Zarr v3 bytes endian at ${path} must be little or big.`);
  }
}

function validateTranspose(configuration, shape, path) {
  validateObject(configuration, `${path} transpose configuration`);
  validateKeys(configuration, ["order"], `${path} transpose configuration`);
  const order = configuration.order;
  if (order === "C" || order === "F") return;
  if (
    !Array.isArray(order) ||
    order.length !== shape.length ||
    order.some((axis) => !Number.isSafeInteger(axis) || axis < 0 || axis >= shape.length) ||
    new Set(order).size !== shape.length
  ) {
    throw new Error(`Zarr v3 transpose order at ${path} must be a complete axis permutation.`);
  }
}

function validateZstd(configuration, path) {
  validateObject(configuration, `${path} Zstd configuration`);
  validateKeys(configuration, ["level", "checksum"], `${path} Zstd configuration`);
  if (
    configuration.level !== undefined &&
    (!Number.isSafeInteger(configuration.level) || configuration.level < -131072 || configuration.level > 22)
  ) {
    throw new Error(`Zarr v3 Zstd level at ${path} is invalid.`);
  }
  if (configuration.checksum !== undefined && typeof configuration.checksum !== "boolean") {
    throw new Error(`Zarr v3 Zstd checksum flag at ${path} must be boolean.`);
  }
}

function validateEmptyConfiguration(configuration, path, name) {
  if (configuration === undefined) return;
  validateObject(configuration, `${path} ${name} configuration`);
  validateKeys(configuration, [], `${path} ${name} configuration`);
}

function crc32c(bytes) {
  let value = 0xffffffff;
  for (const byte of bytes) value = CRC32C_TABLE[(value ^ byte) & 0xff] ^ (value >>> 8);
  return (value ^ 0xffffffff) >>> 0;
}

async function validatePipeline(codecs, meta, path, allowSharding, indexPipeline = false) {
  if (!Array.isArray(codecs) || codecs.length === 0) {
    throw new Error(`Zarr v3 array ${path} must declare a codec pipeline.`);
  }
  if (codecs.some((codec) => codec?.name === "sharding_indexed")) {
    if (!allowSharding || codecs.length !== 1 || codecs[0]?.name !== "sharding_indexed") {
      throw new Error(`Zarr v3 sharding at ${path} must be the complete outer codec pipeline.`);
    }
    const declaration = validateDeclaration(codecs[0], path);
    await validateSharding(declaration, meta, path);
    const configuration = declaration.configuration;
    return { ...meta, shape: [...configuration.chunk_shape], codecs: configuration.codecs };
  }

  let currentMeta = { ...meta };
  let arrayToBytes = false;
  let sawCompressor = false;
  let sawCrc32c = false;
  for (const declaration of codecs) {
    const { name, configuration } = validateDeclaration(declaration, path);
    if (name === "transpose") {
      if (arrayToBytes || indexPipeline) throw new Error(`Zarr v3 transpose codec is misplaced at ${path}.`);
      validateTranspose(configuration, currentMeta.shape, path);
    } else if (name === "bytes") {
      if (arrayToBytes) throw new Error(`Zarr v3 array ${path} declares more than one array-to-bytes codec.`);
      validateBytes(configuration, currentMeta.dataType, path);
      arrayToBytes = true;
    } else if (ZSTD_NAMES.includes(name)) {
      if (indexPipeline || !arrayToBytes || sawCompressor || sawCrc32c) {
        throw new Error(`Zarr v3 Zstd codec is misplaced at ${path}.`);
      }
      validateZstd(configuration, path);
      sawCompressor = true;
    } else if (BLOSC_NAMES.includes(name)) {
      if (indexPipeline || !arrayToBytes || sawCompressor || sawCrc32c) {
        throw new Error(`Zarr v3 Blosc codec is misplaced at ${path}.`);
      }
      validateBlosc(configuration, path, name === "blosc");
      sawCompressor = true;
    } else if (name === "crc32c") {
      if (!arrayToBytes || sawCrc32c) throw new Error(`Zarr v3 CRC32C codec is misplaced at ${path}.`);
      validateEmptyConfiguration(configuration, path, "CRC32C");
      sawCrc32c = true;
    } else {
      throw new Error(`Unsupported Zarr v3 codec at ${path}: ${name}.`);
    }

    const entry = await registry.get(name)();
    const codec = entry.fromConfig(configuration, currentMeta);
    if (codec.getEncodedMeta !== undefined) currentMeta = codec.getEncodedMeta(currentMeta);
  }
  if (!arrayToBytes) throw new Error(`Zarr v3 array ${path} must declare one bytes codec.`);
  return currentMeta;
}

function validateChunkKeyEncoding(encoding, path) {
  validateObject(encoding, `${path} chunk key encoding`);
  validateKeys(encoding, ["name", "configuration"], `${path} chunk key encoding`);
  if (encoding.name !== "default" && encoding.name !== "v2") {
    throw new Error(`Unsupported Zarr v3 chunk key encoding at ${path}: ${encoding.name}.`);
  }
  if (encoding.configuration === undefined) return;
  validateObject(encoding.configuration, `${path} chunk key encoding configuration`);
  validateKeys(encoding.configuration, ["separator"], `${path} chunk key encoding configuration`);
  if (encoding.configuration.separator !== undefined && !["/", "."].includes(encoding.configuration.separator)) {
    throw new Error(`Zarr v3 chunk key separator at ${path} must be slash or period.`);
  }
}

function validateShape(shape, label) {
  if (!Array.isArray(shape) || shape.length === 0 || shape.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new Error(`Zarr v3 ${label} shape must contain positive integers.`);
  }
  return [...shape];
}

function decodedByteLength(meta) {
  const itemBytes = DATA_TYPE_BYTES.get(meta.dataType);
  if (itemBytes === undefined) throw new Error(`Unsupported direct Zarr v3 data type: ${meta.dataType}.`);
  const bytes = meta.shape.reduce((total, value) => total * value, itemBytes);
  if (!Number.isSafeInteger(bytes) || bytes > WIRE_CHUNK_BYTE_LIMIT) {
    throw new Error(`Decoded Zarr v3 chunk exceeds the ${WIRE_CHUNK_BYTE_LIMIT}-byte viewer limit.`);
  }
  return bytes;
}

function encodedByteLimit(meta) {
  let bytes = decodedByteLength(meta);
  for (const codec of meta.codecs) {
    if (ZSTD_NAMES.includes(codec.name)) bytes += Math.ceil(bytes / 128) + 128 * 1024;
    else if (BLOSC_NAMES.includes(codec.name)) bytes += BLOSC_MAX_OVERHEAD;
    else if (codec.name === "crc32c") bytes += 4;
  }
  if (!Number.isSafeInteger(bytes)) throw new Error("Encoded Zarr v3 chunk byte limit is not a safe integer.");
  return bytes;
}

export async function prepareDirectCodecPipeline(metadata, path) {
  validateChunkKeyEncoding(metadata.chunk_key_encoding, path);
  const outerShape = validateShape(metadata.chunk_grid?.configuration?.chunk_shape, `${path} outer chunks`);
  const meta = {
    dataType: metadata.data_type,
    shape: outerShape,
    codecs: metadata.codecs,
    fillValue: metadata.fill_value,
  };
  const chunkMeta = await validatePipeline(metadata.codecs, meta, path, true);
  return encodedByteLimit(chunkMeta);
}

async function validateSharding(declaration, meta, path) {
  const configuration = declaration.configuration;
  validateObject(configuration, `${path} sharding configuration`);
  validateKeys(
    configuration,
    ["chunk_shape", "codecs", "index_codecs", "index_location"],
    `${path} sharding configuration`,
  );
  if ((configuration.index_location ?? "end") !== "end") {
    throw new Error(`Zarr v3 shard index at ${path} must be stored at the end for direct browser reads.`);
  }
  const innerShape = validateShape(configuration.chunk_shape, `${path} shard chunks`);
  if (
    innerShape.length !== meta.shape.length ||
    innerShape.some((value, axis) => value > meta.shape[axis] || meta.shape[axis] % value !== 0)
  ) {
    throw new Error(`Zarr v3 shard chunk shape at ${path} must divide the outer chunk shape exactly.`);
  }
  await validatePipeline(
    configuration.codecs,
    { ...meta, shape: innerShape, codecs: configuration.codecs },
    `${path} shard chunks`,
    false,
  );
  const indexShape = meta.shape.map((value, axis) => value / innerShape[axis]);
  await validatePipeline(
    configuration.index_codecs,
    {
      dataType: "uint64",
      shape: [...indexShape, 2],
      codecs: configuration.index_codecs,
      fillValue: null,
    },
    `${path} shard index`,
    false,
    true,
  );
  const indexBytes =
    indexShape.reduce((total, value) => total * value, 16) +
    (configuration.index_codecs.some((codec) => codec.name === "crc32c") ? 4 : 0);
  if (!Number.isSafeInteger(indexBytes) || indexBytes > ZARR_SHARD_INDEX_BYTE_LIMIT) {
    throw new Error(`Zarr v3 shard index at ${path} exceeds the direct decoder byte limit.`);
  }
}

function validateZstdFrame(bytes, expectedBytes) {
  if (
    !(bytes instanceof Uint8Array) ||
    bytes.byteLength < 8 ||
    bytes[0] !== 0x28 ||
    bytes[1] !== 0xb5 ||
    bytes[2] !== 0x2f ||
    bytes[3] !== 0xfd
  ) {
    throw new Error("Zstd payload has no supported frame header.");
  }
  const descriptor = bytes[4];
  if ((descriptor & 0x18) !== 0) throw new Error("Zstd payload uses reserved frame descriptor bits.");
  const singleSegment = (descriptor & 0x20) !== 0;
  const checksum = (descriptor & 0x04) !== 0;
  const dictionaryBytes = [0, 1, 2, 4][descriptor & 0x03];
  const sizeFlag = descriptor >>> 6;
  const contentSizeBytes = sizeFlag === 0 ? (singleSegment ? 1 : 0) : sizeFlag === 1 ? 2 : sizeFlag === 2 ? 4 : 8;
  if (contentSizeBytes === 0)
    throw new Error("Zstd frames without a declared content size are unsupported in the browser.");
  let offset = 5 + (singleSegment ? 0 : 1) + dictionaryBytes;
  if (offset + contentSizeBytes > bytes.byteLength) throw new Error("Zstd frame header is truncated.");
  let contentSize = 0n;
  for (let index = 0; index < contentSizeBytes; index++) {
    contentSize |= BigInt(bytes[offset + index]) << BigInt(index * 8);
  }
  if (contentSizeBytes === 2) contentSize += 256n;
  if (contentSize !== BigInt(expectedBytes)) {
    throw new Error(`Zstd frame declares ${contentSize} decoded bytes, expected ${expectedBytes}.`);
  }
  offset += contentSizeBytes;
  let lastBlock = false;
  while (!lastBlock) {
    if (offset + 3 > bytes.byteLength) throw new Error("Zstd frame block header is truncated.");
    const header = bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
    offset += 3;
    lastBlock = (header & 1) !== 0;
    const blockType = (header >>> 1) & 3;
    const blockSize = header >>> 3;
    if (blockType === 3) throw new Error("Zstd frame contains a reserved block type.");
    offset += blockType === 1 ? 1 : blockSize;
    if (offset > bytes.byteLength) throw new Error("Zstd frame block is truncated.");
  }
  if (checksum) offset += 4;
  if (offset !== bytes.byteLength) throw new Error("Zstd payload must contain exactly one complete frame.");
}

for (const name of ZSTD_NAMES) {
  const load = registry.get(name);
  registry.set(name, async () => {
    const entry = await load();
    return {
      ...entry,
      fromConfig(configuration, meta) {
        const codec = entry.fromConfig(configuration, meta);
        const decode = codec.decode.bind(codec);
        const expectedBytes = decodedByteLength(meta);
        codec.decode = async (bytes) => {
          validateZstdFrame(bytes, expectedBytes);
          const decoded = await decode(bytes);
          if (!(decoded instanceof Uint8Array) || decoded.byteLength !== expectedBytes) {
            throw new Error(`Zstd decoded ${decoded?.byteLength ?? "invalid"} bytes, expected ${expectedBytes}.`);
          }
          return decoded;
        };
        return codec;
      },
    };
  });
}
for (const name of BLOSC_NAMES) {
  const load = registry.get(name);
  registry.set(name, async () => {
    const entry = await load();
    return {
      ...entry,
      fromConfig(configuration, meta) {
        const codec = entry.fromConfig(configuration, meta);
        // v3 frames carry their own typesize: zarr-python before numcodecs 0.16 wrote 1 whatever was configured.
        codec.decode = boundedBloscDecode(codec, decodedByteLength(meta));
        return codec;
      },
    };
  });
}
registry.set("crc32c", async () => ({
  fromConfig() {
    return {
      kind: "bytes_to_bytes",
      computeEncodedSize: (decodedSize) => decodedSize + 4,
      decode(bytes) {
        if (!(bytes instanceof Uint8Array) || bytes.byteLength < 4) throw new Error("CRC32C payload is truncated.");
        const data = bytes.subarray(0, bytes.byteLength - 4);
        const expected = new DataView(bytes.buffer, bytes.byteOffset + bytes.byteLength - 4, 4).getUint32(0, true);
        const actual = crc32c(data);
        if (actual !== expected) throw new Error("CRC32C checksum does not match the decoded payload.");
        return data;
      },
    };
  },
}));
