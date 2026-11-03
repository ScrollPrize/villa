const SUPPORTED_DTYPES = new Set(["|u1", "|i1", "<u2", ">u2", "<i2", ">i2", "<f4", ">f4"]);
// Every browser runs little-endian; a big-endian host would take the DataView path.
const HOST_LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

export function decodeNumericChunk(bytes, dtype) {
  const numericType = numericTypeForDtype(dtype, "chunk");
  if (!(bytes instanceof Uint8Array) || bytes.byteLength % numericType.itemSize !== 0) {
    throw new Error(`Chunk byte length must be divisible by dtype ${dtype} size ${numericType.itemSize}.`);
  }
  const length = bytes.byteLength / numericType.itemSize;
  // A single-byte dtype or one in host byte order is viewed in place, copied only when the offset is not aligned to
  // its size; the per-element loop below costs the main thread much on multi-megabyte chunks.
  if (numericType.itemSize === 1 || numericType.littleEndian === HOST_LITTLE_ENDIAN) {
    const aligned =
      bytes.byteOffset % numericType.itemSize === 0
        ? { buffer: bytes.buffer, byteOffset: bytes.byteOffset }
        : { buffer: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength), byteOffset: 0 };
    return new (typedArrayConstructorFor(numericType))(aligned.buffer, aligned.byteOffset, length);
  }
  const output = createNumericArray(numericType, length);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < output.length; index++) {
    const offset = index * numericType.itemSize;
    if (numericType.kind === "u") output[index] = view.getUint16(offset, numericType.littleEndian);
    else if (numericType.kind === "i") output[index] = view.getInt16(offset, numericType.littleEndian);
    else output[index] = view.getFloat32(offset, numericType.littleEndian);
  }
  return output;
}

export function typedArrayConstructorFor(numericType) {
  if (numericType.kind === "u" && numericType.itemSize === 1) return Uint8Array;
  if (numericType.kind === "i" && numericType.itemSize === 1) return Int8Array;
  if (numericType.kind === "u") return Uint16Array;
  if (numericType.kind === "i") return Int16Array;
  return Float32Array;
}

export function numericTypeForDtype(dtype, path) {
  if (typeof dtype !== "string" || !SUPPORTED_DTYPES.has(dtype)) {
    throw new Error(`Unsupported numeric dtype at ${path}: ${dtype}.`);
  }
  const kind = dtype[1];
  const itemSize = Number(dtype[2]);
  return { dtype, kind, itemSize, littleEndian: dtype[0] !== ">" };
}

export function createNumericArray(numericType, length) {
  return new (typedArrayConstructorFor(numericType))(length);
}
