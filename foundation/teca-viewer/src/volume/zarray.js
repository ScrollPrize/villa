import { WIRE_CHUNK_BYTE_LIMIT } from "../config.js";
import { createNumericArray, numericTypeForDtype } from "./numeric-type.js";

function v3TransportDtype(dataType, path) {
  const dtypes = {
    uint8: "|u1",
    int8: "|i1",
    uint16: "<u2",
    int16: "<i2",
    float32: "<f4",
  };
  const dtype = dtypes[dataType];
  if (dtype === undefined) throw new Error(`Unsupported Zarr v3 data type at ${path}: ${dataType}.`);
  return dtype;
}

export function validateZarray(zarray, axisNames, path) {
  let dtype;
  let chunks;
  let chunkSeparator = "/";
  if (zarray.zarr_format === 2) {
    dtype = zarray.dtype;
    chunks = zarray.chunks;
    if (zarray.filters?.length > 0) {
      throw new Error(`Filtered Zarr array at ${path} uses unsupported Zarr v2 filters.`);
    }
    if (zarray.order !== "C") throw new Error(`Unsupported array order at ${path}: ${zarray.order}`);
    chunkSeparator = zarray.dimension_separator ?? ".";
    if (chunkSeparator !== "/" && chunkSeparator !== ".") {
      throw new Error(`Unsupported chunk separator at ${path}: ${chunkSeparator}`);
    }
  } else if (zarray.zarr_format === 3) {
    if (zarray.node_type !== "array") throw new Error(`Zarr v3 node at ${path} must be an array.`);
    const names = zarray.dimension_names ?? axisNames;
    if (names.length !== axisNames.length || names.some((name, index) => name !== null && name !== axisNames[index])) {
      throw new Error(`Zarr v3 dimension_names at ${path} must exactly match the authored NGFF axes.`);
    }
    dtype = v3TransportDtype(zarray.data_type, path);
    if (zarray.chunk_grid?.name !== "regular") {
      throw new Error(`Unsupported Zarr v3 chunk grid at ${path}: ${zarray.chunk_grid?.name}.`);
    }
    const shard = zarray.codecs?.find((codec) => codec?.name === "sharding_indexed");
    chunks = shard?.configuration?.chunk_shape ?? zarray.chunk_grid?.configuration?.chunk_shape;
  } else {
    throw new Error(`Unsupported Zarr format at ${path}: ${zarray.zarr_format}`);
  }
  const numericType = numericTypeForDtype(dtype, path);
  // A fill_value of null declares none, so a missing chunk reads as 0.
  const declaredFill = zarray.fill_value === null ? 0 : zarray.fill_value;
  const fillValue = declaredFill === "NaN" && numericType.kind === "f" ? NaN : declaredFill;
  if (
    typeof fillValue !== "number" ||
    (!Number.isFinite(fillValue) && !(numericType.kind === "f" && Number.isNaN(fillValue)))
  ) {
    throw new Error(`Unsupported or absent fill value at ${path}.`);
  }
  const fillProbe = createNumericArray(numericType, 1);
  fillProbe[0] = fillValue;
  if (numericType.kind !== "f" && fillProbe[0] !== fillValue)
    throw new Error(`Fill value at ${path} is outside its dtype.`);
  if (!Array.isArray(zarray.shape) || !Array.isArray(chunks)) {
    throw new Error(`Array ${path} shape and chunks must be arrays.`);
  }
  if (zarray.shape.length !== axisNames.length || chunks.length !== axisNames.length) {
    throw new Error(`Array ${path} rank does not match axes ${axisNames.join(",")}.`);
  }
  if (
    zarray.shape.some((value) => !Number.isSafeInteger(value) || value < 1) ||
    chunks.some((value) => !Number.isSafeInteger(value) || value < 1)
  ) {
    throw new Error(`Array ${path} shape and chunks must contain positive integers.`);
  }
  const chunkBytes = chunks.reduce((total, size) => total * size, 1) * numericType.itemSize;
  if (!Number.isSafeInteger(chunkBytes) || chunkBytes > WIRE_CHUNK_BYTE_LIMIT) {
    throw new Error(`Array ${path} chunk size ${chunkBytes} exceeds the ${WIRE_CHUNK_BYTE_LIMIT}-byte viewer limit.`);
  }
  return {
    numericType,
    fillValue,
    chunks: [...chunks],
    chunkSeparator,
    compressor: zarray.zarr_format === 2 ? zarray.compressor : null,
  };
}
