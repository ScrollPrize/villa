// A big-endian dtype shares its little-endian format, since decoded chunks are in host byte order.

import * as THREE from "../../vendor/three.module.js";

const FORMATS = {
  u8: Object.freeze({
    normalizedDtype: "|u1",
    ArrayType: Uint8Array,
    bytesPerVoxel: Uint8Array.BYTES_PER_ELEMENT,
    format: THREE.RedIntegerFormat,
    type: THREE.UnsignedByteType,
    internalFormat: "R8UI",
    samplerClass: "uint",
  }),
  u16: Object.freeze({
    normalizedDtype: "<u2",
    ArrayType: Uint16Array,
    bytesPerVoxel: Uint16Array.BYTES_PER_ELEMENT,
    format: THREE.RedIntegerFormat,
    type: THREE.UnsignedShortType,
    internalFormat: "R16UI",
    samplerClass: "uint",
  }),
  i8: Object.freeze({
    normalizedDtype: "|i1",
    ArrayType: Int8Array,
    bytesPerVoxel: Int8Array.BYTES_PER_ELEMENT,
    format: THREE.RedIntegerFormat,
    type: THREE.ByteType,
    internalFormat: "R8I",
    samplerClass: "int",
  }),
  i16: Object.freeze({
    normalizedDtype: "<i2",
    ArrayType: Int16Array,
    bytesPerVoxel: Int16Array.BYTES_PER_ELEMENT,
    format: THREE.RedIntegerFormat,
    type: THREE.ShortType,
    internalFormat: "R16I",
    samplerClass: "int",
  }),
  f32: Object.freeze({
    normalizedDtype: "<f4",
    ArrayType: Float32Array,
    bytesPerVoxel: Float32Array.BYTES_PER_ELEMENT,
    format: THREE.RedFormat,
    type: THREE.FloatType,
    internalFormat: "R32F",
    samplerClass: "float",
  }),
};

const FORMAT_BY_DTYPE = new Map([
  ["|u1", FORMATS.u8],
  ["<u2", FORMATS.u16],
  [">u2", FORMATS.u16],
  ["|i1", FORMATS.i8],
  ["<i2", FORMATS.i16],
  [">i2", FORMATS.i16],
  ["<f4", FORMATS.f32],
  [">f4", FORMATS.f32],
]);

export function atlasFormatForDtype(dtype) {
  const format = FORMAT_BY_DTYPE.get(dtype);
  if (format === undefined) throw new Error(`GPU chunk atlases do not support dtype ${dtype}.`);
  return format;
}

export function atlasFormatIdentity(dtype) {
  const format = atlasFormatForDtype(dtype);
  return `${dtype}\0${format.normalizedDtype}\0${format.samplerClass}`;
}
