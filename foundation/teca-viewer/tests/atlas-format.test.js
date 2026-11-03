import assert from "node:assert/strict";
import { test } from "node:test";
import { atlasFormatForDtype, atlasFormatIdentity } from "../src/gpu/atlas-format.js";
import * as THREE from "../vendor/three.module.js";

test("each supported dtype maps to one frozen sized WebGL2 format", () => {
  for (const [dtype, normalizedDtype, ArrayType, bytesPerVoxel, format, type, internalFormat, samplerClass] of [
    ["|u1", "|u1", Uint8Array, 1, THREE.RedIntegerFormat, THREE.UnsignedByteType, "R8UI", "uint"],
    ["<u2", "<u2", Uint16Array, 2, THREE.RedIntegerFormat, THREE.UnsignedShortType, "R16UI", "uint"],
    [">u2", "<u2", Uint16Array, 2, THREE.RedIntegerFormat, THREE.UnsignedShortType, "R16UI", "uint"],
    ["|i1", "|i1", Int8Array, 1, THREE.RedIntegerFormat, THREE.ByteType, "R8I", "int"],
    ["<i2", "<i2", Int16Array, 2, THREE.RedIntegerFormat, THREE.ShortType, "R16I", "int"],
    [">i2", "<i2", Int16Array, 2, THREE.RedIntegerFormat, THREE.ShortType, "R16I", "int"],
    ["<f4", "<f4", Float32Array, 4, THREE.RedFormat, THREE.FloatType, "R32F", "float"],
    [">f4", "<f4", Float32Array, 4, THREE.RedFormat, THREE.FloatType, "R32F", "float"],
  ]) {
    const descriptor = atlasFormatForDtype(dtype);
    assert.deepEqual(
      { ...descriptor },
      { normalizedDtype, ArrayType, bytesPerVoxel, format, type, internalFormat, samplerClass },
      dtype,
    );
    assert.equal(Object.isFrozen(descriptor), true);
  }
});

test("a big-endian dtype shares its little-endian format and keeps its own name in the identity", () => {
  assert.equal(atlasFormatForDtype(">u2"), atlasFormatForDtype("<u2"));
  assert.equal(atlasFormatForDtype(">i2"), atlasFormatForDtype("<i2"));
  assert.equal(atlasFormatForDtype(">f4"), atlasFormatForDtype("<f4"));
  assert.equal(atlasFormatIdentity(">i2"), ">i2\0<i2\0int");
  assert.equal(atlasFormatIdentity("<f4"), "<f4\0<f4\0float");
});

test("a dtype outside the eight throws", () => {
  for (const dtype of ["<u1", ">i1", "|f4", "<u4", "<f8", "", null, undefined]) {
    assert.throws(() => atlasFormatForDtype(dtype), /do not support dtype/);
  }
});
