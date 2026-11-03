import assert from "node:assert/strict";
import { test } from "node:test";
import { validateZarray } from "../src/volume/zarray.js";

const v2 = {
  zarr_format: 2,
  shape: [64, 96],
  chunks: [32, 32],
  dtype: "<u2",
  filters: null,
  order: "C",
  fill_value: 0,
};
const v3 = {
  zarr_format: 3,
  node_type: "array",
  shape: [256, 256],
  dimension_names: ["y", "x"],
  data_type: "float32",
  chunk_grid: { name: "regular", configuration: { chunk_shape: [128, 128] } },
  codecs: [{ name: "sharding_indexed", configuration: { chunk_shape: [32, 64] } }],
  fill_value: "NaN",
};

test("a v2 array without a dimension separator joins its chunk keys with dots and keeps its compressor", () => {
  const { numericType, fillValue, chunks, chunkSeparator, compressor } = validateZarray(
    { ...v2, compressor: { id: "zstd", level: 3 } },
    ["y", "x"],
    "0",
  );
  assert.deepEqual(
    [numericType.dtype, fillValue, chunks, chunkSeparator, compressor],
    ["<u2", 0, [32, 32], ".", { id: "zstd", level: 3 }],
  );
});

test("a sharded v3 array's chunks are its shards' inner chunks, and a float's NaN fill is NaN", () => {
  const { numericType, fillValue, chunks, chunkSeparator, compressor } = validateZarray(v3, ["y", "x"], "0");
  assert.deepEqual([numericType.dtype, chunks, chunkSeparator, compressor], ["<f4", [32, 64], "/", null]);
  assert.ok(Number.isNaN(fillValue));
});

test("a fill value outside the dtype, and v3 dimension names other than the axes', each throw", () => {
  assert.throws(
    () => validateZarray({ ...v2, dtype: "|u1", fill_value: 256, compressor: null }, ["y", "x"], "0"),
    /Fill value at 0 is outside its dtype/,
  );
  assert.throws(() => validateZarray(v3, ["x", "y"], "1"), /dimension_names at 1 must exactly match/);
});
