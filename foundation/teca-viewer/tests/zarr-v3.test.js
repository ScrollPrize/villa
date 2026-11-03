import assert from "node:assert/strict";
import { test } from "node:test";
import { openDirectZarrV3 } from "../src/volume/zarr-v3.js";

const encoder = new TextEncoder();

function memoryStore(objects) {
  return {
    metadata: new Map(),
    get: async (key) => objects.get(key),
    resolve: (key) => `memory:${key}`,
    setDataLimit: () => {},
  };
}

test("an unsharded Zarr v3 array reads a chunk as its values in C order", async () => {
  const group = {
    zarr_format: 3,
    node_type: "group",
    attributes: {
      ome: {
        version: "0.5",
        multiscales: [
          {
            axes: [
              { name: "z", type: "space" },
              { name: "y", type: "space" },
              { name: "x", type: "space" },
            ],
            datasets: [{ path: "0", coordinateTransformations: [{ type: "scale", scale: [1, 1, 1] }] }],
          },
        ],
      },
    },
  };
  const array = {
    zarr_format: 3,
    node_type: "array",
    shape: [4, 4, 4],
    data_type: "uint8",
    chunk_grid: { name: "regular", configuration: { chunk_shape: [2, 2, 2] } },
    chunk_key_encoding: { name: "default", configuration: { separator: "/" } },
    codecs: [{ name: "bytes", configuration: { endian: "little" } }],
    fill_value: 0,
  };
  const chunk_DHW = Uint8Array.from({ length: 8 }, (_, index) => 10 + index);
  const store = memoryStore(
    new Map([
      ["/zarr.json", encoder.encode(JSON.stringify(group))],
      ["/0/zarr.json", encoder.encode(JSON.stringify(array))],
      ["/0/c/1/0/1", chunk_DHW],
    ]),
  );
  const opened = await openDirectZarrV3(store, null);
  const scale = { storageChunks: [2, 2, 2], numericType: { kind: "u", itemSize: 1 } };
  const values = await opened.arrays.get("0").getChunk([1, 0, 1], scale, new AbortController().signal);
  assert.deepEqual([...values], [...chunk_DHW]);
});
