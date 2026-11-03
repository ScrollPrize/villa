import assert from "node:assert/strict";
import { test } from "node:test";
import { declaredDatasetPaths, readVolumeDocument } from "../src/volume/ngff.js";

const withPaths = (...paths) => ({ multiscales: [{ datasets: paths.map((path) => ({ path })) }] });

test("datasets 0 and 1 give the paths 0 and 1", () => {
  assert.deepEqual(declaredDatasetPaths(withPaths("0", "1")), ["0", "1"]);
});

test("no multiscales entry, a parent path and a repeated path each throw", () => {
  assert.throws(() => declaredDatasetPaths({ multiscales: [] }), /a multiscales entry/);
  assert.throws(() => declaredDatasetPaths(withPaths("../0")), /relative path without empty or parent components/);
  assert.throws(() => declaredDatasetPaths(withPaths("0", "0")), /must be unique/);
});

test("levels of c, z, y and x take their transform in ZYX, and in voxels of the finest level", () => {
  const level = (path, scale, translation) => ({
    path,
    coordinateTransformations: [
      { type: "scale", scale },
      { type: "translation", translation },
    ],
  });
  const attrs = {
    multiscales: [
      {
        axes: [
          { name: "c", type: "channel" },
          { name: "z", type: "space" },
          { name: "y", type: "space" },
          { name: "x", type: "space" },
        ],
        datasets: [level("0", [1, 2, 3, 4], [0, 0.5, 1, 1.5]), level("1", [1, 4, 6, 8], [0, 1.5, 4, 5.5])],
      },
    ],
  };
  const zarray = { zarr_format: 2, shape: [1, 64, 64, 64], chunks: [1, 32, 32, 32], dtype: "|u1" };
  const array = { ...zarray, compressor: null, filters: null, order: "C", fill_value: 0 };
  const { scales } = readVolumeDocument(attrs, { 0: array, 1: array });
  assert.deepEqual(
    scales.map(({ scale_ZYX, translation_ZYX, downsample_ZYX, offset_ZYX }) => [
      scale_ZYX,
      translation_ZYX,
      downsample_ZYX,
      offset_ZYX,
    ]),
    [
      [
        [2, 3, 4],
        [0.5, 1, 1.5],
        [1, 1, 1],
        [0, 0, 0],
      ],
      [
        [4, 6, 8],
        [1.5, 4, 5.5],
        [2, 2, 2],
        [0.5, 1, 1],
      ],
    ],
  );
});
