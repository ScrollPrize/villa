import assert from "node:assert/strict";
import { test } from "node:test";
import {
  copyBrick,
  layerPresentationScales,
  presentationChunkRequest,
  presentationScale,
} from "../src/gpu/presentation-bricks.js";

const bricks = (chunks_ZYX, downsample_ZYX) => [
  ...presentationScale({ path: "0", chunks_ZYX }, downsample_ZYX).chunks_ZYX,
];

test("a 256³ chunk is shown as 64³ bricks; a chunk of at most 64³ voxels, or with a 257-voxel axis, is shown whole", () => {
  assert.deepEqual(bricks([256, 256, 256]), [64, 64, 64]);
  assert.deepEqual(bricks([128, 64, 32]), [128, 64, 32]);
  assert.deepEqual(bricks([257, 256, 256]), [257, 256, 256]);
});

test("voxels twice as deep in Z cut a 256³ chunk into 64×32×32 bricks, and a 100-voxel axis into its largest divisor under 64", () => {
  assert.deepEqual(bricks([256, 256, 256], [2, 1, 1]), [64, 32, 32]);
  assert.deepEqual(bricks([256, 100, 256]), [64, 50, 64]);
});

test("a scale its bricks leave whole is its own presentation scale, and a bricked one points to its source", () => {
  const whole = { path: "0", chunks_ZYX: [128, 64, 32] };
  assert.equal(presentationScale(whole), whole);
  const bricked = { path: "0", chunks_ZYX: [256, 256, 256] };
  assert.equal(presentationScale(bricked).sourceScale, bricked);
});

test("brick 3/2/1 of 2³ in 4³ chunks is chunk 1/1/0 from voxel 2/0/2, and copies its eight voxels", () => {
  const sourceScale = { chunks_ZYX: [4, 4, 4] };
  assert.deepEqual(presentationChunkRequest({ chunks_ZYX: [2, 2, 2], sourceScale }, [3, 2, 1]), {
    scale: sourceScale,
    chunk_ZYX: [1, 1, 0],
    localStart_ZYX: [2, 0, 2],
  });
  const source_DHW = Uint16Array.from({ length: 64 }, (_, index) => index);
  assert.deepEqual([...copyBrick(source_DHW, [4, 4, 4], [2, 2, 2], [2, 0, 2])], [34, 35, 38, 39, 50, 51, 54, 55]);
});

test("a layer on the CT's grid is shown in the CT's bricks up to the CT's last level; one off names its axis", () => {
  const level = (path, shape_ZYX, chunks_ZYX, downsample) => ({
    path,
    shape_ZYX,
    chunks_ZYX,
    downsample_ZYX: [downsample, downsample, downsample],
    offset_ZYX: [0, 0, 0],
  });
  const ct = [level("0", [512, 512, 512], [128, 128, 128], 1), level("1", [256, 256, 256], [128, 128, 128], 2)];
  const layer = [
    level("0", [512, 512, 512], [256, 256, 256], 1),
    level("1", [256, 256, 256], [256, 256, 256], 2),
    level("2", [128, 128, 128], [128, 128, 128], 4),
  ];
  const bricked = (scales) => layerPresentationScales(ct, scales, presentationScale).map((scale) => scale.chunks_ZYX);
  assert.deepEqual(bricked(layer), [
    [64, 64, 64],
    [64, 64, 64],
  ]);
  assert.throws(() => bricked(layer.slice(0, 1)), { message: "The layer has 1 levels; the CT has 2." });
  assert.throws(() => bricked([level("0", [512, 512, 510], [256, 256, 256], 1), layer[1]]), {
    message: "Layer level 0 has shape_ZYX 512,512,510; CT level 0 has 512,512,512.",
  });
});
