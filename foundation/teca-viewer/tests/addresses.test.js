import assert from "node:assert/strict";
import { test } from "node:test";
import {
  DISPLAY_WINDOW_SAMPLE_PRIORITY,
  PREFETCH_CHUNK_PRIORITY,
  SHOWN_CHUNK_PRIORITY_LIMIT,
  SLICE_AHEAD_PREFETCH_PRIORITY,
  SURFACE_GEOMETRY_PRIORITY,
} from "../src/config.js";
import {
  chunkKey,
  chunkKeyParts,
  levelSamplePosition,
  rotatedPlaneChunkAddresses,
  toLevelSamplePoint_ZYX,
  visibleChunkPriority,
  visibleChunkRanges,
  visiblePlaneSpan,
} from "../src/gpu/addresses.js";
import { IDENTITY_ROTATION, quatFromAxisAngle } from "../src/math.js";
import { rotatedPlaneBasis } from "../src/planes.js";

const keys = (addresses) => addresses.map(({ chunk_ZYX }) => chunk_ZYX.join("/"));
const axisChunks = (addresses, axis) => [...new Set(addresses.map(({ chunk_ZYX }) => chunk_ZYX[axis]))].sort();
const ANISOTROPIC = { path: "2", shape_ZYX: [32, 32, 32], chunks_ZYX: [8, 8, 8], downsample_ZYX: [4, 2, 1] };
const CUBE_256 = {
  path: "0",
  shape_ZYX: [256, 256, 256],
  chunks_ZYX: [128, 128, 128],
  downsample_ZYX: [1, 1, 1],
  offset_ZYX: [0, 0, 0],
};

test("chunk 3/12/7 of a scale at path s0/c comes back from its key with the path whole", () => {
  assert.deepEqual(chunkKeyParts(chunkKey({ path: "s0/c" }, [3, 12, 7])), { path: "s0/c", chunk_ZYX: [3, 12, 7] });
});

test("a pane at fixed texel 7 reads chunk 0, at 7.25 and 7.999 chunks 0 and 1, and at 8 chunk 1", () => {
  const scale = { ...ANISOTROPIC, offset_ZYX: [3, 1, -7] };
  for (const [plane, axis] of [
    ["xy", 0],
    ["xz", 1],
    ["zy", 2],
  ]) {
    const basis = rotatedPlaneBasis(plane, IDENTITY_ROTATION);
    for (const [texel, chunks] of [
      [7, [0]],
      [7.25, [0, 1]],
      [7.999, [0, 1]],
      [8, [1]],
    ]) {
      const center_ZYX = [0, 1, 2].map(
        (index) =>
          scale.offset_ZYX[index] +
          (index === axis ? texel : 2.25) * scale.downsample_ZYX[index] +
          (scale.downsample_ZYX[index] - 1) / 2,
      );
      const addresses = rotatedPlaneChunkAddresses(center_ZYX, basis, 2, scale, 1, 1);
      assert.deepEqual(axisChunks(addresses, axis), chunks, `${plane} at texel ${texel}`);
      assert.equal(addresses.length, chunks.length);
    }
  }
});

test("a wide native ZY view needs 132 chunks at an integer X and 264 at a fractional X across a chunk edge", () => {
  const scale = {
    path: "0",
    shape_ZYX: [75784, 32693, 32693],
    chunks_ZYX: [64, 64, 64],
    downsample_ZYX: [1, 1, 1],
    offset_ZYX: [0, 0, 0],
  };
  const basis = rotatedPlaneBasis("zy", IDENTITY_ROTATION);
  for (const [x, count] of [
    [17932, 132],
    [17983, 132],
    [17983.25, 264],
    [17984, 132],
    [18047, 132],
    [18047.25, 264],
    [18048, 132],
  ]) {
    const addresses = rotatedPlaneChunkAddresses(
      [53256.65316, 14551.11497, x],
      basis,
      768 / 1.1607142857142858,
      scale,
      1,
      1509 / 1396,
    );
    assert.equal(addresses.length, count, `X=${x}`);
    assert.equal(axisChunks(addresses, 2).length, count / 132);
  }
});

test("a diagonal plane through a 32768³ volume of 128³ chunks meets 4975 chunks", () => {
  const scale = {
    shape_ZYX: [32768, 32768, 32768],
    chunks_ZYX: [128, 128, 128],
    downsample_ZYX: [1, 1, 1],
    offset_ZYX: [0, 0, 0],
  };
  const basis = {
    right_XYZ: [1 / Math.sqrt(2), -1 / Math.sqrt(2), 0],
    up_XYZ: [1 / Math.sqrt(6), 1 / Math.sqrt(6), -2 / Math.sqrt(6)],
    normal_XYZ: [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)],
  };
  assert.equal(rotatedPlaneChunkAddresses([16384, 16384, 16384], basis, 6144, scale, 1, 1).length, 4975);
});

test("a view over more than 65536 chunks throws a RangeError", () => {
  const scale = { shape_ZYX: [1, 512, 512], chunks_ZYX: [1, 1, 1], downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] };
  assert.throws(
    () => rotatedPlaneChunkAddresses([0, 256, 256], rotatedPlaneBasis("xy", IDENTITY_ROTATION), 800, scale, 1, 1),
    { name: "RangeError", message: /intersects more than 65536 chunks/ },
  );
});

test("an XY plane at z 127.25 and 127.75 reads Z chunks 0 and 1 for its trilinear neighbours", () => {
  const basis = rotatedPlaneBasis("xy", IDENTITY_ROTATION);
  for (const z of [127.25, 127.75]) {
    assert.deepEqual(axisChunks(rotatedPlaneChunkAddresses([z, 128, 128], basis, 16, CUBE_256, 1, 1), 0), [0, 1]);
  }
});

test("the chunks of a translated anisotropic level are the shader's trilinear footprint", () => {
  const scale = { ...ANISOTROPIC, offset_ZYX: [3, 1, 0] };
  const d = Math.SQRT1_2;
  const flat = rotatedPlaneChunkAddresses(
    [50.9, 30.2, 14.6],
    rotatedPlaneBasis("xy", IDENTITY_ROTATION),
    12,
    scale,
    1,
    1,
  );
  assert.deepEqual(keys(flat).sort(), ["1/1/1", "1/1/2", "1/2/1", "1/2/2"]);
  const turned = { right_XYZ: [d, d, 0], up_XYZ: [0, 0, 1], normal_XYZ: [d, -d, 0] };
  assert.deepEqual(keys(rotatedPlaneChunkAddresses([63.4, 27.8, 20.1], turned, 12, scale, 1, 1)).sort(), [
    "1/1/1",
    "1/1/2",
    "1/1/3",
    "1/2/2",
    "1/2/3",
    "2/1/1",
    "2/1/2",
    "2/1/3",
    "2/2/2",
    "2/2/3",
  ]);
});

test("an XY plane tilted 30 degrees about X lists its nine chunks nearest first", () => {
  const scale = {
    path: "0",
    shape_ZYX: [40, 40, 40],
    chunks_ZYX: [10, 10, 10],
    downsample_ZYX: [1, 1, 1],
    offset_ZYX: [0, 0, 0],
  };
  const basis = rotatedPlaneBasis("xy", quatFromAxisAngle([1, 0, 0], 30));
  assert.deepEqual(keys(rotatedPlaneChunkAddresses([15, 20, 20], basis, 20, scale, 1, 1)), [
    "1/2/2",
    "1/1/2",
    "1/2/1",
    "1/1/1",
    "2/2/2",
    "2/2/1",
    "1/2/3",
    "1/1/3",
    "2/2/3",
  ]);
});

test("visible chunk ranges of an XY view at the origin clamp to the volume", () => {
  assert.deepEqual(visibleChunkRanges("xy", [0, 0, 0], 100, 100, [40, 40, 40], [10, 10, 10], 0), {
    z: [0, 0],
    y: [0, 3],
    x: [0, 3],
  });
});

test("world Z 1021 at downsample 8 samples level texel 127.1875", () => {
  const scale = { downsample_ZYX: [8, 8, 8], offset_ZYX: [0, 0, 0] };
  assert.deepEqual(toLevelSamplePoint_ZYX(scale, [1021, 0, 0]), [127.1875, -0.4375, -0.4375]);
});

test("a texel centroid returns its own level position", () => {
  const scale = { downsample_ZYX: [2, 4, 8], offset_ZYX: [0.5, 1.5, 3.5] };
  for (const texel of [0, 1, 7]) {
    for (const axis of [0, 1, 2]) {
      const d = scale.downsample_ZYX[axis];
      assert.equal(levelSamplePosition(scale, axis, scale.offset_ZYX[axis] + texel * d + (d - 1) / 2), texel);
    }
  }
});

test("the priority of a shown chunk stays below the shown-chunk limit", () => {
  assert.equal(visibleChunkPriority(1e9), SHOWN_CHUNK_PRIORITY_LIMIT - 1);
});

test("a hostname reads shown chunks, then slice-ahead ones, then surface geometry and the display sample, then prefetch", () => {
  for (const later of [SURFACE_GEOMETRY_PRIORITY, DISPLAY_WINDOW_SAMPLE_PRIORITY]) {
    const order = [SHOWN_CHUNK_PRIORITY_LIMIT, SLICE_AHEAD_PREFETCH_PRIORITY, later, PREFETCH_CHUNK_PRIORITY];
    assert.ok(
      order.every((priority, index) => index === 0 || order[index - 1] <= priority),
      `${order} are out of order`,
    );
  }
});

test("a view 100 voxels wide spans 200 by 100 at aspect 2 and 100 by 200 at aspect 0.5", () => {
  assert.deepEqual(visiblePlaneSpan(100, 2), [200, 100]);
  assert.deepEqual(visiblePlaneSpan(100, 0.5), [100, 200]);
});
