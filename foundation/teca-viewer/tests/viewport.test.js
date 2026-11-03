import assert from "node:assert/strict";
import { test } from "node:test";
import {
  constrainSurfacePose,
  surfaceOffsetAtScreen,
  surfaceVisibleSpan,
  validateSurfaceWork,
} from "../src/surface/viewport.js";
import {
  defaultSurfaceVolumetricSettings,
  surfaceFlatCamera,
  surfaceVolumetricCamera,
} from "../src/surface/volumetric.js";

test("a pose outside the grid and the limits is pulled onto the last cell, zoom 512 and offset -256", () => {
  assert.deepEqual(
    constrainSurfacePose({ center_UV: [-10, 2000], zoom: 1000, normalOffset: -400, normalSign: -1 }, [1000, 2000]),
    { center_UV: [0, 999], zoom: 512, normalOffset: -256, normalSign: -1 },
  );
});

test("zoom 1 shows 1024 cells across the shorter side, and anisotropic cells keep their UV scale", () => {
  assert.deepEqual(surfaceVisibleSpan([2472, 6704], [1, 1], 1, 2), [2048, 1024]);
  assert.deepEqual(surfaceVisibleSpan([2472, 6704], [1, 1], 2, 0.5), [512, 1024]);
  const [u, v] = surfaceVisibleSpan([1000, 1000], [2, 1], 2, 2);
  assert.equal(u / 2 / v, 2);
});

test("a point a quarter pane right and up of center is offset (20, -10) in an 80 by 40 flat view, and (20, 10) with V flipped", () => {
  const settings = { ...defaultSurfaceVolumetricSettings(), enabled: true, azimuth: 20, tilt: 30 };
  const flat = surfaceFlatCamera(surfaceVolumetricCamera(settings, [1, 1]));
  assert.deepEqual(surfaceOffsetAtScreen([0.25, -0.25], [80, 40], flat), [20, -10]);
  assert.deepEqual(surfaceOffsetAtScreen([0.25, -0.25], [80, 40], { ...flat, flipV: true }), [20, 10]);
});

test("64 layers on an 1800² pane fit the frame budget, and a tilt that doubles their samples does not", () => {
  const volumetric = { ...defaultSurfaceVolumetricSettings(), enabled: true, layersFront: 64 };
  validateSurfaceWork(volumetric, [1800, 1800]);
  assert.throws(() => validateSurfaceWork({ ...volumetric, tilt: 10 }, [1800, 1800]), /interactive budget/);
});
