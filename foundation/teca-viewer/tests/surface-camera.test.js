import assert from "node:assert/strict";
import { test } from "node:test";
import {
  surfaceCameraDemandBounds,
  surfaceCameraPointToScreen,
  surfaceCameraScreenToPoint,
} from "../src/surface/camera.js";
import { defaultSurfaceVolumetricSettings, surfaceVolumetricCamera } from "../src/surface/volumetric.js";

const rounded = (values) => values.map((value) => Number(value.toFixed(6)));

test("a point 14.5 below a tilted perspective slab maps to the screen and back", () => {
  const settings = {
    ...defaultSurfaceVolumetricSettings(),
    enabled: true,
    layersFront: 0,
    layersBehind: 32,
    spacing: 0.5,
    azimuth: 23,
    tilt: 42,
    perspective: 0.35,
    relief: 2.5,
  };
  const camera = surfaceVolumetricCamera(settings, [0.05, 0.05]);
  const screen_UV = surfaceCameraPointToScreen([125, -80], -14.5, [800, 500], camera);
  assert.deepEqual(rounded(surfaceCameraScreenToPoint(screen_UV, -14.5, [800, 500], camera)), [125, -80]);
});

test("the screen center of a slab tilted 30 degrees meets depth -120 at V -3.4641016 on a grid of scale 0.05", () => {
  const settings = {
    ...defaultSurfaceVolumetricSettings(),
    enabled: true,
    layersFront: 0,
    layersBehind: 48,
    tilt: 30,
    relief: 2.5,
  };
  const [u, v] = surfaceCameraScreenToPoint([0, 0], -120, [800, 500], surfaceVolumetricCamera(settings, [0.05, 0.05]));
  assert.equal(u, 0);
  assert.ok(Math.abs(v + 3.4641016) < 1e-6, `${v}`);
});

test("a view 800 by 500 demands 800 by 500 flat and 800 by 5738.86 at a tilt of 85 degrees", () => {
  const flat = surfaceVolumetricCamera(defaultSurfaceVolumetricSettings(), [1, 1]);
  assert.deepEqual(surfaceCameraDemandBounds([800, 500], 0, 0, flat), { centerOffset_UV: [0, 0], span_UV: [800, 500] });
  const tilted = { azimuth: 0, tilt: 85, perspective: 0, minimumDepth: 0, maximumDepth: 0, scale_UV: [1, 1] };
  const { centerOffset_UV, span_UV } = surfaceCameraDemandBounds([800, 500], 0, 0, tilted);
  assert.deepEqual(centerOffset_UV, [0, 0]);
  assert.deepEqual([span_UV[0], Number(span_UV[1].toFixed(2))], [800, 5738.86]);
});

test("the camera shows V downward unless flipped, and flipping U mirrors it", () => {
  for (const enabled of [false, true]) {
    for (const flipU of [false, true]) {
      for (const flipV of [false, true]) {
        const camera = surfaceVolumetricCamera(
          { ...defaultSurfaceVolumetricSettings(), enabled, flipU, flipV },
          [0.5, 0.25],
        );
        const screen_UV = surfaceCameraPointToScreen([7, 11], 0, [100, 80], camera);
        assert.deepEqual(screen_UV, [flipU ? -7 : 7, flipV ? 11 : -11]);
        assert.deepEqual(surfaceCameraScreenToPoint(screen_UV, 0, [100, 80], camera), [7, 11]);
      }
    }
  }
});
