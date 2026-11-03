import assert from "node:assert/strict";
import { test } from "node:test";
import {
  defaultSurfaceVolumetricSettings,
  rotatedSurfaceCamera,
  surfaceVolumetricCamera,
  surfaceVolumetricDepthAtFraction,
  surfaceVolumetricRadius,
  validateSurfaceVolumetricSettings,
} from "../src/surface/volumetric.js";

const settings = (patch) => ({ ...defaultSurfaceVolumetricSettings(), enabled: true, ...patch });

test("three layers behind at spacing 0.5 reach 1.5 from the surface", () => {
  assert.equal(surfaceVolumetricRadius(settings({ layersFront: 0, layersBehind: 3, spacing: 0.5 })), 1.5);
});

test("one layer in front and three behind at spacing 0.5 and relief 2 span depths -3 to 1, the surface at 0.75", () => {
  const twoSided = settings({ layersFront: 1, layersBehind: 3, spacing: 0.5, relief: 2 });
  assert.equal(surfaceVolumetricDepthAtFraction(twoSided, 0), -3);
  assert.equal(surfaceVolumetricDepthAtFraction(twoSided, 0.75), 0);
  assert.equal(surfaceVolumetricDepthAtFraction(twoSided, 1), 1);
  assert.equal(surfaceVolumetricDepthAtFraction({ ...twoSided, reverse: true }, 0), -3);
});

test("32 layers behind at spacing 0.5 and relief 2.5 give the camera depths -40 to 0", () => {
  const camera = surfaceVolumetricCamera(
    settings({ layersFront: 0, layersBehind: 32, spacing: 0.5, relief: 2.5 }),
    [1, 1],
  );
  assert.deepEqual([camera.minimumDepth, camera.maximumDepth], [-40, 0]);
});

test("an opacity range that does not rise, or a flip that is not boolean, is malformed", () => {
  const malformed = /settings are malformed/;
  assert.throws(() => validateSurfaceVolumetricSettings(settings({ opacityMinimum: 1, opacityMaximum: 1 })), malformed);
  assert.throws(() => validateSurfaceVolumetricSettings(settings({ flipU: 1 })), malformed);
});

test("a drag turns the camera 0.4 degrees a pixel, the azimuth wrapping at 180 and the tilt held in 0 to 85", () => {
  assert.deepEqual(rotatedSurfaceCamera({ azimuth: 170, tilt: 80 }, 50, 25), { azimuth: -170, tilt: 85 });
  assert.deepEqual(rotatedSurfaceCamera({ azimuth: 0, tilt: 10 }, -10, -100), { azimuth: -4, tilt: 0 });
});
