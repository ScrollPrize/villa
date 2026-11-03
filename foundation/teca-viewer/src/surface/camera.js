const DEGREES_TO_RADIANS = Math.PI / 180;

function cameraCoefficients(visibleSpan_UV, camera) {
  if (
    !Array.isArray(visibleSpan_UV) ||
    visibleSpan_UV.length !== 2 ||
    visibleSpan_UV.some((value) => !Number.isFinite(value) || value <= 0)
  ) {
    throw new Error("Surface camera span is malformed.");
  }
  if (
    camera === null ||
    typeof camera !== "object" ||
    !Number.isFinite(camera.azimuth) ||
    camera.azimuth < -180 ||
    camera.azimuth > 180 ||
    !Number.isFinite(camera.tilt) ||
    camera.tilt < 0 ||
    camera.tilt > 85 ||
    !Number.isFinite(camera.perspective) ||
    camera.perspective < 0 ||
    camera.perspective > 1 ||
    !Number.isFinite(camera.minimumDepth) ||
    !Number.isFinite(camera.maximumDepth) ||
    camera.minimumDepth > camera.maximumDepth ||
    !Array.isArray(camera.scale_UV) ||
    camera.scale_UV.length !== 2 ||
    camera.scale_UV.some((value) => !Number.isFinite(value) || value <= 0)
  ) {
    throw new Error("Surface camera settings are malformed.");
  }
  const azimuth = camera.azimuth * DEGREES_TO_RADIANS;
  const tilt = camera.tilt * DEGREES_TO_RADIANS;
  const sineTilt = Math.sin(tilt);
  const cosineTilt = Math.cos(tilt);
  const halfSpan = Math.max(...visibleSpan_UV) / 2;
  const halfHeight = visibleSpan_UV[1] / 2;
  const fovTangent = Math.tan(camera.perspective * 45 * DEGREES_TO_RADIANS);
  const cameraDistance = Math.max(
    Math.max(halfSpan, 1) / Math.max(fovTangent, 0.0001),
    Math.tan(tilt) * halfHeight + 1,
    Math.max(camera.maximumDepth, 0) / cosineTilt + 1,
  );
  const inverseDistance = camera.perspective > 0 ? 1 / cameraDistance : 0;
  return {
    cosineAzimuth: Math.cos(azimuth),
    sineAzimuth: Math.sin(azimuth),
    cosineTilt,
    sineTilt,
    inverseDistance,
    perspectiveTilt: sineTilt * inverseDistance,
  };
}

function validateMappingInputs(point_UV, depth, visibleSpan_UV, camera) {
  if (
    !Array.isArray(point_UV) ||
    point_UV.length !== 2 ||
    point_UV.some((value) => !Number.isFinite(value)) ||
    !Number.isFinite(depth) ||
    camera === null ||
    typeof camera !== "object"
  ) {
    throw new Error("Surface camera mapping inputs are malformed.");
  }
  cameraCoefficients(visibleSpan_UV, camera);
}

export function surfaceCameraPointToScreen(point_UV, depth, visibleSpan_UV, camera) {
  validateMappingInputs(point_UV, depth, visibleSpan_UV, camera);
  return surfaceCameraProjector(visibleSpan_UV, camera)(point_UV, depth);
}

function clampDenominator(value) {
  return Math.max(value, 0.0001);
}

export function surfaceCameraProjector(visibleSpan_UV, camera) {
  const [scaleU, scaleV] = camera.scale_UV;
  const { cosineAzimuth, sineAzimuth, cosineTilt, sineTilt, inverseDistance, perspectiveTilt } = cameraCoefficients(
    visibleSpan_UV.map((value, axis) => value / camera.scale_UV[axis]),
    camera,
  );
  const signU = camera.flipU ? -1 : 1;
  const signV = camera.flipV ? 1 : -1;
  return (point_UV, depth) => {
    const [pointU, pointV] = [point_UV[0] / scaleU, point_UV[1] / scaleV];
    const rotatedU = cosineAzimuth * pointU - sineAzimuth * pointV;
    const rotatedV = sineAzimuth * pointU + cosineAzimuth * pointV;
    const screenV =
      (rotatedV * cosineTilt - depth * sineTilt) /
      clampDenominator(1 - depth * cosineTilt * inverseDistance - rotatedV * perspectiveTilt);
    const screenU =
      (rotatedU * (cosineTilt + perspectiveTilt * screenV)) / clampDenominator(cosineTilt - depth * inverseDistance);
    return [screenU * scaleU * signU, screenV * scaleV * signV];
  };
}

export function surfaceCameraScreenToPoint(screen_UV, depth, visibleSpan_UV, camera) {
  validateMappingInputs(screen_UV, depth, visibleSpan_UV, camera);
  const screenFull_UV = [
    (screen_UV[0] * (camera.flipU ? -1 : 1)) / camera.scale_UV[0],
    (screen_UV[1] * (camera.flipV ? 1 : -1)) / camera.scale_UV[1],
  ];
  const visibleSpanFull_UV = visibleSpan_UV.map((value, axis) => value / camera.scale_UV[axis]);
  const { cosineAzimuth, sineAzimuth, cosineTilt, sineTilt, inverseDistance, perspectiveTilt } = cameraCoefficients(
    visibleSpanFull_UV,
    camera,
  );
  const denominator = clampDenominator(cosineTilt + perspectiveTilt * screenFull_UV[1]);
  const rotatedU = (screenFull_UV[0] * (cosineTilt - depth * inverseDistance)) / denominator;
  const rotatedV = (screenFull_UV[1] * (1 - depth * cosineTilt * inverseDistance) + depth * sineTilt) / denominator;
  return [
    (cosineAzimuth * rotatedU + sineAzimuth * rotatedV) * camera.scale_UV[0],
    (-sineAzimuth * rotatedU + cosineAzimuth * rotatedV) * camera.scale_UV[1],
  ];
}

export function surfaceCameraDemandBounds(visibleSpan_UV, minimumDepth, maximumDepth, camera) {
  if (
    !Number.isFinite(minimumDepth) ||
    !Number.isFinite(maximumDepth) ||
    minimumDepth > maximumDepth ||
    minimumDepth < camera?.minimumDepth ||
    maximumDepth > camera?.maximumDepth
  ) {
    throw new Error("Surface camera depth bounds are malformed.");
  }
  const half_UV = visibleSpan_UV.map((value) => value / 2);
  const visibleSpanFull_UV = visibleSpan_UV.map((value, axis) => value / camera.scale_UV[axis]);
  const halfFull_UV = visibleSpanFull_UV.map((value) => value / 2);
  const { cosineTilt, inverseDistance, perspectiveTilt } = cameraCoefficients(visibleSpanFull_UV, camera);
  if (
    cosineTilt - Math.abs(perspectiveTilt) * halfFull_UV[1] <= 0.0001 ||
    maximumDepth * inverseDistance >= cosineTilt - 0.0001
  ) {
    throw new Error("Surface camera does not keep the complete slab in front of the visible viewport.");
  }
  const bounds = camera.compareFlat
    ? [-half_UV[0], -half_UV[1], half_UV[0], half_UV[1]]
    : [Infinity, Infinity, -Infinity, -Infinity];
  for (const depth of new Set([minimumDepth, maximumDepth])) {
    for (const screenU of [-half_UV[0], half_UV[0]]) {
      for (const screenV of [-half_UV[1], half_UV[1]]) {
        const point_UV = surfaceCameraScreenToPoint([screenU, screenV], depth, visibleSpan_UV, camera);
        bounds[0] = Math.min(bounds[0], point_UV[0]);
        bounds[1] = Math.min(bounds[1], point_UV[1]);
        bounds[2] = Math.max(bounds[2], point_UV[0]);
        bounds[3] = Math.max(bounds[3], point_UV[1]);
      }
    }
  }
  const centerOffset_UV = [0, 1].map((axis) => (bounds[axis] + bounds[axis + 2]) / 2);
  const span_UV = [0, 1].map((axis) => {
    const transformedSpan = bounds[axis + 2] - bounds[axis];
    const fitsOriginal = bounds[axis] >= -half_UV[axis] - 1e-6 && bounds[axis + 2] <= half_UV[axis] + 1e-6;
    return fitsOriginal ? visibleSpan_UV[axis] : transformedSpan + 2;
  });
  return { centerOffset_UV, span_UV };
}

export function surfaceCameraShader() {
  return `
    uniform vec2 surfaceScale_UV;
    uniform vec2 surfaceDisplaySign_UV;

    vec2 surfaceCameraScreenToPoint(vec2 screen_UV, float depth, vec2 visibleSpan_UV) {
      vec2 screenFull_UV = screen_UV * surfaceDisplaySign_UV / surfaceScale_UV;
      vec2 visibleSpanFull_UV = visibleSpan_UV / surfaceScale_UV;
      float azimuth = radians(volumetricAzimuth);
      float tilt = radians(volumetricTilt);
      float ca = cos(azimuth);
      float sa = sin(azimuth);
      float ct = cos(tilt);
      float st = sin(tilt);
      float halfSpan = max(max(visibleSpanFull_UV.x, visibleSpanFull_UV.y) * 0.5, 1.0);
      float halfHeight = visibleSpanFull_UV.y * 0.5;
      float fovTangent = tan(volumetricPerspective * radians(45.0));
      float cameraDistance = max(
        halfSpan / max(fovTangent, 0.0001),
        max(tan(tilt) * halfHeight + 1.0, max(volumetricMaximumDepth, 0.0) / ct + 1.0)
      );
      float inverseDistance = volumetricPerspective > 0.0 ? 1.0 / cameraDistance : 0.0;
      float perspectiveTilt = st * inverseDistance;
      float denominator = max(ct + perspectiveTilt * screenFull_UV.y, 0.0001);
      float rotatedU = screenFull_UV.x * (ct - depth * inverseDistance) / denominator;
      float rotatedV = (
        screenFull_UV.y * (1.0 - depth * ct * inverseDistance) + depth * st
      ) / denominator;
      return vec2(ca * rotatedU + sa * rotatedV, -sa * rotatedU + ca * rotatedV) * surfaceScale_UV;
    }

    vec2 surfaceCameraPointToScreen(vec2 point_UV, float depth, vec2 visibleSpan_UV) {
      vec2 pointFull_UV = point_UV / surfaceScale_UV;
      vec2 visibleSpanFull_UV = visibleSpan_UV / surfaceScale_UV;
      float azimuth = radians(volumetricAzimuth);
      float tilt = radians(volumetricTilt);
      float ca = cos(azimuth);
      float sa = sin(azimuth);
      float ct = cos(tilt);
      float st = sin(tilt);
      float halfSpan = max(max(visibleSpanFull_UV.x, visibleSpanFull_UV.y) * 0.5, 1.0);
      float halfHeight = visibleSpanFull_UV.y * 0.5;
      float fovTangent = tan(volumetricPerspective * radians(45.0));
      float cameraDistance = max(
        halfSpan / max(fovTangent, 0.0001),
        max(tan(tilt) * halfHeight + 1.0, max(volumetricMaximumDepth, 0.0) / ct + 1.0)
      );
      float inverseDistance = volumetricPerspective > 0.0 ? 1.0 / cameraDistance : 0.0;
      float perspectiveTilt = st * inverseDistance;
      float rotatedU = ca * pointFull_UV.x - sa * pointFull_UV.y;
      float rotatedV = sa * pointFull_UV.x + ca * pointFull_UV.y;
      float screenV = (rotatedV * ct - depth * st) /
        max(1.0 - depth * ct * inverseDistance - rotatedV * perspectiveTilt, 0.0001);
      float screenU = rotatedU * (ct + perspectiveTilt * screenV) /
        max(ct - depth * inverseDistance, 0.0001);
      return vec2(screenU, screenV) * surfaceScale_UV * surfaceDisplaySign_UV;
    }
  `;
}
