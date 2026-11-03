import { SURFACE_ROTATION_DEGREES_PER_PIXEL, SURFACE_TILT_LIMIT } from "../config.js";
import { defaultSurfaceReliefMask, validateSurfaceReliefMask } from "./relief-mask.js";
import { defaultSurfaceTransfer, validateSurfaceTransfer } from "./transfer.js";

export const SURFACE_VOLUMETRIC_LAYER_LIMIT = 64;

export function defaultSurfaceVolumetricSettings() {
  return {
    enabled: false,
    compareFlat: false,
    layersFront: 8,
    layersBehind: 0,
    spacing: 1,
    reverse: false,
    opacityMinimum: 0,
    opacityMaximum: 0.86,
    opacity: 0.9,
    gamma: 1.5,
    azimuth: 0,
    tilt: 0,
    perspective: 0,
    relief: 1,
    lighting: 0,
    lightAzimuth: -135,
    feature: "intensity",
    gradientGain: 16,
    constantEmission: false,
    transfer: defaultSurfaceTransfer(),
    mask: defaultSurfaceReliefMask(),
  };
}

export function validateSurfaceVolumetricSettings(settings) {
  const keys = [
    "enabled",
    "compareFlat",
    "layersFront",
    "layersBehind",
    "spacing",
    "reverse",
    "opacityMinimum",
    "opacityMaximum",
    "opacity",
    "gamma",
    "azimuth",
    "tilt",
    "perspective",
    "relief",
    "lighting",
    "lightAzimuth",
    "feature",
    "gradientGain",
    "constantEmission",
    "transfer",
    "mask",
  ];
  if (
    settings === null ||
    typeof settings !== "object" ||
    Array.isArray(settings) ||
    Object.keys(settings).filter((key) => key !== "flipU" && key !== "flipV").length !== keys.length ||
    ["flipU", "flipV"].some((key) => key in settings && typeof settings[key] !== "boolean") ||
    keys.some((key) => !Object.hasOwn(settings, key)) ||
    typeof settings.enabled !== "boolean" ||
    typeof settings.compareFlat !== "boolean" ||
    !Number.isSafeInteger(settings.layersFront) ||
    settings.layersFront < 0 ||
    settings.layersFront > SURFACE_VOLUMETRIC_LAYER_LIMIT ||
    !Number.isSafeInteger(settings.layersBehind) ||
    settings.layersBehind < 0 ||
    settings.layersBehind > SURFACE_VOLUMETRIC_LAYER_LIMIT ||
    settings.layersFront + settings.layersBehind < 1 ||
    settings.layersFront + settings.layersBehind > SURFACE_VOLUMETRIC_LAYER_LIMIT ||
    !Number.isFinite(settings.spacing) ||
    settings.spacing < 0.05 ||
    settings.spacing > 8 ||
    typeof settings.reverse !== "boolean" ||
    !Number.isFinite(settings.opacityMinimum) ||
    !Number.isFinite(settings.opacityMaximum) ||
    settings.opacityMinimum < 0 ||
    settings.opacityMaximum > 1 ||
    settings.opacityMinimum >= settings.opacityMaximum ||
    !Number.isFinite(settings.opacity) ||
    settings.opacity < 0.001 ||
    settings.opacity > 1 ||
    !Number.isFinite(settings.gamma) ||
    settings.gamma < 0.1 ||
    settings.gamma > 5 ||
    !Number.isFinite(settings.azimuth) ||
    settings.azimuth < -180 ||
    settings.azimuth > 180 ||
    !Number.isFinite(settings.tilt) ||
    settings.tilt < 0 ||
    settings.tilt > SURFACE_TILT_LIMIT ||
    !Number.isFinite(settings.perspective) ||
    settings.perspective < 0 ||
    settings.perspective > 1 ||
    !Number.isFinite(settings.relief) ||
    settings.relief < 0.1 ||
    settings.relief > 8 ||
    !Number.isFinite(settings.lighting) ||
    settings.lighting < 0 ||
    settings.lighting > 1 ||
    !Number.isFinite(settings.lightAzimuth) ||
    Math.abs(settings.lightAzimuth) > 180 ||
    !["intensity", "normal-gradient", "tangent-gradient", "gradient-magnitude"].includes(settings.feature) ||
    !Number.isFinite(settings.gradientGain) ||
    settings.gradientGain < 0.1 ||
    settings.gradientGain > 256 ||
    typeof settings.constantEmission !== "boolean"
  ) {
    throw new Error("Surface volumetric settings are malformed.");
  }
  return {
    ...settings,
    transfer: validateSurfaceTransfer(settings.transfer),
    mask: validateSurfaceReliefMask(settings.mask),
  };
}

export function surfaceVolumetricRadius(settings) {
  return settings.enabled ? Math.max(settings.layersFront, settings.layersBehind) * settings.spacing : 0;
}

export function surfaceVolumetricDepthAtFraction(settings, depth) {
  const intervals = settings.layersFront + settings.layersBehind;
  return (depth * intervals - settings.layersBehind) * settings.spacing * settings.relief;
}

export function surfaceVolumetricCamera(settings, scale_UV) {
  const orientation = { flipU: settings.flipU === true, flipV: settings.flipV === true };
  return settings.enabled
    ? {
        ...orientation,
        compareFlat: settings.compareFlat,
        azimuth: settings.azimuth,
        tilt: settings.tilt,
        perspective: settings.perspective,
        minimumDepth: -settings.layersBehind * settings.spacing * settings.relief,
        maximumDepth: settings.layersFront * settings.spacing * settings.relief,
        scale_UV: [...scale_UV],
      }
    : {
        ...orientation,
        azimuth: 0,
        tilt: 0,
        perspective: 0,
        minimumDepth: 0,
        maximumDepth: 0,
        scale_UV: [...scale_UV],
      };
}

export function surfaceVolumetricDemandKey(settings, camera) {
  if (!settings.enabled) return "disabled";
  return [
    settings.compareFlat,
    surfaceVolumetricRadius(settings),
    camera.azimuth,
    camera.tilt,
    camera.perspective,
    camera.minimumDepth,
    camera.maximumDepth,
  ].join(":");
}

export function rotatedSurfaceCamera({ azimuth, tilt }, dx, dy) {
  return {
    azimuth: ((((azimuth + dx * SURFACE_ROTATION_DEGREES_PER_PIXEL + 180) % 360) + 360) % 360) - 180,
    tilt: Math.min(SURFACE_TILT_LIMIT, Math.max(0, tilt + dy * SURFACE_ROTATION_DEGREES_PER_PIXEL)),
  };
}

export function surfaceFlatCamera(camera) {
  return { ...camera, azimuth: 0, tilt: 0, perspective: 0 };
}
