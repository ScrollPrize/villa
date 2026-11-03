export const SURFACE_TRANSFER_POINT_LIMIT = 8;

const PRESETS = Object.freeze({
  linear: Object.freeze([
    Object.freeze({ value: 0, color: "#ffffff", opacity: 0 }),
    Object.freeze({ value: 1, color: "#ffffff", opacity: 1 }),
  ]),
  ink: Object.freeze([
    Object.freeze({ value: 0, color: "#ff4242", opacity: 0 }),
    Object.freeze({ value: 0.22, color: "#ff4242", opacity: 0 }),
    Object.freeze({ value: 0.34, color: "#dbc34d", opacity: 0.12 }),
    Object.freeze({ value: 0.37, color: "#cbb962", opacity: 0.22 }),
    Object.freeze({ value: 1, color: "#ffffff", opacity: 0.35 }),
  ]),
  fiber: Object.freeze([
    Object.freeze({ value: 0, color: "#17212b", opacity: 0 }),
    Object.freeze({ value: 0.2, color: "#35566d", opacity: 0.04 }),
    Object.freeze({ value: 0.5, color: "#8db6af", opacity: 0.2 }),
    Object.freeze({ value: 1, color: "#f3e1bd", opacity: 0.4 }),
  ]),
});

export function surfaceTransferPreset(name) {
  if (!Object.hasOwn(PRESETS, name)) throw new Error(`Unknown Surface transfer preset: ${name}.`);
  return PRESETS[name].map((point) => ({ ...point }));
}

export function defaultSurfaceTransfer() {
  return { preset: "linear", points: surfaceTransferPreset("linear") };
}

export function validateSurfaceTransfer(transfer) {
  if (
    transfer === null ||
    typeof transfer !== "object" ||
    Array.isArray(transfer) ||
    Object.keys(transfer).length !== 2 ||
    !Object.hasOwn(transfer, "preset") ||
    !Object.hasOwn(transfer, "points") ||
    (transfer.preset !== "custom" && !Object.hasOwn(PRESETS, transfer.preset)) ||
    !Array.isArray(transfer.points) ||
    transfer.points.length < 2 ||
    transfer.points.length > SURFACE_TRANSFER_POINT_LIMIT
  ) {
    throw new Error("Surface transfer curve is malformed.");
  }
  let previous = -1;
  const points = transfer.points.map((point) => {
    if (
      point === null ||
      typeof point !== "object" ||
      Array.isArray(point) ||
      Object.keys(point).length !== 3 ||
      !Object.hasOwn(point, "value") ||
      !Object.hasOwn(point, "color") ||
      !Object.hasOwn(point, "opacity") ||
      !Number.isFinite(point.value) ||
      point.value < 0 ||
      point.value > 1 ||
      point.value <= previous ||
      !/^#[0-9a-fA-F]{6}$/.test(point.color) ||
      !Number.isFinite(point.opacity) ||
      point.opacity < 0 ||
      point.opacity > 1
    ) {
      throw new Error("Surface transfer curve points are malformed.");
    }
    previous = point.value;
    return { ...point, color: point.color.toLowerCase() };
  });
  if (points[0].value !== 0 || points.at(-1).value !== 1) {
    throw new Error("Surface transfer curves must start at 0 and end at 1.");
  }
  if (
    transfer.preset !== "custom" &&
    JSON.stringify(points) !== JSON.stringify(surfaceTransferPreset(transfer.preset))
  ) {
    throw new Error("Named Surface transfer presets must retain their canonical points.");
  }
  return { preset: transfer.preset, points };
}

export function surfaceTransferColor(color) {
  if (!/^#[0-9a-fA-F]{6}$/.test(color)) throw new Error("Surface transfer color is malformed.");
  return [1, 3, 5].map((start) => Number.parseInt(color.slice(start, start + 2), 16) / 255);
}

export function evaluateSurfaceTransfer(value, transfer) {
  const { points } = validateSurfaceTransfer(transfer);
  if (!Number.isFinite(value)) throw new Error("Surface transfer input must be finite.");
  const admitted = Math.min(1, Math.max(0, value));
  let high = 1;
  while (high < points.length - 1 && admitted > points[high].value) high += 1;
  const low = high - 1;
  const weight = (admitted - points[low].value) / (points[high].value - points[low].value);
  const lowColor = surfaceTransferColor(points[low].color);
  const highColor = surfaceTransferColor(points[high].color);
  return {
    color: lowColor.map((channel, index) => channel + (highColor[index] - channel) * weight),
    opacity: points[low].opacity + (points[high].opacity - points[low].opacity) * weight,
  };
}
