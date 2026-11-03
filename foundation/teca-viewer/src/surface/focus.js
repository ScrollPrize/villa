import { FOCUS_STRIP_BYTES, SURFACE_SEARCH_SLICE_MS } from "../config.js";
import { surfaceFrameAt } from "./frame.js";
import { surfaceNormalProjection } from "./normal-projection.js";
import { STRIP_TEXEL_BYTES } from "./source.js";
import { yieldTask } from "./yield-task.js";

export async function projectSurfaceFocus(point_ZYX, candidates, source, registration, residentFrame, signal) {
  signal.throwIfAborted();
  if (candidates.length === 0) return null;
  // The ±1 normal halo spans at most five vertices when floating-point addition rounds across an integer.
  const sampleWidth = 5;
  const samples_HWRGBA = new Float32Array(sampleWidth * sampleWidth * 4);
  let sampleBounds = null;
  const strips = new Map();
  const stripLimit = Math.max(
    1,
    Math.floor(FOCUS_STRIP_BYTES / (source.stripRows * source.grid_HW[1] * STRIP_TEXEL_BYTES)),
  );
  let deadline = performance.now() + SURFACE_SEARCH_SLICE_MS;
  const frameAt = async (surface_UV) => {
    signal.throwIfAborted();
    if (performance.now() >= deadline) {
      await yieldTask(signal);
      signal.throwIfAborted();
      deadline = performance.now() + SURFACE_SEARCH_SLICE_MS;
    }
    if (surface_UV.some((value, axis) => value < 0 || value > source.grid_HW[1 - axis] - 1)) return null;
    const resident = residentFrame(surface_UV);
    if (resident !== null) return resident;
    const low_UV = surface_UV.map((value) => Math.max(0, Math.floor(value - 1)));
    const high_UV = surface_UV.map((value, axis) => Math.min(source.grid_HW[1 - axis] - 1, Math.floor(value + 1) + 1));
    if (
      sampleBounds === null ||
      low_UV.some((value, axis) => value < sampleBounds.low_UV[axis]) ||
      high_UV.some((value, axis) => value > sampleBounds.high_UV[axis])
    ) {
      for (
        let index = Math.floor(low_UV[1] / source.stripRows);
        index <= Math.floor(high_UV[1] / source.stripRows);
        index++
      ) {
        let strip_HWRGBA = strips.get(index);
        if (strip_HWRGBA === undefined) {
          if (strips.size === stripLimit) strips.delete(strips.keys().next().value);
          strip_HWRGBA = await source.getStrip(index, signal);
          signal.throwIfAborted();
        }
        strips.delete(index);
        strips.set(index, strip_HWRGBA);
        const firstRow = Math.max(low_UV[1], index * source.stripRows);
        const lastRow = Math.min(high_UV[1], (index + 1) * source.stripRows - 1);
        for (let v = firstRow; v <= lastRow; v++) {
          const start = ((v % source.stripRows) * source.grid_HW[1] + low_UV[0]) * 4;
          samples_HWRGBA.set(
            strip_HWRGBA.subarray(start, start + (high_UV[0] - low_UV[0] + 1) * 4),
            (v - low_UV[1]) * sampleWidth * 4,
          );
        }
      }
      sampleBounds = { low_UV, high_UV };
    }
    return surfaceFrameAt(surface_UV, 1, source.grid_HW, ([u, v]) => {
      const offset = ((v - sampleBounds.low_UV[1]) * sampleWidth + u - sampleBounds.low_UV[0]) * 4;
      if (samples_HWRGBA[offset + 3] < 0.5) return null;
      return [0, 1, 2].map(
        (axis) => samples_HWRGBA[offset + axis] * registration.scale_ZYX[axis] + registration.offset_ZYX[axis],
      );
    });
  };
  const refine = async (seed) => {
    const projection = surfaceNormalProjection(point_ZYX, seed, source.grid_HW);
    let step = projection.next();
    while (!step.done) step = projection.next(await frameAt(step.value));
    return step.value;
  };
  const solutions = [];
  for (const seed of candidates) {
    const solution = await refine(seed);
    if (
      solution !== null &&
      !solutions.some(
        (other) => Math.hypot(...other.surface_UV.map((value, axis) => value - solution.surface_UV[axis])) < 1e-3,
      )
    )
      solutions.push(solution);
  }
  solutions.sort((a, b) => a.distanceSquared - b.distanceSquared);
  if (
    solutions.length === 0 ||
    (solutions.length > 1 && solutions[1].distanceSquared - solutions[0].distanceSquared < 1e-3)
  )
    return null;
  return solutions[0];
}
