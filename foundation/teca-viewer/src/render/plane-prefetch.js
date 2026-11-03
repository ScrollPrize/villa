import {
  ROTATION_PREDICTION_FIRST_LATENCY_MS,
  ROTATION_PREDICTION_LATENCY_FACTOR,
  ROTATION_PREDICTION_LIFETIME_MS,
  ROTATION_PREDICTION_MIN_HORIZON_MS,
  ROTATION_PREDICTION_TURN_LIMIT_RADIANS,
} from "../config.js";
import { visiblePlaneSpan } from "../gpu/addresses.js";
import { quatFromAxisAngle, quatMultiply, rotateVectorByQuat } from "../math.js";

export class PlanePrefetch {
  constructor() {
    this.reset();
  }

  reset() {
    this.identity = null;
    this.planes = new Map();
  }

  observe(identity, views, timestamp = performance.now()) {
    if (identity !== this.identity) this.reset();
    this.identity = identity;
    const visible = new Set(views.map(({ plane }) => plane));
    for (const plane of this.planes.keys()) if (!visible.has(plane)) this.planes.delete(plane);
    for (const view of views) {
      const previous = this.planes.get(view.plane)?.current;
      const current = { ...view, center_ZYX: [...view.center_ZYX], rotation: [...view.rotation], timestamp };
      const compatible =
        previous !== undefined &&
        previous.viewWidthVoxels === view.viewWidthVoxels &&
        previous.viewAspect === view.viewAspect &&
        timestamp > previous.timestamp &&
        timestamp - previous.timestamp <= ROTATION_PREDICTION_LIFETIME_MS;
      if (
        compatible &&
        previous.center_ZYX.every((value, axis) => value === view.center_ZYX[axis]) &&
        previous.rotation.every((value, axis) => value === view.rotation[axis])
      )
        continue;
      this.planes.set(view.plane, { previous: compatible ? previous : null, current });
    }
  }

  prediction(plane, scale, averageLatencyMs, timestamp = performance.now()) {
    const motion = this.planes.get(plane);
    if (motion === undefined || motion.previous === null) return null;
    const { previous, current } = motion;
    if (timestamp - current.timestamp > ROTATION_PREDICTION_LIFETIME_MS) return null;
    const horizon = Math.min(
      ROTATION_PREDICTION_LIFETIME_MS,
      Math.max(
        ROTATION_PREDICTION_MIN_HORIZON_MS,
        (averageLatencyMs ?? ROTATION_PREDICTION_FIRST_LATENCY_MS) * ROTATION_PREDICTION_LATENCY_FACTOR,
      ),
    );
    const factor = horizon / (current.timestamp - previous.timestamp);
    const chunkSize_ZYX = scale.chunks_ZYX.map((size, axis) => size * scale.downsample_ZYX[axis]);
    const displacement_ZYX = current.center_ZYX.map((value, axis) => (value - previous.center_ZYX[axis]) * factor);
    const translationLimit = Math.max(
      1,
      ...displacement_ZYX.map((value, axis) => Math.abs(value) / chunkSize_ZYX[axis]),
    );
    const center_ZYX = current.center_ZYX.map((value, axis) => value + displacement_ZYX[axis] / translationLimit);
    const conjugate = previous.rotation.map((value, axis) => (axis === 3 ? value : -value));
    let delta = quatMultiply(current.rotation, conjugate);
    if (delta[3] < 0) delta = delta.map((value) => -value);
    const sine = Math.hypot(...delta.slice(0, 3));
    let basis = current.basis;
    if (sine > 1e-10) {
      const halfDiagonal = Math.hypot(...visiblePlaneSpan(current.viewWidthVoxels, current.viewAspect)) / 2;
      const radians = Math.min(
        2 * Math.atan2(sine, delta[3]) * factor,
        ROTATION_PREDICTION_TURN_LIMIT_RADIANS,
        Math.min(...chunkSize_ZYX) / Math.max(1, halfDiagonal),
      );
      const rotation = quatFromAxisAngle(delta.slice(0, 3), (radians * 180) / Math.PI);
      basis = {
        right_XYZ: rotateVectorByQuat(current.basis.right_XYZ, rotation),
        up_XYZ: rotateVectorByQuat(current.basis.up_XYZ, rotation),
        normal_XYZ: rotateVectorByQuat(current.basis.normal_XYZ, rotation),
      };
    }
    return { center_ZYX, basis };
  }
}
