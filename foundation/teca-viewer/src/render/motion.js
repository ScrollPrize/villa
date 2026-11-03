import {
  SLICE_AHEAD_FIRST_LATENCY_MS,
  SLICE_AHEAD_HORIZON_MARGIN,
  SLICE_AHEAD_MAX_HORIZON_MS,
  SLICE_AHEAD_MIN_HORIZON_MS,
  SLICE_AHEAD_VELOCITY_WEIGHT,
} from "../config.js";

export class MotionProfile {
  constructor() {
    this.position_ZYX = null;
    this.timestamp = null;
    this.velocity_ZYX = [0, 0, 0];
  }

  observe(position_ZYX, timestamp) {
    if (this.position_ZYX === null) {
      this.position_ZYX = [...position_ZYX];
      this.timestamp = timestamp;
      return;
    }
    const displacement_ZYX = position_ZYX.map((value, axis) => value - this.position_ZYX[axis]);
    if (displacement_ZYX.every((value) => value === 0)) return;
    const elapsed = timestamp - this.timestamp;
    if (elapsed > 0 && elapsed <= SLICE_AHEAD_MAX_HORIZON_MS) {
      this.velocity_ZYX = this.velocity_ZYX.map(
        (velocity, axis) =>
          velocity * (1 - SLICE_AHEAD_VELOCITY_WEIGHT) +
          (displacement_ZYX[axis] / elapsed) * SLICE_AHEAD_VELOCITY_WEIGHT,
      );
    } else {
      this.velocity_ZYX = [0, 0, 0];
    }
    this.position_ZYX = [...position_ZYX];
    this.timestamp = timestamp;
  }

  prediction(position_ZYX, averageLatencyMs, scale, maxLeadChunks, requestWaves) {
    const completionLatencyMs = (averageLatencyMs ?? SLICE_AHEAD_FIRST_LATENCY_MS) * requestWaves;
    const horizonMs = Math.min(
      SLICE_AHEAD_MAX_HORIZON_MS * requestWaves,
      Math.max(SLICE_AHEAD_MIN_HORIZON_MS, completionLatencyMs * SLICE_AHEAD_HORIZON_MARGIN),
    );
    const displacement_ZYX = this.velocity_ZYX.map((velocity, axis) => {
      const chunkWorldSize = scale.chunks_ZYX[axis] * scale.downsample_ZYX[axis];
      const limit = chunkWorldSize * maxLeadChunks;
      return Math.min(limit, Math.max(-limit, velocity * horizonMs));
    });
    const leadChunks = Math.max(
      1,
      ...displacement_ZYX.map(
        (distance, axis) => Math.ceil(Math.abs(distance) / (scale.chunks_ZYX[axis] * scale.downsample_ZYX[axis])) + 1,
      ),
    );
    return {
      center_ZYX: position_ZYX.map((value, axis) => value + displacement_ZYX[axis]),
      displacement_ZYX,
      leadChunks: Math.min(maxLeadChunks, leadChunks),
      horizonMs,
    };
  }
}
