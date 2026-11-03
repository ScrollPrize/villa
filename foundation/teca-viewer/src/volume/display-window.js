import {
  DISPLAY_WINDOW_SAMPLE_CHUNKS,
  DISPLAY_WINDOW_SAMPLE_PRIORITY,
  DISPLAY_WINDOW_SAMPLE_VALUES,
  DISPLAY_WINDOW_TAIL,
} from "../config.js";

export async function displaySamples(volume, signal) {
  const scale = volume.scales.at(-1);
  const grid_ZYX = scale.shape_ZYX.map((size, axis) => Math.ceil(size / scale.chunks_ZYX[axis]));
  const gridCount = grid_ZYX[0] * grid_ZYX[1] * grid_ZYX[2];
  const count = Math.min(DISPLAY_WINDOW_SAMPLE_CHUNKS, gridCount);
  const samples = new Array(count);
  const chunkBytes = scale.storageChunks.reduce((total, size) => total * size, scale.numericType.itemSize);
  const concurrency = Math.min(count, Math.max(1, Math.floor(volume.chunkCacheByteLimit / chunkBytes)));
  const controller = new AbortController();
  const abort = () => controller.abort(signal.reason);
  signal?.addEventListener("abort", abort, { once: true });
  if (signal?.aborted) abort();
  let next = 0;
  const read = async () => {
    while (next < count) {
      controller.signal.throwIfAborted();
      const sample = next++;
      const index = count === 1 ? 0 : Math.floor((sample * (gridCount - 1)) / (count - 1));
      const z = Math.floor(index / (grid_ZYX[1] * grid_ZYX[2]));
      const y = Math.floor(index / grid_ZYX[2]) % grid_ZYX[1];
      const x = index % grid_ZYX[2];
      const values = await volume.getChunk(z, y, x, 0, scale, controller.signal, DISPLAY_WINDOW_SAMPLE_PRIORITY);
      controller.signal.throwIfAborted();
      const stride = Math.max(1, Math.ceil(values.length / DISPLAY_WINDOW_SAMPLE_VALUES));
      const sampled = new Float64Array(Math.ceil(values.length / stride));
      let length = 0;
      for (let i = 0; i < values.length; i += stride) {
        const lz = Math.floor(i / (scale.chunks_ZYX[1] * scale.chunks_ZYX[2]));
        const ly = Math.floor(i / scale.chunks_ZYX[2]) % scale.chunks_ZYX[1];
        const lx = i % scale.chunks_ZYX[2];
        if (
          z * scale.chunks_ZYX[0] + lz < scale.shape_ZYX[0] &&
          y * scale.chunks_ZYX[1] + ly < scale.shape_ZYX[1] &&
          x * scale.chunks_ZYX[2] + lx < scale.shape_ZYX[2]
        )
          sampled[length++] = values[i];
      }
      samples[sample] = sampled.subarray(0, length);
    }
  };
  try {
    await Promise.allSettled(
      Array.from({ length: concurrency }, () =>
        read().catch((error) => {
          controller.abort(error);
          throw error;
        }),
      ),
    );
    controller.signal.throwIfAborted();
  } finally {
    signal?.removeEventListener("abort", abort);
  }
  const combined = new Float64Array(samples.reduce((total, values) => total + values.length, 0));
  let offset = 0;
  for (const values of samples) {
    combined.set(values, offset);
    offset += values.length;
  }
  return combined;
}

export function displayWindow(samples) {
  const values = Float64Array.from(samples).filter(Number.isFinite).sort();
  if (values.length === 0) throw new Error("Volume display sample has no finite values.");
  const bounds = [DISPLAY_WINDOW_TAIL, 1 - DISPLAY_WINDOW_TAIL].map((fraction) => {
    const position = fraction * (values.length - 1);
    const index = Math.floor(position);
    const next = Math.min(index + 1, values.length - 1);
    return values[index] + (values[next] - values[index]) * (position - index);
  });
  return { kind: "linear_window", minimum: bounds[0], maximum: bounds[1] };
}

export function displayWindowShader(raw, minimum, maximum) {
  return `(${maximum} > ${minimum} ? clamp((${raw} - ${minimum}) / (${maximum} - ${minimum}), 0.0, 1.0) : 0.0)`;
}
