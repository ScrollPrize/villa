import { PREDICTED_CHUNK_LIMIT } from "../config.js";
import { rotatedPlaneChunkAddresses } from "./addresses.js";

export function planPlaneDemand({ center_ZYX, scales, views }) {
  return views.map((view) =>
    scales.map((scale, index) => {
      const addresses = rotatedPlaneChunkAddresses(
        center_ZYX,
        view.basis,
        view.width,
        scale,
        view.coverage,
        view.aspect,
      );
      const prediction = view.predictions[index];
      if (prediction === null) return { addresses, prefetch: [] };
      let predicted;
      try {
        predicted = rotatedPlaneChunkAddresses(
          prediction.center_ZYX,
          prediction.basis,
          view.width,
          scale,
          view.coverage,
          view.aspect,
        );
      } catch (error) {
        if (!(error instanceof RangeError)) throw error;
        return { addresses, prefetch: [] };
      }
      const shown = new Set(addresses.map(({ chunk_ZYX }) => chunk_ZYX.join("/")));
      const prefetch = predicted
        .filter(({ chunk_ZYX }) => !shown.has(chunk_ZYX.join("/")))
        .slice(0, PREDICTED_CHUNK_LIMIT)
        .map((address) => ({ ...address, fetchOrder: 1 }));
      return { addresses, prefetch };
    }),
  );
}
