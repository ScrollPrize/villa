import { displayWindow } from "./display-window.js";
import { readResponseBytes } from "./response-bytes.js";

self.onmessage = async ({ data }) => {
  const { id, kind, bytes, expectedBytes, samples } = data;
  try {
    if (kind === "window") {
      self.postMessage({ id, result: displayWindow(samples) });
    } else if (kind === "gzip") {
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
      const result = await readResponseBytes(new Response(stream), expectedBytes, null);
      if (result.byteLength !== expectedBytes) throw new Error("Gzip chunk decoded to an unexpected size.");
      self.postMessage({ id, result }, [result.buffer]);
    } else throw new Error(`Unsupported source decode operation: ${kind}.`);
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
