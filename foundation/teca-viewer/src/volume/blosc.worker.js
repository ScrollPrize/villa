import { openDirectBloscDecoder } from "./zarr-codecs.js";

self.onmessage = async ({ data: { id, compressor, decodedBytes, path, bytes } }) => {
  try {
    const decode = await openDirectBloscDecoder(compressor, decodedBytes, path);
    const data = await decode(bytes);
    self.postMessage({ id, data }, [data.buffer]);
  } catch (error) {
    self.postMessage({ id, error: error.message });
  }
};
