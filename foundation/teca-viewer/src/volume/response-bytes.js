// Content-Length cannot bound decoded bytes: HTTP compression and CORS can hide their size.

import { responseDownloads } from "./download-stats.js";

const FIRST_BUFFER_BYTES = 64 * 1024;

export async function readResponseBytes(response, byteLimit, signal = null) {
  const contentLength = response.headers.get("Content-Length");
  const declaredLength = contentLength === null ? null : Number(contentLength);
  const reader = response.body?.getReader();
  let cancellation = null;
  const cancel = () => {
    cancellation ??=
      reader === undefined
        ? Promise.resolve(null)
        : reader.cancel().then(
            () => null,
            (error) => error,
          );
  };
  signal?.addEventListener("abort", cancel, { once: true });
  let bytes = null;
  let byteLength = 0;
  try {
    if (
      declaredLength !== null &&
      (!Number.isSafeInteger(declaredLength) || declaredLength < 0 || declaredLength > byteLimit)
    ) {
      throw new Error(`HTTP response declares ${contentLength} bytes, exceeding the ${byteLimit}-byte limit.`);
    }
    while (true) {
      if (signal?.aborted) throw new DOMException("Fetch aborted.", "AbortError");
      if (reader === undefined) break;
      const { done, value } = await reader.read();
      if (signal?.aborted) throw new DOMException("Fetch aborted.", "AbortError");
      if (done) break;
      const nextLength = byteLength + value.byteLength;
      if (nextLength > byteLimit) throw new Error(`HTTP response exceeds the ${byteLimit}-byte limit.`);
      if (bytes === null) bytes = value;
      else {
        if (nextLength > bytes.byteLength) {
          const capacity = Math.min(
            byteLimit,
            Math.max(nextLength, declaredLength ?? FIRST_BUFFER_BYTES, bytes.byteLength * 2),
          );
          const grown = new Uint8Array(capacity);
          grown.set(bytes.subarray(0, byteLength));
          bytes = grown;
        }
        bytes.set(value, byteLength);
      }
      byteLength = nextLength;
    }
  } catch (error) {
    cancel();
    const cause = await cancellation;
    if (cause !== null && cause !== error) error.cause = cause;
    throw error;
  } finally {
    signal?.removeEventListener("abort", cancel);
    reader?.releaseLock();
  }
  if (/^https?:/.test(response.url)) responseDownloads.download(byteLength);
  return bytes === null ? new Uint8Array(0) : byteLength === bytes.byteLength ? bytes : bytes.subarray(0, byteLength);
}
