// Floating-point predictor 3 has big-endian byte planes whatever the file's byte order.

import { TIFF_TILE_BYTE_LIMIT } from "../config.js";
import { yieldTask } from "./yield-task.js";

async function decodeLzw(encoded, length, signal) {
  const output = new Uint8Array(length);
  const prefix = new Uint16Array(4096);
  const suffix = new Uint8Array(4096);
  const stack = new Uint8Array(4096);
  let width = 9,
    next = 258,
    previous = -1,
    first = 0;
  let input = 0,
    bits = 0,
    accumulator = 0,
    written = 0,
    codes = 0;
  let started = performance.now();
  for (;;) {
    if ((++codes & 8191) === 0) {
      signal?.throwIfAborted();
      if (performance.now() - started >= 4) {
        await yieldTask(signal);
        signal?.throwIfAborted();
        started = performance.now();
      }
    }
    while (bits < width) {
      if (input === encoded.length) throw new Error("TIFF LZW stream is truncated.");
      accumulator = (accumulator << 8) | encoded[input++];
      bits += 8;
    }
    bits -= width;
    let code = (accumulator >>> bits) & ((1 << width) - 1);
    if (code === 257) {
      if (written !== length) throw new Error("TIFF LZW output has an incorrect decoded size.");
      return output;
    }
    if (code === 256) {
      width = 9;
      next = 258;
      previous = -1;
      continue;
    }
    if (code > next || code >= 4096 || (previous === -1 && code >= 256))
      throw new Error("TIFF LZW stream contains an invalid code.");
    const current = code;
    let count = 0;
    if (code === next) {
      stack[count++] = first;
      code = previous;
    }
    while (code >= 258) {
      if (count >= stack.length) throw new Error("TIFF LZW dictionary chain exceeds its bound.");
      stack[count++] = suffix[code];
      code = prefix[code];
    }
    if (code >= 256) throw new Error("TIFF LZW dictionary contains a control code.");
    first = code;
    stack[count++] = first;
    if (written + count > length) throw new Error("TIFF LZW output exceeds the decoded tile size.");
    while (count > 0) output[written++] = stack[--count];
    if (previous !== -1 && next < 4096) {
      prefix[next] = previous;
      suffix[next++] = first;
      // TIFF changes code width one dictionary entry earlier than GIF LZW.
      if (next === (1 << width) - 1 && width < 12) width++;
    }
    previous = current;
  }
}

export async function decodeFloat32Tile(
  encoded,
  { compression, predictor, littleEndian, tileWidth, tileHeight },
  signal,
) {
  signal?.throwIfAborted();
  const length = tileWidth * tileHeight * 4;
  if (!Number.isSafeInteger(length) || length < 4 || length > TIFF_TILE_BYTE_LIMIT)
    throw new Error(`TIFF tile exceeds the ${TIFF_TILE_BYTE_LIMIT / 2 ** 20} MiB decoded limit.`);
  const bytes = compression === 5 ? await decodeLzw(encoded, length, signal) : encoded;
  if (bytes.byteLength !== length) throw new Error("TIFF tile has an incorrect decoded size.");
  const result = new Float32Array(tileWidth * tileHeight);
  const rowBytes = new Uint8Array(tileWidth * 4);
  const rowView = new DataView(rowBytes.buffer);
  let started = performance.now();
  for (let row = 0; row < tileHeight; row++) {
    signal?.throwIfAborted();
    const source = bytes.subarray(row * tileWidth * 4, (row + 1) * tileWidth * 4);
    if (predictor === 3) {
      for (let i = 1; i < source.length; i++) source[i] += source[i - 1];
      for (let column = 0; column < tileWidth; column++)
        for (let byte = 0; byte < 4; byte++) rowBytes[column * 4 + byte] = source[byte * tileWidth + column];
    } else rowBytes.set(source);
    for (let column = 0; column < tileWidth; column++)
      result[row * tileWidth + column] = rowView.getFloat32(column * 4, predictor === 3 ? false : littleEndian);
    if (performance.now() - started >= 4) {
      await yieldTask(signal);
      started = performance.now();
    }
  }
  return result;
}
