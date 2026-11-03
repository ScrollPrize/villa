// Arrays of zeros for the chunks the labels first write, made ahead by a worker (spares.worker.js), which writes each
// of their pages, so the kernel zeroes the pages on its thread: writing a byte on each page of 2,684 new 64³ arrays
// took the page 202 to 224 ms, and 2.4 to 2.7 ms in arrays the worker made.

const HELD_BYTES = 32 * 2 ** 20;
const BATCH_BYTES = 8 * 2 ** 20;

export class Spares {
  constructor(ask) {
    this.ask = ask;
    this.length = 0;
    this.held = [];
    this.asked = 0;
  }

  zeroed(length) {
    if (length !== this.length) {
      this.length = length;
      this.held = [];
    }
    const array = this.held.pop() ?? new Uint8Array(length);
    const batch = Math.max(1, Math.floor(BATCH_BYTES / length));
    if ((this.held.length + this.asked + batch) * length <= HELD_BYTES) {
      this.asked += batch;
      this.ask(batch, length);
    }
    return array;
  }

  received(length, buffers) {
    this.asked -= buffers.length;
    if (length === this.length) for (const buffer of buffers) this.held.push(new Uint8Array(buffer));
  }
}
