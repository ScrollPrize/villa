const FIRST_RUNS = 256;

function withBytes(packed) {
  return { ...packed, bytes: Object.values(packed).reduce((total, array) => total + array.byteLength, 0) };
}

export class Runs {
  constructor(axis, stride) {
    this.axis = axis;
    this.stride = stride;
    this.offsets_N = new Uint32Array(FIRST_RUNS);
    this.lengths_N = new Uint32Array(FIRST_RUNS);
    this.values_N = new Uint8Array(FIRST_RUNS);
    this.count = 0;
    this.end = -1;
  }

  add(offset, length, value) {
    if (offset === this.end && this.values_N[this.count - 1] === value) {
      this.lengths_N[this.count - 1] += length;
      this.end += length * this.stride;
      return;
    }
    if (this.count === this.offsets_N.length) {
      for (const name of ["offsets_N", "lengths_N", "values_N"]) {
        const grown = new this[name].constructor(this.count * 2);
        grown.set(this[name]);
        this[name] = grown;
      }
    }
    this.offsets_N[this.count] = offset;
    this.lengths_N[this.count] = length;
    this.values_N[this.count] = value;
    this.count += 1;
    this.end = offset + length * this.stride;
  }

  along(axis, stride) {
    this.axis = axis;
    this.stride = stride;
    this.end = -1;
  }

  clear() {
    this.count = 0;
    this.end = -1;
  }

  reset(axis, stride) {
    this.along(axis, stride);
    this.clear();
  }
}

// A grid index may pass 2³², so a piece's three numbers are doubles.
export class Pieces {
  constructor() {
    this.runs = new Runs(0, 1);
    this.chunks_N3 = new Float64Array(FIRST_RUNS * 3);
    this.count = 0;
  }

  enter(index, axis, stride) {
    const last = (this.count - 1) * 3;
    if (this.count > 0 && this.chunks_N3[last] === index && this.chunks_N3[last + 1] === axis) return;
    if (this.count * 3 === this.chunks_N3.length) {
      const grown = new Float64Array(this.count * 6);
      grown.set(this.chunks_N3);
      this.chunks_N3 = grown;
    }
    this.chunks_N3[last + 3] = index;
    this.chunks_N3[last + 4] = axis;
    this.chunks_N3[last + 5] = this.runs.count;
    this.count += 1;
    this.runs.along(axis, stride);
  }

  add(offset, length, value) {
    this.runs.add(offset, length, value);
  }

  views() {
    const { offsets_N, lengths_N, values_N, count } = this.runs;
    return withBytes({
      chunks_N3: this.chunks_N3.subarray(0, this.count * 3),
      offsets_N: offsets_N.subarray(0, count),
      lengths_N: lengths_N.subarray(0, count),
      values_N: values_N.subarray(0, count),
    });
  }

  packed() {
    const { bytes, ...views } = this.views();
    return { ...Object.fromEntries(Object.entries(views).map(([name, view]) => [name, view.slice()])), bytes };
  }
}

export function* piecesOf({ chunks_N3, offsets_N, lengths_N, values_N }) {
  for (let piece = 0; piece < chunks_N3.length; piece += 3) {
    const first = chunks_N3[piece + 2];
    const last = piece + 3 < chunks_N3.length ? chunks_N3[piece + 5] : offsets_N.length;
    yield [
      chunks_N3[piece],
      {
        axis: chunks_N3[piece + 1],
        offsets_N: offsets_N.subarray(first, last),
        lengths_N: lengths_N.subarray(first, last),
        values_N: values_N.subarray(first, last),
      },
    ];
  }
}

export function setTo(runs, value) {
  return { ...runs, values_N: new Uint8Array(runs.offsets_N.length).fill(value) };
}

export class History {
  constructor(byteLimit) {
    this.byteLimit = byteLimit;
    this.done = [];
    this.undone = [];
    this.forgotten = [];
  }

  record(edit) {
    this.done.push(edit);
    this.forgotten.push(...this.undone.map(({ id }) => id));
    this.undone = [];
    this.trim();
  }

  undo() {
    const edit = this.done.pop() ?? null;
    if (edit !== null) this.undone.push(edit);
    return edit;
  }

  redo() {
    const edit = this.undone.pop() ?? null;
    if (edit !== null) this.done.push(edit);
    return edit;
  }

  restore(edits, head) {
    const named = [...head.done, ...head.undone];
    if (named.every((id) => edits.has(id))) {
      this.done = head.done.map((id) => edits.get(id));
      this.undone = head.undone.map((id) => edits.get(id));
    }
    const held = new Set([...this.done, ...this.undone].map(({ id }) => id));
    this.forgotten.push(...[...edits.keys()].filter((id) => !held.has(id)));
    this.trim();
  }

  forget() {
    this.forgotten.push(...[...this.done, ...this.undone].map(({ id }) => id));
    this.done = [];
    this.undone = [];
  }

  head() {
    return { done: this.done.map(({ id }) => id), undone: this.undone.map(({ id }) => id) };
  }

  dropped() {
    const ids = this.forgotten;
    this.forgotten = [];
    return ids;
  }

  refused(ids) {
    this.forgotten.push(...ids);
  }

  bytes() {
    return [...this.done, ...this.undone].reduce((total, edit) => total + edit.bytes, 0);
  }

  trim() {
    while (this.bytes() > this.byteLimit && this.done.length + this.undone.length > 1)
      this.forgotten.push((this.done.length > 0 ? this.done.shift() : this.undone.shift()).id);
  }
}
