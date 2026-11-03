export class SurfacePyramidCache {
  constructor(byteLimit) {
    this.byteLimit = byteLimit;
    this.entries = new Map();
    this.bytes = 0;
  }

  take(key) {
    const entry = this.entries.get(key);
    if (entry === undefined) return null;
    this.entries.delete(key);
    this.bytes -= entry.bytes;
    return entry.pyramid;
  }

  retain(key, pyramid) {
    if (key === null) return;
    const bytes = pyramid.levels[0].data_HWRGBA.buffer.byteLength + pyramid.intersectionBounds_B6.byteLength;
    this.take(key);
    if (bytes > this.byteLimit) return;
    while (this.bytes + bytes > this.byteLimit) this.take(this.entries.keys().next().value);
    this.entries.set(key, { pyramid, bytes });
    this.bytes += bytes;
  }
}
