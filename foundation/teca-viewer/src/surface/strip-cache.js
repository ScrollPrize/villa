export class SurfaceStripCache {
  constructor(byteLimit) {
    if (!Number.isSafeInteger(byteLimit) || byteLimit < 16)
      throw new Error("Surface strip cache must hold at least one RGBA32F texel.");
    this.byteLimit = byteLimit;
    this.bytes = 0;
    this.entries = new Set();
    this.sources = new Map();
  }

  get(source, index) {
    const entry = this.sources.get(source)?.get(index);
    if (entry === undefined) return null;
    this.entries.delete(entry);
    this.entries.add(entry);
    return entry.data_HWRGBA;
  }

  admit(source, index, data_HWRGBA) {
    const existing = this.get(source, index);
    if (existing !== null) return existing;
    if (data_HWRGBA.byteLength > this.byteLimit) return data_HWRGBA;
    while (this.bytes + data_HWRGBA.byteLength > this.byteLimit) {
      const oldest = this.entries.values().next().value;
      this.entries.delete(oldest);
      this.sources.get(oldest.source).delete(oldest.index);
      this.bytes -= oldest.data_HWRGBA.byteLength;
    }
    const entry = { source, index, data_HWRGBA };
    this.entries.add(entry);
    if (!this.sources.has(source)) this.sources.set(source, new Map());
    this.sources.get(source).set(index, entry);
    this.bytes += data_HWRGBA.byteLength;
    return data_HWRGBA;
  }

  release(source) {
    for (const entry of this.sources.get(source)?.values() ?? []) {
      this.entries.delete(entry);
      this.bytes -= entry.data_HWRGBA.byteLength;
    }
    this.sources.delete(source);
  }
}
