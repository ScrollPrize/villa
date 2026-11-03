export class SurfaceReaders {
  constructor({ openVolume }) {
    this.openVolume = openVolume;
    this.kept = null;
  }

  async take(rootUrl, signal) {
    if (this.kept?.rootUrl === rootUrl) return this.kept;
    return this.openVolume(rootUrl, signal);
  }

  keep(reader, byteLimit) {
    if (this.kept !== reader) this.kept?.dispose();
    this.kept = reader;
    if (reader.sourceByteLimit !== byteLimit) reader.setChunkCacheByteLimit(byteLimit);
  }

  discard(reader) {
    if (reader !== this.kept) reader.dispose();
  }

  heldChunk(rootUrl, level, chunk_ZYX, channel) {
    if (this.kept?.rootUrl !== rootUrl) return null;
    return this.kept.heldChunk(...chunk_ZYX, channel, this.kept.scales[level]);
  }

  release() {
    this.kept?.cancelPendingChunks(0);
  }

  drop() {
    this.kept?.dispose();
    this.kept = null;
  }

  setChunkCacheByteLimit(bytes) {
    this.kept?.setChunkCacheByteLimit(bytes);
  }
}
