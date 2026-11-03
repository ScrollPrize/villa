export class DownloadStats {
  constructor() {
    this.recording = false;
    this.downloadBytes = 0;
    this.downloadRequests = 0;
    this.abandonedDownloadBytes = 0;
    this.abandonedDownloadRequests = 0;
    this.reset();
  }

  reset() {
    this.requestSizeHistogram = {};
    this.maximumAbandonedDownloadBytesPerEvent = 0;
  }

  start() {
    this.recording = true;
  }

  stop() {
    this.recording = false;
  }

  download(bytes) {
    this.downloadBytes += bytes;
    this.downloadRequests++;
    if (this.recording) this.requestSizeHistogram[bytes] = (this.requestSizeHistogram[bytes] ?? 0) + 1;
  }

  abandon(bytes) {
    this.abandonedDownloadBytes += bytes;
    this.abandonedDownloadRequests++;
  }

  abandonmentEvent(bytes) {
    if (this.recording)
      this.maximumAbandonedDownloadBytesPerEvent = Math.max(this.maximumAbandonedDownloadBytesPerEvent, bytes);
  }

  snapshot({ includeHistogram = true } = {}) {
    return {
      downloadBytes: this.downloadBytes,
      downloadRequests: this.downloadRequests,
      abandonedDownloadBytes: this.abandonedDownloadBytes,
      abandonedDownloadRequests: this.abandonedDownloadRequests,
      maximumAbandonedDownloadBytesPerEvent: this.maximumAbandonedDownloadBytesPerEvent,
      ...(includeHistogram ? { requestSizeHistogram: { ...this.requestSizeHistogram } } : {}),
    };
  }
}

export const responseDownloads = new DownloadStats();
