import { NETWORK_SPEED_SAMPLES, NETWORK_STATUS_REFRESH_MS } from "../config.js";
import { formatBytes } from "./format-bytes.js";
import { showPieces } from "./readouts.js";

export class NetworkStatus {
  constructor(element, downloads, queueStats) {
    this.element = element;
    this.downloads = downloads;
    this.queueStats = queueStats;
    this.samples = [];
    this.update();
    window.setInterval(() => this.update(), NETWORK_STATUS_REFRESH_MS);
  }

  update() {
    if (document.hidden) {
      this.samples.length = 0;
      return;
    }
    const now = performance.now();
    const bytes = this.downloads.downloadBytes;
    this.samples.push({ time: now, bytes });
    if (this.samples.length > NETWORK_SPEED_SAMPLES) this.samples.shift();
    const oldest = this.samples[0];
    const seconds = (now - oldest.time) / 1000;
    const speed = seconds > 0 ? (bytes - oldest.bytes) / seconds : 0;
    const { active, pending } = this.queueStats();
    const pieces = [`↓ ≈${formatBytes(speed)}/s`, `${formatBytes(bytes)} total`];
    const detail =
      `Streaming data since opening this page: ${formatBytes(bytes)} in ${this.downloads.downloadRequests} completed responses.\n` +
      `${active} active and ${pending} queued data requests.\n` +
      "Estimated speed averages completed response data over 3 seconds, including HTTP cache hits. Bytes read back from download storage or the stored surface geometry are excluded.";
    showPieces(this.element, pieces);
    if (this.element.title !== detail) this.element.title = detail;
  }
}
