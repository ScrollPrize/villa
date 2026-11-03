import { saveFile } from "../storage.js";

export class PerformanceControls {
  constructor(meter, container) {
    this.meter = meter;
    this.timer = null;
    this.result = null;
    container.innerHTML = `<h2>Performance</h2>
      <button type="button" data-record>Record performance</button>
      <button type="button" data-export disabled>Save report…</button>
      <pre class="readout-line" data-results role="status"></pre>`;
    this.record = container.querySelector("[data-record]");
    this.export = container.querySelector("[data-export]");
    this.output = container.querySelector("[data-results]");
    this.record.addEventListener("click", () => {
      if (this.timer === null) {
        this.meter.restart();
        this.record.textContent = "Stop recording";
        this.export.disabled = true;
        this.output.textContent = "Recording. Use the viewer, then stop here.\nStops automatically after 60 seconds.";
        this.timer = setTimeout(() => this.stop(), 60_000);
      } else this.stop();
    });
    this.export.addEventListener("click", () => {
      saveFile(
        new Blob([JSON.stringify(this.result, null, 2)], { type: "application/json" }),
        "viewer-performance.json",
      );
    });
  }

  stop() {
    clearTimeout(this.timer);
    this.timer = null;
    this.meter.stop();
    this.result = { metrics: this.meter.stats(), ...this.meter.rawFrames() };
    const { frames, gpu } = this.result.metrics;
    const lines = [`Frame interval p95: ${frames.p95Ms === null ? "no frames" : `${frames.p95Ms.toFixed(2)} ms`}`];
    for (const label of new Set(gpu.samples.map((sample) => sample.label))) {
      const values = gpu.samples
        .filter((sample) => sample.label === label)
        .map((sample) => sample.milliseconds)
        .sort((a, b) => a - b);
      lines.push(
        `${label} GPU p95: ${values[Math.ceil(values.length * 0.95) - 1].toFixed(2)} ms (${values.length} samples)`,
      );
    }
    if (gpu.samples.length === 0) lines.push("GPU timing unavailable or no timed draws.");
    lines.push(`Discarded GPU samples: ${gpu.discarded}`);
    this.output.textContent = lines.join("\n");
    this.record.textContent = "Record performance";
    this.export.disabled = false;
  }
}
