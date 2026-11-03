// A disjoint GPU discards what was pending.

const SAMPLE_LIMIT = 4096;
const PENDING_LIMIT = 8;

export class GpuTimer {
  constructor(gl) {
    this.gl = gl;
    this.extension = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    this.active = false;
    this.pending = [];
    this.samples = [];
    this.discarded = 0;
  }

  start() {
    this.stop();
    this.samples = [];
    this.discarded = 0;
    this.active = true;
  }

  stop() {
    this.active = false;
    this.collect();
    for (const receipt of this.pending) {
      this.gl.deleteQuery(receipt.query);
      this.discarded++;
    }
    this.pending.length = 0;
  }

  begin(label) {
    const { gl, extension } = this;
    if (!this.active || extension === null || gl.isContextLost()) return null;
    this.collect();
    if (this.pending.length >= PENDING_LIMIT || this.samples.length >= SAMPLE_LIMIT) {
      this.discarded++;
      return null;
    }
    const query = gl.createQuery();
    if (query === null) {
      this.discarded++;
      return null;
    }
    gl.beginQuery(extension.TIME_ELAPSED_EXT, query);
    return { query, label };
  }

  end(receipt) {
    if (receipt === null) return;
    this.gl.endQuery(this.extension.TIME_ELAPSED_EXT);
    this.pending.push(receipt);
  }

  collect() {
    const { gl, extension, pending } = this;
    if (extension === null) return;
    const invalid = gl.isContextLost() || gl.getParameter(extension.GPU_DISJOINT_EXT);
    while (pending.length > 0) {
      const receipt = pending[0];
      if (!invalid && !gl.getQueryParameter(receipt.query, gl.QUERY_RESULT_AVAILABLE)) break;
      pending.shift();
      if (invalid || this.samples.length >= SAMPLE_LIMIT) this.discarded++;
      else
        this.samples.push({
          label: receipt.label,
          milliseconds: gl.getQueryParameter(receipt.query, gl.QUERY_RESULT) / 1e6,
        });
      gl.deleteQuery(receipt.query);
    }
  }

  snapshot() {
    this.collect();
    return {
      supported: this.extension !== null,
      pending: this.pending.length,
      discarded: this.discarded,
      samples: this.samples.map((sample) => ({ ...sample })),
    };
  }
}
