import {
  IDLE_UPLOAD_BYTES_PER_FRAME,
  INTERACTION_SETTLE_DELAY_MS,
  INTERACTIVE_UPLOAD_BYTES_LIMIT,
  UPLOAD_BYTES_PER_FRAME,
  UPLOAD_TIME_LIMIT_MS,
} from "../config.js";

// Quotas come in whole multiples of this: one RGBA32F texel, the widest texel an atlas uploads.
export const UPLOAD_QUANTUM_BYTES = 16;

export class FrameUploadBudget {
  constructor(onError) {
    this.onError = onError;
    this.byteLimit = UPLOAD_BYTES_PER_FRAME;
    this.idleByteLimit = IDLE_UPLOAD_BYTES_PER_FRAME;
    this.interactiveByteLimit = this.byteLimit;
    this.interactiveMaxByteLimit = INTERACTIVE_UPLOAD_BYTES_LIMIT;
    this.timeLimitMs = UPLOAD_TIME_LIMIT_MS;
    this.deadline = Infinity;
    this.now = () => performance.now();
    this.interactionUntil = 0;
    this.timestamp = null;
    this.frameByteLimit = this.byteLimit;
    this.taskByteLimit = Infinity;
    this.bytes = 0;
    this.frameObserver = null;
    this.tasks = new Map();
    this.frame = null;
  }

  interact() {
    this.interactionUntil = this.now() + INTERACTION_SETTLE_DELAY_MS;
  }

  setFrameObserver(observer) {
    this.frameObserver = observer;
  }

  available(timestamp) {
    if (timestamp !== this.timestamp) {
      this.timestamp = timestamp;
      this.bytes = 0;
      this.frameByteLimit = this.now() < this.interactionUntil ? this.interactiveByteLimit : this.idleByteLimit;
    }
    if (this.now() >= this.deadline) return 0;
    return Math.max(0, Math.min(this.frameByteLimit, this.taskByteLimit) - this.bytes);
  }

  spend(bytes) {
    if (
      !Number.isSafeInteger(bytes) ||
      bytes < 0 ||
      this.bytes + bytes > Math.min(this.frameByteLimit, this.taskByteLimit)
    )
      throw new Error("Texture uploads exceeded the shared animation-frame byte quota.");
    this.bytes += bytes;
  }

  schedule(owner, priority, run) {
    this.tasks.set(owner, { priority, run });
    if (this.frame !== null) return;
    this.frame = window.requestAnimationFrame((timestamp) => {
      this.frame = null;
      this.available(timestamp);
      const beforeBytes = this.bytes;
      const started = this.now();
      this.deadline = started + this.timeLimitMs;
      let jobs = 0;
      const tasks = [...this.tasks].sort((a, b) => a[1].priority - b[1].priority);
      for (let index = 0; index < tasks.length; index++) {
        const [owner, task] = tasks[index];
        if (this.tasks.get(owner) !== task) continue;
        const peers = tasks
          .slice(index)
          .filter(
            ([otherOwner, other]) => other.priority === task.priority && this.tasks.get(otherOwner) === other,
          ).length;
        this.tasks.delete(owner);
        const available = this.available(timestamp);
        const taskBytes = this.bytes;
        this.taskByteLimit =
          this.bytes +
          Math.min(
            available,
            Math.max(UPLOAD_QUANTUM_BYTES, Math.floor(available / peers / UPLOAD_QUANTUM_BYTES) * UPLOAD_QUANTUM_BYTES),
          );
        try {
          task.run(timestamp);
          if (this.bytes > taskBytes) jobs++;
        } catch (error) {
          this.onError(error);
        } finally {
          this.taskByteLimit = Infinity;
        }
      }
      const durationMs = this.now() - started;
      this.deadline = Infinity;
      if (started < this.interactionUntil && this.bytes > beforeBytes) {
        if (durationMs > this.timeLimitMs)
          this.interactiveByteLimit = Math.max(
            this.byteLimit,
            Math.floor(this.interactiveByteLimit / 2 / UPLOAD_QUANTUM_BYTES) * UPLOAD_QUANTUM_BYTES,
          );
        else if (durationMs < this.timeLimitMs / 2 && this.bytes >= this.frameByteLimit)
          this.interactiveByteLimit = Math.min(
            this.interactiveMaxByteLimit,
            this.interactiveByteLimit + this.byteLimit,
          );
      }
      if (this.bytes > beforeBytes)
        this.frameObserver?.({
          timestampMs: timestamp,
          bytes: this.bytes - beforeBytes,
          jobs,
          durationMs,
          pending: this.tasks.size,
        });
    });
  }

  cancel(owner) {
    this.tasks.delete(owner);
    if (this.tasks.size === 0 && this.frame !== null) {
      window.cancelAnimationFrame(this.frame);
      this.frame = null;
    }
  }
}
