function percentile(sortedValues_N, fraction) {
  if (sortedValues_N.length === 0) return null;
  return sortedValues_N[Math.ceil(sortedValues_N.length * fraction) - 1];
}

function maximumNullable(values_N) {
  const present_N = values_N.filter((value) => value !== null);
  return present_N.length === 0 ? null : Math.max(...present_N);
}

function compactSnapshot(snapshot, includeHistogram = true) {
  const overview = snapshot.renderers["3d"];
  const programCounts = Object.fromEntries(
    Object.entries(snapshot.renderers).map(([name, renderer]) => [name, renderer.programs]),
  );
  return {
    programCounts,
    programs: Object.values(programCounts).reduce((total, count) => total + count, 0),
    chunkQueueActive: snapshot.chunkQueue?.active ?? 0,
    chunkQueuePending: snapshot.chunkQueue?.pending ?? 0,
    chunkQueueConcurrency: snapshot.chunkQueue?.concurrency ?? 0,
    chunkQueueCompleted: snapshot.chunkQueue?.completed ?? 0,
    downloadBytes: snapshot.chunkQueue?.downloadBytes ?? 0,
    downloadRequests: snapshot.chunkQueue?.downloadRequests ?? 0,
    abandonedDownloadBytes: snapshot.chunkQueue?.abandonedDownloadBytes ?? 0,
    abandonedDownloadRequests: snapshot.chunkQueue?.abandonedDownloadRequests ?? 0,
    maximumAbandonedDownloadBytesPerEvent: snapshot.chunkQueue?.maximumAbandonedDownloadBytesPerEvent ?? 0,
    ...(includeHistogram ? { requestSizeHistogram: snapshot.chunkQueue?.requestSizeHistogram ?? {} } : {}),
    textureUploadPending: snapshot.textureUploads.pending,
    overviewCanvasCopies: overview === undefined ? 0 : overview.canvasCopies,
    overviewCanvasCopyBytes: overview === undefined ? 0 : overview.canvasCopyBytes,
    gpuTextureBytes: snapshot.gpuTextureBytes,
    jsHeapBytes: snapshot.jsHeapBytes,
    displayScaleIndex: snapshot.displayScaleIndex,
    displayedScaleIndex: snapshot.displayedScaleIndex,
    missingRequiredVisibleChunks: snapshot.missingRequiredVisibleChunks,
    scaleGap:
      snapshot.displayScaleIndex === null || snapshot.displayedScaleIndex === null
        ? null
        : snapshot.displayedScaleIndex - snapshot.displayScaleIndex,
  };
}

function weightedBusyFraction(frames_N) {
  let weightedBusy = 0;
  let durationMs = 0;
  for (const frame of frames_N) {
    if (frame.missingRequiredVisibleChunks === 0) continue;
    if (frame.chunkQueueConcurrency === 0) {
      durationMs += frame.deltaMs;
      continue;
    }
    weightedBusy += (frame.deltaMs * frame.chunkQueueActive) / frame.chunkQueueConcurrency;
    durationMs += frame.deltaMs;
  }
  return durationMs === 0 ? null : weightedBusy / durationMs;
}

function requestHistogramDelta(current, baseline) {
  return Object.fromEntries(
    Object.entries(current).flatMap(([size, count]) => {
      const delta = count - (baseline[size] ?? 0);
      return delta === 0 ? [] : [[size, delta]];
    }),
  );
}

function maximumRollingBytesPerSecond(frames_N, baselineBytes, startedAtMs) {
  const windowMs = 1000;
  let maximum = 0;
  let startIndex = 0;
  for (let endIndex = 0; endIndex < frames_N.length; endIndex++) {
    const end = frames_N[endIndex];
    while (startIndex < endIndex && end.atMs - frames_N[startIndex].atMs > windowMs) startIndex += 1;
    const startBytes = startIndex === 0 ? baselineBytes : frames_N[startIndex - 1].downloadBytes;
    const durationMs = end.atMs - (startIndex === 0 ? startedAtMs : frames_N[startIndex - 1].atMs);
    if (durationMs >= windowMs * 0.9) {
      maximum = Math.max(maximum, ((end.downloadBytes - startBytes) * 1000) / durationMs);
    }
  }
  return maximum;
}

export class InteractionMeter {
  constructor({ scheduleFrame, cancelFrame, now, snapshot, maxSamples, downloads, gpuTimer }) {
    if (!Number.isInteger(maxSamples) || maxSamples <= 0) {
      throw new Error("Interaction meter maxSamples must be a positive integer.");
    }
    this.scheduleFrame = scheduleFrame;
    this.cancelFrame = cancelFrame;
    this.now = now;
    this.snapshot = snapshot;
    this.downloads = downloads;
    this.gpuTimer = gpuTimer;
    this.maxSamples = maxSamples;
    this.frameHandle = null;
    this.lastTimestampMs = null;
    this.frames = [];
    this.uploadFrames = [];
    this.lastUploadFrame = null;
    this.longTasks = [];
    this.droppedLongTaskSamples = 0;
    this.droppedFrameSamples = 0;
    this.droppedUploadSamples = 0;
    this.startedAtMs = null;
    this.baseline = null;
    this.current = null;
    this.maxPrograms = 0;
    this.boundFrame = (timestampMs) => this.recordFrame(timestampMs);
    this.longTaskObserver = globalThis.PerformanceObserver?.supportedEntryTypes.includes("longtask")
      ? new PerformanceObserver((list) => this.recordLongTasks(list.getEntries()))
      : null;
  }

  start() {
    if (this.frameHandle !== null) throw new Error("Interaction meter is already running.");
    this.reset();
    this.downloads.start();
    this.gpuTimer.start();
    this.frameHandle = this.scheduleFrame(this.boundFrame);
    this.longTaskObserver?.observe({ type: "longtask" });
  }

  restart() {
    if (this.frameHandle === null) {
      this.start();
      return;
    }
    this.reset();
    this.gpuTimer.start();
  }

  stop() {
    if (this.frameHandle === null) return;
    this.current = compactSnapshot(this.snapshot());
    this.downloads.stop();
    this.cancelFrame(this.frameHandle);
    if (this.longTaskObserver !== null) {
      this.recordLongTasks(this.longTaskObserver.takeRecords());
      this.longTaskObserver.disconnect();
    }
    this.frameHandle = null;
    this.gpuTimer.stop();
  }

  reset() {
    this.downloads.reset();
    const snapshot = compactSnapshot(this.snapshot());
    this.frames = [];
    this.uploadFrames = [];
    this.lastUploadFrame = null;
    this.longTasks = [];
    this.droppedLongTaskSamples = 0;
    this.droppedFrameSamples = 0;
    this.droppedUploadSamples = 0;
    this.startedAtMs = this.now();
    this.lastTimestampMs = null;
    this.baseline = snapshot;
    this.current = snapshot;
    this.maxPrograms = snapshot.programs;
  }

  recordLongTasks(entries) {
    for (const entry of entries) {
      if (this.frameHandle === null || entry.startTime < this.startedAtMs) continue;
      this.appendBounded(
        this.longTasks,
        { atMs: entry.startTime, durationMs: entry.duration },
        "droppedLongTaskSamples",
      );
    }
  }

  recordFrame(timestampMs) {
    const snapshot = compactSnapshot(this.snapshot({ includeHistogram: false }), false);
    if (this.lastTimestampMs !== null) {
      this.appendBounded(
        this.frames,
        {
          atMs: timestampMs,
          deltaMs: timestampMs - this.lastTimestampMs,
          ...snapshot,
        },
        "droppedFrameSamples",
      );
    }
    this.current = snapshot;
    this.maxPrograms = Math.max(this.maxPrograms, snapshot.programs);
    this.lastTimestampMs = timestampMs;
    this.frameHandle = this.scheduleFrame(this.boundFrame);
  }

  recordTextureUploadFrame({ timestampMs, bytes, jobs, durationMs, pending }) {
    if (this.frameHandle === null) return;
    if (timestampMs !== undefined && this.lastUploadFrame?.atMs === timestampMs) {
      this.lastUploadFrame.bytes += bytes;
      this.lastUploadFrame.jobs += jobs;
      this.lastUploadFrame.durationMs += durationMs;
      this.lastUploadFrame.pending += pending;
      return;
    }
    const frame = { atMs: timestampMs ?? this.now(), bytes, jobs, durationMs, pending };
    this.lastUploadFrame = frame;
    this.appendBounded(this.uploadFrames, frame, "droppedUploadSamples");
  }

  stats() {
    const current = this.frameHandle === null ? this.current : compactSnapshot(this.snapshot());
    this.current = current;
    this.maxPrograms = Math.max(this.maxPrograms, current.programs);
    const frames = this.orderedSamples(this.frames, this.droppedFrameSamples);
    const uploadFrames = this.orderedSamples(this.uploadFrames, this.droppedUploadSamples);
    const frameTimes_N = frames.map((frame) => frame.deltaMs).sort((a, b) => a - b);
    const uploadBytes_N = uploadFrames.map((frame) => frame.bytes);
    const uploadDurations_N = uploadFrames.map((frame) => frame.durationMs);
    const contexts = Object.fromEntries(
      Object.keys(current.programCounts).map((name) => [
        name,
        {
          baseline: this.baseline.programCounts[name],
          current: current.programCounts[name],
        },
      ]),
    );
    return {
      gpu: this.gpuTimer.snapshot(),
      startedAtMs: this.startedAtMs,
      durationMs: this.now() - this.startedAtMs,
      frames: {
        samples: this.frames.length,
        droppedSamples: this.droppedFrameSamples,
        p50Ms: percentile(frameTimes_N, 0.5),
        p95Ms: percentile(frameTimes_N, 0.95),
        p99Ms: percentile(frameTimes_N, 0.99),
        maximumMs: frameTimes_N.length === 0 ? null : frameTimes_N.at(-1),
        hitchesOver50Ms: frameTimes_N.filter((value) => value > 50).length,
      },
      longTasks: {
        supported: this.longTaskObserver !== null,
        samples: this.longTasks.length,
        droppedSamples: this.droppedLongTaskSamples,
        totalMs: this.longTasks.reduce((sum, entry) => sum + entry.durationMs, 0),
        maximumMs: this.longTasks.reduce((maximum, entry) => Math.max(maximum, entry.durationMs), 0),
      },
      programs: {
        baseline: this.baseline.programs,
        current: current.programs,
        maximum: this.maxPrograms,
        delta: current.programs - this.baseline.programs,
        contexts,
      },
      queues: {
        current: {
          chunkActive: current.chunkQueueActive,
          chunkPending: current.chunkQueuePending,
          chunkConcurrency: current.chunkQueueConcurrency,
          textureUploadPending: current.textureUploadPending,
        },
        maximum: {
          chunkActive: maximumNullable(this.frames.map((frame) => frame.chunkQueueActive)),
          chunkPending: maximumNullable(this.frames.map((frame) => frame.chunkQueuePending)),
          textureUploadPending: maximumNullable(this.frames.map((frame) => frame.textureUploadPending)),
        },
      },
      network: {
        bytes: current.downloadBytes - this.baseline.downloadBytes,
        requests: current.downloadRequests - this.baseline.downloadRequests,
        abandonedBytes: current.abandonedDownloadBytes - this.baseline.abandonedDownloadBytes,
        abandonedRequests: current.abandonedDownloadRequests - this.baseline.abandonedDownloadRequests,
        maximumAbandonedBytesPerEvent: current.maximumAbandonedDownloadBytesPerEvent,
        completedQueueRequests: current.chunkQueueCompleted - this.baseline.chunkQueueCompleted,
        requestSizeHistogram: requestHistogramDelta(current.requestSizeHistogram, this.baseline.requestSizeHistogram),
        maximumRollingBytesPerSecond: maximumRollingBytesPerSecond(
          frames,
          this.baseline.downloadBytes,
          this.startedAtMs,
        ),
        connectionBusyFractionWhileMissing: weightedBusyFraction(this.frames),
      },
      scaleGap: {
        current: current.scaleGap,
        maximum: maximumNullable(this.frames.map((frame) => frame.scaleGap)),
      },
      uploads: {
        frames: this.uploadFrames.length,
        droppedSamples: this.droppedUploadSamples,
        jobs: this.uploadFrames.reduce((total, frame) => total + frame.jobs, 0),
        bytes: this.uploadFrames.reduce((total, frame) => total + frame.bytes, 0),
        maximumFrameBytes: uploadBytes_N.length === 0 ? 0 : Math.max(...uploadBytes_N),
        maximumFrameMs: uploadDurations_N.length === 0 ? 0 : Math.max(...uploadDurations_N),
        overviewCanvasCopies: current.overviewCanvasCopies - this.baseline.overviewCanvasCopies,
        overviewCanvasBytes: current.overviewCanvasCopyBytes - this.baseline.overviewCanvasCopyBytes,
      },
      residency: {
        currentGpuTextureBytes: current.gpuTextureBytes,
        maximumGpuTextureBytes: maximumNullable(this.frames.map((frame) => frame.gpuTextureBytes)),
        currentJsHeapBytes: current.jsHeapBytes,
        maximumJsHeapBytes: maximumNullable(this.frames.map((frame) => frame.jsHeapBytes)),
      },
    };
  }

  rawFrames() {
    return {
      frames: this.orderedSamples(this.frames, this.droppedFrameSamples).map((frame) => ({
        ...frame,
        programCounts: { ...frame.programCounts },
      })),
      uploadFrames: this.orderedSamples(this.uploadFrames, this.droppedUploadSamples).map((frame) => ({ ...frame })),
      droppedFrameSamples: this.droppedFrameSamples,
      droppedUploadSamples: this.droppedUploadSamples,
      longTasks: this.orderedSamples(this.longTasks, this.droppedLongTaskSamples).map((entry) => ({ ...entry })),
      droppedLongTaskSamples: this.droppedLongTaskSamples,
    };
  }

  appendBounded(samples, sample, droppedCounter) {
    if (samples.length === this.maxSamples) {
      samples[this[droppedCounter] % this.maxSamples] = sample;
      this[droppedCounter] += 1;
    } else {
      samples.push(sample);
    }
  }

  orderedSamples(samples, dropped) {
    const start = dropped % this.maxSamples;
    return start === 0 ? samples : [...samples.slice(start), ...samples.slice(0, start)];
  }
}
