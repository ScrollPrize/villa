import { drawnPanes } from "../views/pose.js";
import { originRequestStats } from "../volume/request-queue.js";
import { InteractionMeter } from "./interaction-meter.js";
import { SurfaceDemandStats } from "./streaming.js";

const METER_SAMPLE_LIMIT = 72_000;

export function createViewerMeter({ state, view, surfaceController, navigationDemand, volumeDownloads }) {
  const surfaceDemandStats = new SurfaceDemandStats();

  const interactionMeter = new InteractionMeter({
    downloads: volumeDownloads,
    gpuTimer: view.rendering.gpuTimer,
    scheduleFrame: (callback) => window.requestAnimationFrame(callback),
    cancelFrame: (handle) => window.cancelAnimationFrame(handle),
    now: () => performance.now(),
    snapshot: ({ includeHistogram = true } = {}) => {
      const navigation = navigationDemand.telemetry();
      const panes = drawnPanes(state, view.paneNames());
      const missing =
        view.slicedPlane.demandStats(panes).missing +
        surfaceDemandStats.sample(panes.includes("surface") ? surfaceController.presentation : null).missing;
      return {
        renderers: view.getRendererStats(),
        chunkQueue: { ...originRequestStats(), ...volumeDownloads.snapshot({ includeHistogram }) },
        textureUploads: { pending: view.rendering.uploadBudget.tasks.size },
        gpuTextureBytes: view.slicedPlane.allocatedBytes() + surfaceController.allocatedBytes(),
        displayScaleIndex: navigation.targetScaleIndex,
        displayedScaleIndex:
          navigation.targetScaleIndex === null
            ? null
            : (view.getVisibleScaleCoverage(navigation.plane, navigation.renderToken)?.scaleIndex ?? null),
        missingRequiredVisibleChunks: missing,
        jsHeapBytes: performance.memory?.usedJSHeapSize ?? null,
      };
    },
    maxSamples: METER_SAMPLE_LIMIT,
  });
  view.rendering.uploadBudget.setFrameObserver((frame) => interactionMeter.recordTextureUploadFrame(frame));
  interactionMeter.reset();
  return interactionMeter;
}
