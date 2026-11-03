import { DEFAULT_ZOOM } from "../config.js";

function validateRestoredView(state, volume) {
  if (state.center_ZYX.some((value, axis) => value < 0 || value >= volume.fullShape_ZYX[axis])) {
    throw new Error(
      `Restored center ${state.center_ZYX.join(",")} is outside source shape ${volume.fullShape_ZYX.join("×")}.`,
    );
  }
  for (const key of ["manualScaleIndex", "manualCoarsestScaleIndex"]) {
    if (state[key] !== null && state[key] >= volume.scales.length) {
      throw new Error(`Restored level ${state[key]} is outside 0..${volume.scales.length - 1}.`);
    }
  }
  if (state.selectedChannel >= volume.metadata.channelCount) {
    throw new Error(`Restored channel ${state.selectedChannel} is outside 0..${volume.metadata.channelCount - 1}.`);
  }
}

export class PrimarySourceController {
  constructor({
    state,
    openVolume,
    cancelCompetingWork,
    resetView,
    presentVolume,
    clearPlaneTiles,
    markViewDirty,
    renderVolume,
    publishSession,
    refreshScaleControls,
    refreshSourceMetadata,
    setSourceInput,
    setStatus,
    reportError,
  }) {
    this.state = state;
    this.openVolume = openVolume;
    this.cancelCompetingWork = cancelCompetingWork;
    this.resetView = resetView;
    this.presentVolume = presentVolume;
    this.clearPlaneTiles = clearPlaneTiles;
    this.markViewDirty = markViewDirty;
    this.renderVolume = renderVolume;
    this.publishSession = publishSession;
    this.refreshScaleControls = refreshScaleControls;
    this.refreshSourceMetadata = refreshSourceMetadata;
    this.setSourceInput = setSourceInput;
    this.setStatus = setStatus;
    this.reportError = reportError;
    this.loadAbortController = null;
  }

  async open(input, { restoreView }) {
    const location = input.trim();
    if (!/^https?:\/\//.test(location)) {
      this.setStatus(`Enter an HTTP(S) Zarr URL, not ${JSON.stringify(location)}.`);
      return false;
    }
    const state = this.state;
    this.setStatus(`Loading metadata for ${location}...`);
    this.setSourceInput(location);
    const generation = ++state.workGeneration;
    this.loadAbortController?.abort();
    const controller = new AbortController();
    this.loadAbortController = controller;
    this.cancelCompetingWork();
    let volume = null;
    try {
      volume = await this.openVolume(location, controller.signal);
      if (generation !== state.workGeneration || controller.signal.aborted) {
        volume.dispose();
        return false;
      }
      if (restoreView) validateRestoredView(state, volume);
      const previous = state.volume;
      if (!state.sources.includes(location)) state.sources.push(location);
      state.activeSource = location;
      state.contentGeneration += 1;
      volume.contentNamespace = `content-${state.contentGeneration}`;
      state.volume = volume;
      if (!restoreView) {
        state.selectedChannel = 0;
        state.manualScaleIndex = null;
        state.manualCoarsestScaleIndex = null;
        this.resetView(
          volume.fullShape_ZYX.map((value) => Math.floor(value / 2)),
          DEFAULT_ZOOM,
        );
      }
      this.presentVolume();
      this.renderVolume();
      previous?.dispose();
      this.publishSession();
      return true;
    } catch (error) {
      if (volume !== state.volume) volume?.dispose();
      if (error.name === "AbortError") return false;
      this.reportError(error);
      return false;
    } finally {
      if (this.loadAbortController === controller) this.loadAbortController = null;
    }
  }

  selectChannel(channel) {
    if (this.state.volume === null) return;
    this.state.selectedChannel = channel;
    this.clearPlaneTiles();
    this.markViewDirty();
    this.renderVolume();
    this.refreshSourceMetadata();
    this.cancelCompetingWork();
  }

  selectScaleBounds({ finest, coarsest }) {
    if (this.state.volume === null) return;
    if (finest !== null && coarsest !== null && finest > coarsest) {
      throw new Error("Finest level must not be coarser than the selected coarsest level.");
    }
    this.state.manualScaleIndex = finest;
    this.state.manualCoarsestScaleIndex = coarsest;
    this.refreshScaleControls();
    this.clearPlaneTiles();
    this.markViewDirty();
    this.renderVolume();
  }
}
