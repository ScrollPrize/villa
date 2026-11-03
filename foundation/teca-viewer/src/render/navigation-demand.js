import { PREFETCH_CHUNK_PRIORITY, SCALE_PICK_TOLERANCE, SHOWN_CHUNK_PRIORITY_LIMIT } from "../config.js";
import { orthogonalStepPlane, viewIsRotated, viewWidth } from "../views/pose.js";
import { resolveScaleIndex } from "../volume/levels.js";
import { sliceNavigationPolicy } from "./navigation-policy.js";

export class NavigationDemand {
  constructor({ state, view, displayScalePixels, plannedDemand }) {
    this.state = state;
    this.view = view;
    this.displayScalePixels = displayScalePixels;
    this.plannedDemand = plannedDemand;
    this.demand = null;
  }

  update() {
    const state = this.state;
    const plane = orthogonalStepPlane(state);
    if (state.volume === null) {
      this.demand = null;
      return;
    }
    const viewWidthVoxels = viewWidth(state);
    const requestedScaleIndex = resolveScaleIndex(
      state.volume.pickScaleIndexForView(
        viewWidthVoxels,
        this.displayScalePixels(viewWidthVoxels),
        SCALE_PICK_TOLERANCE,
      ),
      state.manualScaleIndex,
      state.manualCoarsestScaleIndex,
      state.volume.scales.length,
    );
    const planned = this.plannedDemand(plane, requestedScaleIndex);
    this.demand =
      planned === null ? null : { renderToken: state.renderToken, plane, targetScaleIndex: planned.scaleIndex };
  }

  activeSliceNavigation() {
    return this.slicePolicy().axisAligned;
  }

  retainRequests() {
    return this.demand !== null && this.view.slicedPlane.retainCurrentRequests();
  }

  slicePolicy() {
    const { state } = this;
    const plane = orthogonalStepPlane(state);
    return sliceNavigationPolicy({
      plane,
      rotated: viewIsRotated(state, plane),
      heldSliceDirection: state.heldSliceDirection,
      dirtyPlanes: state.dirtyPlanes,
    });
  }

  cancelSupersededChunks() {
    const volume = this.state.volume;
    if (volume === null) return;
    if (this.demand !== null) {
      volume.cancelPendingChunks(PREFETCH_CHUNK_PRIORITY, Infinity, true);
    } else if (this.slicePolicy().held) {
      volume.cancelPendingChunks(0, SHOWN_CHUNK_PRIORITY_LIMIT - 1, true);
      volume.cancelPendingChunks(PREFETCH_CHUNK_PRIORITY, Infinity, true);
    } else {
      volume.cancelPendingChunks(0, Infinity, true);
    }
  }

  telemetry() {
    if (this.demand === null)
      return { renderToken: this.state.renderToken, plane: this.state.plane, targetScaleIndex: null };
    const { renderToken, plane, targetScaleIndex } = this.demand;
    return { renderToken, plane, targetScaleIndex };
  }
}
