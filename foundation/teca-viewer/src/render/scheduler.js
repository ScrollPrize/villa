import {
  CT_LEVEL_LIMIT,
  CT_REPLAN_INTERVAL_MS,
  INTERACTION_SETTLE_DELAY_MS,
  ORTHOGONAL_PLANES,
  SCALE_PICK_TOLERANCE,
  SETTLE_DELAY_MS,
  VIEW_RESIZE_RENDER_DELAY_MS,
} from "../config.js";
import { volumeBounds_ZYX } from "../gpu/addresses.js";
import { atlasFormatIdentity } from "../gpu/atlas-format.js";
import { filterMargin } from "../gpu/filter-chain.js";
import { presentationScale } from "../gpu/presentation-bricks.js";
import { atlasAdmission, settledLevelStack } from "../gpu/sliced-plane-atlas.js";
import { drawnPlanes, planeDisplayBasis, planeRotation, shownPanes, viewIsRotated, viewWidth } from "../views/pose.js";
import { demandScale, resolveScaleIndex, scalePickPixels } from "../volume/levels.js";
import { createWorkerPool } from "../worker-pool.js";
import { NavigationDemand } from "./navigation-demand.js";
import { PlanePrefetch } from "./plane-prefetch.js";

function slicedIdentity(volume, scale, channel) {
  return `${volume.contentNamespace}\0${atlasFormatIdentity(scale.numericType.dtype)}\0${channel}`;
}

function prioritizedOrthogonalPlanes(state, planes) {
  return [...planes].sort((a, b) => {
    if (a === state.plane) return -1;
    if (b === state.plane) return 1;
    return 0;
  });
}

export function createRenderScheduler({ ui, state, view, slicePrefetch, queueOrthogonalRedraw, publishSession }) {
  let renderFramePending = false;
  let resizeTimer = null;
  let planeDemand = null;
  let planningController = null;
  let interactionPlanTimer = null;
  let admittedPaneLayoutRevision = null;
  let admittedPaneLayoutSupported = false;
  let lastInteractionPlanAt = 0;
  let settleTimer = null;
  let settleFrame = null;
  let settleArmedAt = 0;
  let settleFrameStarts = 0;
  let deferredSettle = null;
  const planePrefetch = new PlanePrefetch();
  const navigationDemand = new NavigationDemand({
    state,
    view,
    displayScalePixels: orthogonalDisplayScalePixels,
    plannedDemand: plannedNavigationDemand,
  });

  function currentCheck(token, volume) {
    return () => token === state.renderToken && state.volume === volume;
  }

  function markPlanesDirty(planes) {
    state.dirtyPlanes = planes;
    state.lastInteractionDirtyPlanes = planes;
  }

  function markViewDirty() {
    markPlanesDirty(ORTHOGONAL_PLANES);
  }

  function resetDirtyPlanes() {
    state.dirtyPlanes = null;
  }

  function scheduleResizeRender() {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      resizeTimer = null;
      if (state.volume === null) return;
      markViewDirty();
      scheduleRender();
    }, VIEW_RESIZE_RENDER_DELAY_MS);
  }

  function scheduleRender({ interactive = false } = {}) {
    ++state.renderToken;
    slicePrefetch.cancelHiddenPanes();
    if ((interactive || isInteracting()) && presentationsCurrent() && updateSlicedPlane()) {
      scheduleInteractionPlan();
      return true;
    }
    window.clearTimeout(interactionPlanTimer);
    interactionPlanTimer = null;
    prepareRenderGeneration({ persistSession: !interactive && !isInteracting(), redraw: true });
    scheduleInteractiveRender();
    return false;
  }

  function presentationsCurrent() {
    if (state.volume === null) return false;
    const viewWidthVoxels = viewWidth(state);
    return presentationCurrent(state.volume.scales[presentedViewScaleIndex(viewWidthVoxels)]);
  }

  function presentationCurrent(scale) {
    if (!admittedPaneLayoutSupported || admittedPaneLayoutRevision !== view.paneLayoutRevision) return false;
    return view.slicedPlane.identity === slicedIdentity(state.volume, scale, state.selectedChannel);
  }

  function plannedNavigationDemand(plane, requestedScaleIndex) {
    const demand = view.slicedPlane.navigationDemand(plane, requestedScaleIndex);
    return demand !== null && presentationCurrent(demand.scale) ? demand : null;
  }

  function prepareRenderGeneration({ persistSession, redraw }) {
    if (state.volume) {
      slicePrefetch.observeMotion();
      planePrefetch.observe(
        `${state.volume.contentNamespace}\0${state.selectedChannel}`,
        shownPanes(state, ORTHOGONAL_PLANES)
          .filter((plane) => viewIsRotated(state, plane))
          .map((plane) => ({
            plane,
            center_ZYX: state.center_ZYX,
            rotation: planeRotation(state, plane),
            basis: planeDisplayBasis(state, plane),
            viewWidthVoxels: viewWidth(state),
            viewAspect: view.getOrthogonalPanelAspect(plane),
          })),
      );
    }
    navigationDemand.update();
    state.volume?.beginNavigation(state.renderToken);
    if (redraw) updateSlicedPlane();
    navigationDemand.retainRequests();
    if (persistSession) publishSession();
    // Held slice navigation keeps the slice-ahead reads alive: aborting them on
    // every step would guarantee a stall at each chunk boundary.
    if (!navigationDemand.activeSliceNavigation()) slicePrefetch.cancelSliceAhead();
    navigationDemand.cancelSupersededChunks();
    if (redraw && state.volume) queueOrthogonalRedraw(drawnPlanes(state));
  }

  function scheduleInteractiveRender() {
    if (renderFramePending) return;
    renderFramePending = true;
    window.requestAnimationFrame(() => {
      renderFramePending = false;
      if (!state.volume) return;
      if (settleTimer !== null) window.clearTimeout(settleTimer);
      const token = state.renderToken;
      const dirtyPlanes = state.dirtyPlanes;
      renderAtQuality(token, "interactive", dirtyPlanes);
      if (deferredSettle?.token === token) {
        scheduleDeferredSettle();
        return;
      }
      settleTimer = window.setTimeout(() => {
        settleTimer = null;
        if (isInteracting()) {
          deferredSettle = { token, dirtyPlanes };
          return;
        }
        scheduleRenderAtQuality(token, "settled", dirtyPlanes);
      }, SETTLE_DELAY_MS);
    });
  }

  function scheduleInteractionPlan() {
    if (interactionPlanTimer !== null) return;
    const elapsed = performance.now() - lastInteractionPlanAt;
    const delay = Math.max(0, CT_REPLAN_INTERVAL_MS - elapsed);
    interactionPlanTimer = window.setTimeout(() => {
      interactionPlanTimer = null;
      lastInteractionPlanAt = performance.now();
      if (!state.volume) return;
      prepareRenderGeneration({ persistSession: false, redraw: false });
      renderAtQuality(state.renderToken, "interactive", state.dirtyPlanes);
    }, delay);
  }

  function scheduleDeferredSettle() {
    if (!state.volume) return;
    if (!deferredSettle || deferredSettle.token !== state.renderToken) {
      deferredSettle = {
        token: state.renderToken,
        dirtyPlanes: state.dirtyPlanes ?? state.lastInteractionDirtyPlanes,
      };
    }
    if (settleTimer !== null) {
      window.clearTimeout(settleTimer);
    }
    settleArmedAt = performance.now();
    settleFrameStarts = 0;
    if (settleFrame === null) probeSettleFrame();
    settleTimer = window.setTimeout(() => {
      settleTimer = null;
      // Two frame starts after arming prove a frame completed inside the window;
      // fewer means the main thread was blocked by an interactive render longer
      // than the delay, not that input paused.
      const framesStalled = document.visibilityState === "visible" && settleFrameStarts < 2;
      if (isInteracting() || framesStalled) {
        scheduleDeferredSettle();
        return;
      }
      const deferred = deferredSettle;
      deferredSettle = null;
      if (!deferred || deferred.token !== state.renderToken) return;
      publishSession();
      scheduleRenderAtQuality(deferred.token, "settled", deferred.dirtyPlanes);
    }, INTERACTION_SETTLE_DELAY_MS);
  }

  function probeSettleFrame() {
    settleFrame = window.requestAnimationFrame((frameStart) => {
      settleFrame = null;
      if (frameStart > settleArmedAt) settleFrameStarts++;
      if (settleTimer !== null) probeSettleFrame();
    });
  }

  function isInteracting() {
    return (
      state.heldSliceDirection !== 0 ||
      state.heldZoomDirection !== 0 ||
      state.heldRotation !== null ||
      state.pointerDrag !== null
    );
  }

  function scheduleRenderAtQuality(token, quality, dirtyPlanes) {
    window.requestAnimationFrame(() => renderAtQuality(token, quality, dirtyPlanes));
  }

  async function renderAtQuality(token, quality, dirtyPlanes) {
    try {
      await renderCurrentState(token, quality, dirtyPlanes);
    } catch (error) {
      if (error.name === "AbortError") return;
      ui.error(`${quality} render failed: ${error.message}`);
      console.error(error);
    }
  }

  async function renderCurrentState(token, quality, dirtyPlanes) {
    if (!state.volume) return;
    if (token !== state.renderToken) return;
    const viewWidthVoxels = viewWidth(state);
    const requestedPlanes = drawnPlanes(state, dirtyPlanes ?? ORTHOGONAL_PLANES);
    const visiblePlanes = drawnPlanes(state);
    if (requestedPlanes.length === 0) return;
    const slicing = view.slicedPlane;
    const screenScaleIndex = viewScaleIndex(viewWidthVoxels);
    const volumeBeforePlanning = state.volume;
    const needsPlan =
      view.replanDue(quality, visiblePlanes) ||
      slicing.identity !== slicedIdentity(state.volume, state.volume.scales[screenScaleIndex], state.selectedChannel);
    let settledLevels = null;
    if (needsPlan) {
      planningController?.abort();
      planningController = new AbortController();
      try {
        settledLevels = await planSettledLevels(
          screenScaleIndex,
          visiblePlanes,
          viewWidthVoxels,
          planningController.signal,
        );
      } catch (error) {
        if (error.name === "AbortError") return;
        throw error;
      }
      if (token !== state.renderToken || state.volume !== volumeBeforePlanning) return;
    }
    const planScaleIndex = settledLevels?.levelSelection.scaleIndex ?? slicing.presentedScaleIndex(screenScaleIndex);
    const planScale = state.volume.scales[planScaleIndex];
    const planeViews = slicedPlaneViews(visiblePlanes, viewWidthVoxels);
    const admission = atlasAdmission(
      state.volume.scales
        .slice(planScaleIndex, planScaleIndex + CT_LEVEL_LIMIT)
        .map((scale) => presentationScale(scale)),
      view.rendering.textureLimits,
    );
    admittedPaneLayoutRevision = view.paneLayoutRevision;
    admittedPaneLayoutSupported = admission.supported;
    if (!admission.supported) {
      view.hideSlices(requestedPlanes);
      ui.error(admission.reason);
      return;
    }
    view.hideSlices(ORTHOGONAL_PLANES.filter((plane) => !visiblePlanes.includes(plane)));
    ui.scale(planScaleIndex);
    const volume = state.volume;
    const identity = slicedIdentity(volume, planScale, state.selectedChannel);
    view.updateSliceViews({
      identity,
      token,
      planeViews: planeViews,
      isCurrent: currentCheck(token, volume),
      onPresent: (plane) => queueOrthogonalRedraw([plane]),
    });
    if (!needsPlan) return;
    const settledLevelsByPlane = new Map(settledLevels.planePlans.map(({ plane, levels }) => [plane, levels]));
    const planePlans = planeViews.map((planeView) => {
      const levels = settledLevelsByPlane
        .get(planeView.plane)
        .map(({ scaleIndex, scale, addresses: planned, prefetch }) => {
          const addresses = planned.map((address) => ({ ...address, fetchOrder: 0 }));
          const fetchAddresses = slicePrefetch.planeFetchAddresses(planeView, scale, addresses, prefetch);
          return { scaleIndex, scale, addresses, fetchAddresses };
        });
      return { ...planeView, scaleIndex: levels[0].scaleIndex, levels };
    });
    view.beginSlices({
      identity,
      volume,
      sourceBounds_ZYX: volumeBounds_ZYX(volume),
      channel: state.selectedChannel,
      token,
      planePlans,
      displayEncoding: volume.metadata.displayEncoding,
      displaySettings: view.scene.displaySettings,
      levelSelection: settledLevels.levelSelection,
      isCurrent: currentCheck(token, volume),
      onPresent: (plane) => queueOrthogonalRedraw([plane]),
      onError: (error) => {
        if (error.name === "AbortError") return;
        if (token !== state.renderToken) return;
        ui.error(`CT render failed: ${error.message}`);
        console.error(error);
      },
    });
    navigationDemand.update();
    if (dirtyPlanes === state.dirtyPlanes) state.dirtyPlanes = null;
    ui.status(`View: ${quality}`);
    if (quality === "settled") ui.settled();
  }

  function orthogonalDisplayScalePixels(viewWidthVoxels) {
    return scalePickPixels(viewWidthVoxels, view.getOrthogonalPanelPixelSize(state.plane));
  }

  function planeScaleIndex(plane, viewWidthVoxels) {
    const autoScaleIndex = state.volume.pickScaleIndexForView(
      viewWidthVoxels,
      scalePickPixels(viewWidthVoxels, view.getOrthogonalPanelPixelSize(plane)),
      SCALE_PICK_TOLERANCE,
    );
    return resolveScaleIndex(
      autoScaleIndex,
      state.manualScaleIndex,
      state.manualCoarsestScaleIndex,
      state.volume.scales.length,
    );
  }

  function viewScaleIndex(viewWidthVoxels) {
    const planes = drawnPlanes(state);
    return Math.min(
      ...(planes.length > 0 ? planes : [state.plane]).map((plane) => planeScaleIndex(plane, viewWidthVoxels)),
    );
  }

  function presentedViewScaleIndex(viewWidthVoxels) {
    return view.slicedPlane.presentedScaleIndex(viewScaleIndex(viewWidthVoxels));
  }

  function updateSlicedPlane() {
    if (state.volume === null) return false;
    const viewWidthVoxels = viewWidth(state);
    const token = state.renderToken;
    const planes = drawnPlanes(state, state.dirtyPlanes ?? ORTHOGONAL_PLANES);
    if (planes.length === 0) return false;
    const scale = state.volume.scales[presentedViewScaleIndex(viewWidthVoxels)];
    const volume = state.volume;
    const identity = slicedIdentity(volume, scale, state.selectedChannel);
    if (view.slicedPlane.identity !== identity) return false;
    return view.updateSliceViews({
      identity,
      token,
      planeViews: slicedPlaneViews(planes, viewWidthVoxels),
      isCurrent: currentCheck(token, volume),
      onPresent: (plane) => queueOrthogonalRedraw([plane]),
    });
  }

  async function planSettledLevels(scaleIndex, planes, viewWidthVoxels, signal) {
    const scales = state.volume.scales.map((scale) => presentationScale(scale));
    const coarsestScaleIndex = resolveScaleIndex(
      state.volume.scales.length - 1,
      state.manualScaleIndex,
      state.manualCoarsestScaleIndex,
      state.volume.scales.length,
    );
    const candidates = scales.slice(scaleIndex, coarsestScaleIndex + 1);
    const latencyMs = state.volume.getQueueStats().averageLatencyMs;
    planeDemand ??= createWorkerPool(new URL("../gpu/plane-demand.worker.js", import.meta.url), "Plane demand", {
      workerCount: 1,
      terminateOnAbort: true,
    });
    const demands = await planeDemand.run(
      {
        center_ZYX: [...state.center_ZYX],
        scales: candidates.map(demandScale),
        views: planes.map((plane) => ({
          basis: planeDisplayBasis(state, plane),
          width: viewWidthVoxels,
          aspect: view.getOrthogonalPanelAspect(plane),
          coverage:
            1 +
            (2 * filterMargin(view.scene.filterReach, viewWidthVoxels / view.getOrthogonalPanelPixelSize(plane))) /
              viewWidthVoxels,
          predictions: candidates.map((scale) =>
            state.heldSliceDirection === 0 && viewIsRotated(state, plane)
              ? planePrefetch.prediction(plane, scale, latencyMs)
              : null,
          ),
        })),
      },
      [],
      signal,
    );
    const settled = settledLevelStack({
      scales,
      scaleIndex,
      coarsestScaleIndex,
      planes,
      planeScaleIndex: (plane) => planeScaleIndex(plane, viewWidthVoxels),
      addressesFor: (scale, plane) => demands[planes.indexOf(plane)][candidates.indexOf(scale)].addresses,
      limits: view.rendering.textureLimits,
    });
    for (const plan of settled.planePlans) {
      for (const level of plan.levels) {
        level.prefetch = demands[planes.indexOf(plan.plane)][candidates.indexOf(level.scale)].prefetch;
      }
    }
    return settled;
  }

  function slicedPlaneViews(planes, viewWidthVoxels) {
    return prioritizedOrthogonalPlanes(state, planes).map((plane) => ({
      plane,
      shown: shownPanes(state, [plane]).length > 0,
      center_ZYX: [...state.center_ZYX],
      basis: planeDisplayBasis(state, plane),
      viewWidthVoxels,
      viewAspect: view.getOrthogonalPanelAspect(plane),
    }));
  }

  return {
    navigationDemand,
    markPlanesDirty,
    markViewDirty,
    resetDirtyPlanes,
    scheduleRender,
    scheduleDeferredSettle,
    scheduleResizeRender,
    updateSlicedPlane,
  };
}
