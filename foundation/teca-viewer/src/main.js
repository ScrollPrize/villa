import { installProbe } from "./debug/probe.js";
import { createViewerMeter } from "./debug/viewer-meter.js";
import createBookmarks from "./extensions/bookmarks.js";
import createCatalog from "./extensions/catalog.js";
import createCtBox from "./extensions/ct-box.js";
import createFilters from "./extensions/filters.js";
import createInk from "./extensions/ink.js";
import createLayers from "./extensions/layers.js";
import createMeshEdit from "./extensions/mesh-edit.js";
import createPaint from "./extensions/paint.js";
import createPoints from "./extensions/points.js";
import createSurfaceMeshes from "./extensions/surface-meshes.js";
import { createRedraw } from "./render/redraw.js";
import { createRenderScheduler } from "./render/scheduler.js";
import { SlicePrefetch } from "./render/slice-prefetch.js";
import { createSessionAddress, linkedSession, openedKept } from "./session.js";
import { createViewerState } from "./state.js";
import { SurfaceController } from "./surface/controller.js";
import { GeometryByteCache } from "./surface/geometry-byte-cache.js";
import { SurfaceMarkers } from "./surface/markers.js";
import { createOverlayStore } from "./surface/overlay-store.js";
import { SurfaceReaders } from "./surface/readers.js";
import { readChunkBudget } from "./ui/cache-settings.js";
import { createCamera } from "./ui/camera.js";
import { bindViewerControls } from "./ui/controls.js";
import { createDisplay } from "./ui/display.js";
import { initDownloadCacheControls } from "./ui/download-cache.js";
import { viewerElements } from "./ui/elements.js";
import { createExtensions } from "./ui/extensions.js";
import { readGpuBudget } from "./ui/gpu-settings.js";
import { createHeldNavigation } from "./ui/held-navigation.js";
import { createKeyboardControls } from "./ui/keyboard.js";
import { NetworkStatus } from "./ui/network-status.js";
import { createOverlayControls } from "./ui/overlays.js";
import { PerformanceControls } from "./ui/performance.js";
import { createPointerControls } from "./ui/pointer.js";
import { createPopovers } from "./ui/popover.js";
import { createReadouts } from "./ui/readouts.js";
import { createShellControls } from "./ui/shell.js";
import { SourcePanel } from "./ui/source-panel.js";
import { createStatusLine } from "./ui/status-line.js";
import { SurfaceControls } from "./ui/surface-controls.js";
import { createSurfaceLayout } from "./ui/surface-layout.js";
import { createTemporaryCamera } from "./ui/temporary-camera.js";
import { createView } from "./views/composed-view.js";
import { createMarks } from "./views/marks.js";
import { createOverlayRenderer } from "./views/overlay-lines.js";
import { PrimarySourceController } from "./volume/active-volume.js";
import { DownloadStats, responseDownloads } from "./volume/download-stats.js";
import { DownloadStorage } from "./volume/download-storage.js";
import { originRequestStats } from "./volume/request-queue.js";
import { ZarrVolume } from "./volume/zarr-volume.js";

const storage = window.localStorage;
const query = new URLSearchParams(location.search);
const { session, refused } = linkedSession(location);
const state = createViewerState(session);
let chunkBudgetBytes = readChunkBudget(storage);
const downloadStorage = new DownloadStorage(window.indexedDB !== undefined);
const volumeDownloads = new DownloadStats();
const openVolume = (url, signal) => ZarrVolume.open(url, signal, chunkBudgetBytes, downloadStorage, volumeDownloads);
const geometryByteCache = new GeometryByteCache({
  storage: window.caches,
  locks: navigator.locks,
  origin: location.origin,
});

const elements = viewerElements(document);
const wirePopover = createPopovers(document);
const statusLine = createStatusLine(elements.status);
const { publishSession, sessionAddress, sessionFile, openSessionFile, holdPublication, publicationHeld } =
  createSessionAddress({
    state,
    setStatus: statusLine.setStatus,
    clearStatus: statusLine.clearStatus,
    stash: window.sessionStorage,
  });
const view = createView({
  paneGrid: elements.orthoGrid,
  document,
  window,
  gpuByteLimit: readGpuBudget(storage),
  onUploadError: (error) => {
    statusLine.setStatus(`Texture upload failed: ${error.message}`);
    console.error(error);
  },
});
const readouts = createReadouts({ state, view, elements });

const overlayStore = createOverlayStore({
  state,
  rendering: view.rendering,
  geometryByteCache,
  linkedPaths: query.getAll("overlay"),
});
const overlayRenderer = createOverlayRenderer({
  state,
  view,
  overlays: overlayStore.overlays,
  overlayRevision: overlayStore.overlayRevision,
});
const surfaceControls = new SurfaceControls({ state, view, elements, storage, setStatus: statusLine.setStatus });
const surfaceMarkers = new SurfaceMarkers({
  state,
  pane: view.getSurfacePane(),
  onError: (error) => surfaceControls.setStatus(error.message, true),
});
const marks = createMarks({ state, view });
const { queueOrthogonalRedraw } = createRedraw({
  updateOverlays: overlayRenderer.updateOverlays,
  updateMarks: marks.updateMarks,
  surfaceMarkers,
  surfaceViewDrawn: () => extensions.surfaceViewDrawn(),
  state,
  view,
});
const slicePrefetch = new SlicePrefetch({ state, view });
const sourcePanel = new SourcePanel({
  elements,
  state,
  syncPositionInputs: readouts.syncPositionInputs,
  updateViewerShellState: readouts.updateViewerShellState,
  setStatus: statusLine.setStatus,
  activate: (source) => sources.open(source, { restoreView: false }),
});
const scheduler = createRenderScheduler({
  ui: {
    status: (message) => {
      statusLine.setAmbientStatus(message);
      readouts.refreshViewReadout();
    },
    error: statusLine.setStatus,
    scale: (scaleIndex) => {
      sourcePanel.showDrawnScale(scaleIndex);
      readouts.refreshViewReadout();
    },
    settled: () => extensions.focusSettled(),
  },
  state,
  view,
  slicePrefetch,
  queueOrthogonalRedraw,
  publishSession,
});
const { scheduleRender, scheduleDeferredSettle } = scheduler;
const display = createDisplay({
  state,
  elements,
  view,
  publishSession,
  queueOrthogonalRedraw,
  setStatus: statusLine.setStatus,
});

const surfaceReaders = new SurfaceReaders({ openVolume });
const surfaceController = new SurfaceController({
  state,
  view,
  overlays: overlayStore.overlays,
  panel: {
    status: (message, error) => surfaceControls.setStatus(message, error),
    clearError: () => surfaceControls.clearError(),
    clearRestoreStatus: () => surfaceControls.clearRestoreStatus(),
    promptForSurface: () => surfaceControls.promptForSurface(),
    sync: () => surfaceControls.syncControls(),
    scheduleSync: () => surfaceControls.scheduleControlSync(),
    syncNormalOffset: () => surfaceControls.syncNormalOffset(),
    syncOverlay: () => surfaceControls.syncOverlay(),
  },
  markers: surfaceMarkers,
  displaySettings: display.displaySettings,
  refreshViewReadout: readouts.refreshViewReadout,
  invalidatePointerInspection: readouts.invalidatePointerInspection,
  setViewports: (activeViewport, fullscreenViewport) => camera.setViewports(activeViewport, fullscreenViewport),
  queueOrthogonalRedraw,
  scheduleRender,
  publishSession,
  holdPublication,
  publicationHeld,
  readers: surfaceReaders,
});
surfaceControls.bind(surfaceController);
const camera = createCamera({
  state,
  view,
  surfaceController,
  scheduleRender,
  queueOrthogonalRedraw,
  updateSlicedPlane: scheduler.updateSlicedPlane,
  markPlanesDirty: scheduler.markPlanesDirty,
  markViewDirty: scheduler.markViewDirty,
  resetDirtyPlanes: scheduler.resetDirtyPlanes,
  slicePrefetch,
  publishSession,
  setStatus: statusLine.setStatus,
  syncPositionInputs: readouts.syncPositionInputs,
  syncZoomInputs: readouts.syncZoomInputs,
  syncRotationReadout: readouts.syncRotationReadout,
  updateScaleBars: readouts.updateScaleBars,
  updateViewerShellState: readouts.updateViewerShellState,
});
const temporaryCamera = createTemporaryCamera({
  state,
  surfaceController,
  restorePose: camera.restorePose,
  setStatus: statusLine.setStatus,
});
const shell = createShellControls({
  window,
  document,
  storage,
  state,
  elements,
  wirePopover,
  publishSession,
  setActiveViewport: camera.setActiveViewport,
  setPlane: camera.setPlane,
  toggleActiveFullscreen: camera.toggleActiveFullscreen,
  openSurface: () =>
    void surfaceController.openMode("surface").catch((error) => surfaceController.handleOpenError(error)),
  closeSurface: () => surfaceController.close(),
});
const pointer = createPointerControls({
  state,
  view,
  elements,
  surfaceController,
  surfaceControls,
  surfaceMarkers,
  setPhysicalFocus: camera.setPhysicalFocus,
  panByPixels: camera.panByPixels,
  tiltPlaneByPixels: camera.tiltPlaneByPixels,
  rotateCamera3dByPixels: camera.rotateCamera3dByPixels,
  syncRotationReadout: readouts.syncRotationReadout,
  showPointerInspection: readouts.showPointerInspection,
  invalidatePointerInspection: readouts.invalidatePointerInspection,
  inspectPoint: readouts.inspectPoint,
  cancelInspection: readouts.cancelInspection,
  setActivePlane: camera.setActivePlane,
  setActiveViewport: camera.setActiveViewport,
  setZoomAtClientPoint: camera.setZoomAtClientPoint,
  scheduleDeferredSettle,
  publishSession,
  setStatus: statusLine.setStatus,
});
const heldNavigation = createHeldNavigation({
  state,
  view,
  surfaceController,
  slicePrefetch,
  scheduleDeferredSettle,
  stepSlice: camera.stepSlice,
  setZoomAtLastPointer: camera.setZoomAtLastPointer,
  rotateActiveViewport: camera.rotateActiveViewport,
  setStatus: statusLine.setStatus,
});
const keyboard = createKeyboardControls({
  state,
  view,
  surfaceController,
  adjustHeldSliceStep: camera.adjustHeldSliceStep,
  resetRotation: camera.resetRotation,
  setActiveViewport: camera.setActiveViewport,
  setPlane: camera.setPlane,
  toggleActiveFullscreen: camera.toggleActiveFullscreen,
  toggleCrosshair: camera.toggleCrosshair,
  toggleNormalLock: () => surfaceControls.toggleNormalLock(),
  toggleAlone: display.toggleAlone,
  saveTemporaryCamera: temporaryCamera.saveTemporaryCamera,
  restoreTemporaryCamera: temporaryCamera.restoreTemporaryCamera,
  cancelPointerInteraction: pointer.cancelPointerInteraction,
  focusAtPointer: pointer.focusAtPointer,
  holdPan: pointer.holdPan,
  releasePan: pointer.releasePan,
  startHeldSlice: heldNavigation.startSlice,
  startHeldZoom: heldNavigation.startZoom,
  startHeldRotation: heldNavigation.startRotation,
  stepOneSlice: heldNavigation.stepOnce,
  stopHeldSlice: heldNavigation.stopSlice,
  stopHeldZoom: heldNavigation.stopZoom,
  stopHeldRotation: heldNavigation.stopRotation,
  modalIsOpen: shell.modalIsOpen,
  toggleLeftPanel: shell.toggleLeftPanel,
  toggleRightPanel: shell.toggleRightPanel,
  shortcuts: elements.shortcuts,
});
const overlayControls = createOverlayControls({
  state,
  view,
  elements,
  wirePopover,
  loadOverlay: overlayStore.loadOverlay,
  syncOverlayLoads: overlayStore.syncLoads,
  removeOverlay: overlayStore.removeOverlay,
  setOverlayVisible: overlayStore.setOverlayVisible,
  setOverlayLineWidth: overlayStore.setOverlayLineWidth,
  surfaceController,
  surfaceControls,
  setActiveViewport: camera.setActiveViewport,
  onLayoutChange: handleViewerLayoutChange,
  queueOrthogonalRedraw,
  publishSession,
  surfacesChanged: () => extensions.surfacesChanged(),
  setStatus: statusLine.setStatus,
});
const sources = new PrimarySourceController({
  state,
  openVolume,
  cancelCompetingWork: () => {
    pointer.cancelPointerInteraction();
    slicePrefetch.cancelSliceAhead();
  },
  resetView: camera.resetView,
  presentVolume,
  clearPlaneTiles: () => view.clearPlaneTiles(),
  markViewDirty: scheduler.markViewDirty,
  renderVolume: scheduleRender,
  publishSession,
  refreshScaleControls: () => sourcePanel.populateScaleSelect(),
  refreshSourceMetadata: () => sourcePanel.renderSourceMetadata(),
  setSourceInput: (value) => {
    elements.url.value = value;
  },
  setStatus: statusLine.setStatus,
  reportError: (error) => {
    statusLine.setStatus(error.message);
    console.error(error);
  },
});
const showVolume = (url) => sources.open(url, { restoreView: false });
const extensions = createExtensions({
  state,
  opened: openedKept(window.sessionStorage),
  view,
  marks,
  surfaceMarkers,
  dataPanels: elements.dataPanels,
  viewPanels: elements.viewPanels,
  rightTabs: elements.rightTabs,
  infoTab: elements.infoTab,
  infoPanel: elements.infoPanel,
  openVolume,
  showVolume,
  addSurface: overlayControls.addSurface,
  readSurface: overlayStore.readSurface,
  heldSurfaceChunk: (rootUrl, level, chunk_ZYX, channel) =>
    surfaceReaders.heldChunk(rootUrl, level, chunk_ZYX, channel),
  setEditedSurface: overlayStore.setEditedSurface,
  surfaceChanged: overlayStore.surfaceChanged,
  setTool: pointer.setTool,
  setFocus: camera.setPhysicalFocus,
  addKeys: keyboard.addKeys,
  markViewDirty: scheduler.markViewDirty,
  scheduleRender,
  queueOrthogonalRedraw,
  publishSession,
  setStatus: statusLine.setStatus,
});
extensions.extend("layers", createLayers);
extensions.extend("ink", createInk);
extensions.extend("points", createPoints);
extensions.extend("paint", createPaint);
extensions.extend("ct-box", createCtBox);
extensions.extend("catalog", createCatalog);
extensions.extend("filters", createFilters);
extensions.extend("mesh-edit", createMeshEdit);
extensions.extend("surface-meshes", createSurfaceMeshes);
extensions.extend("bookmarks", createBookmarks);

function presentVolume() {
  sourcePanel.renderSourceItems();
  shell.updateSourceName();
  readouts.syncZoomInputs();
  view.clearPlaneTiles();
  view.clearLayers();
  display.showAlone(null);
  sourcePanel.populateScaleSelect();
  sourcePanel.populateChannelSelect();
  sourcePanel.configurePositionInputs();
  sourcePanel.configureSpatialCapabilities();
  if (state.volume.fullShape_ZYX[0] === 1 && state.plane !== "xy") camera.setPlane("xy");
  sourcePanel.renderSourceMetadata();
  readouts.updateViewerShellState();
  view.resize();
  readouts.updateScaleBars();
  scheduler.resetDirtyPlanes();
  queueOrthogonalRedraw(["3d"]);
  statusLine.setStatus("Metadata loaded. Rendering visible slices...");
  surfaceController.volumeChanged();
  overlayControls.syncLoads();
  extensions.volumeChanged(state.volume);
}

function handleViewerLayoutChange() {
  view.resize();
  surfaceController.refreshLayout();
  readouts.updateScaleBars();
  queueOrthogonalRedraw();
  scheduler.scheduleResizeRender();
}

function applyGpuBudget(bytes) {
  if (bytes !== view.rendering.byteLimit) {
    pointer.cancelPointerInteraction();
    const surfacePath = surfaceController.active() ? state.surfaceView.path : null;
    if (surfacePath !== null) surfaceController.close();
    view.rendering.setByteLimit(bytes);
    scheduler.markViewDirty();
    scheduleRender();
    if (surfacePath !== null)
      void surfaceController.openPath(surfacePath).catch((error) => surfaceController.handleOpenError(error));
  }
  overlayControls.syncLoads();
}

function applyChunkBudget(bytes) {
  chunkBudgetBytes = bytes;
  state.volume?.setChunkCacheByteLimit(bytes);
  surfaceReaders.setChunkCacheByteLimit(bytes);
}

const interactionMeter = createViewerMeter({
  state,
  view,
  surfaceController,
  navigationDemand: scheduler.navigationDemand,
  volumeDownloads,
});
new PerformanceControls(interactionMeter, elements.performanceControls);
new NetworkStatus(elements.networkStatus, responseDownloads, originRequestStats);
installProbe({ state, view, surfaceController, getOverlay2dStats: overlayRenderer.getOverlay2dStats });

view.setDisplaySettings(display.displaySettings());
view.resize();
const downloadStorageControls = initDownloadCacheControls(elements, downloadStorage);
shell.initShell({ onViewerLayoutChange: handleViewerLayoutChange, onSettingsOpen: downloadStorageControls.refresh });
createSurfaceLayout({
  state,
  view,
  elements,
  resetPaneOrientation: camera.resetPaneOrientation,
  setActivePlane: camera.setActivePlane,
  surfaceController,
  surfaceControls,
  publishSession,
}).initSurfaceLayout({
  handleViewerLayoutChange,
  cancelPointerInteraction: pointer.cancelPointerInteraction,
});
bindViewerControls({
  state,
  elements,
  wirePopover,
  setStatus: statusLine.setStatus,
  resetRotation: camera.resetRotation,
  resetAllRotations: camera.resetAllRotations,
  setRotationMode: camera.setRotationMode,
  syncRotationReadout: readouts.syncRotationReadout,
  setPositionAxis: camera.setPositionAxis,
  setPositionXYZ: camera.setPositionXYZ,
  setZoom: camera.setZoom,
  syncZoomInputs: readouts.syncZoomInputs,
  setDisplayControlValue: display.setDisplayControlValue,
  syncDisplayInputs: display.syncDisplayInputs,
  sources,
  surfaceController,
  openSource: showVolume,
  publishSession,
  sessionAddress,
  sessionFile,
  openSessionFile,
  kept: extensions.kept,
  gpuBudgetBytes: view.rendering.byteLimit,
  applyGpuBudget,
  chunkBudgetBytes,
  applyChunkBudget,
  storage,
  geometryByteCache,
});
keyboard.initKeyboard();
pointer.initPointer();
performance.mark("viewer:controls-ready");
elements.startupMessage.close();
elements.app.inert = false;

const linkedVolume = query.get("volume");
if (state.activeSource !== null) await sources.open(state.activeSource, { restoreView: true });
else if (refused === null && linkedVolume !== null) await sources.open(linkedVolume, { restoreView: false });
else {
  sourcePanel.configureNoSourceState(refused ?? "Open a remote Zarr URL.");
  sourcePanel.renderSourceItems();
}
