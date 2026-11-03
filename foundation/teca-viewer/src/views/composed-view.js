import { CT_ATLAS_BYTES, PRECOMPILE_IDLE_TIMEOUT_MS } from "../config.js";
import { invalidateElementRects } from "../element-rect.js";
import { atlasFormatForDtype } from "../gpu/atlas-format.js";
import { layerPresentationScales, presentationScale } from "../gpu/presentation-bricks.js";
import { ViewerRenderer } from "../gpu/renderer.js";
import { SlicedPlane } from "../gpu/sliced-plane.js";
import { OrthogonalPane } from "./orthogonal-pane.js";
import { CT_RENDER_ORDER, OrthogonalScene } from "./scene.js";
import { SurfaceLayout } from "./surface-layout.js";
import { SurfacePane } from "./surface-pane.js";
import { ThreePane } from "./three-pane.js";

function layerIdentity(identity, layer) {
  return `${identity}\0layer ${layer.id}`;
}

class ComposedView {
  constructor({ panes: composedPanes, threePane, surfacePane, rendering, scene, paneGrid, pixelRatio }) {
    this.scene = scene;
    this.threePane = threePane;
    this.surfacePane = surfacePane;
    this.surfaceButton = threePane.surfaceButton;
    this.pixelRatio = pixelRatio;
    this.panes = composedPanes;
    this.rendering = rendering;
    this.surfaceLayout = new SurfaceLayout(paneGrid, composedPanes);
    const planes = composedPanes.filter((pane) => pane.mode === "plane");
    const precompileJobs = planes.map((pane) => pane.renderer);
    const scheduleNextPrecompile = () => {
      const renderer = precompileJobs.shift();
      if (renderer === undefined) return;
      const compile = async () => {
        await this.scene.precompileSlicedPlaneMaterial(renderer);
        scheduleNextPrecompile();
      };
      if (typeof window.requestIdleCallback === "function") {
        window.requestIdleCallback(compile, { timeout: PRECOMPILE_IDLE_TIMEOUT_MS });
      } else {
        window.setTimeout(compile, 0);
      }
    };
    scheduleNextPrecompile();
    this.renderStats = { orthogonalFrames: 0 };
    this.paneLayoutSignature = null;
    this.paneLayoutRevision = 0;
    this.planeRenderers = new Map(planes.map((pane) => [pane.name, pane.renderer]));
    this.slicedPlane = new SlicedPlane({
      rendering,
      tileGroups: scene.tileGroups,
      renderers: this.planeRenderers,
      renderOrder: CT_RENDER_ORDER,
      atlasBytes: CT_ATLAS_BYTES,
      style: null,
    });
    this.layers = [];
    this.addedLayers = 0;
    this.alone = null;
  }

  addLayer({ slices, surface }, primary, style) {
    const scales = layerPresentationScales(primary.scales, slices.scales, presentationScale);
    const bytesPerVoxel = (source) => atlasFormatForDtype(source.scales[0].numericType.dtype).bytesPerVoxel;
    const atlasBytes = Math.ceil((CT_ATLAS_BYTES * bytesPerVoxel(slices)) / bytesPerVoxel(primary));
    this.addedLayers += 1;
    const layer = {
      id: this.addedLayers,
      volume: slices,
      scales,
      planned: false,
      surface: surface === null ? null : this.surfacePane.addLayer(surface, primary.scales, style, this.addedLayers),
      slicedPlane: new SlicedPlane({
        rendering: this.rendering,
        tileGroups: this.scene.tileGroups,
        renderers: this.planeRenderers,
        renderOrder: CT_RENDER_ORDER + this.addedLayers,
        atlasBytes,
        style,
      }),
    };
    layer.slicedPlane.setDrawn(this.alone !== "ct");
    this.layers.push(layer);
    return layer;
  }

  showAlone(alone) {
    this.alone = alone;
    this.slicedPlane.setDrawn(alone !== "layers");
    for (const layer of this.layers) layer.slicedPlane.setDrawn(alone !== "ct");
    this.surfacePane.showAlone(alone);
  }

  hasLayers() {
    return this.layers.length > 0 || this.surfacePane.hasImages();
  }

  addSurfaceImage(path, image, style) {
    this.addedLayers += 1;
    return this.surfacePane.addImage(path, image, style, this.addedLayers);
  }

  setSurfaceImageStyle(image, style) {
    this.surfacePane.setImageStyle(image, style);
  }

  setSurfaceImageDetail(image, detail) {
    this.surfacePane.setImageDetail(image, detail);
  }

  removeSurfaceImage(image) {
    this.surfacePane.removeImage(image);
  }

  filterCT(passes, reach) {
    this.scene.filterCT(passes, reach);
    this.surfacePane.filterCT(passes, reach);
  }

  setLayerStyle(layer, style) {
    layer.slicedPlane.setStyle(style);
    if (layer.surface !== null) this.surfacePane.setLayerStyle(layer.surface, style);
  }

  refreshLayer(layer, changes) {
    layer.slicedPlane.refresh(changes);
    if (layer.surface !== null) this.surfacePane.refreshLayer(layer.surface, changes);
  }

  removeLayer(layer) {
    const index = this.layers.indexOf(layer);
    if (index < 0) throw new Error(`Layer ${layer.volume.rootUrl} is already closed.`);
    this.layers.splice(index, 1);
    layer.slicedPlane.dispose();
    if (layer.surface !== null) this.surfacePane.removeLayer(layer.surface);
  }

  clearLayers() {
    for (const layer of this.layers) layer.slicedPlane.dispose();
    this.surfacePane.clearLayers();
    this.layers = [];
  }

  replanDue(quality, planes) {
    return this.slicedPlane.replanDue(quality, planes) || this.layers.some((layer) => !layer.planned);
  }

  beginSlices(plan) {
    this.slicedPlane.begin(plan);
    for (const layer of this.layers) {
      const planePlans = plan.planePlans.map((planePlan) => ({
        ...planePlan,
        levels: planePlan.levels.map((level) => ({ ...level, scale: layer.scales[level.scaleIndex] })),
      }));
      layer.planned = true;
      layer.volume.beginNavigation(plan.token);
      layer.slicedPlane.begin({
        identity: layerIdentity(plan.identity, layer),
        volume: layer.volume,
        sourceBounds_ZYX: plan.sourceBounds_ZYX,
        channel: 0,
        token: plan.token,
        planePlans,
        displayEncoding: layer.volume.metadata.displayEncoding,
        displaySettings: plan.displaySettings,
        levelSelection: plan.levelSelection,
        isCurrent: plan.isCurrent,
        onPresent: plan.onPresent,
        onError: plan.onError,
      });
    }
  }

  updateSliceViews(update) {
    for (const layer of this.layers) {
      layer.slicedPlane.updateViews({
        identity: layerIdentity(update.identity, layer),
        token: update.token,
        planeViews: update.planeViews,
        isCurrent: update.isCurrent,
        onPresent: update.onPresent,
      });
    }
    return this.slicedPlane.updateViews(update);
  }

  layerStats() {
    return this.layers.map((layer) => layer.slicedPlane.stats());
  }

  hideSlices(planes) {
    this.slicedPlane.hide(planes);
    for (const layer of this.layers) layer.slicedPlane.hide(planes);
  }

  paneNames() {
    return this.panes.map((pane) => pane.name);
  }

  getCanvas(name) {
    return this.panes.find((pane) => pane.name === name)?.canvas ?? null;
  }

  captureOrthogonalSnapshots() {
    return this.panes
      .filter((pane) => pane.mode === "plane" && !pane.element.hidden)
      .map((pane) => {
        pane.render(this.scene);
        return { plane: pane.name, canvas: pane.output.snapshot(), rect: pane.canvas.getBoundingClientRect() };
      });
  }

  getScaleBar(name) {
    return this.panes.find((pane) => pane.name === name)?.scaleBar ?? null;
  }

  add3d(object) {
    this.threePane.addObject(object);
  }

  remove3d(object) {
    this.threePane.removeObject(object);
  }

  pickThreePoint(clientX, clientY) {
    return this.threePane.pointAtClient(clientX, clientY);
  }

  setDisplaySettings(settings) {
    this.scene.setDisplaySettings(settings);
    this.slicedPlane.setDisplaySettings(settings);
    this.surfacePane.setDisplaySettings(settings);
  }

  clearPlaneTiles() {
    this.slicedPlane.clear();
    for (const layer of this.layers) layer.slicedPlane.clear();
  }

  getVisibleScaleCoverage(name, renderToken) {
    return this.slicedPlane.visibleScaleCoverage(name, renderToken);
  }

  getRendererStats() {
    return Object.fromEntries(
      this.panes.map((pane) => [
        pane.name,
        {
          programs: pane.renderer.info.programs.length,
          textures: pane.renderer.info.memory.textures,
          geometries: pane.renderer.info.memory.geometries,
          sortObjects: pane.renderer.sortObjects,
        },
      ]),
    );
  }

  getSurfacePane() {
    return this.surfacePane;
  }

  nextFrame() {
    return new Promise((resolve) => window.requestAnimationFrame(resolve));
  }

  setAuxiliaryPane(name) {
    if (name !== "3d" && name !== "surface") throw new Error(`Unknown auxiliary pane: ${name}.`);
    for (const pane of this.panes) {
      if (pane.name === "3d" || pane.name === "surface") pane.setVisible(pane.name === name);
    }
  }

  showOrthogonalGrid(planesToRender, cameraOrientation, overlayContext) {
    const filter = planesToRender === null ? null : new Set(planesToRender);
    for (const pane of this.panes) {
      if (filter !== null && !filter.has(pane.name)) continue;
      if (pane.element.hidden && pane.mode !== "plane") continue;
      pane.render(this.scene, cameraOrientation, overlayContext);
      if (pane.mode !== "plane" || pane.element.hidden) continue;
      pane.showLoading(overlayContext.sourceActive ? this.slicedPlane.loadingScaleIndex(pane.name) : null);
    }
    this.renderStats.orthogonalFrames += 1;
  }

  getOrthogonalPanels() {
    return this.panes.filter((pane) => !pane.element.hidden && pane.output.shown).map((pane) => pane.panel());
  }

  getOrthogonalPanelPixelSize(name) {
    const [, , , width, height] = this.panes.find((pane) => pane.name === name).panel();
    return Math.max(1, Math.min(width, height));
  }

  getOrthogonalPanelAspect(name) {
    const panel = this.getOrthogonalPanels().find(([paneName]) => paneName === name);
    if (!panel) return 1;
    return Math.max(0.001, panel[3] / Math.max(1, panel[4]));
  }

  orthogonalPanelAt(clientX, clientY) {
    for (const pane of this.panes) {
      if (pane.contains(clientX, clientY)) return pane.panel();
    }
    return null;
  }

  resize() {
    invalidateElementRects();
    this.rendering.resize();
    const hiddenSide = this.threePane.hiddenPlaneSide();
    for (const pane of this.panes) {
      if (pane.mode === "plane") pane.resize(hiddenSide);
      else pane.resize();
    }
    const paneLayoutSignature = this.panes
      .filter((pane) => !pane.element.hidden)
      .map((pane) => {
        const [, , , width, height] = pane.panel();
        return `${pane.name}:${width}x${height}`;
      })
      .join("|");
    if (paneLayoutSignature !== this.paneLayoutSignature) {
      this.paneLayoutSignature = paneLayoutSignature;
      this.paneLayoutRevision += 1;
    }
  }
}

export function createView({ paneGrid, document, window, gpuByteLimit, onUploadError }) {
  const pixelRatio = Math.min(window.devicePixelRatio, 2);
  const rendering = new ViewerRenderer(paneGrid, pixelRatio, gpuByteLimit, onUploadError);
  for (const type of ["pointerdown", "pointermove", "wheel", "keydown", "input"])
    document.addEventListener(
      type,
      (event) => {
        if (type !== "pointermove" || event.buttons !== 0) rendering.uploadBudget.interact();
      },
      { capture: true, passive: true },
    );
  const scene = new OrthogonalScene();
  const panes = [
    new OrthogonalPane(paneGrid, pixelRatio, rendering, "xy", "XY"),
    new OrthogonalPane(paneGrid, pixelRatio, rendering, "zy", "YZ"),
    new OrthogonalPane(paneGrid, pixelRatio, rendering, "xz", "XZ"),
  ];
  const threePane = new ThreePane(paneGrid, panes, pixelRatio, rendering);
  panes.push(threePane);
  const surfacePane = new SurfacePane(paneGrid, pixelRatio, rendering);
  panes.push(surfacePane);

  return new ComposedView({
    panes,
    threePane,
    surfacePane,
    rendering,
    scene,
    paneGrid,
    pixelRatio,
  });
}
