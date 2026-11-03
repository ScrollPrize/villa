import * as THREE from "../../vendor/three.module.js";
import { elementRect } from "../element-rect.js";
import { FilterChain } from "../gpu/filter-chain.js";
import { PaneFramebuffer } from "../gpu/pane-framebuffer.js";
import { layerPresentationScales } from "../gpu/presentation-bricks.js";
import { greyTexture } from "../surface/layers.js";
import { drawSurfaceMaterialFlat } from "../surface/material.js";
import { surfacePresentationScale } from "../surface/scale-stack.js";
import { SurfaceMarks } from "./surface-marks.js";

export class SurfacePane {
  constructor(container, pixelRatio, rendering) {
    this.name = "surface";
    this.mode = "surface";
    this.element = document.createElement("div");
    this.element.className = "viewer-pane pane-surface";
    this.element.dataset.plane = this.name;
    this.element.hidden = true;
    this.canvas = document.createElement("canvas");
    this.canvas.id = "viewer-surface";
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute("aria-label", "Interactive tifxyz surface viewport");
    const paneName = document.createElement("div");
    paneName.className = "pane-name";
    paneName.textContent = "Surface";
    const returnButton = document.createElement("button");
    returnButton.type = "button";
    returnButton.className = "pane-return";
    returnButton.textContent = "← CT panes";
    returnButton.title = "Return to CT panes (U)";
    this.element.append(returnButton);
    this.scaleBar = document.createElement("div");
    this.scaleBar.className = "scale-bar";
    this.scaleBar.hidden = true;
    this.curtain = document.createElement("canvas");
    this.curtain.className = "surface-curtain";
    this.curtain.hidden = true;
    this.curtainLabel = document.createElement("div");
    this.curtainLabel.className = "surface-curtain-label";
    this.curtainLabel.hidden = true;
    this.reachRing = document.createElement("div");
    this.reachRing.className = "pane-reach";
    this.reachRing.hidden = true;
    this.element.append(this.canvas, paneName, this.scaleBar, this.curtain, this.curtainLabel, this.reachRing);
    container.append(this.element);
    this.pixelRatio = pixelRatio;
    this.shownWith = [];
    this.compareFlat = false;
    this.interactionX = null;
    this.compareLabels = ["Flat", "Slab"].map((text, half) => {
      const label = document.createElement("div");
      label.className = `surface-compare-label surface-compare-${half === 0 ? "flat" : "slab"}`;
      label.textContent = text;
      label.hidden = true;
      this.element.append(label);
      return label;
    });
    this.focusMarkers = [0, 1].map(() => {
      const marker = document.createElement("div");
      marker.className = "surface-focus";
      marker.hidden = true;
      marker.setAttribute("aria-label", "Shared physical focus");
      this.element.append(marker);
      return marker;
    });
    this.pointerMarkers = [0, 1].map(() => {
      const marker = document.createElement("div");
      marker.className = "plane-pointer";
      marker.hidden = true;
      marker.setAttribute("aria-label", "Projected mouse position");
      this.element.append(marker);
      return marker;
    });
    this.loadingStatus = document.createElement("div");
    this.loadingStatus.className = "surface-loading-status";
    this.loadingStatus.setAttribute("role", "status");
    this.loadingStatus.hidden = true;
    this.element.append(this.loadingStatus);
    this.loadingStartedAt = null;
    this.loadingRevision = null;
    this.rendering = rendering;
    this.renderer = rendering.renderer;
    this.gpuTimer = rendering.gpuTimer;
    this.output = new PaneFramebuffer(rendering, this.canvas);
    this.baseScene = new THREE.Scene();
    this.baseRevision = 0;
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    this.camera.position.set(0, 0, 2);
    this.presentation = null;
    this.presentationPath = null;
    this.layers = [];
    this.images = [];
    this.filterPasses = [];
    this.filterReach = 0;
    this.filterChain = new FilterChain();
    this.alone = null;
    this.marks = new SurfaceMarks(pixelRatio);
  }

  setPresentation(presentation, path) {
    this.presentation = presentation;
    this.presentationPath = path;
    for (const layer of this.layers)
      layer.drawn = presentation?.layers.add(layer.volume, layer.scales, layer.style, layer.renderOrder) ?? null;
    for (const image of this.images) image.drawn = this.drawImage(image);
    this.baseRevision++;
    this.loadingStartedAt = performance.now();
    this.loadingRevision = null;
    this.loadingStatus.hidden = presentation === null;
  }

  addLayer(volume, primaryScales, style, renderOrder) {
    const scales = layerPresentationScales(primaryScales, volume.scales, surfacePresentationScale);
    const drawn = this.presentation?.layers.add(volume, scales, style, renderOrder) ?? null;
    const layer = { volume, scales, style, renderOrder, drawn };
    this.layers.push(layer);
    return layer;
  }

  setLayerStyle(layer, style) {
    layer.style = style;
    if (layer.drawn !== null) this.presentation.layers.setStyle(layer.drawn, style);
  }

  refreshLayer(layer, changes) {
    if (layer.drawn !== null) this.presentation.layers.refresh(layer.drawn, changes);
  }

  removeLayer(layer) {
    this.layers = this.layers.filter((candidate) => candidate !== layer);
    if (layer.drawn !== null) this.presentation.layers.remove(layer.drawn);
  }

  clearLayers() {
    for (const layer of this.layers) if (layer.drawn !== null) this.presentation.layers.remove(layer.drawn);
    this.layers = [];
    for (const image of this.images) this.removeImage(image);
  }

  addImage(path, pixels, style, renderOrder) {
    const { texture, bytes } = greyTexture(this.rendering, pixels, 0);
    const image = { path, texture, shape_HW: pixels.shape_HW, detail: null, style, renderOrder, drawn: null };
    image.allocatedBytes = () => bytes + (image.detail?.bytes ?? 0);
    this.rendering.addAllocation(image);
    image.drawn = this.drawImage(image);
    this.images.push(image);
    return image;
  }

  setImageDetail(image, detail) {
    const freed = image.detail?.bytes ?? 0;
    const next = detail === null ? null : { ...detail, ...greyTexture(this.rendering, detail, freed) };
    image.detail?.texture.dispose();
    image.detail = next;
    if (image.drawn !== null) this.presentation.layers.setDetail(image.drawn, next);
  }

  setImageStyle(image, style) {
    image.style = style;
    if (image.drawn !== null) this.presentation.layers.setStyle(image.drawn, style);
  }

  removeImage(image) {
    this.images = this.images.filter((candidate) => candidate !== image);
    if (image.drawn !== null) this.presentation.layers.removeImage(image.drawn);
    image.drawn = null;
    image.texture.dispose();
    image.detail?.texture.dispose();
    this.rendering.removeAllocation(image);
  }

  drawImage(image) {
    if (this.presentation === null || image.path !== this.presentationPath) return null;
    return this.presentation.layers.addImage(image);
  }

  filterCT(passes, reach) {
    this.filterPasses = passes;
    this.filterReach = reach;
    if (passes.length === 0) this.filterChain.release();
    this.presentation?.requestRender();
    this.presentation?.scheduleReplan();
  }

  showAlone(alone) {
    this.alone = alone;
    this.sceneChanged();
  }

  hasImages() {
    return this.images.length > 0;
  }

  setCompareFlat(compare) {
    this.compareFlat = compare;
    for (const label of this.compareLabels) label.hidden = !compare;
    this.element.classList.toggle("surface-compare", compare);
  }

  holdInteraction(clientX) {
    this.interactionX = clientX;
  }

  setDisplaySettings(settings) {
    this.presentation?.setDisplaySettings(settings);
  }

  setVisible(visible) {
    this.element.hidden = !visible;
    for (const element of this.shownWith) element.hidden = !visible;
  }

  showWith(...elements) {
    this.shownWith.push(...elements);
    for (const element of elements) element.hidden = this.element.hidden;
  }

  panel() {
    return [this.name, 0, 0, ...this.panelSize()];
  }

  showReach(x, y, radius) {
    if (this.reachRing.hidden !== (radius === null)) this.reachRing.hidden = radius === null;
    if (radius === null) return;
    const rect = elementRect(this.element);
    this.reachRing.style.left = `${x - rect.left}px`;
    this.reachRing.style.top = `${y - rect.top}px`;
    this.reachRing.style.width = this.reachRing.style.height = `${2 * radius}px`;
  }

  placeMarkers(markers, positions, title) {
    for (const [index, marker] of markers.entries()) {
      const position = positions[index];
      marker.hidden = position === null || position.some((value) => value < 0 || value > 1);
      if (marker.hidden) continue;
      marker.style.left = `${(100 * (position[0] + index)) / (this.compareFlat ? 2 : 1)}%`;
      marker.style.top = `${100 * position[1]}%`;
      marker.title = title;
    }
  }

  addMarks(options) {
    return this.marks.add(options);
  }

  snapshot() {
    return { plane: this.name, canvas: this.output.snapshot(), rect: this.canvas.getBoundingClientRect() };
  }

  panelSize() {
    return [
      Math.max(
        1,
        Math.floor(
          (this.canvas.width || Math.round(this.canvas.clientWidth * this.pixelRatio)) / (this.compareFlat ? 2 : 1),
        ),
      ),
      this.canvas.height || Math.max(1, Math.round(this.canvas.clientHeight * this.pixelRatio)),
    ];
  }

  interactionRect(clientX) {
    const rect = elementRect(this.canvas);
    if (!this.compareFlat) return rect;
    const width = rect.width / 2;
    const left = (this.interactionX ?? clientX) < rect.left + width ? rect.left : rect.left + width;
    return { left, right: left + width, top: rect.top, bottom: rect.bottom, width, height: rect.height };
  }

  flatAt(clientX) {
    const rect = elementRect(this.canvas);
    return this.compareFlat && (this.interactionX ?? clientX) < rect.left + rect.width / 2;
  }

  aspect() {
    const [width, height] = this.panelSize();
    return Math.max(0.001, width / Math.max(1, height));
  }

  contains(clientX, clientY) {
    if (this.element.hidden) return false;
    const rect = elementRect(this.canvas);
    return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom;
  }

  resize() {
    if (this.element.hidden) return;
    this.output.resize();
    this.configureCamera();
  }

  shownScale() {
    return this.presentation?.shownScale() ?? null;
  }

  render() {
    if (this.presentation !== null) {
      this.presentation.render();
      const now = performance.now();
      const revision = [
        this.presentation.poseRevision,
        this.presentation.geometryRevision,
        this.presentation.dataRevision,
        this.presentation.planCount,
      ].join("/");
      if (revision !== this.loadingRevision) {
        const progress = this.presentation.loadingProgress();
        const elapsed = (now - this.loadingStartedAt) / 1000;
        this.loadingStatus.textContent = `${progress.phase}, ${elapsed.toFixed(1)} s after preparation`;
        this.loadingStatus.hidden = progress.ready;
        this.loadingRevision = revision;
      }
    }
    if (!this.curtain.hidden && this.presentation?.presentedPrimaryFrame() === true) this.hideCurtain();
    this.output.complete();
  }

  showCurtain(snapshots, label) {
    if (snapshots.length === 0) return;
    const left = Math.min(...snapshots.map(({ rect }) => rect.left));
    const top = Math.min(...snapshots.map(({ rect }) => rect.top));
    const width = Math.max(...snapshots.map(({ rect }) => rect.right)) - left;
    const height = Math.max(...snapshots.map(({ rect }) => rect.bottom)) - top;
    this.curtain.width = Math.max(1, Math.round(width * this.pixelRatio));
    this.curtain.height = Math.max(1, Math.round(height * this.pixelRatio));
    const context = this.curtain.getContext("2d");
    context.fillStyle = "#000";
    context.fillRect(0, 0, this.curtain.width, this.curtain.height);
    for (const { canvas, rect } of snapshots) {
      context.drawImage(
        canvas,
        (rect.left - left) * this.pixelRatio,
        (rect.top - top) * this.pixelRatio,
        rect.width * this.pixelRatio,
        rect.height * this.pixelRatio,
      );
    }
    this.curtainLabel.textContent = label;
    this.curtain.hidden = false;
    this.curtainLabel.hidden = false;
  }

  hideCurtain() {
    this.curtain.hidden = true;
    this.curtainLabel.hidden = true;
  }

  renderScene(mesh, baseImage) {
    this.output.begin(false);
    this.configureCamera();
    for (const child of this.baseScene.children)
      child.material.uniforms.drawn.value = this.alone === (child === mesh ? "layers" : "ct") ? 0 : 1;
    this.renderer.setClearColor(0x000000, 1);
    this.renderer.clear(true, true, true);
    if (this.compareFlat) {
      this.baseRevision++;
      const width = this.canvas.width / this.pixelRatio;
      const height = this.canvas.height / this.pixelRatio;
      this.renderer.setScissorTest(true);
      this.renderer.setViewport(0, 0, width / 2, height);
      this.renderer.setScissor(0, 0, width / 2, height);
      drawSurfaceMaterialFlat(mesh.material, () => {
        this.renderer.render(this.baseScene, this.camera);
        this.marks.render(this.renderer, mesh.material.uniforms, this.presentationPath);
      });
      this.renderer.setViewport(width / 2, 0, width / 2, height);
      this.renderer.setScissor(width / 2, 0, width / 2, height);
      this.renderer.render(this.baseScene, this.camera);
      this.marks.render(this.renderer, mesh.material.uniforms, this.presentationPath);
      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, width, height);
    } else {
      const { surfaceVisibleSpan_UV, surfaceScale_UV, volumetricEnabled } = mesh.material.uniforms;
      const draw = () => {
        const span = surfaceVisibleSpan_UV.value.x / surfaceScale_UV.value.x;
        if (
          this.filterPasses.length === 0 ||
          volumetricEnabled.value !== 0 ||
          this.alone === "layers" ||
          !this.filterChain.draw(this.renderer, this.filterPasses, this.filterReach, mesh, this.camera, span)
        ) {
          this.renderer.render(this.baseScene, this.camera);
          return;
        }
        mesh.visible = false;
        this.renderer.render(this.baseScene, this.camera);
        mesh.visible = true;
      };
      const { width, height } = this.output.target;
      if (baseImage.prepare(width, height)) baseImage.render(this.renderer, draw, this.baseRevision);
      else draw();
      this.marks.render(this.renderer, mesh.material.uniforms, this.presentationPath);
    }
  }

  sceneChanged() {
    this.baseRevision++;
  }

  configureCamera() {
    const aspect = this.aspect();
    if (aspect >= 1) {
      this.camera.left = -aspect;
      this.camera.right = aspect;
      this.camera.top = 1;
      this.camera.bottom = -1;
    } else {
      this.camera.left = -1;
      this.camera.right = 1;
      this.camera.top = 1 / aspect;
      this.camera.bottom = -1 / aspect;
    }
    this.camera.updateProjectionMatrix();
    for (const child of this.baseScene.children) {
      if (!child.isMesh) continue;
      child.scale.set(aspect >= 1 ? aspect * 2 : 2, aspect >= 1 ? 2 : 2 / aspect, 1);
    }
  }
}
