import { elementRect } from "../element-rect.js";
import { PaneFramebuffer } from "../gpu/pane-framebuffer.js";

export class OrthogonalPane {
  constructor(container, pixelRatio, rendering, name, label) {
    this.name = name;
    this.mode = "plane";
    this.rasterPose = null;
    this.element = document.createElement("div");
    this.element.className = `viewer-pane pane-${label.toLowerCase()}`;
    this.element.dataset.plane = this.name;
    this.canvas = document.createElement("canvas");
    this.canvas.id = `viewer-${name}`;
    this.canvas.tabIndex = 0;
    this.canvas.setAttribute("aria-label", `Interactive ${name.toUpperCase()} slice viewport`);
    const paneName = document.createElement("div");
    paneName.className = "pane-name";
    paneName.textContent = label;
    const maximize = document.createElement("button");
    maximize.type = "button";
    maximize.className = "pane-maximize";
    maximize.dataset.plane = this.name;
    maximize.textContent = "⛶";
    maximize.setAttribute("aria-label", `Maximize ${this.name.toUpperCase()} pane`);
    maximize.title = "Maximize pane (Z)";
    const orientation = document.createElement("select");
    orientation.className = "pane-orientation";
    orientation.dataset.plane = this.name;
    orientation.setAttribute("aria-label", `Reset ${this.name.toUpperCase()} plane orientation`);
    orientation.title = "Reset orientation. Middle-drag tilts this plane.";
    orientation.innerHTML =
      '<option value="">Axes…</option><option value="xy">XY</option><option value="zy">YZ</option><option value="xz">XZ</option>';
    const corner = document.createElement("div");
    corner.className = "pane-corner-controls";
    corner.append(orientation, maximize);
    this.element.append(corner);
    this.loadingStatus = document.createElement("div");
    this.loadingStatus.className = "pane-loading";
    this.loadingStatus.hidden = true;
    this.element.append(this.loadingStatus);
    this.scaleBar = document.createElement("div");
    this.scaleBar.className = "scale-bar";
    this.element.append(this.canvas, paneName, this.scaleBar);
    this.pointerPoint_ZYX = null;
    this.pointerRadius = null;
    this.reachRing = document.createElement("div");
    this.reachRing.className = "pane-reach";
    this.reachRing.hidden = true;
    this.element.append(this.reachRing);
    this.pointerMarker = document.createElement("div");
    this.pointerMarker.className = "plane-pointer";
    this.pointerMarker.hidden = true;
    this.pointerMarker.setAttribute("aria-label", "Projected mouse position");
    this.element.append(this.pointerMarker);
    container.append(this.element);
    this.pixelRatio = pixelRatio;
    this.renderer = rendering.renderer;
    this.gpuTimer = rendering.gpuTimer;
    this.output = new PaneFramebuffer(rendering, this.canvas);
  }

  panel() {
    return [
      this.name,
      0,
      0,
      this.canvas.width || Math.max(1, Math.round(this.canvas.clientWidth * this.pixelRatio)),
      this.canvas.height || Math.max(1, Math.round(this.canvas.clientHeight * this.pixelRatio)),
    ];
  }

  contains(clientX, clientY) {
    const rect = elementRect(this.canvas);
    return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom;
  }

  resize(hiddenSide) {
    this.output.resize(hiddenSide);
  }

  showPointer(point_ZYX, radius) {
    const pose = this.rasterPose;
    this.pointerPoint_ZYX = pose === null ? null : point_ZYX;
    this.pointerRadius = radius;
    // Each frame redraws the pointer: an element is hidden or shown only when that changes, since each change
    // restyles the page.
    const hide = (element) => {
      if (!element.hidden) element.hidden = true;
    };
    if (this.pointerPoint_ZYX === null || this.element.hidden) {
      hide(this.pointerMarker);
      hide(this.reachRing);
      return;
    }
    const [, , , width, height] = this.panel();
    const spanX = pose.viewWidthVoxels * Math.max(1, width / height);
    const spanY = pose.viewWidthVoxels * Math.max(1, height / width);
    const delta_ZYX = point_ZYX.map((value, axis) => value - pose.center_ZYX[axis]);
    const u = delta_ZYX.reduce((sum, value, axis) => sum + value * pose.basis.right_XYZ[2 - axis], 0);
    const v = delta_ZYX.reduce((sum, value, axis) => sum + value * pose.basis.up_XYZ[2 - axis], 0);
    const x = 0.5 + u / spanX;
    const y = 0.5 - v / spanY;
    if (y < 0 || y > 1 || x < 0 || x > 1) {
      hide(this.pointerMarker);
      hide(this.reachRing);
      return;
    }
    if (this.pointerMarker.hidden) this.pointerMarker.hidden = false;
    this.pointerMarker.style.left = `${100 * x}%`;
    this.pointerMarker.style.top = `${100 * y}%`;
    if (radius === null) {
      hide(this.reachRing);
      return;
    }
    if (this.reachRing.hidden) this.reachRing.hidden = false;
    this.reachRing.style.left = `${100 * x}%`;
    this.reachRing.style.top = `${100 * y}%`;
    this.reachRing.style.width = `${(200 * radius) / spanX}%`;
    this.reachRing.style.height = `${(200 * radius) / spanY}%`;
  }

  showLoading(scaleIndex) {
    this.loadingStatus.hidden = scaleIndex === null;
    const text = `Loading level ${scaleIndex}…`;
    if (scaleIndex !== null && this.loadingStatus.textContent !== text) this.loadingStatus.textContent = text;
  }

  render(scene, _cameraOrientation, overlayContext = null) {
    const [, , , width, height] = this.panel();
    this.output.begin(true);
    const timing = this.gpuTimer.begin(`ct-${this.name}`);
    try {
      scene.renderPane(this.renderer, this.name, width, height);
    } finally {
      this.gpuTimer.end(timing);
    }
    if (overlayContext !== null) {
      this.rasterPose = overlayContext.sourceActive
        ? {
            center_ZYX: [...overlayContext.center_ZYX],
            viewWidthVoxels: overlayContext.viewWidthVoxels,
            basis: overlayContext.planeBases[this.name],
          }
        : null;
    }
    this.output.complete();
    this.showPointer(this.pointerPoint_ZYX, this.pointerRadius);
  }
}
