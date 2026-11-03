import { RELIEF_MASK_KEY_STEP } from "../config.js";
import { surfaceCameraPointToScreen, surfaceCameraScreenToPoint } from "../surface/camera.js";
import { surfaceVolumetricDepthAtFraction } from "../surface/volumetric.js";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const CURVE_SAMPLES_PER_SEGMENT = 8;

export class SurfaceReliefMaskOverlay {
  constructor({ container, canvas, onSelect, onEdit }) {
    this.canvas = canvas;
    this.onSelect = onSelect;
    this.onEdit = onEdit;
    this.svg = document.createElementNS(SVG_NAMESPACE, "svg");
    this.svg.classList.add("surface-relief-overlay");
    this.svg.setAttribute("aria-label", "Editable Surface relief-mask boundaries");
    this.svg.setAttribute("hidden", "");
    this.lines = new Map();
    this.controls = new Map();
    this.model = null;
    this.selectedKey = null;
    this.drag = null;
    this.pendingEdit = null;
    this.editFrame = null;
    this.syncFrame = null;
    container.append(this.svg);
    new ResizeObserver(() => this.scheduleSync()).observe(canvas);
  }

  update(model) {
    this.model = model;
    this.scheduleSync();
  }

  clear() {
    if (
      this.model === null &&
      this.svg.hasAttribute("hidden") &&
      this.drag === null &&
      this.editFrame === null &&
      this.syncFrame === null
    )
      return;
    this.model = null;
    this.releaseDrag();
    this.selectedKey = null;
    this.pendingEdit = null;
    if (this.editFrame !== null) window.cancelAnimationFrame(this.editFrame);
    this.editFrame = null;
    if (this.syncFrame !== null) window.cancelAnimationFrame(this.syncFrame);
    this.syncFrame = null;
    this.svg.setAttribute("hidden", "");
  }

  scheduleSync() {
    if (this.syncFrame !== null) return;
    this.syncFrame = window.requestAnimationFrame(() => {
      this.syncFrame = null;
      this.sync();
    });
  }

  sync() {
    const model = this.model;
    this.svg.toggleAttribute("hidden", model === null || !model.visible);
    if (this.svg.hasAttribute("hidden")) return;
    const rect = this.svg.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    this.svg.setAttribute("viewBox", `0 0 ${rect.width} ${rect.height}`);
    const retainedLines = new Set();
    const retainedControls = new Set();
    for (const boundary of ["top", "bottom"]) {
      for (const [ridgeIndex, ridge] of model.mask[boundary].entries()) {
        const lineKey = `${boundary}:${ridgeIndex}`;
        retainedLines.add(lineKey);
        let path = this.lines.get(lineKey);
        if (path === undefined) {
          path = document.createElementNS(SVG_NAMESPACE, "path");
          path.classList.add("surface-relief-line", `surface-relief-${boundary}`);
          this.lines.set(lineKey, path);
          this.svg.prepend(path);
        }
        path.setAttribute("d", this.ridgePath(ridge, rect));
        for (const [pointIndex, point] of ridge.points.entries()) {
          const controlKey = `${boundary}:${ridgeIndex}:${pointIndex}`;
          retainedControls.add(controlKey);
          let control = this.controls.get(controlKey);
          if (control === undefined) {
            control = this.createControl(controlKey);
            this.controls.set(controlKey, control);
            this.svg.append(control);
          }
          const [x, y] = this.project(point.u, ridge.v, point.depth, rect);
          control.setAttribute("cx", x);
          control.setAttribute("cy", y);
          control.setAttribute(
            "aria-label",
            `${boundary} relief control at U ${point.u.toFixed(3)}, V ${ridge.v.toFixed(3)}, depth ${point.depth.toFixed(3)}`,
          );
          control.classList.toggle("selected", controlKey === this.selectedKey);
        }
      }
    }
    for (const [key, path] of this.lines) {
      if (retainedLines.has(key)) continue;
      path.remove();
      this.lines.delete(key);
    }
    for (const [key, control] of this.controls) {
      if (retainedControls.has(key)) continue;
      control.remove();
      this.controls.delete(key);
      if (this.selectedKey === key) this.selectedKey = null;
    }
  }

  ridgePath(ridge, rect) {
    const positions = [];
    for (let segment = 0; segment < ridge.points.length - 1; segment++) {
      const start = ridge.points[segment];
      const stop = ridge.points[segment + 1];
      for (let sample = segment === 0 ? 0 : 1; sample <= CURVE_SAMPLES_PER_SEGMENT; sample++) {
        const weight = sample / CURVE_SAMPLES_PER_SEGMENT;
        positions.push(
          this.project(
            start.u + (stop.u - start.u) * weight,
            ridge.v,
            start.depth + (stop.depth - start.depth) * weight,
            rect,
          ),
        );
      }
    }
    return positions.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x} ${y}`).join(" ");
  }

  project(u, v, depth, rect) {
    const model = this.model;
    const point_UV = [u * (model.grid_HW[1] - 1) - model.center_UV[0], v * (model.grid_HW[0] - 1) - model.center_UV[1]];
    const screen_UV = surfaceCameraPointToScreen(point_UV, this.cameraDepth(depth), model.visibleSpan_UV, model.camera);
    return [
      (0.5 + screen_UV[0] / model.visibleSpan_UV[0]) * rect.width,
      (0.5 - screen_UV[1] / model.visibleSpan_UV[1]) * rect.height,
    ];
  }

  surfacePosition(clientX, clientY, depth) {
    const rect = this.svg.getBoundingClientRect();
    const model = this.model;
    const screen_UV = [
      ((clientX - rect.left) / rect.width - 0.5) * model.visibleSpan_UV[0],
      (0.5 - (clientY - rect.top) / rect.height) * model.visibleSpan_UV[1],
    ];
    const relative_UV = surfaceCameraScreenToPoint(
      screen_UV,
      this.cameraDepth(depth),
      model.visibleSpan_UV,
      model.camera,
    );
    return [
      Math.min(1, Math.max(0, (relative_UV[0] + model.center_UV[0]) / Math.max(1, model.grid_HW[1] - 1))),
      Math.min(1, Math.max(0, (relative_UV[1] + model.center_UV[1]) / Math.max(1, model.grid_HW[0] - 1))),
    ];
  }

  cameraDepth(depth) {
    return surfaceVolumetricDepthAtFraction(this.model.volumetric, depth);
  }

  createControl(key) {
    const control = document.createElementNS(SVG_NAMESPACE, "circle");
    control.classList.add("surface-relief-control", `surface-relief-${key.split(":")[0]}`);
    control.setAttribute("r", "5");
    control.setAttribute("role", "button");
    control.setAttribute("tabindex", "0");
    control.dataset.key = key;
    control.addEventListener("pointerdown", (event) => this.beginDrag(event));
    control.addEventListener("keydown", (event) => this.nudge(event));
    return control;
  }

  beginDrag(event) {
    if (this.model === null || this.drag !== null || !event.isPrimary || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const key = event.currentTarget.dataset.key;
    const [boundary, ridgeIndexText, pointIndexText] = key.split(":");
    const ridgeIndex = Number(ridgeIndexText);
    const pointIndex = Number(pointIndexText);
    const point = this.model.mask[boundary][ridgeIndex].points[pointIndex];
    this.selectedKey = key;
    this.onSelect({ boundary, ridgeIndex, pointIndex, depth: point.depth });
    if (this.syncFrame !== null) window.cancelAnimationFrame(this.syncFrame);
    this.syncFrame = null;
    this.sync();
    this.drag = {
      pointerId: event.pointerId,
      boundary,
      ridgeIndex,
      pointIndex,
      depth: point.depth,
      depthOrigin: point.depth,
      depthOriginClientY: event.clientY,
      shifted: event.shiftKey,
      target: event.currentTarget,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.currentTarget.addEventListener("pointermove", this.handlePointerMove);
    event.currentTarget.addEventListener("pointerup", this.finishDrag);
    event.currentTarget.addEventListener("pointercancel", this.finishDrag);
  }

  handlePointerMove = (event) => {
    if (this.drag === null || event.pointerId !== this.drag.pointerId) return;
    let patch;
    if (event.shiftKey) {
      if (!this.drag.shifted) {
        this.drag.depthOrigin = this.drag.depth;
        this.drag.depthOriginClientY = event.clientY;
      }
      this.drag.shifted = true;
      this.drag.depth = Math.min(
        1,
        Math.max(0, this.drag.depthOrigin + (this.drag.depthOriginClientY - event.clientY) / this.canvas.clientHeight),
      );
      patch = { depth: this.drag.depth };
    } else {
      const changedFromDepth = this.drag.shifted;
      this.drag.shifted = false;
      const [u, v] = this.surfacePosition(event.clientX, event.clientY, this.drag.depth);
      patch = changedFromDepth ? { u, v, depth: this.drag.depth } : { u, v };
    }
    const pendingPatch = this.pendingEdit?.patch ?? {};
    this.pendingEdit = {
      boundary: this.drag.boundary,
      ridgeIndex: this.drag.ridgeIndex,
      pointIndex: this.drag.pointIndex,
      patch: { ...pendingPatch, ...patch },
    };
    if (this.editFrame === null) this.editFrame = window.requestAnimationFrame(() => this.flushEdit());
  };

  finishDrag = (event) => {
    if (this.drag === null || event.pointerId !== this.drag.pointerId) return;
    if (this.editFrame !== null) window.cancelAnimationFrame(this.editFrame);
    this.editFrame = null;
    this.flushEdit();
    this.releaseDrag();
  };

  releaseDrag() {
    if (this.drag === null) return;
    const { pointerId, target } = this.drag;
    if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
    target.removeEventListener("pointermove", this.handlePointerMove);
    target.removeEventListener("pointerup", this.finishDrag);
    target.removeEventListener("pointercancel", this.finishDrag);
    this.drag = null;
  }

  flushEdit() {
    this.editFrame = null;
    if (this.pendingEdit === null) return;
    const edit = this.pendingEdit;
    this.pendingEdit = null;
    this.onEdit(edit);
  }

  nudge(event) {
    if (!new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]).has(event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    const [boundary, ridgeIndexText, pointIndexText] = event.currentTarget.dataset.key.split(":");
    const ridgeIndex = Number(ridgeIndexText);
    const pointIndex = Number(pointIndexText);
    const ridge = this.model.mask[boundary][ridgeIndex];
    const point = ridge.points[pointIndex];
    const direction = new Set(["ArrowRight", "ArrowUp"]).has(event.key) ? 1 : -1;
    const patch = event.shiftKey
      ? { depth: Math.min(1, Math.max(0, point.depth + direction * RELIEF_MASK_KEY_STEP)) }
      : new Set(["ArrowLeft", "ArrowRight"]).has(event.key)
        ? { u: Math.min(1, Math.max(0, point.u + direction * RELIEF_MASK_KEY_STEP)) }
        : { v: Math.min(1, Math.max(0, ridge.v + direction * RELIEF_MASK_KEY_STEP)) };
    this.selectedKey = event.currentTarget.dataset.key;
    this.onSelect({ boundary, ridgeIndex, pointIndex, depth: point.depth });
    this.onEdit({ boundary, ridgeIndex, pointIndex, patch });
  }
}
