import { SURFACE_MARKER_OFF_NORMAL_LIMIT } from "../config.js";
import { surfaceCameraDemandBounds, surfaceCameraProjector } from "./camera.js";
import { projectSurfaceFocus } from "./focus.js";
import { surfaceNormalProjection } from "./normal-projection.js";
import { surfaceVisibleSpan } from "./viewport.js";
import { surfaceFlatCamera, surfaceVolumetricCamera } from "./volumetric.js";

export class SurfaceMarkers {
  constructor({ state, pane, onError }) {
    this.state = state;
    this.pane = pane;
    this.onError = onError;
    this.surface = null;
    this.focus = null;
    this.focusMarkerKey = null;
    this.pointer_ZYX = null;
    this.pointerMarkerKey = null;
  }

  setSurface(surface) {
    this.focus?.controller.abort();
    this.focus = null;
    this.focusMarkerKey = null;
    this.surface = surface;
    this.showPointer(null);
  }

  addMarks({ diameter, width, voxels }) {
    return this.pane.addMarks({ diameter, width, voxels });
  }

  shown() {
    return this.surface !== null && this.state.fullscreenViewport === "surface";
  }

  points() {
    const { normalOffset, normalSign } = this.state.surfaceView;
    const presentation = this.surface?.presentation ?? null;
    return (cell_HW) => {
      const frame = presentation?.surfaceFrameAt([cell_HW[1], cell_HW[0]], normalSign) ?? null;
      if (frame === null) return null;
      const point_ZYX = frame.point_ZYX.map((value, axis) => value + frame.normal_ZYX[axis] * normalOffset);
      return { point_ZYX, normal_ZYX: frame.normal_ZYX };
    };
  }

  resolveFocus(point_ZYX) {
    const key = `${point_ZYX.join()}:${this.surface.overlay.pyramid.revision}`;
    if (key === this.focus?.key) return this.focus.promise;
    this.focus?.controller.abort();
    const controller = new AbortController();
    const { presentation, projector, overlay } = this.surface;
    const focus = { key, controller, promise: null, projection: null };
    this.focus = focus;
    this.focusMarkerKey = null;
    focus.promise = Promise.resolve()
      .then(async () => {
        controller.signal.throwIfAborted();
        return projectSurfaceFocus(
          point_ZYX,
          await projector.focusCandidates(point_ZYX, controller.signal),
          overlay.surfaceSource,
          overlay.registration,
          (uv) => presentation.surfaceFrameAt(uv, 1),
          controller.signal,
        );
      })
      .then((projection) => {
        if (controller.signal.aborted) return null;
        focus.projection = projection;
        this.focusMarkerKey = null;
        this.syncFocus();
        return projection;
      })
      .catch((error) => {
        if (!controller.signal.aborted && error.name !== "AbortError") this.onError(error);
        return null;
      });
    return focus.promise;
  }

  syncFocus() {
    if (!this.shown()) return;
    this.resolveFocus(this.state.center_ZYX);
    const view = this.state.surfaceView;
    const key = [
      this.focus.key,
      JSON.stringify(view.volumetric),
      ...view.center_UV,
      view.normalOffset,
      view.normalSign,
      view.zoom,
      ...this.pane.panelSize(),
    ].join(":");
    if (this.focusMarkerKey === key) return;
    this.focusMarkerKey = key;
    this.place(this.pane.focusMarkers, this.focus.projection, this.state.center_ZYX, "Physical focus");
  }

  showPointer(point_ZYX) {
    this.pointer_ZYX = point_ZYX;
    this.syncPointer();
  }

  syncPointer() {
    const point_ZYX = this.pointer_ZYX;
    const presentation = this.surface?.presentation;
    if (point_ZYX === null || !this.shown() || !presentation.presented()) {
      this.pointerMarkerKey = null;
      this.pane.placeMarkers(this.pane.pointerMarkers, [null, null], "");
      return;
    }
    const key = [presentation.poseRevision, presentation.geometryRevision, ...this.pane.panelSize(), ...point_ZYX].join(
      ":",
    );
    if (key === this.pointerMarkerKey) return;
    this.pointerMarkerKey = key;
    const seed = this.surface.projector.project(point_ZYX);
    let projection = null;
    if (seed !== null && !seed.ambiguous) {
      const steps = surfaceNormalProjection(point_ZYX, seed, this.surface.overlay.surfaceSource.grid_HW);
      let step = steps.next();
      while (!step.done) {
        const frame = presentation.surfaceFrameAt(step.value, 1);
        if (frame === null) break;
        step = steps.next(frame);
      }
      projection = step.done ? step.value : null;
    }
    this.place(this.pane.pointerMarkers, projection, point_ZYX, "Mouse position");
  }

  view() {
    if (!this.shown()) return null;
    const view = this.state.surfaceView;
    const { grid_HW, scale_UV } = this.surface.overlay.surfaceSource;
    const span = surfaceVisibleSpan(grid_HW, scale_UV, view.zoom, this.pane.aspect());
    const box = surfaceCameraDemandBounds(span, 0, 0, surfaceVolumetricCamera(view.volumetric, scale_UV));
    const [u, v] = view.center_UV.map((value, axis) => value + box.centerOffset_UV[axis]);
    return {
      surface: this.surface.path,
      grid_HW: [...grid_HW],
      start_HW: [v - box.span_UV[1] / 2, u - box.span_UV[0] / 2],
      stop_HW: [v + box.span_UV[1] / 2, u + box.span_UV[0] / 2],
      scale_UV: [...scale_UV],
      pixel: span[0] / scale_UV[0] / this.pane.panelSize()[0],
    };
  }

  // [x, y] in 0-1 of each half, y down.
  screen() {
    const view = this.state.surfaceView;
    const source = this.surface.overlay.surfaceSource;
    const span = surfaceVisibleSpan(source.grid_HW, source.scale_UV, view.zoom, this.pane.aspect());
    const camera = surfaceVolumetricCamera(view.volumetric, source.scale_UV);
    const halves = this.pane.compareFlat ? [surfaceFlatCamera(camera), camera] : [camera];
    const projectors = halves.map((halfCamera) => surfaceCameraProjector(span, halfCamera));
    const [centerU, centerV] = view.center_UV;
    const relief = view.volumetric.relief * (view.volumetric.reverse ? -1 : 1);
    return (surface_UV, relativeDepth) => {
      const positions = [null, null];
      for (const [half, project] of projectors.entries()) {
        const screen_UV = project([surface_UV[0] - centerU, surface_UV[1] - centerV], relativeDepth * relief);
        positions[half] = [0.5 + screen_UV[0] / span[0], 0.5 - screen_UV[1] / span[1]];
      }
      return positions;
    };
  }

  place(markers, projection, point_ZYX, label) {
    if (
      projection === null ||
      projection.ambiguous ||
      projection.distanceSquared - projection.signedDepth ** 2 > SURFACE_MARKER_OFF_NORMAL_LIMIT ** 2
    ) {
      this.pane.placeMarkers(markers, [null, null], "");
      return;
    }
    const view = this.state.surfaceView;
    const relativeDepth = projection.signedDepth * view.normalSign - view.normalOffset;
    const positions = this.screen()(projection.surface_UV, relativeDepth);
    this.pane.placeMarkers(
      markers,
      positions,
      `${label}\nZYX ${point_ZYX.map((value) => value.toFixed(2)).join(", ")}\nslab depth ${relativeDepth.toFixed(2)}`,
    );
  }
}
