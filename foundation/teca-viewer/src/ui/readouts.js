import { ORTHOGONAL_PLANES, PREFETCH_CHUNK_PRIORITY } from "../config.js";
import { quatToEulerXYZDegrees } from "../math.js";
import {
  orthogonalStepPlane,
  planeDisplayBasis,
  planeRotation,
  shownView,
  viewerDock,
  viewWidth,
} from "../views/pose.js";
import { rotationReadout } from "./rotation-readout.js";
import { formatScaleLengthUm, scaleBarMeasurement } from "./scale-bar.js";

export function formatControlNumber(value) {
  return Number.parseFloat(Number(value).toFixed(3)).toString();
}

async function readRawFinestVoxel(volume, channel, point_ZYX, signal) {
  const scale = volume.scales[0];
  const pointInScale_ZYX = point_ZYX.map((value, axis) =>
    Math.round((value - scale.offset_ZYX[axis]) / scale.downsample_ZYX[axis]),
  );
  if (pointInScale_ZYX.some((value, axis) => value < 0 || value >= scale.shape_ZYX[axis])) {
    throw new Error(`ZYX ${point_ZYX.join(", ")} is outside the finest source grid.`);
  }
  const chunk_ZYX = pointInScale_ZYX.map((value, axis) => Math.floor(value / scale.chunks_ZYX[axis]));
  const chunk = await volume.getChunk(...chunk_ZYX, channel, scale, signal, PREFETCH_CHUNK_PRIORITY);
  const local_ZYX = pointInScale_ZYX.map(
    (value, axis) => ((value % scale.chunks_ZYX[axis]) + scale.chunks_ZYX[axis]) % scale.chunks_ZYX[axis],
  );
  return chunk[(local_ZYX[0] * scale.chunks_ZYX[1] + local_ZYX[1]) * scale.chunks_ZYX[2] + local_ZYX[2]];
}

export function showPieces(element, pieces) {
  const spans = [...element.children];
  if (spans.length === pieces.length && spans.every((span, index) => span.textContent === pieces[index])) return;
  element.replaceChildren(
    ...pieces.map((piece) => Object.assign(document.createElement("span"), { textContent: piece })),
  );
}

export function createReadouts({ state, view, elements }) {
  let inspectionAbortController = null;
  let inspectionGeneration = 0;

  function showPointerInspection(...pieces) {
    showPieces(elements.cursorValue, pieces);
  }

  function invalidatePointerInspection() {
    showPieces(elements.cursorValue, state.volume === null ? ["cursor: —"] : []);
  }

  function cancelInspection() {
    inspectionGeneration += 1;
    inspectionAbortController?.abort();
    inspectionAbortController = null;
  }

  async function inspectPoint(plane, point_ZYX) {
    cancelInspection();
    const generation = inspectionGeneration;
    if (plane === "3d") {
      showPointerInspection("cursor: unavailable in 3D overview");
      return;
    }
    if (plane === "surface") {
      showPointerInspection(
        point_ZYX === null
          ? "cursor: no published surface at this point (mesh hole, edge, or loading)"
          : `surface XYZ [${[...point_ZYX].reverse().join(", ")}]`,
      );
      return;
    }
    if (point_ZYX.some((value, axis) => value < 0 || value >= state.volume.fullShape_ZYX[axis])) {
      showPointerInspection(`cursor ZYX ${point_ZYX.join(", ")}`, "outside source");
      return;
    }
    const abortController = new AbortController();
    inspectionAbortController = abortController;
    const volume = state.volume;
    const channel = state.selectedChannel;
    try {
      const value = await readRawFinestVoxel(volume, channel, point_ZYX, abortController.signal);
      if (
        abortController.signal.aborted ||
        generation !== inspectionGeneration ||
        volume !== state.volume ||
        channel !== state.selectedChannel
      )
        return;
      showPointerInspection(`cursor ZYX ${point_ZYX.join(", ")}`, `value ${value}`);
    } catch (error) {
      if (error.name !== "AbortError" && generation === inspectionGeneration) showPointerInspection(error.message);
    } finally {
      if (inspectionAbortController === abortController) inspectionAbortController = null;
    }
  }

  function refreshViewReadout() {
    const pieces = [];
    if (state.volume !== null) {
      const name = state.activeViewport;
      let label = name === "zy" ? "YZ" : name.toUpperCase();
      let scale = null;
      let zoom = state.zoom;
      if (name === "surface") {
        label = "Surface";
        scale = view.getSurfacePane().shownScale();
        zoom = state.surfaceView.zoom;
      } else if (name !== "3d") {
        scale = view.slicedPlane.shownScale(name);
      }
      pieces.push(
        label,
        ...(scale === null ? [] : [`mag ${scale.downsample_ZYX.join("×")}`]),
        `zoom ${zoom.toFixed(2)}×`,
      );
    }
    showPieces(elements.magZoom, pieces);
  }

  function syncPositionInputs() {
    invalidatePointerInspection();
    for (const [axis, input] of [elements.posZ, elements.posY, elements.posX].entries()) {
      const value = String(state.center_ZYX[axis]);
      if (input.value !== value) input.value = value;
    }
    const position_XYZ = [...state.center_ZYX].reverse().map(formatControlNumber);
    if (elements.positionNumber.validity.customError) elements.positionNumber.setCustomValidity("");
    const value = position_XYZ.join(", ");
    if (elements.positionNumber.value !== value) elements.positionNumber.value = value;
    const text = `XYZ [${value}]`;
    if (elements.positionText.textContent !== text) elements.positionText.textContent = text;
  }

  function syncZoomInputs() {
    invalidatePointerInspection();
    elements.zoom.value = state.zoom;
    elements.zoomNumber.value = formatControlNumber(state.zoom);
  }

  function syncRotationReadout() {
    invalidatePointerInspection();
    for (const pane of view.surfaceLayout.planes) {
      if (pane.name === "3d") continue;
      const normal = planeDisplayBasis(state, pane.name).normal_XYZ;
      const axis = normal.findIndex((value) => Math.abs(value) > 1 - 1e-6);
      const name = axis < 0 ? "Oblique" : ["YZ", "XZ", "XY"][axis];
      pane.element.querySelector(".pane-name").textContent = name;
      pane.element.querySelector(".pane-orientation").options[0].textContent = name;
    }
    const plane = orthogonalStepPlane(state);
    const rotation = state.activeViewport === "3d" ? state.cameraOrientation : planeRotation(state, plane);
    elements.rotationNumber.value = quatToEulerXYZDegrees(rotation).map(formatControlNumber).join(", ");
    elements.rotationText.textContent = rotationReadout({
      plane,
      rotationMode: state.rotationMode,
      rotation: state.rotation,
      paneRotations: state.paneRotations,
    });
    for (const button of elements.rotationMode.querySelectorAll("button[data-rotation-mode]")) {
      const active = button.dataset.rotationMode === state.rotationMode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    }
  }

  function updateScaleBars() {
    refreshViewReadout();
    if (!state.volume) return;
    const viewWidthVoxels = viewWidth(state);
    const pixelRatio = view.pixelRatio;
    for (const [plane, , , panelWidth, panelHeight] of view.getOrthogonalPanels()) {
      const scaleBar = view.getScaleBar(plane);
      if (!scaleBar) continue;
      const measurement = !ORTHOGONAL_PLANES.includes(plane)
        ? null
        : scaleBarMeasurement(
            plane,
            planeRotation(state, plane),
            state.volume.metadata.physicalScale_ZYX,
            state.volume.metadata.physicalUnits_ZYX,
            viewWidthVoxels,
            panelWidth / pixelRatio,
            Math.max(0.001, panelWidth / panelHeight),
          );
      if (measurement === null) {
        scaleBar.hidden = true;
        delete scaleBar.dataset.label;
        scaleBar.style.removeProperty("--scale-bar-width");
        continue;
      }
      scaleBar.hidden = false;
      scaleBar.title =
        measurement.unit === null
          ? "Voxel units: this source does not declare physical spacing for this direction."
          : "Physical distance along the displayed horizontal axis.";
      scaleBar.style.setProperty("--scale-bar-width", `${measurement.widthPixels}px`);
      scaleBar.dataset.label =
        measurement.unit === null
          ? `${formatControlNumber(measurement.lengthUnits)} vox`
          : ["micrometer", "micrometers", "um", "µm"].includes(measurement.unit)
            ? formatScaleLengthUm(measurement.lengthUnits)
            : `${formatControlNumber(measurement.lengthUnits)} ${measurement.unit}`;
    }
  }

  function updateViewerShellState() {
    const dock = viewerDock(state);
    view.surfaceLayout.apply(dock?.layout ?? state.surfaceView.layout, dock !== null, dock?.primary ?? "surface");
    for (const button of document.querySelectorAll(".pane-maximize")) {
      const maximized =
        dock !== null && dock.layout.mode !== "solo"
          ? dock.layout.maximized === button.dataset.plane
          : state.fullscreenViewport === button.dataset.plane;
      button.textContent = maximized ? "⊡" : "⛶";
      button.title = maximized ? "Restore panes (Z)" : "Maximize pane (Z)";
      button.setAttribute(
        "aria-label",
        maximized ? "Restore panes" : `Maximize ${button.dataset.plane.toUpperCase()} pane`,
      );
    }
    const shown = shownView(state);
    for (const button of elements.viewStrip.querySelectorAll("[data-view]"))
      button.setAttribute("aria-pressed", String(button.dataset.view === shown));
    for (const viewport of view.paneNames()) {
      elements.viewerShell.classList.toggle(`active-${viewport}`, state.activeViewport === viewport);
      elements.viewerShell.classList.toggle(`fullscreen-${viewport}`, state.fullscreenViewport === viewport);
    }
    elements.viewerShell.classList.toggle("crosshair-hidden", !state.crosshairVisible);
    syncRotationReadout();
    refreshViewReadout();
  }

  return {
    showPointerInspection,
    invalidatePointerInspection,
    cancelInspection,
    inspectPoint,
    refreshViewReadout,
    syncPositionInputs,
    syncZoomInputs,
    syncRotationReadout,
    updateScaleBars,
    updateViewerShellState,
  };
}
