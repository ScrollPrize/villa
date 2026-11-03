import { CLICK_SLOP, WHEEL_ZOOM_FACTOR } from "../config.js";
import { elementRect } from "../element-rect.js";
import { baseViewWidth, panelVoxelOffset, planeDisplayBasis } from "../views/pose.js";

const WHEEL_IGNORED = "button, input, select, summary, a, [role=separator], .surface-navigation-controls";
const PRESS_IGNORED = `#status, ${WHEEL_IGNORED}`;
const LABEL_OFFSET = 14;

export function createPointerControls({
  state,
  view,
  elements,
  surfaceController,
  surfaceControls,
  surfaceMarkers,
  setPhysicalFocus,
  panByPixels,
  tiltPlaneByPixels,
  rotateCamera3dByPixels,
  syncRotationReadout,
  showPointerInspection,
  invalidatePointerInspection,
  inspectPoint,
  cancelInspection,
  setActivePlane,
  setActiveViewport,
  setZoomAtClientPoint,
  scheduleDeferredSettle,
  publishSession,
  setStatus,
}) {
  let hoverFrame = null;
  let hoverEvent = null;
  let pointerCaptureId = null;
  let panPressed = false;

  function cancelHover() {
    if (hoverFrame !== null) cancelAnimationFrame(hoverFrame);
    hoverFrame = hoverEvent = null;
    cancelInspection();
  }

  function releasePointerCapture() {
    if (hoverFrame !== null) cancelAnimationFrame(hoverFrame);
    hoverFrame = hoverEvent = null;
    updatePointerMarkers();
    view.getSurfacePane().holdInteraction(null);
    cancelInspection();
    if (pointerCaptureId === null) return;
    if (elements.viewerShell.hasPointerCapture(pointerCaptureId)) {
      elements.viewerShell.releasePointerCapture(pointerCaptureId);
    }
    pointerCaptureId = null;
  }

  function cancelPointerInteraction() {
    const drag = state.pointerDrag;
    invalidatePointerInspection();
    state.pointerDrag = null;
    releasePointerCapture();
    if (drag?.kind === "tool") drag.tool.up(drag.at);
  }

  function setTool(tool) {
    const previous = state.tool;
    if (tool !== previous) cancelPointerInteraction();
    state.tool = tool;
    if (tool !== previous) previous?.putAway();
    const pointer = elements.viewerShell.matches(":hover") ? state.lastPointerClient : null;
    updatePointerMarkers(pointer === null ? null : { clientX: pointer.x, clientY: pointer.y });
  }

  function holdPan() {
    if (state.panHeld) return;
    state.panHeld = true;
    panPressed = false;
  }

  function releasePan() {
    const unused = state.panHeld && !panPressed;
    state.panHeld = false;
    return unused;
  }

  function capturePointer(pointerId) {
    if (state.pointerDrag.plane === "surface") view.getSurfacePane().holdInteraction(state.pointerDrag.startX);
    elements.viewerShell.setPointerCapture(pointerId);
    pointerCaptureId = pointerId;
  }

  function pointFromClient(plane, clientX, clientY, panel, { round = true } = {}) {
    const [, panelX, panelY, panelWidth, panelHeight] = panel;
    const rect = elementRect(view.getCanvas(plane));
    const offset = panelVoxelOffset(
      (clientX - rect.left) * view.pixelRatio,
      (rect.bottom - clientY) * view.pixelRatio,
      panelX,
      panelY,
      panelWidth,
      panelHeight,
      state.zoom,
      baseViewWidth(state),
    );
    const basis = planeDisplayBasis(state, plane);
    const point_ZYX = [
      state.center_ZYX[0] + basis.right_XYZ[2] * offset.u + basis.up_XYZ[2] * offset.v,
      state.center_ZYX[1] + basis.right_XYZ[1] * offset.u + basis.up_XYZ[1] * offset.v,
      state.center_ZYX[2] + basis.right_XYZ[0] * offset.u + basis.up_XYZ[0] * offset.v,
    ];
    return round ? point_ZYX.map(Math.round) : point_ZYX;
  }

  function toolAt(plane, clientX, clientY, panel) {
    if (plane === "surface") return surfaceController.toolAt(clientX, clientY);
    const point_ZYX = pointFromClient(plane, clientX, clientY, panel, { round: false });
    const next_ZYX = pointFromClient(plane, clientX + 1, clientY, panel, { round: false });
    return {
      pane: plane,
      point_ZYX,
      normal_ZYX: [...planeDisplayBasis(state, plane).normal_XYZ].reverse(),
      pixel: Math.hypot(...next_ZYX.map((value, axis) => value - point_ZYX[axis])),
    };
  }

  function pointAtClient(pane, clientX, clientY, panel) {
    if (pane === "surface") return surfaceController.pointAtClient(clientX, clientY);
    if (pane === "3d") {
      if (document.elementFromPoint(clientX, clientY) !== view.getCanvas("3d")) return null;
      return view.pickThreePoint(clientX, clientY);
    }
    return pointFromClient(pane, clientX, clientY, panel, { round: false });
  }

  function focusAt(point_ZYX, options) {
    try {
      setPhysicalFocus(point_ZYX, options);
    } catch (error) {
      showPointerInspection(error.message);
    }
  }

  function focusAtPointer() {
    const pointer = hoverEvent ? { x: hoverEvent.clientX, y: hoverEvent.clientY } : state.lastPointerClient;
    const panel = pointer === null ? null : view.orthogonalPanelAt(pointer.x, pointer.y);
    if (
      panel === null ||
      !elements.viewerShell.matches(":hover") ||
      document.elementFromPoint(pointer.x, pointer.y) !== view.getCanvas(panel[0])
    ) {
      setStatus("Move the mouse over a view to focus there.");
      return;
    }
    const point_ZYX = pointAtClient(panel[0], pointer.x, pointer.y, panel);
    if (point_ZYX === null) {
      setStatus("Focus unavailable here: no displayed volume or surface point.");
      return;
    }
    if (hoverEvent !== null) {
      rememberPointer(hoverEvent);
      updateActivePane(hoverEvent, panel);
    }
    cancelPointerInteraction();
    focusAt(point_ZYX);
  }

  function updateActivePane(event, panel = view.orthogonalPanelAt(event.clientX, event.clientY)) {
    if (!state.volume || !panel) return;
    if (panel[0] === "3d" || panel[0] === "surface") setActiveViewport(panel[0], { redrawOnly: true });
    else setActivePlane(panel[0], { redrawOnly: true });
  }

  function rememberPointer(event) {
    const rect = elementRect(elements.viewerShell);
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    ) {
      updatePointerMarkers();
      return;
    }
    state.lastPointerClient = { x: event.clientX, y: event.clientY };
    updatePointerMarkers(event);
  }

  function updatePointerMarkers(event = null) {
    const panel = event === null || state.volume === null ? null : view.orthogonalPanelAt(event.clientX, event.clientY);
    let point_ZYX = null;
    if (panel?.[0] === "surface") point_ZYX = surfaceController.pointAtClient(event.clientX, event.clientY);
    else if (panel !== null && panel[0] !== "3d")
      point_ZYX = pointFromClient(panel[0], event.clientX, event.clientY, panel, { round: false });
    const radius = state.tool === null || state.panHeld ? null : state.tool.radius;
    for (const pane of view.panes)
      if (pane.mode === "plane") pane.showPointer(point_ZYX, pane.name === panel?.[0] ? radius : null);
    surfaceMarkers.showPointer(panel?.[0] === "surface" ? null : point_ZYX);
    const at =
      panel?.[0] === "surface" && radius !== null && state.tool.surface === true
        ? surfaceController.toolAt(event.clientX, event.clientY)
        : null;
    view.getSurfacePane().showReach(event?.clientX, event?.clientY, at === null ? null : radius / at.pixel);
    const label = state.tool === null || state.panHeld ? null : state.tool.label;
    const labelled =
      label !== null && panel !== null && panel[0] !== "3d" && (panel[0] !== "surface" || state.tool.surface === true);
    showToolLabel(labelled ? label : null, event);
  }

  function showToolLabel(label, event) {
    const element = elements.toolLabel;
    element.hidden = label === null;
    if (label === null) return;
    const color = `rgb(${label.color_RGB.map((value) => Math.round(255 * value)).join(", ")})`;
    if (element.textContent !== label.text) element.textContent = label.text;
    if (element.style.color !== color) element.style.color = color;
    element.style.transform = `translate(${event.clientX + LABEL_OFFSET}px, ${event.clientY + LABEL_OFFSET}px)`;
  }

  function onWheel(event) {
    if (!state.volume || event.target.closest(WHEEL_IGNORED)) return;
    event.preventDefault();
    rememberPointer(event);
    updateActivePane(event);
    const factor = event.deltaY < 0 ? WHEEL_ZOOM_FACTOR : 1 / WHEEL_ZOOM_FACTOR;
    if (state.activeViewport === "surface") {
      if (event.shiftKey) surfaceController.stepNormal(event.deltaY < 0 ? 1 : -1);
      else surfaceController.zoomAtClient(state.surfaceView.zoom * factor, event.clientX, event.clientY);
      return;
    }
    setZoomAtClientPoint(state.zoom * factor, event.clientX, event.clientY, { interactive: true });
    scheduleDeferredSettle();
  }

  function onPointerDown(event) {
    cancelHover();
    if (!state.volume || event.target.closest(PRESS_IGNORED)) return;
    if (!event.isPrimary || (event.button !== 0 && event.button !== 1)) return;
    rememberPointer(event);
    const panel = view.orthogonalPanelAt(event.clientX, event.clientY);
    if (!panel) return;
    if (state.panHeld) panPressed = true;
    const [plane, , , panelWidth, panelHeight] = panel;
    updateActivePane(event, panel);
    view.getCanvas(plane).focus({ preventScroll: true });
    const drag = {
      pointerId: event.pointerId,
      kind: plane === "surface" ? "surface" : plane === "3d" ? "rotate3d" : "plane",
      plane: plane === "3d" ? state.plane : plane,
      focusViewport: null,
      panelWidth: panelWidth / view.pixelRatio,
      panelHeight: panelHeight / view.pixelRatio,
      lastX: event.clientX,
      lastY: event.clientY,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
    };
    if (event.button === 1) {
      event.preventDefault();
      if (plane === "3d") return;
      if (plane === "surface" && (!state.surfaceView.volumetric.enabled || view.getSurfacePane().flatAt(event.clientX)))
        return;
      drag.kind = plane === "surface" ? "surface-camera" : "tilt-plane";
      drag.plane = plane;
      cancelPointerInteraction();
    } else if (event.shiftKey && !event.ctrlKey && !event.altKey && !event.metaKey) {
      event.preventDefault();
      drag.focusViewport = plane;
    } else if (
      state.tool !== null &&
      !state.panHeld &&
      (drag.kind === "plane" || (drag.kind === "surface" && state.tool.surface === true))
    ) {
      const at = toolAt(plane, event.clientX, event.clientY, panel);
      if (at !== null) {
        drag.kind = "tool";
        drag.tool = state.tool;
        drag.panel = panel;
        drag.at = at;
        drag.tool.down(at, event.ctrlKey || event.metaKey);
      }
    }
    state.pointerDrag = drag;
    capturePointer(event.pointerId);
  }

  function onPointerMove(event) {
    if (!state.volume) return;
    const drag = state.pointerDrag;
    if (!drag) {
      hoverEvent = event;
      cancelInspection();
      if (hoverFrame === null)
        hoverFrame = requestAnimationFrame(() => {
          const latest = hoverEvent;
          hoverFrame = hoverEvent = null;
          if (!state.volume || state.pointerDrag) return;
          rememberPointer(latest);
          const panel = view.orthogonalPanelAt(latest.clientX, latest.clientY);
          updateActivePane(latest, panel);
          if (state.heldSliceDirection !== 0 || state.heldZoomDirection !== 0 || state.heldRotation !== null) return;
          const plane = panel?.[0] ?? "3d";
          inspectPoint(
            plane,
            plane === "surface"
              ? surfaceController.pointAtClient(latest.clientX, latest.clientY)
              : plane === "3d"
                ? null
                : pointFromClient(plane, latest.clientX, latest.clientY, panel),
          );
        });
      return;
    }
    rememberPointer(event);
    if (drag.pointerId !== event.pointerId) return;
    if (drag.kind === "tool") {
      const samples = event.getCoalescedEvents();
      for (const sample of samples.length > 0 ? samples : [event]) {
        const at = toolAt(drag.plane, sample.clientX, sample.clientY, drag.panel);
        if (at === null) continue;
        drag.at = at;
        drag.tool.move(at);
      }
      return;
    }
    const dx = event.clientX - drag.lastX;
    const dy = event.clientY - drag.lastY;
    drag.lastX = event.clientX;
    drag.lastY = event.clientY;
    if (!drag.moved && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < CLICK_SLOP) return;
    drag.moved = true;
    if (drag.kind === "tilt-plane") tiltPlaneByPixels(drag.plane, dx, dy);
    else if (drag.kind === "surface-camera") surfaceController.rotateCameraByPixels(dx, dy);
    else if (drag.kind === "rotate3d") rotateCamera3dByPixels(dx, dy);
    else if (drag.kind === "surface")
      surfaceController.panByPixels(dx, dy, drag.panelWidth, drag.panelHeight, event.clientX);
    else panByPixels(drag.plane, dx, dy, drag.panelWidth, drag.panelHeight);
  }

  function onPointerEnd(event) {
    const drag = state.pointerDrag;
    if (!state.volume) {
      cancelPointerInteraction();
      return;
    }
    if (drag?.pointerId !== event.pointerId) return;
    state.pointerDrag = null;
    releasePointerCapture();
    if (drag.kind === "tool") {
      drag.tool.up(drag.at);
      return;
    }
    const click = event.type === "pointerup" && event.isPrimary && event.button === 0 && !drag.moved;
    if (drag.moved && (drag.kind === "rotate3d" || drag.kind === "tilt-plane")) {
      syncRotationReadout();
      publishSession();
    }
    if (drag.kind === "surface-camera") publishSession();
    if (drag.focusViewport !== null) {
      if (click && Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < CLICK_SLOP) {
        const panel = view.orthogonalPanelAt(event.clientX, event.clientY);
        const point_ZYX =
          panel?.[0] === drag.focusViewport
            ? pointAtClient(drag.focusViewport, event.clientX, event.clientY, panel)
            : null;
        if (point_ZYX !== null) focusAt(point_ZYX);
        else setStatus("Focus unavailable here: no displayed volume or surface point.");
      }
    } else if (click) {
      let point_ZYX = null;
      if (drag.kind === "surface") point_ZYX = surfaceController.pointAtClient(event.clientX, event.clientY);
      else if (drag.kind === "rotate3d") point_ZYX = view.pickThreePoint(event.clientX, event.clientY);
      else if (drag.kind === "plane") {
        const panel = view.orthogonalPanelAt(event.clientX, event.clientY);
        if (panel?.[0] === drag.plane)
          point_ZYX = pointFromClient(drag.plane, event.clientX, event.clientY, panel, { round: false });
      }
      if (point_ZYX !== null) focusAt(point_ZYX, { surface: drag.kind !== "surface" });
      else if (drag.kind === "surface")
        surfaceControls.setStatus("Focus unavailable here: geometry is missing or still being published.", true);
    }
    scheduleDeferredSettle();
  }

  function onPointerCaptureLost(event) {
    if (pointerCaptureId === event.pointerId) pointerCaptureId = null;
    const drag = state.pointerDrag;
    if (drag?.pointerId !== event.pointerId) return;
    state.pointerDrag = null;
    if (drag.kind === "tool") drag.tool.up(drag.at);
  }

  function initPointer() {
    elements.viewerShell.addEventListener("wheel", onWheel, { passive: false });
    elements.viewerShell.addEventListener("pointerdown", onPointerDown);
    elements.viewerShell.addEventListener("pointermove", onPointerMove);
    elements.viewerShell.addEventListener("pointerleave", () => {
      cancelHover();
      updatePointerMarkers();
    });
    elements.viewerShell.addEventListener("pointerup", onPointerEnd);
    elements.viewerShell.addEventListener("pointercancel", onPointerEnd);
    elements.viewerShell.addEventListener("lostpointercapture", onPointerCaptureLost);
  }

  return {
    initPointer,
    cancelPointerInteraction,
    focusAtPointer,
    setTool,
    holdPan,
    releasePan,
  };
}
