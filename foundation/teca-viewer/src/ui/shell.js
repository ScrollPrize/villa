import { shownView, viewerDock } from "../views/pose.js";
import { createShellModals } from "./shell-modals.js";
import { createShellPanels } from "./shell-panels.js";

export function createShellControls({
  window,
  document,
  storage,
  state,
  elements,
  wirePopover,
  publishSession,
  setActiveViewport,
  setPlane,
  toggleActiveFullscreen,
  openSurface,
  closeSurface,
}) {
  let panels = null;
  let modals = null;

  function showView(view) {
    if (state.volume === null || shownView(state) === view) return;
    if (view === "grid") {
      if (viewerDock(state) !== null) closeSurface();
      else {
        setActiveViewport(state.fullscreenViewport);
        toggleActiveFullscreen();
      }
      return;
    }
    if (view === "surface" && viewerDock(state) === null) {
      openSurface();
      return;
    }
    if (view === "3d" || view === "surface") setActiveViewport(view);
    else setPlane(view);
    if (state.activeViewport === view && shownView(state) !== view) toggleActiveFullscreen();
  }

  function initShell({ onViewerLayoutChange, onSettingsOpen }) {
    panels = createShellPanels({
      window,
      document,
      state,
      elements,
      storage,
      publishSession,
      onLayoutChange: onViewerLayoutChange,
    });
    window.addEventListener("resize", onViewerLayoutChange);
    let toolbarHeight;
    new ResizeObserver(([entry]) => {
      const height = entry.borderBoxSize[0].blockSize;
      if (height === toolbarHeight) return;
      toolbarHeight = height;
      onViewerLayoutChange();
    }).observe(elements.topBar);
    for (const button of document.querySelectorAll(".pane-maximize")) {
      button.addEventListener("click", () => {
        const plane = button.dataset.plane;
        if (state.volume === null) return;
        if (plane === "3d") setActiveViewport(plane);
        else setPlane(plane);
        toggleActiveFullscreen();
      });
    }
    for (const button of elements.viewStrip.querySelectorAll("[data-view]"))
      button.addEventListener("click", () => showView(button.dataset.view));
    modals = createShellModals({ document, elements, onSettingsOpen });
    const { close } = wirePopover(elements.sourceSwitcher, elements.sourcePopover);
    elements.sourcePopover.addEventListener("click", (event) => {
      if (event.target.closest("button") !== null) close();
    });
    elements.surfaceDisplaySettings.addEventListener("click", () => {
      document.querySelector('[data-ltab="view"]').click();
      elements.surfaceControls.scrollIntoView({ block: "start" });
    });
    updateSourceName();
  }

  function updateSourceName() {
    elements.viewerEmpty.hidden = state.volume !== null;
    elements.app.classList.toggle("has-source", state.volume !== null);
    const active = elements.sourceItems.querySelector("button.active");
    elements.sourceSwitcherName.textContent = active === null ? "No source" : active.textContent;
    elements.sourceSwitcher.title = active?.title ?? "";
    if (active !== null) modals.closeModal(elements.openSourceModal);
  }

  return {
    initShell,
    updateSourceName,
    modalIsOpen: () => modals.modalIsOpen(),
    toggleLeftPanel: () => panels.toggleLeftPanel(),
    toggleRightPanel: () => panels.toggleRightPanel(),
  };
}
