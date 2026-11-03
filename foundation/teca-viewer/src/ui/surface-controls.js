import { bindRangeAndNumber } from "./dom.js";
import { showPieces } from "./readouts.js";
import { createReliefEditor } from "./relief-editor.js";
import { createTransferCurve } from "./transfer-curve.js";

const NORMAL_LOCK_STORAGE_KEY = "teca.viewer.surface-normal-locked";

export function surfaceControlsShown(state) {
  return state.surfaceView.path !== null && state.fullscreenViewport === "surface";
}

export class SurfaceControls {
  constructor({ state, view, elements, storage, setStatus }) {
    this.state = state;
    this.view = view;
    this.elements = elements;
    this.storage = storage;
    this.reportStatus = setStatus;
    this.surface = null;
    this.transferCurve = null;
    this.reliefEditor = null;
    this.offsetReset = null;
    this.controlSyncFrame = null;
    this.error = false;
    this.restoreStatus = null;
    this.normalLockStorageStatus = null;
    this.navigationControls = document.createElement("section");
    this.navigationControls.className = "card surface-navigation-controls";
    this.navigationControls.hidden = true;
    this.navigationControls.innerHTML = `
      <h2>Surface navigation</h2>
      <div class="button-row">
        <button type="button" data-surface-reset="framing" title="Reset Surface pan and zoom (Home)">Reset framing</button>
        <button type="button" data-surface-reset="offset" title="Reset normal offset to zero (0 or Shift+Home)">Reset depth <kbd>0</kbd></button>
      </div>
      <div class="surface-camera-controls" hidden>
        <h3>Slab camera</h3>
        <label>Azimuth ° <input aria-label="Slab camera azimuth" data-camera="azimuth" type="number" min="-180" max="180" step="1"></label>
        <label>Tilt ° <input aria-label="Slab camera tilt" data-camera="tilt" type="number" min="0" max="85" step="1"></label>
        <label>Perspective <input aria-label="Slab camera perspective" data-camera="perspective" type="range" min="0" max="1" step="0.01">
          <input aria-label="Slab camera perspective" data-camera="perspective" type="number" min="0" max="1" step="0.01"></label>
        <button type="button" data-surface-reset="camera" title="Reset slab camera (Alt+Home)">Reset camera</button>
      </div>`;
    this.cameraControls = this.navigationControls.querySelector(".surface-camera-controls");
    this.navigationStatus = document.createElement("div");
    this.navigationStatus.className = "surface-navigation-status";
    this.navigationStatus.setAttribute("role", "status");
    this.navigationStatus.hidden = true;
    this.navigationControls.append(this.navigationStatus);
    this.offsetReadout = document.createElement("div");
    this.offsetReadout.className = "surface-offset-readout";
    this.offsetReadout.hidden = true;
  }

  bind(surface) {
    this.surface = surface;
    const elements = this.elements;
    const pane = this.view.getSurfacePane();
    elements.viewPanel.prepend(this.navigationControls);
    elements.cursorValue.before(this.offsetReadout);
    pane.showWith(this.navigationControls, this.offsetReadout);
    pane.element.querySelector(".pane-return").addEventListener("click", () => surface.close());
    this.navigationControls
      .querySelector('[data-surface-reset="framing"]')
      .addEventListener("click", () => surface.resetFraming());
    this.offsetReset = this.navigationControls.querySelector('[data-surface-reset="offset"]');
    this.offsetReset.addEventListener("click", () => surface.setNormalOffset(0));
    this.navigationControls
      .querySelector('[data-surface-reset="camera"]')
      .addEventListener("click", () => surface.resetCamera());
    for (const input of this.cameraControls.querySelectorAll("[data-camera]")) {
      input.addEventListener("input", () => {
        const value = input.valueAsNumber;
        if (value >= Number(input.min) && value <= Number(input.max))
          surface.setVolumetric({ [input.dataset.camera]: value });
      });
      input.addEventListener("blur", () => this.scheduleControlSync());
    }
    this.view.surfaceButton.addEventListener("click", () => {
      void surface.openMode("surface").catch((error) => surface.handleOpenError(error));
    });
    elements.workspaceView.addEventListener("change", () => {
      const mode = elements.workspaceView.value;
      if (surface.active()) surface.setVolumetric({ enabled: mode !== "surface", compareFlat: mode === "both" });
      else void surface.openMode(mode).catch((error) => surface.handleOpenError(error));
    });
    for (const [input, key] of [
      [elements.surfaceFlipU, "flipU"],
      [elements.surfaceFlipV, "flipV"],
    ]) {
      input.addEventListener("change", () => surface.setVolumetric({ [key]: input.checked }));
    }
    elements.surfaceNormalLock.addEventListener("click", () => this.toggleNormalLock());
    this.normalLockStorageStatus = document.createElement("p");
    this.normalLockStorageStatus.className = "readout-line error";
    this.normalLockStorageStatus.setAttribute("role", "status");
    this.normalLockStorageStatus.hidden = true;
    elements.surfaceOffset.closest("label").after(this.normalLockStorageStatus);
    this.restoreNormalLock();
    this.transferCurve = createTransferCurve({ state: this.state, elements, surface });
    this.reliefEditor = createReliefEditor({
      state: this.state,
      elements,
      pane,
      surface,
      setStatus: (message, error) => this.setStatus(message, error),
    });
    bindRangeAndNumber(elements.surfaceOffset, elements.surfaceOffsetNumber, (value) => surface.setNormalOffset(value));
    bindRangeAndNumber(elements.surfaceStep, elements.surfaceStepNumber, (value) => surface.setNormalStep(value));
    elements.surfaceVolumetricEnabled.addEventListener("change", () =>
      surface.setVolumetric({ enabled: elements.surfaceVolumetricEnabled.checked }),
    );
    for (const [range, number, key, round] of [
      [elements.surfaceVolumetricLayersFront, elements.surfaceVolumetricLayersFrontNumber, "layersFront", true],
      [elements.surfaceVolumetricLayersBehind, elements.surfaceVolumetricLayersBehindNumber, "layersBehind", true],
      [elements.surfaceVolumetricSpacing, elements.surfaceVolumetricSpacingNumber, "spacing", false],
      [
        elements.surfaceVolumetricOpacityMinimum,
        elements.surfaceVolumetricOpacityMinimumNumber,
        "opacityMinimum",
        false,
      ],
      [
        elements.surfaceVolumetricOpacityMaximum,
        elements.surfaceVolumetricOpacityMaximumNumber,
        "opacityMaximum",
        false,
      ],
      [elements.surfaceVolumetricOpacity, elements.surfaceVolumetricOpacityNumber, "opacity", false],
      [elements.surfaceVolumetricGamma, elements.surfaceVolumetricGammaNumber, "gamma", false],
      [elements.surfaceVolumetricAzimuth, elements.surfaceVolumetricAzimuthNumber, "azimuth", false],
      [elements.surfaceVolumetricTilt, elements.surfaceVolumetricTiltNumber, "tilt", false],
      [elements.surfaceVolumetricPerspective, elements.surfaceVolumetricPerspectiveNumber, "perspective", false],
      [elements.surfaceVolumetricRelief, elements.surfaceVolumetricReliefNumber, "relief", false],
      [elements.surfaceVolumetricLighting, elements.surfaceVolumetricLightingNumber, "lighting", false],
      [elements.surfaceVolumetricLightAzimuth, elements.surfaceVolumetricLightAzimuthNumber, "lightAzimuth", false],
    ]) {
      bindRangeAndNumber(range, number, (value) => surface.setVolumetric({ [key]: round ? Math.round(value) : value }));
    }
    elements.surfaceVolumetricFeature.addEventListener("change", () =>
      surface.setVolumetric({ feature: elements.surfaceVolumetricFeature.value }),
    );
    elements.surfaceVolumetricGradientGain.addEventListener("change", () =>
      surface.setVolumetric({ gradientGain: Number(elements.surfaceVolumetricGradientGain.value) }),
    );
    elements.surfaceVolumetricConstantEmission.addEventListener("change", () =>
      surface.setVolumetric({ constantEmission: elements.surfaceVolumetricConstantEmission.checked }),
    );
    elements.surfaceVolumetricReverse.addEventListener("change", () =>
      surface.setVolumetric({ reverse: elements.surfaceVolumetricReverse.checked }),
    );
    elements.surfaceFlipNormal.addEventListener("click", () => surface.flipNormal());
    this.syncControls();
  }

  toggleNormalLock() {
    this.surface.setNormalLocked(!this.surface.normalLocked);
    this.saveNormalLock(this.surface.normalLocked);
  }

  restoreNormalLock() {
    try {
      const stored = this.storage.getItem(NORMAL_LOCK_STORAGE_KEY);
      if (stored !== null) {
        if (stored !== "true" && stored !== "false")
          throw new Error(`Saved normal lock is ${stored}, not true or false.`);
        this.surface.setNormalLocked(stored === "true");
      }
      this.normalLockStorageStatus.hidden = true;
    } catch (error) {
      this.normalLockStorageStatus.textContent = `Could not restore normal lock; the current setting is retained. ${error.message}`;
      this.normalLockStorageStatus.hidden = false;
    }
  }

  saveNormalLock(locked) {
    try {
      this.storage.setItem(NORMAL_LOCK_STORAGE_KEY, String(locked));
      this.normalLockStorageStatus.hidden = true;
    } catch (error) {
      this.normalLockStorageStatus.textContent = `Normal lock applies on this page but could not be saved. ${error.message}`;
      this.normalLockStorageStatus.hidden = false;
    }
  }

  syncControls() {
    if (this.controlSyncFrame !== null) window.cancelAnimationFrame(this.controlSyncFrame);
    this.controlSyncFrame = null;
    const elements = this.elements;
    const surfaceView = this.state.surfaceView;
    const volumetric = surfaceView.volumetric;
    elements.surfaceControls.hidden = !surfaceControlsShown(this.state);
    elements.surfaceDisplaySettings.hidden = elements.surfaceControls.hidden;
    elements.surfaceNormalLock.hidden = elements.surfaceControls.hidden;
    elements.surfaceLayoutMode.value = surfaceView.layout.mode;
    elements.surfacePlaneChoices.hidden = surfaceView.layout.mode === "solo";
    for (const input of elements.surfacePlaneInputs) {
      input.checked = surfaceView.layout.planes.includes(input.dataset.surfacePlane);
    }
    elements.surfacePath.textContent = surfaceView.path?.split("/").filter(Boolean).at(-1) ?? "";
    elements.surfacePath.title = surfaceView.path ?? "";
    this.syncNormalOffset();
    elements.surfaceStep.value = surfaceView.normalStep;
    elements.surfaceStepNumber.value = surfaceView.normalStep;
    this.cameraControls.hidden = !volumetric.enabled;
    for (const input of this.cameraControls.querySelectorAll("[data-camera]")) {
      if (document.activeElement !== input) input.value = volumetric[input.dataset.camera];
    }
    elements.workspaceView.value = !volumetric.enabled ? "surface" : volumetric.compareFlat ? "both" : "volume";
    elements.surfaceVolumetricEnabled.checked = volumetric.enabled;
    elements.surfaceVolumetricControls.hidden = !volumetric.enabled;
    for (const [range, number, key] of [
      [elements.surfaceVolumetricLayersFront, elements.surfaceVolumetricLayersFrontNumber, "layersFront"],
      [elements.surfaceVolumetricLayersBehind, elements.surfaceVolumetricLayersBehindNumber, "layersBehind"],
      [elements.surfaceVolumetricSpacing, elements.surfaceVolumetricSpacingNumber, "spacing"],
      [elements.surfaceVolumetricOpacityMinimum, elements.surfaceVolumetricOpacityMinimumNumber, "opacityMinimum"],
      [elements.surfaceVolumetricOpacityMaximum, elements.surfaceVolumetricOpacityMaximumNumber, "opacityMaximum"],
      [elements.surfaceVolumetricOpacity, elements.surfaceVolumetricOpacityNumber, "opacity"],
      [elements.surfaceVolumetricGamma, elements.surfaceVolumetricGammaNumber, "gamma"],
      [elements.surfaceVolumetricAzimuth, elements.surfaceVolumetricAzimuthNumber, "azimuth"],
      [elements.surfaceVolumetricTilt, elements.surfaceVolumetricTiltNumber, "tilt"],
      [elements.surfaceVolumetricPerspective, elements.surfaceVolumetricPerspectiveNumber, "perspective"],
      [elements.surfaceVolumetricRelief, elements.surfaceVolumetricReliefNumber, "relief"],
      [elements.surfaceVolumetricLighting, elements.surfaceVolumetricLightingNumber, "lighting"],
      [elements.surfaceVolumetricLightAzimuth, elements.surfaceVolumetricLightAzimuthNumber, "lightAzimuth"],
    ]) {
      range.value = volumetric[key];
      number.value = volumetric[key];
    }
    this.transferCurve.sync(volumetric.transfer);
    elements.surfaceVolumetricFeature.value = volumetric.feature;
    elements.surfaceVolumetricGradientGain.value = volumetric.gradientGain;
    elements.surfaceVolumetricConstantEmission.checked = volumetric.constantEmission;
    elements.surfaceVolumetricReverse.checked = volumetric.reverse;
    this.reliefEditor.sync();
    elements.surfaceFlipU.checked = volumetric.flipU;
    elements.surfaceFlipV.checked = volumetric.flipV;
    elements.surfaceFlipNormal.setAttribute("aria-pressed", String(surfaceView.normalSign !== 1));
  }

  syncNormalOffset() {
    const locked = this.surface.normalLocked;
    this.elements.surfaceNormalLock.setAttribute("aria-pressed", String(locked));
    this.elements.surfaceOffset.disabled = locked;
    this.elements.surfaceOffsetNumber.disabled = locked;
    this.elements.surfaceFlipNormal.disabled = locked;
    this.offsetReset.disabled = locked;
    this.elements.surfaceOffset.value = this.state.surfaceView.normalOffset;
    this.elements.surfaceOffsetNumber.value = this.state.surfaceView.normalOffset;
    const offset = `normal ${Number(this.state.surfaceView.normalOffset.toFixed(3))}`;
    showPieces(this.offsetReadout, locked ? [offset, "locked"] : [offset]);
  }

  syncOverlay() {
    this.reliefEditor.syncOverlay();
  }

  scheduleControlSync() {
    if (this.controlSyncFrame !== null) return;
    this.controlSyncFrame = window.requestAnimationFrame(() => {
      this.controlSyncFrame = null;
      this.syncControls();
    });
  }

  setStatus(message, error = false) {
    this.error = error;
    this.elements.surfaceStatus.textContent = message;
    this.elements.surfaceStatus.classList.toggle("error", error);
    const status = this.navigationStatus;
    const pending = message.startsWith("Opening ");
    status.textContent = error || pending ? message : "";
    status.hidden = !(error || pending);
    if (error) this.reportStatus(`Surface: ${message}`);
  }

  clearError() {
    if (this.error) this.setStatus("Surface view ready.");
  }

  setRestoreStatus(message, retry = null) {
    if (this.restoreStatus === null) {
      const element = document.createElement("div");
      element.className = "surface-restore-status";
      element.setAttribute("role", "status");
      const description = document.createElement("p");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "Retry surface";
      element.append(description, button);
      this.view.getSurfacePane().element.append(element);
      this.restoreStatus = { element, description, button };
    }
    this.restoreStatus.description.textContent = message;
    this.restoreStatus.button.onclick = retry;
    this.restoreStatus.button.hidden = retry === null;
    this.restoreStatus.element.hidden = false;
    this.setStatus(message, retry !== null);
  }

  clearRestoreStatus() {
    if (this.restoreStatus === null) return;
    this.restoreStatus.element.hidden = true;
    this.restoreStatus.button.onclick = null;
  }

  promptForSurface() {
    this.elements.dataTab.click();
    this.elements.overlayPath.focus();
    this.elements.overlayPath.scrollIntoView({ block: "nearest" });
    this.setStatus("Add a tifxyz surface URL in Data.");
  }
}
