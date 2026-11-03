import { MAX_ZOOM, MIN_ZOOM } from "../config.js";
import { saveFile } from "../storage.js";
import { initChunkBudgetControls, initStoredGeometryControls } from "./cache-settings.js";
import { bindRangeAndNumber } from "./dom.js";
import { initGpuBudgetControls } from "./gpu-settings.js";
import { wireLayerRowMenu } from "./layer-row.js";

export function bindViewerControls({
  state,
  elements,
  wirePopover,
  setStatus,
  resetRotation,
  resetAllRotations,
  setRotationMode,
  syncRotationReadout,
  setPositionAxis,
  setPositionXYZ,
  setZoom,
  syncZoomInputs,
  setDisplayControlValue,
  syncDisplayInputs,
  sources,
  surfaceController,
  openSource,
  publishSession,
  sessionAddress,
  sessionFile,
  openSessionFile,
  kept,
  gpuBudgetBytes,
  applyGpuBudget,
  chunkBudgetBytes,
  applyChunkBudget,
  storage,
  geometryByteCache,
}) {
  elements.form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await openSource(elements.url.value);
  });
  elements.copySession.addEventListener("click", async () => {
    const refused = publishSession();
    try {
      await navigator.clipboard.writeText(sessionAddress());
      setStatus(refused === null ? "Session link copied." : `Session link copied. ${refused}`);
    } catch (error) {
      setStatus(error.message);
    }
  });
  elements.saveSession.addEventListener("click", async () => {
    const refused = publishSession();
    if (refused !== null) {
      setStatus(`Session not saved: the view is past what a session holds. ${refused}`);
      return;
    }
    try {
      const parts = await kept();
      const { name, text } = sessionFile(parts);
      saveFile(new Blob([text], { type: "application/json" }), name);
      const names = Object.keys(parts);
      setStatus(`Session saved as ${name}${names.length === 0 ? "" : `, with ${names.join(", ")}`}.`);
    } catch (error) {
      setStatus(`Session not saved: ${error.message}`);
    }
  });
  elements.openSession.addEventListener("click", () => elements.sessionFileInput.click());
  elements.sessionFileInput.addEventListener("change", async () => {
    const [chosen] = elements.sessionFileInput.files;
    // So the same file can be chosen again.
    elements.sessionFileInput.value = "";
    if (chosen === undefined) return;
    try {
      openSessionFile(await chosen.text());
    } catch (error) {
      setStatus(`Session file ${chosen.name} not opened: ${error.message}`);
    }
  });

  elements.channelSelect.addEventListener("change", () => {
    sources.selectChannel(Number(elements.channelSelect.value));
    surfaceController.refreshPresentation();
  });
  const levelChoice = (select) => (select.value === "auto" ? null : Number(select.value));
  elements.scaleSelect.addEventListener("change", () => {
    sources.selectScaleBounds({
      finest: levelChoice(elements.scaleSelect),
      coarsest: state.manualCoarsestScaleIndex,
    });
    surfaceController.refreshPresentation();
  });
  elements.coarsestScaleSelect.addEventListener("change", () => {
    sources.selectScaleBounds({
      finest: state.manualScaleIndex,
      coarsest: levelChoice(elements.coarsestScaleSelect),
    });
    surfaceController.refreshPresentation();
  });

  elements.rotationReset.addEventListener("click", () => {
    if (state.volume !== null) resetRotation();
  });
  elements.rotationResetAll.addEventListener("click", () => {
    if (state.volume !== null) resetAllRotations();
  });
  elements.rotationMode.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-rotation-mode]");
    if (button !== null) setRotationMode(button.dataset.rotationMode);
  });
  syncRotationReadout();

  for (const [rangeInput, axis] of [
    [elements.posZ, 0],
    [elements.posY, 1],
    [elements.posX, 2],
  ]) {
    const applyPosition = () => {
      if (state.volume !== null) setPositionAxis(axis, Number(rangeInput.value));
    };
    rangeInput.addEventListener("input", applyPosition);
    rangeInput.addEventListener("change", applyPosition);
  }
  elements.positionNumber.addEventListener("change", () => {
    const position_XYZ = elements.positionNumber.value
      .replace(/^\s*\[/, "")
      .replace(/\]\s*$/, "")
      .split(",")
      .map((value) => Number(value.trim()));
    const valid =
      position_XYZ.length === 3 &&
      position_XYZ.every((value) => Number.isFinite(value)) &&
      state.volume !== null &&
      [position_XYZ[2], position_XYZ[1], position_XYZ[0]].every(
        (value, axis) => value >= 0 && value < state.volume.fullShape_ZYX[axis],
      );
    if (!valid) {
      elements.positionNumber.setCustomValidity("Enter three in-bounds XYZ coordinates separated by commas.");
      elements.positionNumber.reportValidity();
      return;
    }
    elements.positionNumber.setCustomValidity("");
    setPositionXYZ(position_XYZ);
  });
  elements.positionNumber.addEventListener("keydown", (event) => {
    if (event.key === "Enter") event.currentTarget.blur();
  });
  for (const input of [elements.zoom, elements.zoomNumber]) Object.assign(input, { min: MIN_ZOOM, max: MAX_ZOOM });
  bindRangeAndNumber(elements.zoom, elements.zoomNumber, (value) => {
    if (state.volume !== null) setZoom(value);
  });
  wireLayerRowMenu(wirePopover, elements.volumeDisplayMenu, elements.volumeDisplayPopover);
  for (const [range, number, name] of [
    [elements.displayBlack, elements.displayBlackNumber, "black"],
    [elements.displayWhite, elements.displayWhiteNumber, "white"],
    [elements.displayGamma, elements.displayGammaNumber, "gamma"],
  ]) {
    bindRangeAndNumber(range, number, (value) => setDisplayControlValue(name, value));
  }

  initGpuBudgetControls({
    input: elements.gpuBudgetMiB,
    apply: elements.gpuBudgetApply,
    status: elements.gpuBudgetStatus,
    bytes: gpuBudgetBytes,
    applyBudget: applyGpuBudget,
    storage,
  });
  initChunkBudgetControls({
    input: elements.chunkBudgetMiB,
    apply: elements.chunkBudgetApply,
    status: elements.chunkBudgetStatus,
    bytes: chunkBudgetBytes,
    applyBudget: applyChunkBudget,
    storage,
  });
  initStoredGeometryControls(
    elements.geometryCacheClear,
    elements.geometryCacheStatus,
    elements.openSettings,
    geometryByteCache,
  );
  syncZoomInputs();
  syncDisplayInputs();
}
