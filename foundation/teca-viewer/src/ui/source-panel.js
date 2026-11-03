import { setSelectValue } from "./dom.js";

function sourceDisplayName(source) {
  return source.split(/[?#]/)[0].split(/[\\/]/).filter(Boolean).at(-1);
}

function displayEncodingText(encoding) {
  if (encoding.kind === "identity_uint8") return "uint8 identity";
  if (encoding.kind === "linear_window") {
    return `linear ${encoding.minimum}…${encoding.maximum} → 0…255`;
  }
  return encoding.kind;
}

export class SourcePanel {
  constructor({ elements, state, syncPositionInputs, updateViewerShellState, setStatus, activate }) {
    this.elements = elements;
    this.state = state;
    this.syncPositionInputs = syncPositionInputs;
    this.updateViewerShellState = updateViewerShellState;
    this.setStatus = setStatus;
    this.activate = activate;
  }

  populateScaleSelect() {
    const state = this.state;
    const autoOption = document.createElement("option");
    autoOption.value = "auto";
    autoOption.textContent = "Auto";
    autoOption.selected = state.manualScaleIndex === null;
    const coarsestAutoOption = document.createElement("option");
    coarsestAutoOption.value = "auto";
    coarsestAutoOption.textContent = "Auto";
    coarsestAutoOption.selected = state.manualCoarsestScaleIndex === null;
    this.elements.scaleSelect.replaceChildren(
      autoOption,
      ...state.volume.scales.map((scale, index) => {
        const option = document.createElement("option");
        option.value = index;
        option.textContent = `${scale.path} or coarser: ${scale.downsample_ZYX.join(" × ")}`;
        option.selected = state.manualScaleIndex === index;
        option.disabled = state.manualCoarsestScaleIndex !== null && index > state.manualCoarsestScaleIndex;
        return option;
      }),
    );
    this.elements.coarsestScaleSelect.replaceChildren(
      coarsestAutoOption,
      ...state.volume.scales.map((scale, index) => {
        const option = document.createElement("option");
        option.value = index;
        option.textContent = `${scale.path} or finer: ${scale.downsample_ZYX.join(" × ")}`;
        option.selected = state.manualCoarsestScaleIndex === index;
        option.disabled = state.manualScaleIndex !== null && index < state.manualScaleIndex;
        return option;
      }),
    );
    this.elements.scaleSelect.disabled = false;
    this.elements.coarsestScaleSelect.disabled = false;
  }

  showDrawnScale(scaleIndex) {
    const state = this.state;
    const scale = state.volume.scales[scaleIndex];
    const auto = this.elements.scaleSelect.querySelector('option[value="auto"]');
    const text = `Auto (${scale.path}: ${scale.downsample_ZYX.join(" x ")})`;
    if (auto.textContent !== text) auto.textContent = text;
    setSelectValue(this.elements.scaleSelect, state.manualScaleIndex ?? "auto");
    setSelectValue(this.elements.coarsestScaleSelect, state.manualCoarsestScaleIndex ?? "auto");
  }

  populateChannelSelect() {
    const state = this.state;
    const metadata = state.volume.metadata;
    this.elements.channelSelect.replaceChildren(
      ...Array.from({ length: metadata.channelCount }, (_, index) => {
        const option = document.createElement("option");
        option.value = index;
        option.textContent = metadata.channelNames === null ? `Channel ${index}` : metadata.channelNames[index];
        option.selected = index === state.selectedChannel;
        return option;
      }),
    );
    this.elements.channelSelect.disabled = metadata.channelCount === 1;
  }

  configurePositionInputs() {
    const state = this.state;
    for (const [rangeInput, axis] of [
      [this.elements.posZ, 0],
      [this.elements.posY, 1],
      [this.elements.posX, 2],
    ]) {
      rangeInput.max = state.volume.fullShape_ZYX[axis] - 1;
      rangeInput.disabled = false;
    }
    this.elements.positionNumber.disabled = false;
    this.elements.zoom.disabled = false;
    this.elements.zoomNumber.disabled = false;
    this.syncPositionInputs();
  }

  configureSpatialCapabilities() {
    const state = this.state;
    const hasDepth = state.volume.fullShape_ZYX[0] > 1;
    this.elements.rotationReset.disabled = !hasDepth;
    this.elements.rotationReset.title = hasDepth ? "" : "Unavailable for a two-dimensional source";
    this.elements.rotationResetAll.disabled = !hasDepth;
    this.elements.rotationResetAll.title = this.elements.rotationReset.title;
    for (const button of this.elements.rotationMode.querySelectorAll("button")) {
      button.disabled = !hasDepth;
    }
    for (const pane of document.querySelectorAll(".viewer-pane")) {
      const unavailable = !hasDepth && pane.dataset.plane !== "xy";
      pane.classList.toggle("unavailable", unavailable);
      pane.setAttribute("aria-disabled", String(unavailable));
    }
  }

  renderSourceMetadata() {
    const state = this.state;
    const metadata = state.volume.metadata;
    const physical =
      metadata.physicalScale_ZYX.every((value) => value !== null) &&
      metadata.physicalUnits_ZYX.every((value) => value !== null)
        ? metadata.physicalScale_ZYX.map((value, axis) => `${value} ${metadata.physicalUnits_ZYX[axis]}`).join(" × ")
        : "unknown";
    const values = [
      ["Axes", metadata.axes.join(", ")],
      ["Shape ZYX", state.volume.fullShape_ZYX.join(" × ")],
      ["Wire chunks", state.volume.scales[0].chunks_ZYX.join(" × ")],
      ["Source dtype", metadata.sourceDtype],
      ["Channels", String(metadata.channelCount)],
      ["Time points", String(metadata.timeCount)],
      ["Levels", String(state.volume.scales.length)],
      ["Physical scale", physical],
      ["Translation ZYX", metadata.physicalTranslation_ZYX.map((value) => value ?? "unknown").join(" × ")],
      ["Display", displayEncodingText(metadata.displayEncoding)],
    ];
    this.elements.sourceMetadata.replaceChildren(
      ...values
        .filter(([, value]) => !/unknown/.test(value))
        .flatMap(([name, value]) => {
          const term = document.createElement("dt");
          term.textContent = name;
          const description = document.createElement("dd");
          description.textContent = value;
          return [term, description];
        }),
    );
  }

  renderSourceItems() {
    const state = this.state;
    this.elements.sourceItems.replaceChildren(
      ...state.sources.map((source) => {
        const open = document.createElement("button");
        open.type = "button";
        open.className = source === state.activeSource ? "active" : "";
        open.textContent = sourceDisplayName(source);
        open.title = source;
        open.addEventListener("click", () => this.activate(source));
        const row = document.createElement("div");
        row.className = "source-item";
        row.append(open);
        return row;
      }),
    );
  }

  configureNoSourceState(message) {
    const elements = this.elements;
    elements.sourceMetadata.replaceChildren();
    const term = document.createElement("dt");
    term.textContent = "State";
    const description = document.createElement("dd");
    description.textContent = "No source";
    elements.sourceMetadata.append(term, description);
    elements.channelSelect.replaceChildren();
    elements.channelSelect.disabled = true;
    elements.scaleSelect.replaceChildren();
    elements.scaleSelect.disabled = true;
    elements.coarsestScaleSelect.replaceChildren();
    elements.coarsestScaleSelect.disabled = true;
    elements.rotationReset.disabled = true;
    elements.rotationReset.title = "";
    elements.rotationResetAll.disabled = true;
    elements.rotationResetAll.title = "";
    for (const button of elements.rotationMode.querySelectorAll("button")) button.disabled = true;
    for (const pane of document.querySelectorAll(".viewer-pane")) {
      pane.classList.remove("unavailable");
      pane.setAttribute("aria-disabled", "true");
      const scaleBar = pane.querySelector(".scale-bar");
      scaleBar.hidden = true;
      delete scaleBar.dataset.label;
      scaleBar.style.removeProperty("--scale-bar-width");
    }
    for (const control of [
      elements.posZ,
      elements.posY,
      elements.posX,
      elements.positionNumber,
      elements.zoom,
      elements.zoomNumber,
    ])
      control.disabled = true;
    this.updateViewerShellState();
    this.setStatus(message);
  }
}
