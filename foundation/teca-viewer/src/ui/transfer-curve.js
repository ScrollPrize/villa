import { evaluateSurfaceTransfer, SURFACE_TRANSFER_POINT_LIMIT, surfaceTransferPreset } from "../surface/transfer.js";

function numberInput(value, title) {
  const input = document.createElement("input");
  input.type = "number";
  input.min = "0";
  input.max = "1";
  input.step = "0.01";
  input.value = value;
  input.title = title;
  return input;
}

function rgbToHex(color) {
  return `#${color
    .map((channel) =>
      Math.round(Math.min(1, Math.max(0, channel)) * 255)
        .toString(16)
        .padStart(2, "0"),
    )
    .join("")}`;
}

export function createTransferCurve({ state, elements, surface }) {
  let shownKey = null;

  function addPoint() {
    const transfer = state.surfaceView.volumetric.transfer;
    let low = 0;
    for (let index = 1; index < transfer.points.length - 1; index++) {
      if (
        transfer.points[index + 1].value - transfer.points[index].value >
        transfer.points[low + 1].value - transfer.points[low].value
      )
        low = index;
    }
    const value = (transfer.points[low].value + transfer.points[low + 1].value) / 2;
    const evaluated = evaluateSurfaceTransfer(value, transfer);
    const points = [...transfer.points, { value, color: rgbToHex(evaluated.color), opacity: evaluated.opacity }].sort(
      (a, b) => a.value - b.value,
    );
    surface.setVolumetric({ transfer: { preset: "custom", points } });
  }

  function updatePoint(index, patch) {
    const points = state.surfaceView.volumetric.transfer.points.map((point, pointIndex) =>
      pointIndex === index ? { ...point, ...patch } : { ...point },
    );
    const candidate = { preset: "custom", points };
    shownKey = JSON.stringify(candidate);
    surface.setVolumetric({ transfer: candidate });
  }

  function sync(transfer) {
    elements.surfaceVolumetricTransferPreset.value = transfer.preset;
    const key = JSON.stringify(transfer);
    if (key === shownKey) return;
    shownKey = key;
    elements.surfaceVolumetricTransferPoints.replaceChildren();
    for (const [index, point] of transfer.points.entries()) {
      const endpoint = index === 0 || index === transfer.points.length - 1;
      const row = document.createElement("div");
      row.className = "surface-transfer-point";
      const value = numberInput(point.value, "Intensity");
      value.disabled = endpoint;
      value.addEventListener("change", () => updatePoint(index, { value: Number(value.value) }));
      const color = document.createElement("input");
      color.type = "color";
      color.value = point.color;
      color.title = "Color";
      color.addEventListener("input", () => updatePoint(index, { color: color.value }));
      const opacity = numberInput(point.opacity, "Opacity");
      opacity.type = "range";
      const opacityNumber = numberInput(point.opacity, "Opacity");
      opacity.addEventListener("input", () => {
        opacityNumber.value = opacity.value;
        updatePoint(index, { opacity: Number(opacity.value) });
      });
      opacityNumber.addEventListener("change", () => {
        opacity.value = opacityNumber.value;
        updatePoint(index, { opacity: Number(opacityNumber.value) });
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.title = "Remove curve point";
      remove.disabled = endpoint;
      remove.addEventListener("click", () => {
        const { points } = state.surfaceView.volumetric.transfer;
        surface.setVolumetric({
          transfer: { preset: "custom", points: points.filter((_, pointIndex) => pointIndex !== index) },
        });
      });
      row.append(value, color, opacity, opacityNumber, remove);
      elements.surfaceVolumetricTransferPoints.append(row);
    }
    elements.surfaceVolumetricTransferAdd.disabled = transfer.points.length >= SURFACE_TRANSFER_POINT_LIMIT;
  }

  elements.surfaceVolumetricTransferPreset.addEventListener("change", () => {
    const preset = elements.surfaceVolumetricTransferPreset.value;
    if (preset === "custom") return;
    surface.setVolumetric({ transfer: { preset, points: surfaceTransferPreset(preset) } });
  });
  elements.surfaceVolumetricTransferAdd.addEventListener("click", addPoint);

  return { sync };
}
