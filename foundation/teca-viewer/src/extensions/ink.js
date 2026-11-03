import { CatalogChoice } from "./ink/catalog.js";
import { Detail } from "./ink/detail.js";
import { VolumeInk } from "./ink/volume.js";

const DEFAULT_COLOR = "#ff3b30";
const DEFAULT_OPACITY = 0.8;
const DEFAULT_RANGE = [0, 255];
// The finest voxels along the normal a new image's band reaches to each side, and the most it may: a chunk of the
// checks' segment's band takes 1 to 4 ms to make at 4 and 4 to 7 ms at 32, 28 at most, so after a jump the panes'
// band is complete 0.45 s on at 4 and 0.70 s at 32.
const DEFAULT_STEPS = 4;
const STEP_LIMIT = 32;

function isRange(range) {
  return (
    Array.isArray(range) &&
    range.length === 2 &&
    range.every(Number.isInteger) &&
    0 <= range[0] &&
    range[0] <= range[1] &&
    range[1] <= 255
  );
}

export function savedImages(saved) {
  if (!Array.isArray(saved)) throw new Error(`Saved ink images must be an array: ${JSON.stringify(saved)}.`);
  for (const image of saved) {
    const keys = Object.keys(image ?? {})
      .sort()
      .join();
    if (
      image === null ||
      typeof image !== "object" ||
      keys !== "color,location,opacity,range,steps,surface,tiff,visible,volume" ||
      typeof image.volume !== "boolean" ||
      !(Number.isInteger(image.steps) && image.steps >= 0 && image.steps <= STEP_LIMIT) ||
      !/^https?:\/\//.test(image.location) ||
      !(image.tiff === null || /^https?:\/\//.test(image.tiff)) ||
      !/^https?:\/\//.test(image.surface) ||
      !/^#[0-9a-f]{6}$/.test(image.color) ||
      typeof image.opacity !== "number" ||
      !(image.opacity >= 0 && image.opacity <= 1) ||
      typeof image.visible !== "boolean"
    )
      throw new Error(`Saved ink image is malformed: ${JSON.stringify(image)}.`);
    if (!isRange(image.range))
      throw new Error(
        `Saved ink image's range is not [low, high] with 0 ≤ low ≤ high ≤ 255: ${JSON.stringify(image.range)}.`,
      );
  }
  return saved;
}

function heldImage(saved) {
  return {
    ...saved,
    pixels: null,
    controller: null,
    handle: null,
    detail: null,
    inVolume: null,
    name: null,
  };
}

function greyOf(data_HWRGBA) {
  const grey_HW = new Uint8Array(data_HWRGBA.length / 4);
  for (let index = 0; index < grey_HW.length; index++) grey_HW[index] = data_HWRGBA[4 * index];
  return grey_HW;
}

async function decoded(location, signal) {
  const response = await fetch(location, { signal });
  if (!response.ok) throw new Error(`${location} answered ${response.status}.`);
  const bitmap = await createImageBitmap(await response.blob(), {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const shape_HW = [bitmap.height, bitmap.width];
  const context = new OffscreenCanvas(shape_HW[1], shape_HW[0]).getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const data_HWRGBA = context.getImageData(0, 0, shape_HW[1], shape_HW[0]).data;
  return { data_HW: greyOf(data_HWRGBA), shape_HW };
}

function style({ color, opacity, range }) {
  return {
    color_RGB: [1, 3, 5].map((start) => Number.parseInt(color.slice(start, start + 2), 16) / 255),
    opacity,
    range,
  };
}

function fileName(url) {
  return url.split("/").filter(Boolean).at(-1);
}

export default function createInk(viewer, entry) {
  const panel = document.createElement("section");
  panel.id = "ink-section";
  panel.className = "card source-dependent";
  panel.setAttribute("aria-label", "Ink");
  panel.innerHTML = `<style>.ink-steps, .ink-range { flex: none; width: 3.5em; }</style>
    <h2>Ink</h2>
    <div id="ink-items" class="overlay-items"></div>
    <label><span class="control-name">Surface</span><select id="ink-surface"></select></label>
    <label><span class="control-name">Open data</span><select id="ink-catalog"></select></label>
    <form id="ink-form" class="overlay-form">
      <input id="ink-location" type="text" placeholder="HTTP URL to the surface's ink image" spellcheck="false"
        autocomplete="off" aria-label="Ink image URL" />
      <input id="ink-tiff" type="text" placeholder="HTTP URL to its full-resolution tiled TIFF, if any"
        spellcheck="false" autocomplete="off" aria-label="Ink TIFF URL" />
      <button id="ink-add" type="submit">Add</button>
    </form>`;
  const items = panel.querySelector("#ink-items");
  const surfaceSelect = panel.querySelector("#ink-surface");
  const catalogSelect = panel.querySelector("#ink-catalog");
  const form = panel.querySelector("#ink-form");
  const input = panel.querySelector("#ink-location");
  const tiffInput = panel.querySelector("#ink-tiff");

  let images = [];
  try {
    images = savedImages(entry.saved ?? []).map(heldImage);
  } catch (error) {
    viewer.setStatus(`The link's ink images were not opened: ${error.message}`);
  }

  function save() {
    entry.save(
      images.map(({ location, tiff, surface, color, opacity, visible, volume, steps, range }) => ({
        location,
        tiff,
        surface,
        color,
        opacity,
        visible,
        volume,
        steps,
        range,
      })),
    );
  }

  function show(image) {
    if (!image.visible || image.pixels === null) return;
    try {
      image.handle = viewer.addSurfaceImage(image.surface, image.pixels, style(image));
    } catch (error) {
      viewer.setStatus(`Ink image ${fileName(image.location)} is not drawn: ${error.message}`);
      return;
    }
    image.detail?.follow(image.handle, viewer.surfaceView());
    if (image.volume) showInVolume(image);
  }

  function showInVolume(image) {
    try {
      image.inVolume = new VolumeInk(viewer, image.surface, image.pixels, style(image), image.steps, (error) =>
        viewer.setStatus(`Ink image ${fileName(image.location)} is not drawn in the volume: ${error.message}`),
      );
    } catch (error) {
      viewer.setStatus(`Ink image ${fileName(image.location)} is not drawn in the volume: ${error.message}`);
    }
  }

  function hideInVolume(image) {
    image.inVolume?.remove();
    image.inVolume = null;
  }

  function hide(image) {
    image.handle?.remove();
    image.handle = null;
    image.detail?.follow(null, null);
    hideInVolume(image);
  }

  async function read(image) {
    image.controller = new AbortController();
    const { signal } = image.controller;
    image.name.textContent = `${fileName(image.location)} …`;
    try {
      const pixels = await decoded(image.location, signal);
      // Removed while the browser decoded it, which the signal does not stop.
      if (signal.aborted) return;
      image.pixels = pixels;
      if (image.tiff !== null)
        image.detail = new Detail(image.tiff, image.surface, image.pixels.shape_HW, (error) =>
          viewer.setStatus(`Ink TIFF ${fileName(image.tiff)}: ${error.message}`),
        );
      show(image);
    } catch (error) {
      if (error.name !== "AbortError") viewer.setStatus(`Ink image ${image.location} did not open: ${error.message}`);
    } finally {
      image.controller = null;
      image.name.textContent = fileName(image.location);
    }
  }

  function addRow(image) {
    const row = document.createElement("div");
    row.className = "layer-row";
    const visible = document.createElement("input");
    visible.type = "checkbox";
    visible.checked = image.visible;
    visible.setAttribute("aria-label", "Show ink image");
    visible.addEventListener("change", () => {
      image.visible = visible.checked;
      if (!image.visible) hide(image);
      else if (image.pixels === null && image.controller === null) void read(image);
      else show(image);
      save();
    });
    const name = document.createElement("span");
    name.className = "layer-row-name";
    name.textContent = fileName(image.location);
    name.title = `${image.location}${image.tiff === null ? "" : ` and ${image.tiff}`} on ${image.surface}`;
    image.name = name;
    const color = document.createElement("input");
    color.type = "color";
    color.value = image.color;
    color.setAttribute("aria-label", "Ink colour");
    const opacity = document.createElement("input");
    opacity.type = "range";
    opacity.min = "0";
    opacity.max = "1";
    opacity.step = "0.05";
    opacity.value = String(image.opacity);
    opacity.setAttribute("aria-label", "Ink opacity");
    for (const control of [color, opacity]) {
      control.addEventListener("input", () => {
        image.color = color.value;
        image.opacity = Number(opacity.value);
        image.handle?.setStyle(style(image));
        image.inVolume?.setStyle(style(image));
      });
      control.addEventListener("change", save);
    }
    const close = document.createElement("button");
    close.type = "button";
    close.className = "icon-btn";
    close.textContent = "×";
    close.setAttribute("aria-label", "Remove ink image");
    close.addEventListener("click", () => {
      image.controller?.abort();
      hide(image);
      row.remove();
      images = images.filter((candidate) => candidate !== image);
      save();
    });
    row.append(visible, name, color, opacity, close);
    const volumeRow = document.createElement("div");
    volumeRow.className = "layer-row";
    const inVolume = document.createElement("input");
    inVolume.type = "checkbox";
    inVolume.checked = image.volume;
    inVolume.setAttribute("aria-label", "Show ink image in the volume and the 3D pane");
    inVolume.addEventListener("change", () => {
      image.volume = inVolume.checked;
      if (!image.volume) hideInVolume(image);
      else if (image.handle !== null) showInVolume(image);
      save();
    });
    const volumeName = document.createElement("span");
    volumeName.className = "layer-row-name";
    volumeName.textContent = "In the volume";
    const steps = document.createElement("input");
    steps.type = "number";
    steps.className = "ink-steps";
    steps.min = "0";
    steps.max = String(STEP_LIMIT);
    steps.step = "1";
    steps.value = String(image.steps);
    steps.setAttribute("aria-label", "Voxels along the normal the ink's band reaches to each side");
    steps.addEventListener("change", () => {
      if (!(Number.isInteger(steps.valueAsNumber) && steps.valueAsNumber >= 0 && steps.valueAsNumber <= STEP_LIMIT)) {
        steps.value = String(image.steps);
        return;
      }
      image.steps = steps.valueAsNumber;
      image.inVolume?.setSteps(image.steps);
      save();
    });
    volumeRow.append(inVolume, volumeName, steps);
    const rangeRow = document.createElement("div");
    rangeRow.className = "layer-row";
    const rangeName = document.createElement("span");
    rangeName.className = "layer-row-name";
    rangeName.textContent = "Show from";
    const to = document.createElement("span");
    to.textContent = "to";
    const [low, high] = ["Lowest pixel shown", "Highest pixel shown"].map((label, end) => {
      const bound = document.createElement("input");
      bound.type = "number";
      bound.className = "ink-range";
      bound.min = "0";
      bound.max = "255";
      bound.step = "1";
      bound.value = String(image.range[end]);
      bound.setAttribute("aria-label", label);
      return bound;
    });
    for (const bound of [low, high]) {
      bound.addEventListener("input", () => {
        const range = [low.valueAsNumber, high.valueAsNumber];
        if (!isRange(range)) return;
        image.range = range;
        image.handle?.setStyle(style(image));
        image.inVolume?.setStyle(style(image));
      });
      bound.addEventListener("change", () => {
        [low.value, high.value] = image.range.map(String);
        save();
      });
    }
    rangeRow.append(rangeName, low, to, high);
    close.addEventListener("click", () => {
      volumeRow.remove();
      rangeRow.remove();
    });
    items.append(row, volumeRow, rangeRow);
  }

  for (const image of images) addRow(image);

  const catalog = new CatalogChoice(
    catalogSelect,
    () => surfaceSelect.value,
    ({ image, tiff }) => {
      input.value = image;
      tiffInput.value = tiff ?? "";
    },
    (error) => viewer.setStatus(`Ink: ${error.message}`),
  );
  surfaceSelect.addEventListener("change", () => catalog.offer());

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const location = input.value.trim();
    const tiff = tiffInput.value.trim() === "" ? null : tiffInput.value.trim();
    const surface = surfaceSelect.value;
    if (!/^https?:\/\//.test(location)) {
      viewer.setStatus(`An ink image needs an HTTP(S) URL, not "${location}".`);
      return;
    }
    if (tiff !== null && !/^https?:\/\//.test(tiff)) {
      viewer.setStatus(`An ink TIFF needs an HTTP(S) URL, not "${tiff}".`);
      return;
    }
    if (surface === "") {
      viewer.setStatus("Add the image's surface in Surfaces first.");
      return;
    }
    const image = heldImage({
      location,
      tiff,
      surface,
      color: DEFAULT_COLOR,
      opacity: DEFAULT_OPACITY,
      visible: true,
      volume: false,
      steps: DEFAULT_STEPS,
      range: DEFAULT_RANGE,
    });
    images.push(image);
    addRow(image);
    save();
    input.value = "";
    tiffInput.value = "";
    void read(image);
  });

  function surfacesChanged() {
    const chosen = surfaceSelect.value;
    surfaceSelect.replaceChildren(...viewer.surfaces().map((url) => new Option(fileName(url), url)));
    if (viewer.surfaces().includes(chosen)) surfaceSelect.value = chosen;
    catalog.offer();
  }

  function volumeChanged() {
    surfacesChanged();
    for (const image of images) {
      image.handle = null;
      image.detail?.follow(null, null);
      image.inVolume?.dispose();
      image.inVolume = null;
      if (image.pixels !== null) show(image);
      else if (image.visible && image.controller === null) void read(image);
    }
  }

  function surfaceViewChanged() {
    for (const image of images) image.detail?.follow(image.handle, viewer.surfaceView());
  }

  function focusChanged() {
    for (const image of images) void image.inVolume?.update();
  }

  return { panel, place: "data", volumeChanged, surfacesChanged, focusChanged, surfaceViewChanged };
}
