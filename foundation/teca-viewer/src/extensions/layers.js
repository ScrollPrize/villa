const DEFAULT_COLOR = "#ff00ff";
const DEFAULT_OPACITY = 0.65;
const DEFAULT_RANGE = [0, 255];
const CATALOG_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json";
const BUCKET = "s3://vesuvius-challenge-open-data";
const BUCKET_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";
const KINDS = { "ink-detection-3d-zarr": "3D ink", "surface-prediction-zarr": "Surface" };

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

function assetUrl(asset) {
  for (const origin of asset.origins)
    if (origin.access_roots.some((root) => root.url === BUCKET)) return `${BUCKET_URL}/${origin.path}`;
  for (const origin of asset.origins)
    for (const root of origin.access_roots) if (root.type === "https") return `${root.url}/${origin.path}`;
  return null;
}

// A surface prediction made on a coarser level of the volume, its `parameters.level` past 0, has that level's shape at
// its own level 0, which a layer refuses.
export function catalogLayers(metadata) {
  const layers = new Map();
  for (const sample of Object.values(metadata.samples))
    for (const volume of Object.values(sample.volumes)) {
      const zarr = volume.data.find((asset) => asset.type === "ome-zarr");
      const url = zarr === undefined ? null : assetUrl(zarr);
      const found = volume.data.filter(
        (asset) => Object.hasOwn(KINDS, asset.type) && !(asset.parameters.level > 0) && assetUrl(asset) !== null,
      );
      if (url === null || found.length === 0) continue;
      layers.set(
        url.endsWith("/") ? url : `${url}/`,
        found.map((asset) => {
          const model = metadata.models[asset.parameters.model_id];
          if (model === undefined)
            throw new Error(`The catalog lists no model ${asset.parameters.model_id}, of ${assetUrl(asset)}.`);
          return { name: `${KINDS[asset.type]} ${model.long_id}`, url: assetUrl(asset) };
        }),
      );
    }
  return layers;
}

export function savedLayers(saved) {
  if (!Array.isArray(saved)) throw new Error(`Saved layers must be an array: ${JSON.stringify(saved)}.`);
  for (const layer of saved) {
    if (
      layer === null ||
      typeof layer !== "object" ||
      Object.keys(layer)
        .filter((key) => key !== "range")
        .sort()
        .join() !== "color,location,opacity,visible" ||
      !/^https?:\/\//.test(layer.location) ||
      !/^#[0-9a-f]{6}$/.test(layer.color) ||
      !(layer.opacity >= 0 && layer.opacity <= 1) ||
      typeof layer.visible !== "boolean"
    )
      throw new Error(`Saved layer is malformed: ${JSON.stringify(layer)}.`);
    if (layer.range !== undefined && !isRange(layer.range))
      throw new Error(
        `Saved layer's range is not [low, high] with 0 ≤ low ≤ high ≤ 255: ${JSON.stringify(layer.range)}.`,
      );
  }
  if (new Set(saved.map((layer) => layer.location)).size !== saved.length)
    throw new Error("Saved layer locations must be unique.");
  return saved.map((layer) => ({ ...layer, range: layer.range ?? DEFAULT_RANGE }));
}

export function layerStyle({ color, opacity, range }) {
  return {
    color_RGB: [1, 3, 5].map((start) => Number.parseInt(color.slice(start, start + 2), 16) / 255),
    opacity,
    range,
  };
}

function heldLayer(saved) {
  return { ...saved, readers: null, controller: null, handle: null };
}

export default function createLayers(viewer, entry) {
  const panel = document.createElement("section");
  panel.id = "layers-section";
  panel.className = "card source-dependent";
  panel.setAttribute("aria-label", "Layers");
  const style = document.createElement("style");
  style.textContent = ".layers-range { flex: none; width: 3.5em; }";
  const heading = document.createElement("h2");
  heading.textContent = "Layers";
  const items = document.createElement("div");
  items.id = "layers-items";
  items.className = "overlay-items";
  const form = document.createElement("form");
  form.id = "layers-form";
  form.className = "overlay-form";
  const input = document.createElement("input");
  input.id = "layers-location";
  input.type = "text";
  input.placeholder = "HTTP URL to a Zarr on the CT's grid";
  input.spellcheck = false;
  input.autocomplete = "off";
  input.setAttribute("aria-label", "Layer URL");
  const add = document.createElement("button");
  add.id = "layers-add";
  add.type = "submit";
  add.textContent = "Add";
  form.append(input, add);
  const choice = document.createElement("label");
  const choiceName = document.createElement("span");
  choiceName.className = "control-name";
  choiceName.textContent = "Open data";
  const catalogSelect = document.createElement("select");
  catalogSelect.id = "layers-catalog";
  choice.append(choiceName, catalogSelect);
  panel.append(style, heading, items, choice, form);

  let layers = [];
  try {
    layers = savedLayers(entry.saved ?? []).map(heldLayer);
  } catch (error) {
    viewer.setStatus(`The link's layers were not opened: ${error.message}`);
  }

  function save() {
    entry.save(
      layers.map(({ location, color, opacity, visible, range }) => ({ location, color, opacity, visible, range })),
    );
  }

  function show(layer) {
    if (!layer.visible || layer.readers === null) return;
    try {
      layer.handle = viewer.addLayer(layer.readers, layerStyle(layer));
    } catch (error) {
      viewer.setStatus(`Layer ${layer.location} is not drawn: ${error.message}`);
    }
  }

  async function open(layer) {
    layer.controller = new AbortController();
    const { signal } = layer.controller;
    const opened = await Promise.allSettled([0, 1].map(() => viewer.openVolume(layer.location, signal)));
    const readers = opened.filter(({ status }) => status === "fulfilled").map(({ value }) => value);
    const failure = opened.find(({ status }) => status === "rejected");
    if (signal.aborted || failure !== undefined) {
      for (const reader of readers) reader.dispose();
      layer.controller = null;
      if (failure !== undefined && failure.reason.name !== "AbortError")
        viewer.setStatus(`Layer ${layer.location} did not open: ${failure.reason.message}`);
      return;
    }
    layer.readers = { slices: readers[0], surface: readers[1] };
    show(layer);
  }

  function addRow(layer) {
    const row = document.createElement("div");
    row.className = "layer-row";
    const visible = document.createElement("input");
    visible.type = "checkbox";
    visible.checked = layer.visible;
    visible.setAttribute("aria-label", "Show layer");
    visible.addEventListener("change", () => {
      layer.visible = visible.checked;
      if (!layer.visible) {
        layer.handle?.remove();
        layer.handle = null;
      } else if (layer.readers === null && layer.controller === null) void open(layer);
      else show(layer);
      save();
    });
    const name = document.createElement("span");
    name.className = "layer-row-name";
    name.textContent = layer.location.split("/").filter(Boolean).at(-1);
    name.title = layer.location;
    const color = document.createElement("input");
    color.type = "color";
    color.value = layer.color;
    color.setAttribute("aria-label", "Layer colour");
    const opacity = document.createElement("input");
    opacity.type = "range";
    opacity.min = "0";
    opacity.max = "1";
    opacity.step = "0.05";
    opacity.value = String(layer.opacity);
    opacity.setAttribute("aria-label", "Layer opacity");
    for (const control of [color, opacity]) {
      control.addEventListener("input", () => {
        layer.color = color.value;
        layer.opacity = Number(opacity.value);
        layer.handle?.setStyle(layerStyle(layer));
      });
      control.addEventListener("change", save);
    }
    const rangeRow = document.createElement("div");
    rangeRow.className = "layer-row";
    const rangeName = document.createElement("span");
    rangeName.className = "layer-row-name";
    rangeName.textContent = "Show from";
    const to = document.createElement("span");
    to.textContent = "to";
    const [low, high] = ["Lowest value shown", "Highest value shown"].map((label, end) => {
      const bound = document.createElement("input");
      bound.type = "number";
      bound.className = "layers-range";
      bound.min = "0";
      bound.max = "255";
      bound.step = "1";
      bound.value = String(layer.range[end]);
      bound.setAttribute("aria-label", label);
      return bound;
    });
    for (const bound of [low, high]) {
      bound.addEventListener("input", () => {
        const range = [low.valueAsNumber, high.valueAsNumber];
        if (!isRange(range)) return;
        layer.range = range;
        layer.handle?.setStyle(layerStyle(layer));
      });
      bound.addEventListener("change", () => {
        [low.value, high.value] = layer.range.map(String);
        save();
      });
    }
    rangeRow.append(rangeName, low, to, high);
    const close = document.createElement("button");
    close.type = "button";
    close.className = "icon-btn";
    close.textContent = "×";
    close.setAttribute("aria-label", "Remove layer");
    close.addEventListener("click", () => {
      layer.controller?.abort();
      layer.handle?.remove();
      layer.readers?.slices.dispose();
      layer.readers?.surface.dispose();
      row.remove();
      rangeRow.remove();
      layers = layers.filter((candidate) => candidate !== layer);
      save();
    });
    row.append(visible, name, color, opacity, close);
    items.append(row, rangeRow);
  }

  for (const layer of layers) addRow(layer);

  let predictions = null;
  let reading = false;
  let ctUrl = null;

  function offer() {
    const found = predictions?.get(ctUrl) ?? [];
    const first = reading
      ? "Reading the catalog…"
      : predictions === null
        ? "The catalog's predictions"
        : found.length === 0
          ? "The catalog has none for this CT"
          : `The catalog's predictions (${found.length})`;
    catalogSelect.replaceChildren(new Option(first, ""), ...found.map(({ name, url }) => new Option(name, url)));
  }

  async function readCatalog() {
    if (predictions !== null || reading) return;
    reading = true;
    offer();
    try {
      const response = await fetch(CATALOG_URL, { cache: "no-cache" });
      if (!response.ok) throw new Error(`The catalog ${CATALOG_URL} answered ${response.status}.`);
      predictions = catalogLayers(await response.json());
    } catch (error) {
      viewer.setStatus(`Layers: ${error.message}`);
    } finally {
      reading = false;
      offer();
    }
  }

  catalogSelect.addEventListener("pointerdown", readCatalog);
  catalogSelect.addEventListener("keydown", readCatalog);
  catalogSelect.addEventListener("change", () => {
    input.value = catalogSelect.value;
    catalogSelect.value = "";
  });
  offer();

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const location = input.value.trim();
    if (!/^https?:\/\//.test(location)) {
      viewer.setStatus(`A layer needs an HTTP(S) Zarr URL, not "${location}".`);
      return;
    }
    if (layers.some((layer) => layer.location === location)) {
      viewer.setStatus(`Layer ${location} is already open.`);
      return;
    }
    const layer = heldLayer({
      location,
      color: DEFAULT_COLOR,
      opacity: DEFAULT_OPACITY,
      visible: true,
      range: DEFAULT_RANGE,
    });
    layers.push(layer);
    addRow(layer);
    save();
    input.value = "";
    void open(layer);
  });

  function volumeChanged(opened) {
    ctUrl = opened.rootUrl;
    offer();
    for (const layer of layers) {
      layer.handle = null;
      if (layer.readers !== null) show(layer);
      else if (layer.controller === null) void open(layer);
    }
  }

  return { panel, place: "data", volumeChanged };
}
