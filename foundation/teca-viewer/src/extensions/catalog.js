// The catalog is one JSON file, 1.3 MB over the network and 17 MB parsed, so it is fetched only when Load is pressed.
// The browser keeps it and downloads none of it while it is unchanged.

const CATALOG_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json";
const BUCKET = "s3://vesuvius-challenge-open-data";
const BUCKET_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";
const LIST_ROWS = 5;
const ROW_PX = 34;

function assetUrl(asset) {
  for (const origin of asset.origins)
    if (origin.access_roots.some((root) => root.url === BUCKET)) return `${BUCKET_URL}/${origin.path}`;
  for (const origin of asset.origins)
    for (const root of origin.access_roots) if (root.type === "https") return `${root.url}/${origin.path}`;
  return null;
}

function measure(volume, key, unit) {
  const value = volume.properties?.[key];
  return Number.isFinite(value) ? `${value} ${unit}` : null;
}

export function catalogVolumes(metadata) {
  const volumes = [];
  for (const sample of Object.values(metadata.samples)) {
    const byId = new Map();
    for (const volume of Object.values(sample.volumes)) {
      const zarr = volume.data.find((asset) => asset.type === "ome-zarr");
      const url = zarr === undefined ? null : assetUrl(zarr);
      if (url === null) continue;
      const time = /^(\d{4})(\d\d)(\d\d)(\d\d)(\d\d)(\d\d)$/.exec(volume.id);
      const entry = {
        id: volume.id,
        title: `${volume.sample_id} ${volume.long_id.replace(/\.zarr$/, "")}`,
        sample: volume.sample_id,
        fields: [measure(volume, "pixel_size_um", "µm"), measure(volume, "energy_keV", "keV")].filter(
          (field) => field !== null,
        ),
        date: time === null ? null : `${time[1]}-${time[2]}-${time[3]}`,
        time: time === null ? null : `${time[4]}:${time[5]}:${time[6]}`,
        url: url.endsWith("/") ? url : `${url}/`,
        segments: [],
      };
      byId.set(volume.id, entry);
      volumes.push(entry);
    }
    for (const segment of Object.values(sample.segments))
      for (const asset of segment.data) {
        if (asset.type !== "tifxyz-transformed") continue;
        const volume = byId.get(asset.parameters.target_volume);
        const url = assetUrl(asset);
        if (volume !== undefined && url !== null) volume.segments.push({ segment: segment.id, url });
      }
  }
  const alike = (volume) => [volume.sample, ...volume.fields, volume.date].join(" ");
  const counts = new Map();
  for (const volume of volumes) counts.set(alike(volume), (counts.get(alike(volume)) ?? 0) + 1);
  const byName = (a, b) => a.name.localeCompare(b.name);
  const entries = volumes.map((volume) => {
    const shared = counts.get(alike(volume)) > 1;
    const fields = [...volume.fields];
    if (volume.date !== null) fields.push(shared ? `${volume.date} ${volume.time}` : volume.date);
    else if (shared || fields.length === 0) fields.push(volume.id);
    const name = `${volume.title} ${fields.join(" ")}`;
    const surfaces = volume.segments.map(({ segment, url }) => ({
      name: `${segment} ${name}`,
      title: volume.title,
      lines: [[segment, volume.sample], fields],
      url,
      volume: volume.url,
    }));
    return {
      name,
      title: volume.title,
      lines: [[volume.sample], fields],
      url: volume.url,
      surfaces: surfaces.sort(byName),
    };
  });
  return entries.sort(byName);
}

export function everySurface(volumes) {
  return volumes.flatMap((volume) => volume.surfaces).sort((a, b) => a.name.localeCompare(b.name));
}

export function matching(items, query) {
  const words = query
    .toLowerCase()
    .split(/\s+/)
    .filter((word) => word !== "");
  return items.filter((item) => words.every((word) => item.name.toLowerCase().includes(word)));
}

function row(item) {
  const element = Object.assign(document.createElement("div"), { className: "catalog-row", title: item.title });
  element.setAttribute("role", "option");
  element.dataset.url = item.url;
  element.append(
    ...item.lines.map((pieces) => {
      const line = document.createElement("div");
      line.append(...pieces.map((piece) => Object.assign(document.createElement("span"), { textContent: piece })));
      return line;
    }),
  );
  return element;
}

export default function createCatalog(viewer, entry) {
  const panel = document.createElement("section");
  panel.id = "catalog-section";
  panel.className = "card";
  panel.setAttribute("aria-label", "Catalog");
  panel.innerHTML = `
    <style>
      #catalog-section:has(> .catalog-lists:not([hidden])) { display: flex; flex: 1 0 auto; flex-direction: column; }
      .catalog-lists { display: flex; flex: 1 0 auto; flex-direction: column; gap: 6px; }
      .catalog-list { flex: 1 1 0; min-height: ${LIST_ROWS * ROW_PX + 2}px; overflow-y: auto;
        border: 1px solid var(--bevel-dark); background: var(--field); font-size: 12px; }
      .catalog-list:focus-visible { outline: 1px solid var(--accent); }
      .catalog-row { height: ${ROW_PX}px; padding: 2px 6px; box-sizing: border-box; cursor: default;
        content-visibility: auto; contain-intrinsic-size: auto ${ROW_PX}px; }
      .catalog-row[aria-selected="true"] { background: var(--accent); color: var(--field); }
      .catalog-row > div { display: flex; column-gap: 1em; white-space: nowrap; }
      .catalog-row > div + div { font-size: 11px; opacity: 0.8; }
    </style>
    <h2>Catalog</h2>
    <button id="catalog-load" type="button">Load</button>
    <div id="catalog-lists" class="catalog-lists" hidden>
      <input id="catalog-search" type="search" placeholder="Sample, µm, keV…" aria-label="Search volumes" />
      <div id="catalog-volumes" class="catalog-list" role="listbox" tabindex="0" aria-label="Volumes"></div>
      <button id="catalog-open" type="button">Open volume</button>
      <h3>Surfaces</h3>
      <label><span class="control-name">On</span>
        <select id="catalog-surface-scope">
          <option value="open">This volume</option>
          <option value="all">All volumes</option>
        </select>
      </label>
      <input id="catalog-surface-search" type="search" placeholder="Segment, sample, µm…" aria-label="Search surfaces" />
      <div id="catalog-surfaces" class="catalog-list" role="listbox" tabindex="0" aria-label="Surfaces"></div>
      <button id="catalog-add" type="button">Add surface</button>
    </div>
    <p id="catalog-note" class="readout-line" role="status"></p>`;
  const [loadButton, openButton, addButton] = panel.querySelectorAll("button");
  const lists = panel.querySelector("#catalog-lists");
  const [volumeSearch, surfaceSearch] = panel.querySelectorAll("input");
  const scopeSelect = panel.querySelector("select");
  const note = panel.querySelector("#catalog-note");

  let volumes = null;
  let surfaces = null;
  let openUrl = null;
  let everyVolume = false;

  function rowList(element, search, enter) {
    let chosen = "";
    const made = new Map();
    const rows = () => element.children;
    function choose(url) {
      chosen = [...rows()].some((each) => each.dataset.url === url) ? url : "";
      for (const each of rows()) each.setAttribute("aria-selected", String(each.dataset.url === url));
      // The list scrolls to show it, and not the panel it lies in, as scrollIntoView would.
      const row = element.querySelector('[aria-selected="true"]');
      if (row !== null) {
        const top = element.getBoundingClientRect().top + element.clientTop;
        const shown = row.getBoundingClientRect();
        if (shown.top < top) element.scrollTop -= top - shown.top;
        else if (shown.bottom > top + element.clientHeight)
          element.scrollTop += shown.bottom - top - element.clientHeight;
      }
      showChoice();
    }
    function act() {
      element.blur();
      search.blur();
      enter(chosen);
    }
    function keyed(event) {
      if (event.key === "Enter" && chosen !== "") act();
      else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        const listed = [...rows()];
        const at = listed.findIndex((each) => each.dataset.url === chosen);
        const next = event.key === "ArrowDown" ? (at < 0 ? 0 : at + 1) : at < 0 ? listed.length - 1 : at - 1;
        if (next >= 0 && next < listed.length) choose(listed[next].dataset.url);
      } else return;
      // So the viewer's keys do not take it too.
      event.preventDefault();
      event.stopPropagation();
    }
    element.addEventListener("keydown", keyed);
    search.addEventListener("keydown", keyed);
    element.addEventListener("click", (event) => {
      const clicked = event.target.closest(".catalog-row");
      if (clicked !== null) choose(clicked.dataset.url);
    });
    return {
      chosen: () => chosen,
      choose,
      act,
      show: (items) => {
        for (const item of items) if (!made.has(item)) made.set(item, row(item));
        element.replaceChildren(...items.map((item) => made.get(item)));
        choose(chosen);
      },
    };
  }

  const volumeList = rowList(panel.querySelector("#catalog-volumes"), volumeSearch, (url) => {
    void viewer.showVolume(url);
  });
  const surfaceList = rowList(panel.querySelector("#catalog-surfaces"), surfaceSearch, (url) => {
    void add(url);
  });

  function showChoice() {
    openButton.disabled = volumeList.chosen() === "";
    addButton.disabled = surfaceList.chosen() === "";
  }

  function describe() {
    const open = volumes?.find((volume) => volume.url === openUrl) ?? null;
    const all = everyVolume || open === null;
    scopeSelect.value = all ? "all" : "open";
    scopeSelect.options[0].disabled = open === null;
    volumeList.show(matching(volumes ?? [], volumeSearch.value));
    surfaceList.show(matching((all ? surfaces : open.surfaces) ?? [], surfaceSearch.value));
    if (volumes !== null && volumeList.chosen() === "" && open !== null) volumeList.choose(open.url);
    if (volumes === null) return;
    const registered = `${surfaces.length} surfaces are registered to them`;
    if (openUrl === null) note.textContent = `${volumes.length} volumes. ${registered}.`;
    else if (open === null) note.textContent = `${volumes.length} volumes. ${registered}; the open volume is not one.`;
    else note.textContent = `${volumes.length} volumes. ${registered}, ${open.surfaces.length} to the open one.`;
  }

  function searched() {
    entry.save({ volumes: volumeSearch.value, surfaces: surfaceSearch.value, all: everyVolume });
    describe();
  }

  async function add(url) {
    const { volume } = surfaces.find((surface) => surface.url === url);
    if (volume !== openUrl) {
      everyVolume = true;
      searched();
      if (!(await viewer.showVolume(volume))) return;
    }
    await viewer.addSurface(url);
  }

  async function load() {
    loadButton.disabled = true;
    loadButton.textContent = "Loading…";
    try {
      // "no-cache" asks the host whether the browser's copy is current; it answers 304, with no body, when it is.
      const response = await fetch(CATALOG_URL, { cache: "no-cache" });
      if (!response.ok) throw new Error(`The catalog ${CATALOG_URL} answered ${response.status}.`);
      volumes = catalogVolumes(await response.json());
      surfaces = everySurface(volumes);
      loadButton.hidden = true;
      lists.hidden = false;
      searched();
    } catch (error) {
      viewer.setStatus(`Catalog: ${error.message}`);
      note.textContent = error.message;
      loadButton.disabled = false;
      loadButton.textContent = "Load";
    }
  }

  loadButton.addEventListener("click", load);
  volumeSearch.addEventListener("input", searched);
  surfaceSearch.addEventListener("input", searched);
  scopeSelect.addEventListener("change", () => {
    everyVolume = scopeSelect.value === "all";
    searched();
  });
  openButton.addEventListener("click", () => volumeList.act());
  addButton.addEventListener("click", () => surfaceList.act());

  function volumeChanged(opened) {
    if (!everyVolume && openUrl !== null && openUrl !== opened.rootUrl) surfaceSearch.value = "";
    openUrl = opened.rootUrl;
    if (volumes === null) describe();
    else searched();
  }

  const saved = entry.saved;
  if (saved !== null) {
    if (
      typeof saved.volumes !== "string" ||
      typeof saved.surfaces !== "string" ||
      ![undefined, true, false].includes(saved.all)
    )
      viewer.setStatus(`The link's catalog searches were not used: ${JSON.stringify(saved)}.`);
    else {
      volumeSearch.value = saved.volumes;
      surfaceSearch.value = saved.surfaces;
      everyVolume = saved.all === true;
    }
    void load();
  }
  describe();
  return { panel, place: "data", volumeChanged };
}
