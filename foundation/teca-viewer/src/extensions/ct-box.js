import * as THREE from "../../vendor/three.module.js";
import { boxMaterial } from "./ct-box/material.js";
import { rayPick } from "./ct-box/pick.js";
import { boxLevel, boxRegion, copyBlock, displayBytes, regionChunks } from "./ct-box/region.js";

const BOX_TEXELS = 256;
const SIDES = [128, 256, 512, 1024, 2048];
// Black and white are display bytes; density is the absorption a white texel gives per texel of ray. On Paris 4 the
// papyrus is 90 to 155 and the brightest of 256 voxels 115 to 170; absorbed above 120, its fibres show.
const DEFAULT_LOOK = { side: 256, maximum: true, black: 120, white: 200, density: 0.5, follow: false };

export function savedLook(saved) {
  if (
    saved === null ||
    typeof saved !== "object" ||
    Object.keys(saved).sort().join() !== "black,density,follow,maximum,side,white" ||
    !SIDES.includes(saved.side) ||
    typeof saved.maximum !== "boolean" ||
    typeof saved.follow !== "boolean" ||
    !(saved.black >= 0 && saved.black < saved.white && saved.white <= 255) ||
    !(saved.density > 0)
  )
    throw new Error(`Saved 3D CT look is malformed: ${JSON.stringify(saved)}.`);
  return saved;
}

export default function createCtBox(viewer, entry) {
  const panel = document.createElement("section");
  panel.id = "ct-box-section";
  panel.className = "card source-dependent";
  panel.setAttribute("aria-label", "3D CT");
  panel.innerHTML = `<h2>3D CT</h2>
    <div class="button-row">
      <button id="ct-box-show" type="button">Show</button>
      <button id="ct-box-here" type="button">Move to focus</button>
    </div>
    <label class="checkbox-row"><input id="ct-box-follow" type="checkbox" /> Follow the focus</label>
    <label><span class="control-name">Side</span>
      <select id="ct-box-side">${SIDES.map((side) => `<option value="${side}">${side}</option>`).join("")}</select>
    </label>
    <label><span class="control-name">Draw</span>
      <select id="ct-box-draw">
        <option value="maximum">Brightest</option>
        <option value="absorbed">Absorbed</option>
      </select>
    </label>
    <label><span class="control-name">Black</span>
      <span class="range-pair">
        <input id="ct-box-black" type="range" min="0" max="254" />
        <input id="ct-box-black-number" type="number" min="0" max="254" step="1" />
      </span>
    </label>
    <label><span class="control-name">White</span>
      <span class="range-pair">
        <input id="ct-box-white" type="range" min="1" max="255" />
        <input id="ct-box-white-number" type="number" min="1" max="255" step="1" />
      </span>
    </label>
    <label><span class="control-name">Density</span>
      <input id="ct-box-density" type="number" min="0.001" max="10" step="0.01" />
    </label>
    <p id="ct-box-note" class="readout-line" role="status"></p>`;
  const [showButton, hereButton] = panel.querySelectorAll("button");
  const [sideSelect, drawSelect] = panel.querySelectorAll("select");
  const [followInput, blackInput, blackNumber, whiteInput, whiteNumber, densityInput] = panel.querySelectorAll("input");
  const note = panel.querySelector("#ct-box-note");

  let look = DEFAULT_LOOK;
  try {
    look = savedLook(entry.saved ?? DEFAULT_LOOK);
  } catch (error) {
    viewer.setStatus(`The link's 3D CT look was not used: ${error.message}`);
  }
  let ct = null;
  let reader = null;
  let loading = null;
  let shown = null;

  function failed(error) {
    console.error(error);
    viewer.setStatus(`3D CT: ${error.message}`);
    note.textContent = error.message;
  }

  function describe() {
    showButton.textContent = shown === null && loading === null ? "Show" : "Hide";
    showButton.disabled = ct === null;
    hereButton.disabled = shown === null && loading === null;
    followInput.checked = look.follow;
    sideSelect.value = String(look.side);
    drawSelect.value = look.maximum ? "maximum" : "absorbed";
    blackInput.value = String(look.black);
    blackNumber.value = String(look.black);
    whiteInput.value = String(look.white);
    whiteNumber.value = String(look.white);
    densityInput.value = String(look.density);
    densityInput.disabled = look.maximum;
  }

  function stopReading() {
    if (loading === null) return;
    loading.abort();
    loading = null;
    reader?.clearChunkCache();
  }

  function hide() {
    stopReading();
    if (shown === null) return;
    shown.handle.remove();
    shown.mesh.geometry.dispose();
    shown.mesh.material.uniforms.box.value.dispose();
    shown.mesh.material.dispose();
    shown = null;
  }

  async function show() {
    stopReading();
    const controller = new AbortController();
    loading = controller;
    const opened = ct;
    const center_ZYX = viewer.focusPoint();
    describe();
    try {
      const index = boxLevel(opened.scales, look.side, BOX_TEXELS);
      const region = boxRegion(opened.scales[index], center_ZYX, look.side);
      const [sizeZ, sizeY, sizeX] = region.size_ZYX;
      const box_ZYX = new Uint8Array(sizeZ * sizeY * sizeX);
      const held = shown?.index === index ? shown : null;
      if (held !== null) copyBlock(box_ZYX, region, held.box_ZYX, held.region.start_ZYX, held.region.size_ZYX);
      const chunks = regionChunks(opened.scales[index], region, held?.region ?? null);
      note.textContent = `Reading ${chunks.length} chunks of level ${index}…`;
      if (reader === null) {
        const opening = await viewer.openVolume(opened.rootUrl, controller.signal);
        if (controller.signal.aborted) {
          opening.dispose();
          return;
        }
        reader = opening;
      }
      const scale = reader.scales[index];
      await Promise.all(
        chunks.map(async (chunk_ZYX) => {
          const values = await reader.getChunk(...chunk_ZYX, 0, scale, controller.signal, 0);
          const origin = chunk_ZYX.map((c, axis) => c * scale.chunks_ZYX[axis]);
          const bytes = displayBytes(values, opened.metadata.displayEncoding);
          copyBlock(box_ZYX, region, bytes, origin, scale.chunks_ZYX);
        }),
      );
      if (controller.signal.aborted) return;
      // The box holds its own copy; the reader's cache would grow by one box a read.
      reader.clearChunkCache();
      const texture = new THREE.Data3DTexture(box_ZYX, sizeX, sizeY, sizeZ);
      texture.format = THREE.RedFormat;
      texture.minFilter = THREE.LinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.unpackAlignment = 1;
      texture.needsUpdate = true;
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        boxMaterial(texture, [sizeX, sizeY, sizeZ], { ...look, black: look.black / 255, white: look.white / 255 }),
      );
      mesh.scale.set(...region.extent_ZYX.toReversed());
      mesh.position.set(...region.min_ZYX.map((min, axis) => min + region.extent_ZYX[axis] / 2).toReversed());
      mesh.raycast = (raycaster, hits) => {
        const ray = raycaster.ray.clone().applyMatrix4(mesh.matrixWorld.clone().invert());
        const found = rayPick(box_ZYX, [sizeX, sizeY, sizeZ], ray.origin.toArray(), ray.direction.toArray(), look);
        if (found === null) return;
        const [front, point] = [found.enter, found.picked].map((t) =>
          ray.at(t, new THREE.Vector3()).applyMatrix4(mesh.matrixWorld),
        );
        hits.push({ distance: raycaster.ray.origin.distanceTo(front), point, object: mesh });
      };
      loading = null;
      hide();
      shown = { handle: viewer.add3d(mesh), mesh, index, region, box_ZYX };
      const mib = (box_ZYX.length / 2 ** 20).toFixed(1);
      const at = center_ZYX.toReversed().map((value, axis) => `${"xyz"[axis]} ${Math.round(value)}`);
      note.textContent =
        `Level ${index}, ${sizeX} × ${sizeY} × ${sizeZ} voxels, ${mib} MiB, ${chunks.length} chunks read, ` +
        `about ${at.join(", ")}.`;
      describe();
    } catch (error) {
      if (controller.signal.aborted) return;
      stopReading();
      failed(error);
      describe();
    }
  }

  function restyle(next) {
    const reread = next.side !== look.side || (next.follow && !look.follow);
    look = next;
    entry.save(look);
    describe();
    if (reread && (shown !== null || loading !== null)) void show();
    if (shown === null) return;
    const uniforms = shown.mesh.material.uniforms;
    uniforms.black.value = look.black / 255;
    uniforms.white.value = look.white / 255;
    uniforms.density.value = look.density;
    uniforms.maximum.value = look.maximum;
    shown.handle.changed();
  }

  showButton.addEventListener("click", () => {
    if (shown === null && loading === null) void show();
    else {
      hide();
      note.textContent = "";
      describe();
    }
  });
  hereButton.addEventListener("click", () => void show());
  followInput.addEventListener("change", () => restyle({ ...look, follow: followInput.checked }));
  sideSelect.addEventListener("change", () => restyle({ ...look, side: Number(sideSelect.value) }));
  drawSelect.addEventListener("change", () => restyle({ ...look, maximum: drawSelect.value === "maximum" }));
  for (const [input, event, end] of [
    [blackInput, "input", "black"],
    [blackNumber, "change", "black"],
    [whiteInput, "input", "white"],
    [whiteNumber, "change", "white"],
  ])
    input.addEventListener(event, () => {
      const value = input.valueAsNumber;
      if (!(Number.isInteger(value) && value >= Number(input.min) && value <= Number(input.max))) describe();
      else if (end === "black") restyle({ ...look, black: Math.min(value, look.white - 1) });
      else restyle({ ...look, white: Math.max(value, look.black + 1) });
    });
  densityInput.addEventListener("change", () => {
    if (densityInput.valueAsNumber > 0) restyle({ ...look, density: densityInput.valueAsNumber });
    else describe();
  });

  function volumeChanged(opened) {
    hide();
    reader?.dispose();
    reader = null;
    ct = opened;
    note.textContent = "";
    describe();
  }

  function focusChanged() {
    if (look.follow && (shown !== null || loading !== null)) void show();
  }

  describe();
  return { panel, place: "view", volumeChanged, focusChanged };
}
