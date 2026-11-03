// Only a box about the focus is drawn, as a surface wraps the scroll and hides the planes from outside it. The mesh is
// shaded by its normal in view, since the pane lights nothing but ambient.

import * as THREE from "../../vendor/three.module.js";

// Triangles of one surface's mesh at most; each surface shown has its own. Paris 4's segment, 1598 × 2103 cells, takes
// every second row and column: 1.52 M triangles, 17.4 MiB of index, and positions and normals for those of its 840 k
// vertices on the surface, 19.2 MiB were all on it.
const TRIANGLE_LIMIT = 2_000_000;
// Triangles at most of the coarser mesh of the same surface a click in the 3D pane is found on. three.js tests each
// triangle of a ray's mesh, 70 ms a click for Paris 4's 1.52 M; its 0.24 M at this limit take 11 ms and find a point
// within 10 voxels of the drawn mesh's.
const PICK_TRIANGLE_LIMIT = 400_000;
const DEFAULT_REACH = 256;
const MAX_REACH = 100_000;

export function meshStep(grid_HW, limit) {
  for (let step = 1; ; step++) {
    const [rows, columns] = grid_HW.map((cells) => Math.ceil(cells / step));
    if (2 * Math.max(0, rows - 1) * Math.max(0, columns - 1) <= limit) return step;
  }
}

// A cell is off the surface where its three values are all -1 or one is not finite.
export function meshArrays({ grid_HW, points_zyx_HW3 }, step) {
  const width = grid_HW[1];
  const [rows, columns] = grid_HW.map((cells) => Math.ceil(cells / step));
  const vertex_HW = new Int32Array(rows * columns).fill(-1);
  let vertices = 0;
  for (let row = 0; row < rows; row++)
    for (let column = 0; column < columns; column++) {
      const cell = 3 * (row * step * width + column * step);
      const z = points_zyx_HW3[cell];
      const y = points_zyx_HW3[cell + 1];
      const x = points_zyx_HW3[cell + 2];
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && (x !== -1 || y !== -1 || z !== -1))
        vertex_HW[row * columns + column] = vertices++;
    }
  const positions_xyz_N3 = new Float32Array(3 * vertices);
  for (let taken = 0; taken < rows * columns; taken++) {
    const vertex = vertex_HW[taken];
    if (vertex === -1) continue;
    const cell = 3 * (Math.floor(taken / columns) * step * width + (taken % columns) * step);
    for (let axis = 0; axis < 3; axis++) positions_xyz_N3[3 * vertex + axis] = points_zyx_HW3[cell + 2 - axis];
  }
  const whole = (first) =>
    vertex_HW[first] !== -1 &&
    vertex_HW[first + 1] !== -1 &&
    vertex_HW[first + columns] !== -1 &&
    vertex_HW[first + columns + 1] !== -1;
  let quads = 0;
  for (let row = 0; row + 1 < rows; row++)
    for (let column = 0; column + 1 < columns; column++) if (whole(row * columns + column)) quads++;
  const triangles_T3 = new Uint32Array(6 * quads);
  let written = 0;
  for (let row = 0; row + 1 < rows; row++)
    for (let column = 0; column + 1 < columns; column++) {
      const first = row * columns + column;
      if (!whole(first)) continue;
      const next = vertex_HW[first + 1];
      const below = vertex_HW[first + columns];
      triangles_T3[written] = vertex_HW[first];
      triangles_T3[written + 1] = below;
      triangles_T3[written + 2] = next;
      triangles_T3[written + 3] = next;
      triangles_T3[written + 4] = below;
      triangles_T3[written + 5] = vertex_HW[first + columns + 1];
      written += 6;
    }
  return { positions_xyz_N3, triangles_T3 };
}

function geometryOf(surface, limit) {
  const step = meshStep(surface.grid_HW, limit);
  const { positions_xyz_N3, triangles_T3 } = meshArrays(surface, step);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions_xyz_N3, 3));
  geometry.setIndex(new THREE.BufferAttribute(triangles_T3, 1));
  return { geometry, step };
}

function boxedMaterial(box) {
  const material = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide });
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, { boxLow: box.low, boxHigh: box.high });
    shader.vertexShader = `varying vec3 vBoxed;\n${shader.vertexShader}`.replace(
      "#include <begin_vertex>",
      "#include <begin_vertex>\nvBoxed = transformed;",
    );
    shader.fragmentShader =
      `uniform vec3 boxLow;\nuniform vec3 boxHigh;\nvarying vec3 vBoxed;\n${shader.fragmentShader}`.replace(
        "void main() {",
        "void main() {\n  if (any(lessThan(vBoxed, boxLow)) || any(greaterThan(vBoxed, boxHigh))) discard;",
      );
  };
  return material;
}

export default function createSurfaceMeshes(viewer, entry) {
  const panel = document.createElement("section");
  panel.id = "surface-meshes-section";
  panel.className = "card source-dependent";
  panel.setAttribute("aria-label", "Surface meshes");
  panel.innerHTML = `<h2>Surface meshes</h2>
    <label class="checkbox-row"><input id="surface-meshes-show" type="checkbox" /> Show surfaces in 3D</label>
    <label><span class="control-name">Around focus</span>
      <input id="surface-meshes-reach" type="number" min="0" max="${MAX_REACH}" step="16" value="${DEFAULT_REACH}"
        aria-label="World units from the focus the meshes are drawn to, 0 for all of them" />
    </label>
    <p id="surface-meshes-note" class="readout-line" role="status"></p>`;
  const showInput = panel.querySelector("#surface-meshes-show");
  const reachInput = panel.querySelector("#surface-meshes-reach");
  const note = panel.querySelector("#surface-meshes-note");

  const saved = entry.saved;
  const valid =
    saved === null ||
    (typeof saved === "object" &&
      Object.keys(saved).sort().join() === "reach,show" &&
      typeof saved.show === "boolean" &&
      typeof saved.reach === "number" &&
      saved.reach >= 0 &&
      saved.reach <= MAX_REACH);
  if (!valid) viewer.setStatus(`The link's surface meshes setting ${JSON.stringify(saved)} was not used.`);
  const kept = valid && saved !== null ? saved : { show: false, reach: DEFAULT_REACH };
  let show = kept.show;
  let reach = kept.reach;
  reachInput.value = String(reach);
  const box = { low: { value: new THREE.Vector3() }, high: { value: new THREE.Vector3() } };
  const meshes = new Map();

  function describe() {
    showInput.checked = show;
    const held = [...meshes.values()];
    const drawn = held.filter(({ mesh }) => mesh !== null);
    const triangles = drawn.reduce((sum, { mesh }) => sum + mesh.geometry.index.count / 3, 0);
    const steps = [...new Set(drawn.map(({ step }) => step))].join(" and ");
    const reading = held.filter(({ controller }) => controller !== null).length;
    const failed = held.filter((one) => one.failed).length;
    const parts = [
      `${drawn.length} ${drawn.length === 1 ? "surface" : "surfaces"}, ${(triangles / 1e6).toFixed(2)} M triangles`,
      drawn.length > 0 ? `grid step ${steps}` : "",
      reading > 0 ? `reading ${reading}…` : "",
      failed > 0 ? `${failed} failed, the status bar says why` : "",
    ];
    note.textContent = show ? parts.filter((part) => part !== "").join(", ") : "";
  }

  function drop(url) {
    const held = meshes.get(url);
    meshes.delete(url);
    held.controller?.abort();
    if (held.mesh === null) return;
    held.handle.remove();
    held.mesh.geometry.dispose();
    held.picker.geometry.dispose();
    held.mesh.material.dispose();
  }

  async function read(url) {
    const controller = new AbortController();
    const held = { controller, handle: null, mesh: null, picker: null, step: 0, failed: false };
    meshes.set(url, held);
    try {
      const surface = await viewer.readSurface(url, controller.signal);
      if (controller.signal.aborted) return;
      const { geometry, step } = geometryOf(surface, TRIANGLE_LIMIT);
      geometry.computeVertexNormals();
      const material = boxedMaterial(box);
      const mesh = new THREE.Mesh(geometry, material);
      const picker = new THREE.Mesh(geometryOf(surface, PICK_TRIANGLE_LIMIT).geometry, material);
      mesh.raycast = (raycaster, hits) => {
        picker.matrixWorld.copy(mesh.matrixWorld);
        const found = [];
        picker.raycast(raycaster, found);
        for (const hit of found) {
          const local = mesh.worldToLocal(hit.point.clone());
          if (local.clone().clamp(box.low.value, box.high.value).equals(local)) hits.push({ ...hit, object: mesh });
        }
      };
      Object.assign(held, { handle: viewer.add3d(mesh), mesh, picker, step });
    } catch (error) {
      if (controller.signal.aborted) return;
      console.error(error);
      viewer.setStatus(`Surface meshes: ${url}: ${error.message}`);
      held.failed = true;
    } finally {
      if (!controller.signal.aborted) {
        held.controller = null;
        describe();
      }
    }
  }

  function sync() {
    const urls = show ? viewer.surfaces() : [];
    for (const url of [...meshes.keys()]) if (!urls.includes(url)) drop(url);
    for (const url of urls) if (!meshes.has(url)) void read(url);
    describe();
  }

  function place() {
    const focus_XYZ = viewer.focusPoint().reverse();
    const half = reach === 0 ? Infinity : reach;
    box.low.value.set(...focus_XYZ.map((value) => value - half));
    box.high.value.set(...focus_XYZ.map((value) => value + half));
    for (const { handle } of meshes.values()) handle?.changed();
  }

  showInput.addEventListener("change", () => {
    show = showInput.checked;
    entry.save({ show, reach });
    sync();
  });

  reachInput.addEventListener("change", () => {
    if (!(reachInput.valueAsNumber >= 0 && reachInput.valueAsNumber <= MAX_REACH)) reachInput.value = String(reach);
    reach = reachInput.valueAsNumber;
    entry.save({ show, reach });
    place();
  });

  function volumeChanged() {
    for (const url of [...meshes.keys()]) drop(url);
    place();
    sync();
  }

  describe();
  return { panel, place: "view", volumeChanged, surfacesChanged: sync, focusChanged: place };
}
