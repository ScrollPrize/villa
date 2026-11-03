import * as THREE from "../../vendor/three.module.js";
import { dot_ZYX } from "../math.js";
import { paneBasis } from "../surface/overlay-geometry.js";
import { planeRotation, viewWidth } from "./pose.js";

export const DISC_DIAMETER = 8;
const DISC_OUTLINE = 1.5;
export const EDGE_WIDTH = 2;
const EDGE_RENDER_ORDER = 2100;
const TEXTURE_WIDTH = 1024;
// A culled vertex: outside the clip volume, so its triangle draws nothing.
const CULLED = "vec4(2.0, 2.0, 2.0, 1.0)";

function markTexture(rows, type, channels = 4) {
  const values = TEXTURE_WIDTH * rows * channels;
  const data = type === THREE.FloatType ? new Float32Array(values) : new Uint8Array(values);
  const format = channels === 2 ? THREE.RGFormat : THREE.RGBAFormat;
  const texture = new THREE.DataTexture(data, TEXTURE_WIDTH, rows, format, type);
  texture.colorSpace = THREE.NoColorSpace;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

export class MarkData {
  constructor() {
    this.count = 0;
    this.selected = -1;
    this.hiding = false;
    this.crowding = false;
    this.places = markTexture(1, THREE.FloatType);
    this.colors = markTexture(1, THREE.UnsignedByteType);
    this.neighbours = markTexture(1, THREE.FloatType, 2);
    this.edges = new THREE.InstancedBufferGeometry();
    this.uniforms = {
      markPlaces: { value: this.places },
      markColors: { value: this.colors },
      markNeighbours: { value: this.neighbours },
    };
  }

  set(count, place, colors_N3, edges_E2, selected, neighbours_N2) {
    const rows = Math.max(1, Math.ceil(count / TEXTURE_WIDTH));
    if (rows > this.places.image.height || rows * 4 <= this.places.image.height) {
      const capacity = 2 ** Math.ceil(Math.log2(rows));
      this.places.dispose();
      this.colors.dispose();
      this.neighbours.dispose();
      this.places = markTexture(capacity, THREE.FloatType);
      this.colors = markTexture(capacity, THREE.UnsignedByteType);
      this.neighbours = markTexture(capacity, THREE.FloatType, 2);
      this.uniforms.markPlaces.value = this.places;
      this.uniforms.markColors.value = this.colors;
      this.uniforms.markNeighbours.value = this.neighbours;
    }
    place(this.places.image.data);
    const colors = this.colors.image.data;
    for (let index = 0; index < count; index++) {
      for (let channel = 0; channel < 3; channel++)
        colors[4 * index + channel] = Math.round(255 * colors_N3[3 * index + channel]);
      colors[4 * index + 3] = 255;
    }
    this.places.needsUpdate = true;
    this.colors.needsUpdate = true;
    if (neighbours_N2 !== null) {
      this.neighbours.image.data.set(neighbours_N2);
      this.neighbours.needsUpdate = true;
    }
    this.edges.dispose();
    this.edges = new THREE.InstancedBufferGeometry();
    this.edges.setAttribute(
      "position",
      new THREE.Float32BufferAttribute([0, 1, 0, 0, -1, 0, 1, -1, 0, 0, 1, 0, 1, -1, 0, 1, 1, 0], 3),
    );
    this.edges.setAttribute("ends", new THREE.InstancedBufferAttribute(Uint32Array.from(edges_E2), 2));
    this.edges.instanceCount = edges_E2.length / 2;
    this.count = count;
    this.selected = selected;
    this.hiding = false;
    this.crowding = neighbours_N2 !== null;
  }

  show(shown_N) {
    if (shown_N === null && !this.hiding) return;
    const colors = this.colors.image.data;
    for (let index = 0; index < this.count; index++)
      colors[4 * index + 3] = shown_N === null || shown_N[index] ? 255 : 0;
    this.colors.needsUpdate = true;
    this.hiding = shown_N !== null;
  }

  dispose() {
    this.places.dispose();
    this.colors.dispose();
    this.neighbours.dispose();
    this.edges.dispose();
  }
}

const MARK_DATA = `
  uniform highp sampler2D markPlaces;
  uniform highp sampler2D markColors;
  uniform int markCount;
  uniform int markSelected;
  ivec2 markTexel(int index) { return ivec2(index % ${TEXTURE_WIDTH}, index / ${TEXTURE_WIDTH}); }
  vec4 markPlace(int index) { return texelFetch(markPlaces, markTexel(index), 0); }
  vec4 markColor(int index) { return texelFetch(markColors, markTexel(index), 0); }
  int markInstance(int instance) {
    if (markSelected < 0) return instance;
    return instance == markCount - 1 ? markSelected : instance + int(instance >= markSelected);
  }
`;

function discShader(placement) {
  return `
    ${MARK_DATA}
    ${placement}
    uniform highp sampler2D markNeighbours;
    uniform bool markCrowding;
    uniform float radiusPixels;
    out vec2 vLocal;
    out vec3 vColor;
    bool crowded(int index, vec2 center) {
      if (!markCrowding || index == markSelected) return false;
      vec2 neighbours = texelFetch(markNeighbours, markTexel(index), 0).xy;
      for (int side = 0; side < 2; side++) {
        int other = int(neighbours[side]);
        vec2 pixels;
        if (other >= 0 && markColor(other).a != 0.0 && placed(markPlace(other), pixels) &&
            distance(pixels, center) < 2.0 * radiusPixels)
          return true;
      }
      return false;
    }
    void main() {
      int index = markInstance(gl_InstanceID);
      vec4 color = markColor(index);
      vec2 center;
      if (color.a == 0.0 || !placed(markPlace(index), center) || crowded(index, center)) {
        gl_Position = ${CULLED};
        return;
      }
      vLocal = position.xy;
      vColor = index == markSelected ? vec3(1.0) : color.rgb;
      gl_Position = clipOf(center + position.xy * radiusPixels);
    }
  `;
}

const DISC_FRAGMENT = `
  uniform float outline;
  in vec2 vLocal;
  in vec3 vColor;
  out vec4 outColor;
  void main() {
    float radius = length(vLocal);
    if (radius > 1.0) discard;
    outColor = vec4(radius > 1.0 - outline ? vec3(0.0) : vColor, 1.0);
  }
`;

function edgeShader(placement) {
  return `
    ${MARK_DATA}
    ${placement}
    in uvec2 ends;
    uniform float halfWidthPixels;
    out vec2 vLocal;
    out float vLength;
    out vec3 vColor;
    void main() {
      vec4 colorA = markColor(int(ends.x));
      vec2 from;
      vec2 to;
      if (colorA.a == 0.0 || markColor(int(ends.y)).a == 0.0 ||
          !placedEdge(markPlace(int(ends.x)), markPlace(int(ends.y)), from, to)) {
        gl_Position = ${CULLED};
        return;
      }
      vec2 direction = to - from;
      vLength = length(direction);
      vec2 along = vLength > 0.0 ? direction / vLength : vec2(1.0, 0.0);
      vLocal = vec2(position.x * (vLength + 2.0 * halfWidthPixels) - halfWidthPixels, position.y * halfWidthPixels);
      vColor = colorA.rgb;
      gl_Position = clipOf(from + along * vLocal.x + vec2(-along.y, along.x) * vLocal.y);
    }
  `;
}

const EDGE_FRAGMENT = `
  uniform float halfWidthPixels;
  in vec2 vLocal;
  in float vLength;
  in vec3 vColor;
  out vec4 outColor;
  void main() {
    float beyond = max(-vLocal.x, vLocal.x - vLength);
    if (beyond > 0.0 && beyond * beyond + vLocal.y * vLocal.y > halfWidthPixels * halfWidthPixels) discard;
    outColor = vec4(vColor, 1.0);
  }
`;

export function edgeWidth({ width, voxels }, pixelsPerVoxel, pixelRatio) {
  if (voxels === undefined) return width * pixelRatio;
  return Math.min(Math.max(voxels * pixelsPerVoxel, width[0] * pixelRatio), width[1] * pixelRatio);
}

function discGeometry() {
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0], 3),
  );
  return geometry;
}

export function markMeshes(data, placement, uniforms, diameter, pixelRatio) {
  const shared = {
    ...data.uniforms,
    ...uniforms,
    markCount: { value: 0 },
    markSelected: { value: -1 },
    markCrowding: { value: false },
  };
  const material = (vertexShader, fragmentShader, own) =>
    new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: { ...shared, ...own },
      vertexShader,
      fragmentShader,
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
  const discs = new THREE.Mesh(
    discGeometry(),
    material(discShader(placement), DISC_FRAGMENT, {
      radiusPixels: { value: (diameter / 2) * pixelRatio },
      outline: { value: (2 * DISC_OUTLINE) / diameter },
    }),
  );
  const halfWidth = { value: 0 };
  const edges = new THREE.Mesh(
    data.edges,
    material(edgeShader(placement), EDGE_FRAGMENT, { halfWidthPixels: halfWidth }),
  );
  edges.renderOrder = EDGE_RENDER_ORDER;
  discs.renderOrder = EDGE_RENDER_ORDER + 1;
  for (const mesh of [edges, discs]) {
    mesh.frustumCulled = false;
    mesh.visible = false;
  }
  return { discs, edges, shared, halfWidth };
}

export function updateMarkMeshes(meshes, data, diameter, drawn) {
  meshes.shared.markCount.value = data.count;
  meshes.shared.markSelected.value = data.selected;
  meshes.shared.markCrowding.value = data.crowding;
  meshes.discs.geometry.instanceCount = data.count;
  meshes.discs.visible = drawn && diameter > 0 && data.count > 0;
  meshes.edges.geometry = data.edges;
  meshes.edges.visible = drawn && data.edges.instanceCount > 0;
}

// from that origin, so the numbers the GPU takes stay small.
const SLICE_PLACEMENT = `
  uniform vec3 paneNormal;
  uniform vec3 paneRight;
  uniform vec3 paneUp;
  uniform vec2 paneSize;
  uniform float pixelsPerVoxel;
  uniform float paneIso;
  uniform vec2 paneCenter;
  uniform float marksDepth;
  float across(vec3 place) { return dot(place, paneNormal) - paneIso; }
  vec2 pixelsOf(vec3 place) { return (vec2(dot(place, paneRight), dot(place, paneUp)) - paneCenter) * pixelsPerVoxel; }
  vec4 clipOf(vec2 pixels) { return vec4(2.0 * pixels / paneSize, 0.0, 1.0); }
  bool placed(vec4 place, out vec2 pixels) {
    pixels = pixelsOf(place.xyz);
    return abs(across(place.xyz)) <= marksDepth;
  }
  bool placedEdge(vec4 a, vec4 b, out vec2 from, out vec2 to) {
    float acrossA = across(a.xyz);
    float change = across(b.xyz) - acrossA;
    float low = 0.0;
    float high = 1.0;
    if (change == 0.0 && abs(acrossA) > marksDepth) return false;
    if (change != 0.0) {
      float first = (-marksDepth - acrossA) / change;
      float last = (marksDepth - acrossA) / change;
      low = max(0.0, min(first, last));
      high = min(1.0, max(first, last));
      if (low >= high) return false;
    }
    from = pixelsOf(mix(a.xyz, b.xyz, low));
    to = pixelsOf(mix(a.xyz, b.xyz, high));
    return true;
  }
`;

export function createMarks({ state, view }) {
  const drawn = [];
  const paneUniforms = new Map();

  function uniformsFor(pane) {
    if (!paneUniforms.has(pane))
      paneUniforms.set(pane, {
        paneNormal: { value: new THREE.Vector3() },
        paneRight: { value: new THREE.Vector3() },
        paneUp: { value: new THREE.Vector3() },
        paneSize: { value: new THREE.Vector2(1, 1) },
        pixelsPerVoxel: { value: 1 },
      });
    return paneUniforms.get(pane);
  }

  function paneMeshes(entry, pane) {
    const own = { paneIso: { value: 0 }, paneCenter: { value: new THREE.Vector2() }, marksDepth: { value: 0 } };
    const meshes = markMeshes(
      entry.data,
      SLICE_PLACEMENT,
      { ...uniformsFor(pane), ...own },
      entry.diameter,
      view.pixelRatio,
    );
    for (const mesh of [meshes.edges, meshes.discs]) view.scene.tileGroups.get(pane).add(mesh);
    return { ...meshes, own };
  }

  function set(entry, marks) {
    const { points_zyx_N3, colors_N3, edges_E2, selected } = marks;
    const neighbours_N2 = marks.neighbours_N2 ?? null;
    const count = points_zyx_N3.length / 3;
    const [z0, y0, x0] = count === 0 ? [0, 0, 0] : [points_zyx_N3[0], points_zyx_N3[1], points_zyx_N3[2]];
    const place = (places_N4) => {
      for (let index = 0; index < count; index++) {
        places_N4[4 * index] = points_zyx_N3[3 * index] - z0;
        places_N4[4 * index + 1] = points_zyx_N3[3 * index + 1] - y0;
        places_N4[4 * index + 2] = points_zyx_N3[3 * index + 2] - x0;
      }
    };
    entry.data.set(count, place, colors_N3, edges_E2, selected, neighbours_N2);
    Object.assign(entry, { marks, origin_ZYX: [z0, y0, x0] });
    for (const meshes of entry.panes.values()) updateMarkMeshes(meshes, entry.data, entry.diameter, true);
  }

  function addMarks({ diameter = DISC_DIAMETER, width = EDGE_WIDTH, voxels } = {}) {
    const entry = { marks: null, origin_ZYX: [0, 0, 0], data: new MarkData(), diameter, width, voxels };
    entry.panes = new Map([...view.scene.tileGroups.keys()].map((pane) => [pane, paneMeshes(entry, pane)]));
    drawn.push(entry);
    return {
      set: (marks) => set(entry, marks),
      show: (shown_N) => entry.data.show(shown_N),
      remove: () => {
        drawn.splice(drawn.indexOf(entry), 1);
        for (const { discs, edges } of entry.panes.values())
          for (const mesh of [discs, edges]) {
            mesh.removeFromParent();
            mesh.material.dispose();
          }
        for (const { discs } of entry.panes.values()) discs.geometry.dispose();
        entry.data.dispose();
      },
    };
  }

  function updateMarks(planes) {
    if (state.volume === null) return;
    const span = viewWidth(state);
    for (const pane of planes) {
      const basis = paneBasis(pane, planeRotation(state, pane));
      const side = view.getOrthogonalPanelPixelSize(pane);
      const aspect = view.getOrthogonalPanelAspect(pane);
      const uniforms = uniformsFor(pane);
      uniforms.paneNormal.value.set(...basis.normal_ZYX);
      uniforms.paneRight.value.set(...basis.right_ZYX);
      uniforms.paneUp.value.set(...basis.up_ZYX);
      uniforms.paneSize.value.set(side * Math.max(1, aspect), side * Math.max(1, 1 / aspect));
      uniforms.pixelsPerVoxel.value = side / span;
      for (const entry of drawn) {
        if (entry.marks === null) continue;
        const { own, halfWidth } = entry.panes.get(pane);
        halfWidth.value = edgeWidth(entry, uniforms.pixelsPerVoxel.value, view.pixelRatio) / 2;
        const center_ZYX = state.center_ZYX.map((value, axis) => value - entry.origin_ZYX[axis]);
        own.paneIso.value = dot_ZYX(center_ZYX, basis.normal_ZYX);
        own.paneCenter.value.set(dot_ZYX(center_ZYX, basis.right_ZYX), dot_ZYX(center_ZYX, basis.up_ZYX));
        own.marksDepth.value = entry.marks.depth;
      }
    }
  }

  return { addMarks, updateMarks };
}
