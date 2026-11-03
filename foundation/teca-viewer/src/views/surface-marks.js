import * as THREE from "../../vendor/three.module.js";
import { surfaceCameraShader } from "../surface/camera.js";
import { DISC_DIAMETER, EDGE_WIDTH, edgeWidth, MarkData, markMeshes, updateMarkMeshes } from "./marks.js";

const VIEW_UNIFORMS = [
  "volumetricAzimuth",
  "volumetricTilt",
  "volumetricPerspective",
  "volumetricMaximumDepth",
  "volumetricRelief",
  "volumetricDirection",
  "normalOffset",
  "surfaceScale_UV",
  "surfaceDisplaySign_UV",
  "surfaceCenter_UV",
  "surfaceVisibleSpan_UV",
];

// A point's place is its grid cell, column then row.
const SURFACE_PLACEMENT = `
  uniform float volumetricAzimuth;
  uniform float volumetricTilt;
  uniform float volumetricPerspective;
  uniform float volumetricMaximumDepth;
  uniform float volumetricRelief;
  uniform float volumetricDirection;
  uniform float normalOffset;
  ${surfaceCameraShader()}
  uniform vec2 surfaceCenter_UV;
  uniform vec2 surfaceVisibleSpan_UV;
  uniform vec2 paneSize;
  vec4 clipOf(vec2 pixels) { return vec4(2.0 * pixels / paneSize, 0.0, 1.0); }
  vec2 pixelsOf(vec2 cell) {
    float depth = -normalOffset * volumetricDirection * volumetricRelief;
    vec2 screen_UV = surfaceCameraPointToScreen(cell - surfaceCenter_UV, depth, surfaceVisibleSpan_UV);
    return screen_UV / surfaceVisibleSpan_UV * paneSize;
  }
  bool placed(vec4 place, out vec2 pixels) {
    pixels = pixelsOf(place.xy);
    return true;
  }
  bool placedEdge(vec4 a, vec4 b, out vec2 from, out vec2 to) {
    from = pixelsOf(a.xy);
    to = pixelsOf(b.xy);
    return true;
  }
`;

export class SurfaceMarks {
  constructor(pixelRatio) {
    this.pixelRatio = pixelRatio;
    this.scene = new THREE.Scene();
    this.edges = new THREE.Group();
    this.discs = new THREE.Group();
    this.scene.add(this.edges, this.discs);
    this.camera = new THREE.OrthographicCamera();
    this.uniforms = { paneSize: { value: new THREE.Vector2(1, 1) } };
    for (const name of VIEW_UNIFORMS) this.uniforms[name] = { value: 0 };
    this.viewport = new THREE.Vector4();
    this.entries = [];
  }

  add({ diameter = DISC_DIAMETER, width = EDGE_WIDTH, voxels }) {
    const entry = { surface: null, data: new MarkData(), diameter, width, voxels };
    entry.meshes = markMeshes(entry.data, SURFACE_PLACEMENT, this.uniforms, diameter, this.pixelRatio);
    this.edges.add(entry.meshes.edges);
    this.discs.add(entry.meshes.discs);
    this.entries.push(entry);
    return {
      set: ({ surface, cells_N2, colors_N3, edges_E2, selected, neighbours_N2 }) => {
        const count = cells_N2.length / 2;
        const place = (places_N4) => {
          for (let index = 0; index < count; index++) {
            places_N4[4 * index] = cells_N2[2 * index + 1];
            places_N4[4 * index + 1] = cells_N2[2 * index];
          }
        };
        entry.surface = surface;
        entry.data.set(count, place, colors_N3, edges_E2, selected, neighbours_N2 ?? null);
      },
      show: (shown_N) => entry.data.show(shown_N),
      remove: () => {
        this.entries.splice(this.entries.indexOf(entry), 1);
        for (const mesh of [entry.meshes.edges, entry.meshes.discs]) {
          mesh.removeFromParent();
          mesh.material.dispose();
        }
        entry.meshes.discs.geometry.dispose();
        entry.data.dispose();
      },
    };
  }

  render(renderer, uniforms, path) {
    for (const entry of this.entries)
      updateMarkMeshes(entry.meshes, entry.data, entry.diameter, entry.surface === path);
    for (const name of VIEW_UNIFORMS) this.uniforms[name].value = uniforms[name].value;
    renderer.getCurrentViewport(this.viewport);
    this.uniforms.paneSize.value.set(this.viewport.z, this.viewport.w);
    const pixelsPerVoxel =
      (this.viewport.z * uniforms.surfaceScale_UV.value.x) / uniforms.surfaceVisibleSpan_UV.value.x;
    for (const entry of this.entries)
      entry.meshes.halfWidth.value = edgeWidth(entry, pixelsPerVoxel, this.pixelRatio) / 2;
    renderer.render(this.scene, this.camera);
  }
}
