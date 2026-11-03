import * as THREE from "../../vendor/three.module.js";
import { ORTHOGONAL_PLANES } from "../config.js";
import { atlasFormatForDtype } from "../gpu/atlas-format.js";
import { FilterChain } from "../gpu/filter-chain.js";
import { createSlicedPlaneMaterial } from "../gpu/sliced-plane-material.js";

// The CT slices draw at this render order, and each layer after them in the order it was added; the Surface's layers
// count from 1.
export const CT_RENDER_ORDER = 1100;

export class OrthogonalScene {
  constructor() {
    this.flatScene = new THREE.Scene();
    this.flatCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    this.flatCamera.position.set(0, 0, 2);
    this.tileGroups = new Map();
    for (const plane of ORTHOGONAL_PLANES) {
      const group = new THREE.Group();
      group.visible = false;
      group.userData.plane = plane;
      this.tileGroups.set(plane, group);
      this.flatScene.add(group);
    }
    this.displaySettings = { black: 0, white: 1, gamma: 1 };
    this.filterPasses = [];
    this.filterReach = 0;
    this.filterChains = new Map(ORTHOGONAL_PLANES.map((plane) => [plane, new FilterChain()]));
    const precompileLevel = {
      atlas: { texture: null, width: 1, formatDescriptor: atlasFormatForDtype("|u1") },
      pageTable: { texture: null },
      scale: {
        shape_ZYX: [1, 1, 1],
        chunks_ZYX: [1, 1, 1],
        downsample_ZYX: [1, 1, 1],
        offset_ZYX: [0, 0, 0],
        fillValue: 0,
      },
    };
    this.slicedPlanePrecompileMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      createSlicedPlaneMaterial({
        levels: [precompileLevel],
        sourceBounds_ZYX: {
          minimum_ZYX: [0, 0, 0],
          maximum_ZYX: [1, 1, 1],
        },
        displayEncoding: { kind: "identity_uint8" },
        displaySettings: this.displaySettings,
        style: null,
      }),
    );
  }

  precompileSlicedPlaneMaterial(renderer) {
    return renderer.compileAsync(this.slicedPlanePrecompileMesh, this.flatCamera);
  }

  setDisplaySettings({ black, white, gamma }) {
    this.displaySettings = { black, white, gamma };
  }

  filterCT(passes, reach) {
    this.filterPasses = passes;
    this.filterReach = reach;
    if (passes.length === 0) for (const chain of this.filterChains.values()) chain.release();
  }

  renderPane(renderer, plane, panelWidth, panelHeight) {
    renderer.setClearColor(0x000000, 1);
    renderer.clear(true, true, true);
    for (const [name, group] of this.tileGroups) {
      group.visible = name === plane;
    }
    const aspect = panelWidth / panelHeight;
    this.flatCamera.left = -Math.max(1, aspect);
    this.flatCamera.right = Math.max(1, aspect);
    this.flatCamera.top = Math.max(1, 1 / aspect);
    this.flatCamera.bottom = -Math.max(1, 1 / aspect);
    this.flatCamera.updateProjectionMatrix();
    const slice = this.tileGroups
      .get(plane)
      .children.find((child) => child.renderOrder === CT_RENDER_ORDER && child.visible);
    const chain = this.filterChains.get(plane);
    const span = slice?.material.uniforms.visibleSpan_UV.value.x;
    if (
      this.filterPasses.length === 0 ||
      slice === undefined ||
      slice.material.uniforms.drawn.value === 0 ||
      !chain.draw(renderer, this.filterPasses, this.filterReach, slice, this.flatCamera, span)
    ) {
      renderer.render(this.flatScene, this.flatCamera);
      return;
    }
    slice.visible = false;
    renderer.render(this.flatScene, this.flatCamera);
    slice.visible = true;
  }
}
