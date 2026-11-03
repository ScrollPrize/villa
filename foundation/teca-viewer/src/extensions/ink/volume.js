import * as THREE from "../../../vendor/three.module.js";

// Workers that make the band's chunks, each holding the surface and the image, 65 MB. After a jump across the checks'
// segment the panes' band at 32 steps is complete 1.09 s on with one, 0.70 with two and 0.52 with four; at 4 steps
// 0.62, 0.45 and 0.42.
const WORKERS = 2;
const MESH_REACH = 256;
// World units the mesh is drawn nearer the eye than it lies: surface-meshes' mesh of every second corner lies up to
// 16.3 voxels off the grid's corners on the checks' segment, and 99.9% of them within 6.3.
const LIFT = 16;

const VERTEX = `
  uniform float lift;
  out vec2 vUv;
  out vec3 vWorld;
  void main() {
    vUv = uv;
    vWorld = position;
    vec4 view = modelViewMatrix * vec4(position, 1.0);
    float distance = length(view.xyz);
    // Along its ray toward the eye, so it covers the pixels it would, and never past half the way there.
    view.xyz *= max(distance - lift, 0.5 * distance) / distance;
    gl_Position = projectionMatrix * view;
  }
`;
const FRAGMENT = `
  precision highp float;
  uniform sampler2D image;
  uniform vec3 inkColor;
  uniform float inkOpacity;
  uniform vec2 inkRange;
  uniform vec3 boxLow;
  uniform vec3 boxHigh;
  in vec2 vUv;
  in vec3 vWorld;
  out vec4 outColor;
  void main() {
    if (any(lessThan(vWorld, boxLow)) || any(greaterThan(vWorld, boxHigh))) discard;
    float value = texture(image, vUv).r;
    float grey = value * 255.0;
    if (grey < inkRange.x - 0.5 || grey > inkRange.y + 0.5) discard;
    outColor = vec4(inkColor, inkOpacity * value);
  }
`;

export class VolumeInk {
  constructor(viewer, url, pixels, style, steps, failed) {
    this.viewer = viewer;
    this.url = url;
    this.style = style;
    this.failed = failed;
    this.texture = new THREE.DataTexture(
      pixels.data_HW,
      pixels.shape_HW[1],
      pixels.shape_HW[0],
      THREE.RedFormat,
      THREE.UnsignedByteType,
    );
    this.texture.unpackAlignment = 1;
    this.texture.minFilter = THREE.LinearMipmapLinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.generateMipmaps = true;
    this.texture.needsUpdate = true;
    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: {
        lift: { value: LIFT },
        image: { value: this.texture },
        inkColor: { value: new THREE.Color(...style.color_RGB) },
        inkOpacity: { value: style.opacity },
        inkRange: { value: new THREE.Vector2(...style.range) },
        boxLow: { value: new THREE.Vector3() },
        boxHigh: { value: new THREE.Vector3() },
      },
      vertexShader: VERTEX,
      fragmentShader: FRAGMENT,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), this.material);
    this.mesh.raycast = () => {};
    this.drawn = viewer.add3d(this.mesh);
    this.workers = Array.from({ length: WORKERS }, () => ({
      worker: new Worker(new URL("./band.worker.js", import.meta.url), { type: "module" }),
      job: null,
    }));
    for (const slot of this.workers) {
      slot.worker.onmessage = ({ data }) => this.answered(slot, data, null);
      slot.worker.onerror = (event) =>
        this.answered(slot, null, new Error(`The ink band's worker failed: ${event.message}`));
    }
    this.waiting = [];
    this.ready = false;
    this.controller = new AbortController();
    this.meshing = false;
    this.again = false;
    this.removed = false;
    this.layer = null;
    this.setSteps(steps);
    void this.update();
    void this.start(url, pixels);
  }

  async start(url, pixels) {
    try {
      const { grid_HW, points_zyx_HW3 } = await this.viewer.readSurface(url, this.controller.signal);
      if (this.removed) return;
      const message = { surface: { grid_HW, points_zyx_HW3 }, image: pixels };
      for (const [index, { worker }] of this.workers.entries())
        worker.postMessage(message, index === this.workers.length - 1 ? [points_zyx_HW3.buffer] : []);
      this.ready = true;
      this.pump();
    } catch (error) {
      if (error.name !== "AbortError") this.failed(error);
    }
  }

  ask(message, rank, signal) {
    return new Promise((resolve, reject) => {
      const job = { message, rank, settled: false };
      const abort = () => {
        this.waiting = this.waiting.filter((waiting) => waiting !== job);
        job.settle(null, signal.reason);
      };
      job.settle = (data, error) => {
        if (job.settled) return;
        job.settled = true;
        signal.removeEventListener("abort", abort);
        if (error === null) resolve(data);
        else reject(error);
      };
      signal.addEventListener("abort", abort);
      this.waiting.push(job);
      this.pump();
    });
  }

  pump() {
    if (!this.ready) return;
    for (const slot of this.workers) {
      if (slot.job !== null || this.waiting.length === 0) continue;
      let next = 0;
      for (let index = 1; index < this.waiting.length; index++)
        if (this.waiting[index].rank > this.waiting[next].rank) next = index;
      slot.job = this.waiting.splice(next, 1)[0];
      slot.worker.postMessage(slot.job.message);
    }
  }

  answered(slot, data, error) {
    const { job } = slot;
    slot.job = null;
    if (job === null) this.failed(error);
    else job.settle(data, error);
    this.pump();
  }

  async update() {
    this.again = true;
    if (this.meshing) return;
    this.meshing = true;
    try {
      while (this.again && !this.removed) {
        this.again = false;
        const low_ZYX = this.viewer.focusPoint().map((focus) => focus - MESH_REACH);
        const high_ZYX = low_ZYX.map((low) => low + 2 * MESH_REACH);
        const mesh = await this.ask({ low_ZYX, high_ZYX }, Infinity, this.controller.signal);
        this.draw(mesh, low_ZYX, high_ZYX);
      }
    } catch (error) {
      if (error.name !== "AbortError") this.failed(error);
    } finally {
      this.meshing = false;
    }
  }

  draw({ positions_xyz_N3, uvs_N2, triangles_T3 }, low_ZYX, high_ZYX) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions_xyz_N3, 3));
    geometry.setAttribute("uv", new THREE.BufferAttribute(uvs_N2, 2));
    geometry.setIndex(new THREE.BufferAttribute(triangles_T3, 1));
    this.mesh.geometry.dispose();
    this.mesh.geometry = geometry;
    this.material.uniforms.boxLow.value.set(...[...low_ZYX].reverse());
    this.material.uniforms.boxHigh.value.set(...[...high_ZYX].reverse());
    this.drawn.changed();
  }

  setStyle(style) {
    this.style = style;
    this.layer.setStyle(style);
    this.material.uniforms.inkColor.value.setRGB(...style.color_RGB);
    this.material.uniforms.inkOpacity.value = style.opacity;
    this.material.uniforms.inkRange.value.set(...style.range);
    this.drawn.changed();
  }

  setSteps(steps) {
    this.layer?.remove();
    const volume = this.viewer.madeVolume(`ink ${this.url} ${steps}`, (level, chunk_ZYX, signal) => {
      const { shape_ZYX, chunks_ZYX, downsample_ZYX, offset_ZYX } = volume.scales[level];
      const scale = { shape_ZYX, chunks_ZYX, downsample_ZYX, offset_ZYX };
      return this.ask({ scale, chunk_ZYX, steps }, level, signal);
    });
    this.layer = this.viewer.addLayer({ slices: volume, surface: null }, this.style);
  }

  dispose() {
    this.removed = true;
    this.controller.abort();
    for (const { worker } of this.workers) worker.terminate();
    this.drawn.remove();
    this.mesh.geometry.dispose();
    this.material.dispose();
    this.texture.dispose();
  }

  remove() {
    this.layer.remove();
    this.dispose();
  }
}
