import * as THREE from "../../vendor/three.module.js";
import { BrickPool } from "./brick-pool.js";
import { GpuTimer } from "./gpu-timer.js";
import { FrameUploadBudget } from "./upload-budget.js";

export class ViewerRenderer {
  constructor(container, pixelRatio, byteLimit, onUploadError) {
    this.container = container;
    this.canvas = document.createElement("canvas");
    this.canvas.className = "viewer-compositor";
    this.canvas.setAttribute("aria-hidden", "true");
    container.prepend(this.canvas);
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false, alpha: true });
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.setPixelRatio(pixelRatio);
    this.renderer.autoClear = false;
    const gl = this.renderer.getContext();
    this.textureLimits = {
      maxTextureSize: this.renderer.capabilities.maxTextureSize,
      maxLayers: gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS),
    };
    this.uploadBudget = new FrameUploadBudget(onUploadError);
    this.gpuTimer = new GpuTimer(gl);
    this.brickPool = new BrickPool(this.renderer);
    this.outputs = new Set();
    this.allocations = new Set();
    this.byteLimit = byteLimit;
    this.pending = false;
    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 2);
    this.camera.position.z = 1;
    this.geometry = new THREE.PlaneGeometry(2, 2);
    this.material = new THREE.MeshBasicMaterial({ depthTest: false, depthWrite: false, toneMapped: false });
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    this.scene.add(this.mesh);
    this.layout = null;
    const invalidateLayout = () => {
      this.layout = null;
      this.requestPresent();
    };
    this.layoutObserver = new ResizeObserver(invalidateLayout);
    this.layoutObserver.observe(container);
    document.addEventListener("scroll", invalidateLayout, { capture: true, passive: true });
  }

  attach(output) {
    this.outputs.add(output);
    this.layout = null;
    this.layoutObserver.observe(output.canvas);
  }

  addAllocation(allocation) {
    this.allocations.add(allocation);
  }

  removeAllocation(allocation) {
    this.allocations.delete(allocation);
  }

  resize() {
    this.layout = null;
    const { clientWidth: width, clientHeight: height } = this.container;
    const ratio = this.renderer.getPixelRatio();
    if (this.canvas.width !== Math.floor(width * ratio) || this.canvas.height !== Math.floor(height * ratio))
      this.renderer.setSize(Math.max(1, width), Math.max(1, height), false);
    this.requestPresent();
  }

  requestPresent() {
    if (this.pending) return;
    this.pending = true;
    queueMicrotask(() => {
      if (this.pending) this.present();
    });
  }

  present() {
    this.pending = false;
    const renderer = this.renderer;
    const target = renderer.getRenderTarget();
    const viewport = renderer.getViewport(new THREE.Vector4());
    const scissor = renderer.getScissor(new THREE.Vector4());
    const scissorTest = renderer.getScissorTest();
    const clearColor = renderer.getClearColor(new THREE.Color());
    const clearAlpha = renderer.getClearAlpha();
    if (this.layout === null) {
      const rect = this.canvas.getBoundingClientRect();
      this.layout = [...this.outputs].map((output) => {
        const pane = output.canvas.getBoundingClientRect();
        return {
          output,
          x: pane.left - rect.left,
          y: rect.bottom - pane.bottom,
          width: pane.width,
          height: pane.height,
        };
      });
    }
    try {
      renderer.setRenderTarget(null);
      renderer.setScissorTest(false);
      renderer.setClearColor(0x000000, 0);
      renderer.clear(true, true, true);
      renderer.setScissorTest(true);
      for (const { output, x, y, width, height } of this.layout) {
        if (!output.presented || width <= 0 || height <= 0) continue;
        renderer.setViewport(x, y, width, height);
        renderer.setScissor(x, y, width, height);
        this.material.map = output.target.texture;
        renderer.render(this.scene, this.camera);
      }
    } finally {
      renderer.setRenderTarget(target);
      renderer.setViewport(viewport);
      renderer.setScissor(scissor);
      renderer.setScissorTest(scissorTest);
      renderer.setClearColor(clearColor, clearAlpha);
    }
  }

  setByteLimit(bytes) {
    if (bytes <= this.allocatedBytes()) throw new Error("Shared pane images and meshes exceed this GPU allowance.");
    this.byteLimit = bytes;
  }

  allocatedBytes() {
    let bytes = 0;
    for (const holder of [...this.outputs, ...this.allocations]) bytes += holder.allocatedBytes();
    return bytes;
  }

  availableBytes() {
    return this.byteLimit - this.allocatedBytes();
  }
}
