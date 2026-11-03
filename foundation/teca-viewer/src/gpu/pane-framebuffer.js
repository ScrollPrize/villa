import * as THREE from "../../vendor/three.module.js";

export class PaneFramebuffer {
  constructor(owner, canvas) {
    this.owner = owner;
    this.canvas = canvas;
    this.target = new THREE.WebGLRenderTarget(1, 1, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
      stencilBuffer: false,
      generateMipmaps: false,
    });
    this.target.texture.colorSpace = THREE.NoColorSpace;
    this.presented = false;
    this.shown = false;
    this.owner.attach(this);
  }

  resize(hiddenSide = null) {
    this.shown = this.canvas.clientWidth > 0 && this.canvas.clientHeight > 0;
    if (!this.shown && hiddenSide === null) return;
    const ratio = this.owner.renderer.getPixelRatio();
    const width = this.shown ? Math.max(1, Math.floor(this.canvas.clientWidth * ratio)) : hiddenSide;
    const height = this.shown ? Math.max(1, Math.floor(this.canvas.clientHeight * ratio)) : hiddenSide;
    if (this.canvas.width === width && this.canvas.height === height) return;
    this.canvas.width = width;
    this.canvas.height = height;
    this.target.setSize(width, height);
    this.presented = false;
  }

  begin(sortObjects) {
    this.owner.renderer.sortObjects = sortObjects;
    this.owner.renderer.state.buffers.depth.setMask(true);
    this.owner.renderer.state.buffers.color.setMask(true);
    this.target.viewport.set(0, 0, this.target.width, this.target.height);
    this.target.scissorTest = false;
    this.owner.renderer.setRenderTarget(this.target);
    const ratio = this.owner.renderer.getPixelRatio();
    this.owner.renderer.setViewport(0, 0, this.target.width / ratio, this.target.height / ratio);
    this.owner.renderer.setScissor(0, 0, this.target.width / ratio, this.target.height / ratio);
    this.owner.renderer.setScissorTest(false);
  }

  complete() {
    this.presented = true;
    this.owner.requestPresent();
  }

  snapshot() {
    this.owner.present();
    const source = this.owner.canvas;
    const rect = source.getBoundingClientRect();
    const pane = this.canvas.getBoundingClientRect();
    const image = document.createElement("canvas");
    image.width = this.canvas.width;
    image.height = this.canvas.height;
    const ratio = this.owner.renderer.getPixelRatio();
    image
      .getContext("2d")
      .drawImage(
        source,
        (pane.left - rect.left) * ratio,
        (pane.top - rect.top) * ratio,
        pane.width * ratio,
        pane.height * ratio,
        0,
        0,
        image.width,
        image.height,
      );
    return image;
  }

  allocatedBytes() {
    return this.target.width * this.target.height * 8;
  }
}
