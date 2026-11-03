import * as THREE from "../../vendor/three.module.js";

export class SurfaceBaseImage {
  constructor(byteLimit) {
    this.byteLimit = byteLimit;
    this.target = null;
    this.revision = null;
    this.geometry = new THREE.PlaneGeometry(2, 2);
    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: { sourceTexture: { value: null } },
      vertexShader: "void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: `precision highp float;
        uniform sampler2D sourceTexture;
        out vec4 outColor;
        void main() { outColor = texelFetch(sourceTexture, ivec2(gl_FragCoord.xy), 0); }`,
      blending: THREE.NoBlending,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    this.scene = new THREE.Scene();
    const mesh = new THREE.Mesh(this.geometry, this.material);
    mesh.frustumCulled = false;
    this.scene.add(mesh);
    this.camera = new THREE.OrthographicCamera();
  }

  prepare(width, height) {
    if (width * height * 4 > this.byteLimit) {
      this.release();
      return false;
    }
    if (this.target?.width === width && this.target?.height === height) return true;
    this.release();
    this.target = new THREE.WebGLRenderTarget(width, height, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    });
    this.target.texture.colorSpace = THREE.NoColorSpace;
    this.material.uniforms.sourceTexture.value = this.target.texture;
    return true;
  }

  render(renderer, draw, revision) {
    const output = renderer.getRenderTarget();
    if (this.revision !== revision) {
      this.revision = null;
      try {
        renderer.setRenderTarget(this.target);
        renderer.clear(true, false, false);
        draw();
        this.revision = revision;
      } finally {
        renderer.setRenderTarget(output);
      }
    }
    renderer.render(this.scene, this.camera);
  }

  allocatedBytes() {
    return this.target === null ? 0 : this.target.width * this.target.height * 4;
  }

  release() {
    this.target?.dispose();
    this.target = null;
    this.revision = null;
    this.material.uniforms.sourceTexture.value = null;
  }

  dispose() {
    this.release();
    this.geometry.dispose();
    this.material.dispose();
  }
}
