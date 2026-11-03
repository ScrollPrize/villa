// Half floats keep a pass from rounding the grey to 1/255.

import * as THREE from "../../vendor/three.module.js";
import { FILTER_PICTURE_BYTES } from "../config.js";

export function filterMargin(reach, voxelsPerPixel) {
  return reach * Math.max(1, voxelsPerPixel);
}

export class FilterChain {
  constructor() {
    this.targets = null;
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2));
    this.quad.frustumCulled = false;
    this.wide = new THREE.Mesh(new THREE.PlaneGeometry(1, 1));
    this.wide.frustumCulled = false;
    this.wideCamera = new THREE.OrthographicCamera();
    this.copy = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: { image: { value: null }, margin: { value: 0 } },
      vertexShader: "void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: `precision highp float;
        uniform sampler2D image;
        uniform int margin;
        out vec4 outColor;
        void main() { outColor = vec4(vec3(texelFetch(image, ivec2(gl_FragCoord.xy) + margin, 0).r), 1.0); }`,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NoBlending,
      toneMapped: false,
    });
  }

  draw(renderer, passes, reach, mesh, camera, span) {
    const output = renderer.getRenderTarget();
    const pixel = span / output.width;
    const margin = Math.ceil(reach * Math.max(1, 1 / pixel));
    const width = output.width + 2 * margin;
    const height = output.height + 2 * margin;
    if (Math.max(width, height) > renderer.capabilities.maxTextureSize || width * height * 4 > FILTER_PICTURE_BYTES)
      return false;
    this.targets ??= [0, 1].map(
      () =>
        new THREE.WebGLRenderTarget(width, height, {
          format: THREE.RedFormat,
          type: THREE.HalfFloatType,
          minFilter: THREE.NearestFilter,
          magFilter: THREE.NearestFilter,
          depthBuffer: false,
          generateMipmaps: false,
        }),
    );
    for (const target of this.targets) target.setSize(width, height);
    const spreadX = width / output.width;
    const spreadY = height / output.height;
    this.wideCamera.copy(camera);
    this.wideCamera.left *= spreadX;
    this.wideCamera.right *= spreadX;
    this.wideCamera.top *= spreadY;
    this.wideCamera.bottom *= spreadY;
    this.wideCamera.updateProjectionMatrix();
    this.wide.material = mesh.material;
    this.wide.position.copy(mesh.position);
    this.wide.scale.set(mesh.scale.x * spreadX, mesh.scale.y * spreadY, 1);
    const [u0, u1, v0, v1] = [0.5 - spreadX / 2, 0.5 + spreadX / 2, 0.5 - spreadY / 2, 0.5 + spreadY / 2];
    const uv = this.wide.geometry.attributes.uv;
    uv.array.set([u0, v1, u1, v1, u0, v0, u1, v0]);
    uv.needsUpdate = true;
    let at = 0;
    renderer.setRenderTarget(this.targets[at]);
    renderer.clear(true, false, false);
    renderer.render(this.wide, this.wideCamera);
    for (const pass of passes) {
      pass.uniforms.image.value = this.targets[at].texture;
      pass.uniforms.pixel.value = pixel;
      at = 1 - at;
      renderer.setRenderTarget(this.targets[at]);
      this.quad.material = pass;
      renderer.render(this.quad, this.wideCamera);
    }
    this.copy.uniforms.image.value = this.targets[at].texture;
    this.copy.uniforms.margin.value = margin;
    renderer.setRenderTarget(output);
    this.quad.material = this.copy;
    renderer.render(this.quad, this.wideCamera);
    return true;
  }

  release() {
    for (const target of this.targets ?? []) target.dispose();
    this.targets = null;
  }
}
