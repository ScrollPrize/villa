// A tap is a voxel of the finest level apart, or a device pixel apart once a voxel is smaller, so a stage's cost does
// not grow with zoom.

import * as THREE from "../../../vendor/three.module.js";

const RADIUS_LIMIT = 24;

export const KINDS = {
  gaussian: {
    name: "Gaussian",
    parameter: { label: "σ", value: 1, min: 0.25, max: 8, step: 0.25 },
    reach: (sigma) => 2 * Math.min(RADIUS_LIMIT, Math.ceil(3 * sigma)),
    passes: [
      [1, 0],
      [0, 1],
    ].map(
      (axis) => `
        float sigma = value / (tapPixels * pixel);
        int radius = min(int(ceil(3.0 * sigma)), ${RADIUS_LIMIT});
        float sum = 0.0;
        float weights = 0.0;
        for (int offset = -${RADIUS_LIMIT}; offset <= ${RADIUS_LIMIT}; offset++) {
          if (abs(offset) > radius) continue;
          float weight = exp(-0.5 * float(offset * offset) / (sigma * sigma));
          sum += weight * grey(uv + vec2(${axis[0]}.0, ${axis[1]}.0) * float(offset) * tap);
          weights += weight;
        }
        outColor = vec4(vec3(sum / weights), 1.0);`,
    ),
  },
  median: {
    name: "Median 3 × 3",
    parameter: null,
    reach: () => 1,
    passes: [
      `
        float values[9];
        for (int index = 0; index < 9; index++) values[index] = grey(uv + vec2(index % 3 - 1, index / 3 - 1) * tap);
        for (int index = 1; index < 9; index++) {
          float held = values[index];
          int below = index - 1;
          while (below >= 0 && values[below] > held) {
            values[below + 1] = values[below];
            below--;
          }
          values[below + 1] = held;
        }
        outColor = vec4(vec3(values[4]), 1.0);`,
    ],
  },
  sharpen: {
    name: "Sharpen",
    parameter: { label: "Amount", value: 1, min: 0.1, max: 4, step: 0.1 },
    reach: () => 1,
    passes: [
      `
        float centre = grey(uv);
        float around = grey(uv + vec2(tap.x, 0.0)) + grey(uv - vec2(tap.x, 0.0))
          + grey(uv + vec2(0.0, tap.y)) + grey(uv - vec2(0.0, tap.y));
        outColor = vec4(vec3(clamp(centre + value * (4.0 * centre - around), 0.0, 1.0)), 1.0);`,
    ],
  },
  edges: {
    name: "Edges",
    parameter: null,
    reach: () => 1,
    passes: [
      `
        float g[9];
        for (int index = 0; index < 9; index++) g[index] = grey(uv + vec2(index % 3 - 1, index / 3 - 1) * tap);
        float x = g[2] + 2.0 * g[5] + g[8] - g[0] - 2.0 * g[3] - g[6];
        float y = g[6] + 2.0 * g[7] + g[8] - g[0] - 2.0 * g[1] - g[2];
        outColor = vec4(vec3(clamp(length(vec2(x, y)) / 4.0, 0.0, 1.0)), 1.0);`,
    ],
  },
  threshold: {
    name: "Threshold",
    parameter: { label: "Level", value: 128, min: 0, max: 255, step: 1 },
    reach: () => 0,
    passes: [
      `
        outColor = vec4(vec3(grey(uv) * 255.0 >= value ? 1.0 : 0.0), 1.0);`,
    ],
  },
};

export function stagePasses({ kind, value }) {
  return KINDS[kind].passes.map(
    (body) =>
      new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3,
        uniforms: { image: { value: null }, pixel: { value: 1 }, value: { value: value ?? 0 } },
        vertexShader: "void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }",
        fragmentShader: `
          precision highp float;
          uniform sampler2D image;
          uniform float pixel;
          uniform float value;
          out vec4 outColor;
          float grey(vec2 uv) { return texture(image, uv).r; }
          void main() {
            vec2 size = vec2(textureSize(image, 0));
            vec2 uv = gl_FragCoord.xy / size;
            float tapPixels = max(1.0, 1.0 / pixel);
            vec2 tap = tapPixels / size;
            ${body}
          }`,
        depthTest: false,
        depthWrite: false,
        blending: THREE.NoBlending,
        toneMapped: false,
      }),
  );
}
