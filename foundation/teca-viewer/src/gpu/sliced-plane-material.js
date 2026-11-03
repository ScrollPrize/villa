import * as THREE from "../../vendor/three.module.js";
import { CT_LEVEL_LIMIT } from "../config.js";
import { displayWindowShader } from "../volume/display-window.js";
import { LABEL_COLOR_SHADER } from "./label-colors.js";
import { commonAtlasFormat, levelShader, levelUniforms, nearestShader, updateLevelUniforms } from "./level-sampling.js";

const DISPLAY_WINDOW_SHADER = displayWindowShader("raw", "encodingMinimum", "encodingMaximum");

export const LAYER_COLOR_SHADER = `
  uniform vec3 layerColor_RGB;
  uniform float layerOpacity;
  uniform vec2 layerRange;
  vec4 layerColor(float value) {
    float displayByte = value * 255.0;
    if (displayByte < layerRange.x - 0.5 || displayByte > layerRange.y + 0.5) discard;
    return vec4(layerColor_RGB, layerOpacity * value);
  }
`;

function levelSampler(index) {
  return `
    // World coordinates are fine-voxel indices. Level texel g holds the mean
    // of fine voxels [g*d, g*d + d), centroid g*d + (d - 1)/2, so sampling at
    // that centroid keeps every level laterally aligned with level 0: exact
    // slices at d = 1, no content shift when refinement swaps levels.
    bool sampleLevel${index}(vec3 world_ZYX, out float raw) {
      if (${index} >= levelCount) return false;
      vec3 relative_ZYX = world_ZYX - offset${index};
      vec3 position_ZYX = clamp(
        (relative_ZYX - (downsample${index} - vec3(1.0)) * 0.5) / downsample${index},
        vec3(0.0),
        shape${index} - vec3(1.0)
      );
      ivec3 cachedChunk_ZYX = ivec3(0);
      int slot = -1;
      ivec3 low_ZYX = ivec3(floor(position_ZYX));
      vec3 weight_ZYX = fract(position_ZYX);
      ivec3 high_ZYX = min(low_ZYX + ivec3(ceil(weight_ZYX)), ivec3(shape${index}) - ivec3(1));
      bvec3 split_ZYX = notEqual(high_ZYX, low_ZYX);
      float c000; float c001; float c010; float c011;
      float c100; float c101; float c110; float c111;
      if (!sampleVoxel${index}(low_ZYX, cachedChunk_ZYX, slot, c000)) return false;
      c001 = c000;
      if (split_ZYX.z && !sampleVoxel${index}(ivec3(low_ZYX.x, low_ZYX.y, high_ZYX.z), cachedChunk_ZYX, slot, c001))
        return false;
      c010 = c000;
      c011 = c001;
      if (split_ZYX.y) {
        if (!sampleVoxel${index}(ivec3(low_ZYX.x, high_ZYX.y, low_ZYX.z), cachedChunk_ZYX, slot, c010)) return false;
        c011 = c010;
        if (split_ZYX.z && !sampleVoxel${index}(ivec3(low_ZYX.x, high_ZYX.y, high_ZYX.z), cachedChunk_ZYX, slot, c011))
          return false;
      }
      c100 = c000;
      c101 = c001;
      c110 = c010;
      c111 = c011;
      if (split_ZYX.x) {
        if (!sampleVoxel${index}(ivec3(high_ZYX.x, low_ZYX.y, low_ZYX.z), cachedChunk_ZYX, slot, c100)) return false;
        c101 = c100;
        if (split_ZYX.z && !sampleVoxel${index}(ivec3(high_ZYX.x, low_ZYX.y, high_ZYX.z), cachedChunk_ZYX, slot, c101))
          return false;
        c110 = c100;
        c111 = c101;
        if (split_ZYX.y) {
          if (!sampleVoxel${index}(ivec3(high_ZYX.x, high_ZYX.y, low_ZYX.z), cachedChunk_ZYX, slot, c110)) return false;
          c111 = c110;
          if (split_ZYX.z && !sampleVoxel${index}(ivec3(high_ZYX.x, high_ZYX.y, high_ZYX.z), cachedChunk_ZYX, slot, c111))
            return false;
        }
      }
      float c00 = mix(c000, c001, weight_ZYX.z);
      float c01 = mix(c010, c011, weight_ZYX.z);
      float c10 = mix(c100, c101, weight_ZYX.z);
      float c11 = mix(c110, c111, weight_ZYX.z);
      raw = mix(mix(c00, c01, weight_ZYX.y), mix(c10, c11, weight_ZYX.y), weight_ZYX.x);
      return true;
    }
  `;
}

export function createSlicedPlaneMaterial({ levels, sourceBounds_ZYX, displayEncoding, displaySettings, style }) {
  const atlasFormat = commonAtlasFormat(levels);
  const uniforms = {
    center_ZYX: { value: new THREE.Vector3() },
    right_ZYX: { value: new THREE.Vector3() },
    up_ZYX: { value: new THREE.Vector3() },
    visibleSpan_UV: { value: new THREE.Vector2(1, 1) },
    sourceMinimum_ZYX: { value: new THREE.Vector3(...sourceBounds_ZYX.minimum_ZYX) },
    sourceMaximum_ZYX: { value: new THREE.Vector3(...sourceBounds_ZYX.maximum_ZYX) },
    black: { value: displaySettings.black },
    white: { value: displaySettings.white },
    gamma: { value: displaySettings.gamma },
    encodingMinimum: { value: displayEncoding.kind === "linear_window" ? displayEncoding.minimum : 0 },
    encodingMaximum: { value: displayEncoding.kind === "linear_window" ? displayEncoding.maximum : 255 },
    identityEncoding: { value: displayEncoding.kind === "identity_uint8" ? 1 : 0 },
    layerColor_RGB: { value: new THREE.Vector3(...(style?.color_RGB ?? [0, 0, 0])) },
    layerOpacity: { value: style?.opacity ?? 0 },
    layerLabels: { value: style?.color_RGB === null ? 1 : 0 },
    layerRange: { value: new THREE.Vector2(...(style?.range ?? [0, 255])) },
    drawn: { value: 1 },
    ...levelUniforms(levels, CT_LEVEL_LIMIT),
  };
  const shading =
    style === null
      ? `float displayed = clamp((encoded - black) / max(white - black, 0.0001), 0.0, 1.0);
        displayed = pow(displayed, 1.0 / gamma);
        outColor = vec4(vec3(displayed), 1.0);`
      : "outColor = layerColor(clamp(encoded, 0.0, 1.0));";
  const labels =
    style === null
      ? ""
      : `if (layerLabels == 1) {
          if (!inside || !sampleNearest(world_ZYX, raw) || raw == 0.0) discard;
          outColor = vec4(labelColor(raw), layerOpacity);
          return;
        }`;
  const levelIndices = Array.from({ length: CT_LEVEL_LIMIT }, (_, index) => index);
  const fallthrough = levelIndices.map((index) => `sampleLevel${index}(world_ZYX, raw)`).join(" || ");
  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms,
    vertexShader: `
      out vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      precision highp int;
      precision highp sampler2DArray;
      precision highp isampler2DArray;
      precision highp usampler2DArray;
      precision highp usampler2D;
      uniform vec3 center_ZYX;
      uniform vec3 right_ZYX;
      uniform vec3 up_ZYX;
      uniform vec2 visibleSpan_UV;
      uniform vec3 sourceMinimum_ZYX;
      uniform vec3 sourceMaximum_ZYX;
      uniform float black;
      uniform float white;
      uniform float gamma;
      uniform float encodingMinimum;
      uniform float encodingMaximum;
      uniform int identityEncoding;
      uniform int layerLabels;
      uniform int drawn;
      ${levelShader(CT_LEVEL_LIMIT, atlasFormat.samplerClass)}
      in vec2 vUv;
      out vec4 outColor;

      ${levelIndices.map(levelSampler).join("\n")}
      ${style === null ? "" : `${nearestShader(CT_LEVEL_LIMIT)}${LABEL_COLOR_SHADER}${LAYER_COLOR_SHADER}`}

      bool sampleWorld(vec3 world_ZYX, out float raw) {
        return ${fallthrough};
      }

      void main() {
        if (drawn == 0) discard;
        vec2 local_UV = (vUv - vec2(0.5)) * visibleSpan_UV;
        vec3 world_ZYX = center_ZYX + right_ZYX * local_UV.x + up_ZYX * local_UV.y;
        bool inside = all(greaterThanEqual(world_ZYX, sourceMinimum_ZYX)) && all(lessThan(world_ZYX, sourceMaximum_ZYX));
        float raw = 0.0;
        ${labels}
        if (!inside || !sampleWorld(world_ZYX, raw)) discard;
        float encoded = identityEncoding == 1
          ? raw / 255.0
          : ${DISPLAY_WINDOW_SHADER};
        ${shading}
      }
    `,
    side: THREE.DoubleSide,
    depthTest: false,
    depthWrite: false,
    transparent: style !== null,
  });
  material.toneMapped = false;
  return material;
}

export function updateSlicedPlaneMaterialLevels(material, levels) {
  updateLevelUniforms(material.uniforms, levels, CT_LEVEL_LIMIT);
}
