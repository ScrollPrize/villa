import * as THREE from "../../vendor/three.module.js";
import { commonAtlasFormat, levelShader, levelUniforms, updateLevelUniforms } from "../gpu/level-sampling.js";
import { displayWindowShader } from "../volume/display-window.js";
import { surfaceCameraShader } from "./camera.js";
import { surfaceGeometryShader } from "./geometry-sampling.js";
import { SURFACE_RELIEF_MASK_SIZE } from "./relief-mask.js";
import { SURFACE_TRANSFER_POINT_LIMIT, surfaceTransferColor } from "./transfer.js";
import { SURFACE_VOLUMETRIC_LAYER_LIMIT } from "./volumetric.js";

export const MATERIAL_LEVELS = 2;
const VOLUMETRIC_CAMERA_UNIFORMS = [
  "volumetricEnabled",
  "volumetricAzimuth",
  "volumetricTilt",
  "volumetricPerspective",
  "volumetricMaximumDepth",
];
const DISPLAY_WINDOW_SHADER = displayWindowShader("raw", "encodingMinimum", "encodingMaximum");
const SURFACE_MAIN = `
  void main() {
    if (drawn == 0) discard;
    vec2 screen_UV = (vUv - vec2(0.5)) * surfaceVisibleSpan_UV;
    vec2 surface_UV = surfaceCenter_UV
      + surfaceCameraScreenToPoint(screen_UV, 0.0, surfaceVisibleSpan_UV);
    vec3 point_ZYX = vec3(0.0); vec3 tangentU_ZYX = vec3(0.0); vec3 tangentV_ZYX = vec3(0.0);
    vec3 normal_ZYX = vec3(0.0);
    bool baseFrameValid = (volumetricEnabled == 0 || volumetricLayerDependentCamera == 0) && sampleSurfaceFrame(
      surface_UV,
      point_ZYX,
      tangentU_ZYX,
      tangentV_ZYX,
      normal_ZYX
    );
    if (volumetricEnabled == 1) {
      if (volumetricLayerDependentCamera == 0 && !baseFrameValid) discard;
      vec3 accumulated = vec3(0.0);
      float transmittance = 1.0;
      bool anyValid = false;
      float opacityRange = volumetricOpacityMaximum - volumetricOpacityMinimum;
      vec2 fixedReliefBounds = vec2(0.0, 1.0);
      if (volumetricReliefMaskEnabled == 1 && volumetricLayerDependentCamera == 0) {
        vec2 mask_UV = clamp(surface_UV / max(surfaceGrid_HW - vec2(1.0), vec2(1.0)), vec2(0.0), vec2(1.0));
        fixedReliefBounds = texture(volumetricReliefMask, reliefMaskTextureUv(mask_UV)).rg;
      }
      int totalLayers = volumetricLayersFront + volumetricLayersBehind;
      vec2 raySlope_UV = (surfaceCameraScreenToPoint(screen_UV, volumetricRelief, surfaceVisibleSpan_UV)
        - surfaceCameraScreenToPoint(screen_UV, 0.0, surfaceVisibleSpan_UV)) / surfaceScale_UV;
      float rayLength = length(vec3(raySlope_UV, 1.0));
      for (int layer = 0; layer <= ${SURFACE_VOLUMETRIC_LAYER_LIMIT}; layer++) {
        if (layer > totalLayers) break;
        int canonicalIndex = totalLayers - layer;
        float canonicalOffset = float(canonicalIndex - volumetricLayersBehind) * volumetricSpacing;
        float physicalOffset = canonicalOffset * volumetricDirection;
        vec3 layerPoint_ZYX = point_ZYX;
        vec3 layerTangentU_ZYX = tangentU_ZYX;
        vec3 layerTangentV_ZYX = tangentV_ZYX;
        vec3 layerNormal_ZYX = normal_ZYX;
        vec2 layerSurface_UV = surface_UV;
        if (volumetricLayerDependentCamera == 1) {
          layerSurface_UV = surfaceCenter_UV + surfaceCameraScreenToPoint(
            screen_UV,
            canonicalOffset * volumetricRelief,
            surfaceVisibleSpan_UV
          );
        }
        if (volumetricReliefMaskEnabled == 1) {
          vec2 bounds = fixedReliefBounds;
          if (volumetricLayerDependentCamera == 1) {
            vec2 mask_UV = clamp(layerSurface_UV / max(surfaceGrid_HW - vec2(1.0), vec2(1.0)), vec2(0.0), vec2(1.0));
            bounds = texture(volumetricReliefMask, reliefMaskTextureUv(mask_UV)).rg;
          }
          float depth = float(canonicalIndex) / float(totalLayers);
          if (depth < bounds.x || depth > bounds.y) continue;
        }
        if (volumetricLayerDependentCamera == 1 && !sampleSurfaceFrame(
          layerSurface_UV,
          layerPoint_ZYX,
          layerTangentU_ZYX,
          layerTangentV_ZYX,
          layerNormal_ZYX
        )) continue;
        vec3 sample_ZYX = layerPoint_ZYX
          + layerNormal_ZYX * (normalOffset + physicalOffset);
        bool inside = all(greaterThanEqual(sample_ZYX, sourceMinimum_ZYX))
          && all(lessThan(sample_ZYX, sourceMaximum_ZYX));
        if (!inside) continue;
        float densityValue = 0.0;
        float emission = 0.0;
        vec3 gradient_ZYX = vec3(0.0);
        if (!sampleSurfaceScalarWorld(sample_ZYX, densityValue, emission, gradient_ZYX)) continue;
        anyValid = true;
        float density = clamp((densityValue - volumetricOpacityMinimum) / opacityRange, 0.0, 1.0);
        vec4 transfer = surfaceTransfer(density);
        vec3 gradient_UVW = vec3(
          dot(gradient_ZYX, layerTangentU_ZYX),
          dot(gradient_ZYX, layerTangentV_ZYX),
          dot(gradient_ZYX, layerNormal_ZYX)
        );
        if (volumetricFeature != 0) {
          vec3 feature = gradient_UVW * volumetricGradientGain / max(encodingMaximum - encodingMinimum, 0.000001);
          float strength = volumetricFeature == 1 ? abs(feature.z)
            : volumetricFeature == 2 ? length(feature.xy) : length(feature);
          transfer.a *= clamp(strength, 0.0, 1.0);
          transfer.rgb = volumetricFeature == 1
            ? (feature.z < 0.0 ? vec3(0.1, 0.5, 1.0) : vec3(1.0, 0.4, 0.1)) : vec3(1.0);
        }
        if (volumetricConstantEmission == 1) emission = 1.0;
        float support = canonicalIndex == 0 || canonicalIndex == totalLayers ? 0.5 : 1.0;
        float alpha = 1.0 - pow(1.0 - volumetricOpacity * pow(transfer.a, volumetricGamma),
          volumetricSpacing * rayLength * support);
        float shade = 1.0;
        if (volumetricLighting > 0.0) {
          if (dot(gradient_UVW, gradient_UVW) > 0.000001) {
            float diffuse = max(
              0.0,
              dot(normalize(gradient_UVW), vec3(cos(radians(volumetricLightAzimuth)) * 0.819152, sin(radians(volumetricLightAzimuth)) * 0.819152, 0.573576))
            );
            float fullyLit = 0.1 + 0.9 * diffuse;
            shade = 1.0 + volumetricLighting * (fullyLit - 1.0);
          }
        }
        accumulated += transmittance * alpha * emission * transfer.rgb * shade;
        transmittance *= 1.0 - alpha;
        if (transmittance < 0.004) break;
      }
      if (!anyValid) discard;
      outColor = vec4(accumulated, 1.0);
      return;
    }
    if (!baseFrameValid) discard;
    vec3 sample_ZYX = point_ZYX + normal_ZYX * normalOffset;
    bool inside = all(greaterThanEqual(sample_ZYX, sourceMinimum_ZYX))
      && all(lessThan(sample_ZYX, sourceMaximum_ZYX));
    float raw = 0.0;
    if (!inside || !sampleWorld(sample_ZYX, raw)) discard;
    outColor = vec4(vec3(displayRaw(raw)), 1.0);
  }
`;

function levelSampler(index) {
  return `
    bool sampleLevelData${index}(vec3 world_ZYX, out float raw, out vec3 gradient_ZYX) {
      if (${index} >= levelCount) return false;
      vec3 position_ZYX = clamp(
        (world_ZYX - offset${index} - (downsample${index} - vec3(1.0)) * 0.5) / downsample${index},
        vec3(0.0),
        shape${index} - vec3(1.0)
      );
      ivec3 low_ZYX = ivec3(floor(position_ZYX));
      ivec3 high_ZYX = min(low_ZYX + ivec3(1), ivec3(shape${index}) - ivec3(1));
      vec3 weight_ZYX = fract(position_ZYX);
      float c000; float c001; float c010; float c011; float c100; float c101; float c110; float c111;
      ivec3 cachedChunk_ZYX = ivec3(0);
      int slot = -1;
      if (!sampleVoxel${index}(ivec3(low_ZYX.x, low_ZYX.y, low_ZYX.z), cachedChunk_ZYX, slot, c000)
        || !sampleVoxel${index}(ivec3(low_ZYX.x, low_ZYX.y, high_ZYX.z), cachedChunk_ZYX, slot, c001)
        || !sampleVoxel${index}(ivec3(low_ZYX.x, high_ZYX.y, low_ZYX.z), cachedChunk_ZYX, slot, c010)
        || !sampleVoxel${index}(ivec3(low_ZYX.x, high_ZYX.y, high_ZYX.z), cachedChunk_ZYX, slot, c011)
        || !sampleVoxel${index}(ivec3(high_ZYX.x, low_ZYX.y, low_ZYX.z), cachedChunk_ZYX, slot, c100)
        || !sampleVoxel${index}(ivec3(high_ZYX.x, low_ZYX.y, high_ZYX.z), cachedChunk_ZYX, slot, c101)
        || !sampleVoxel${index}(ivec3(high_ZYX.x, high_ZYX.y, low_ZYX.z), cachedChunk_ZYX, slot, c110)
        || !sampleVoxel${index}(ivec3(high_ZYX.x, high_ZYX.y, high_ZYX.z), cachedChunk_ZYX, slot, c111)) return false;
      float c00 = mix(c000, c001, weight_ZYX.z);
      float c01 = mix(c010, c011, weight_ZYX.z);
      float c10 = mix(c100, c101, weight_ZYX.z);
      float c11 = mix(c110, c111, weight_ZYX.z);
      raw = mix(mix(c00, c01, weight_ZYX.y), mix(c10, c11, weight_ZYX.y), weight_ZYX.x);
      gradient_ZYX = vec3(
        mix(mix(c100 - c000, c101 - c001, weight_ZYX.z),
            mix(c110 - c010, c111 - c011, weight_ZYX.z), weight_ZYX.y),
        mix(mix(c010 - c000, c011 - c001, weight_ZYX.z),
            mix(c110 - c100, c111 - c101, weight_ZYX.z), weight_ZYX.x),
        mix(mix(c001 - c000, c011 - c010, weight_ZYX.y),
            mix(c101 - c100, c111 - c110, weight_ZYX.y), weight_ZYX.x)
      ) / downsample${index};
      return true;
    }
    bool sampleLevel${index}(vec3 world_ZYX, out float raw) {
      vec3 gradient_ZYX;
      return sampleLevelData${index}(world_ZYX, raw, gradient_ZYX);
    }
  `;
}

export function surfaceShaderMaterial(uniforms, levels, main) {
  const levelIndices = Array.from({ length: MATERIAL_LEVELS }, (_, index) => index);
  const fallthrough = levelIndices.map((index) => `sampleLevel${index}(world_ZYX, raw)`).join(" || ");
  const gradientFallthrough = levelIndices
    .map((index) => `sampleLevelData${index}(world_ZYX, raw, gradient_ZYX)`)
    .join(" || ");
  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: { ...uniforms, ...levelUniforms(levels, MATERIAL_LEVELS) },
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
      uniform sampler2DArray surfaceAtlas;
      uniform usampler2D surfacePages;
      uniform vec2 surfaceGrid_HW;
      uniform int surfaceStripRows;
      uniform int surfaceAtlasWidth;
      uniform vec2 surfaceCenter_UV;
      uniform vec2 surfaceVisibleSpan_UV;
      uniform float normalOffset;
      uniform float normalSign;
      uniform vec3 sourceMinimum_ZYX;
      uniform vec3 sourceMaximum_ZYX;
      uniform float black;
      uniform float white;
      uniform float gamma;
      uniform float encodingMinimum;
      uniform float encodingMaximum;
      uniform int identityEncoding;
      uniform int volumetricEnabled;
      uniform int volumetricLayersFront;
      uniform int volumetricLayersBehind;
      uniform float volumetricSpacing;
      uniform float volumetricDirection;
      uniform float volumetricOpacityMinimum;
      uniform float volumetricOpacityMaximum;
      uniform float volumetricOpacity;
      uniform float volumetricGamma;
      uniform float volumetricAzimuth;
      uniform float volumetricTilt;
      uniform float volumetricPerspective;
      uniform float volumetricMaximumDepth;
      uniform float volumetricRelief;
      uniform float volumetricLighting;
      uniform float volumetricLightAzimuth;
      uniform int volumetricFeature;
      uniform float volumetricGradientGain;
      uniform int volumetricConstantEmission;
      uniform int volumetricLayerDependentCamera;
      uniform int volumetricTransferCount;
      uniform float volumetricTransferValues[${SURFACE_TRANSFER_POINT_LIMIT}];
      uniform vec4 volumetricTransferColors[${SURFACE_TRANSFER_POINT_LIMIT}];
      uniform sampler2D volumetricReliefMask;
      uniform int volumetricReliefMaskEnabled;
      uniform int drawn;
      ${levelShader(MATERIAL_LEVELS, commonAtlasFormat(levels).samplerClass)}
      in vec2 vUv;
      out vec4 outColor;

      ${surfaceCameraShader()}

      ${surfaceGeometryShader()}

      ${levelIndices.map(levelSampler).join("\n")}

      bool sampleWorld(vec3 world_ZYX, out float raw) {
        return ${fallthrough};
      }

      bool sampleWorldGradient(vec3 world_ZYX, out float raw, out vec3 gradient_ZYX) {
        return ${gradientFallthrough};
      }

      float encodeRaw(float raw) {
        return identityEncoding == 1 ? raw / 255.0 : ${DISPLAY_WINDOW_SHADER};
      }

      float displayRaw(float raw) {
        float displayed = clamp((encodeRaw(raw) - black) / max(white - black, 0.0001), 0.0, 1.0);
        return pow(displayed, 1.0 / gamma);
      }

      bool sampleSurfaceScalarWorld(
        vec3 world_ZYX,
        out float densityValue,
        out float emission,
        out vec3 gradient_ZYX
      ) {
        float raw = 0.0;
        gradient_ZYX = vec3(0.0);
        bool found = false;
        if (volumetricLighting > 0.0 || volumetricFeature != 0) {
          found = sampleWorldGradient(world_ZYX, raw, gradient_ZYX);
        } else {
          found = sampleWorld(world_ZYX, raw);
        }
        if (!found) return false;
        densityValue = encodeRaw(raw);
        emission = displayRaw(raw);
        return true;
      }

      vec4 surfaceTransfer(float density) {
        vec4 low = volumetricTransferColors[0];
        float lowValue = volumetricTransferValues[0];
        for (int index = 1; index < ${SURFACE_TRANSFER_POINT_LIMIT}; index++) {
          if (index >= volumetricTransferCount) break;
          vec4 high = volumetricTransferColors[index];
          float highValue = volumetricTransferValues[index];
          if (density <= highValue || index == volumetricTransferCount - 1) {
            return mix(low, high, clamp((density - lowValue) / max(highValue - lowValue, 0.0001), 0.0, 1.0));
          }
          low = high;
          lowValue = highValue;
        }
        return low;
      }

      vec2 reliefMaskTextureUv(vec2 surfaceNormalized_UV) {
        return (surfaceNormalized_UV * ${SURFACE_RELIEF_MASK_SIZE - 1}.0 + vec2(0.5))
          / ${SURFACE_RELIEF_MASK_SIZE}.0;
      }

      ${main}
    `,
    side: THREE.DoubleSide,
    depthTest: false,
    depthWrite: false,
  });
  material.toneMapped = false;
  return material;
}

export function surfaceEncodingUniforms(displayEncoding) {
  return {
    encodingMinimum: { value: displayEncoding.kind === "linear_window" ? displayEncoding.minimum : 0 },
    encodingMaximum: { value: displayEncoding.kind === "linear_window" ? displayEncoding.maximum : 255 },
    identityEncoding: { value: displayEncoding.kind === "identity_uint8" ? 1 : 0 },
  };
}

export function createSurfaceMaterial({
  surfaceAtlas,
  surfaceGrid_HW,
  surfaceScale_UV,
  levels,
  sourceBounds_ZYX,
  displayEncoding,
  displaySettings,
  reliefMaskTexture,
}) {
  const uniforms = {
    surfaceAtlas: { value: surfaceAtlas.texture },
    surfacePages: { value: surfaceAtlas.pageTexture },
    surfaceGrid_HW: { value: new THREE.Vector2(surfaceGrid_HW[1], surfaceGrid_HW[0]) },
    surfaceStripRows: { value: surfaceAtlas.stripRows },
    surfaceAtlasWidth: { value: surfaceAtlas.atlasWidth },
    surfaceCenter_UV: { value: new THREE.Vector2() },
    surfaceVisibleSpan_UV: { value: new THREE.Vector2(1, 1) },
    surfaceDisplaySign_UV: { value: new THREE.Vector2(1, -1) },
    surfaceScale_UV: { value: new THREE.Vector2(...surfaceScale_UV) },
    normalOffset: { value: 0 },
    normalSign: { value: 1 },
    sourceMinimum_ZYX: { value: new THREE.Vector3(...sourceBounds_ZYX.minimum_ZYX) },
    sourceMaximum_ZYX: { value: new THREE.Vector3(...sourceBounds_ZYX.maximum_ZYX) },
    black: { value: displaySettings.black },
    white: { value: displaySettings.white },
    gamma: { value: displaySettings.gamma },
    ...surfaceEncodingUniforms(displayEncoding),
    volumetricEnabled: { value: 0 },
    volumetricLayersFront: { value: 0 },
    volumetricLayersBehind: { value: 0 },
    volumetricSpacing: { value: 0 },
    volumetricDirection: { value: 0 },
    volumetricOpacityMinimum: { value: 0 },
    volumetricOpacityMaximum: { value: 0 },
    volumetricOpacity: { value: 0 },
    volumetricGamma: { value: 0 },
    volumetricAzimuth: { value: 0 },
    volumetricTilt: { value: 0 },
    volumetricPerspective: { value: 0 },
    volumetricMaximumDepth: { value: 0 },
    volumetricRelief: { value: 0 },
    volumetricLighting: { value: 0 },
    volumetricLightAzimuth: { value: 0 },
    volumetricFeature: { value: 0 },
    volumetricGradientGain: { value: 0 },
    volumetricConstantEmission: { value: 0 },
    volumetricLayerDependentCamera: { value: 0 },
    volumetricTransferCount: { value: 0 },
    volumetricTransferValues: { value: new Float32Array(SURFACE_TRANSFER_POINT_LIMIT) },
    volumetricTransferColors: {
      value: Array.from({ length: SURFACE_TRANSFER_POINT_LIMIT }, () => new THREE.Vector4()),
    },
    volumetricReliefMask: { value: reliefMaskTexture },
    volumetricReliefMaskEnabled: { value: 0 },
    drawn: { value: 1 },
  };
  return surfaceShaderMaterial(uniforms, levels, SURFACE_MAIN);
}

export function updateSurfaceMaterialDisplay(material, { black, white, gamma }) {
  material.uniforms.black.value = black;
  material.uniforms.white.value = white;
  material.uniforms.gamma.value = gamma;
}

export function updateSurfaceMaterialPose(material, { center_UV, normalOffset, normalSign }, visibleSpan_UV) {
  material.uniforms.surfaceCenter_UV.value.set(...center_UV);
  material.uniforms.surfaceVisibleSpan_UV.value.set(...visibleSpan_UV);
  material.uniforms.normalOffset.value = normalOffset;
  material.uniforms.normalSign.value = normalSign;
}

export function drawSurfaceMaterialFlat(material, draw) {
  const saved = VOLUMETRIC_CAMERA_UNIFORMS.map((key) => material.uniforms[key].value);
  for (const key of VOLUMETRIC_CAMERA_UNIFORMS) material.uniforms[key].value = 0;
  draw();
  for (const [index, key] of VOLUMETRIC_CAMERA_UNIFORMS.entries()) material.uniforms[key].value = saved[index];
}

export function updateSurfaceMaterialLevels(material, levels) {
  updateLevelUniforms(material.uniforms, levels, MATERIAL_LEVELS);
}

export function updateSurfaceMaterialVolumetric(material, settings, camera) {
  material.uniforms.volumetricEnabled.value = settings.enabled ? 1 : 0;
  material.uniforms.volumetricLayersFront.value = settings.layersFront;
  material.uniforms.volumetricLayersBehind.value = settings.layersBehind;
  material.uniforms.volumetricSpacing.value = settings.spacing;
  material.uniforms.volumetricDirection.value = settings.reverse ? -1 : 1;
  material.uniforms.volumetricOpacityMinimum.value = settings.opacityMinimum;
  material.uniforms.volumetricOpacityMaximum.value = settings.opacityMaximum;
  material.uniforms.volumetricOpacity.value = settings.opacity;
  material.uniforms.volumetricGamma.value = settings.gamma;
  material.uniforms.surfaceDisplaySign_UV.value.set(settings.flipU ? -1 : 1, settings.flipV ? 1 : -1);
  material.uniforms.volumetricAzimuth.value = settings.enabled ? settings.azimuth : 0;
  material.uniforms.volumetricTilt.value = settings.enabled ? settings.tilt : 0;
  material.uniforms.volumetricPerspective.value = settings.enabled ? settings.perspective : 0;
  material.uniforms.volumetricMaximumDepth.value = camera.maximumDepth;
  material.uniforms.volumetricRelief.value = settings.relief;
  material.uniforms.volumetricLighting.value = settings.lighting;
  material.uniforms.volumetricLightAzimuth.value = settings.lightAzimuth;
  material.uniforms.volumetricFeature.value = [
    "intensity",
    "normal-gradient",
    "tangent-gradient",
    "gradient-magnitude",
  ].indexOf(settings.feature);
  material.uniforms.volumetricGradientGain.value = settings.gradientGain;
  material.uniforms.volumetricConstantEmission.value = settings.constantEmission ? 1 : 0;
  material.uniforms.volumetricLayerDependentCamera.value =
    settings.enabled && (settings.tilt !== 0 || settings.perspective !== 0) ? 1 : 0;
  const transferValues = material.uniforms.volumetricTransferValues.value;
  const transferColors = material.uniforms.volumetricTransferColors.value;
  transferValues.fill(0);
  for (const color of transferColors) color.set(0, 0, 0, 0);
  material.uniforms.volumetricTransferCount.value = settings.transfer.points.length;
  for (const [index, point] of settings.transfer.points.entries()) {
    transferValues[index] = point.value;
    transferColors[index].set(...surfaceTransferColor(point.color), point.opacity);
  }
  updateSurfaceMaterialReliefMask(material, settings.mask.enabled);
}

export function updateSurfaceMaterialReliefMask(material, enabled) {
  material.uniforms.volumetricReliefMaskEnabled.value = enabled ? 1 : 0;
}
