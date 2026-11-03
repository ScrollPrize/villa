import * as THREE from "../../vendor/three.module.js";
import { atlasFormatForDtype } from "../gpu/atlas-format.js";
import { LABEL_COLOR_SHADER } from "../gpu/label-colors.js";
import { nearestShader } from "../gpu/level-sampling.js";
import { LAYER_COLOR_SHADER } from "../gpu/sliced-plane-material.js";
import { SurfaceChunkStream } from "./chunk-stream.js";
import {
  MATERIAL_LEVELS,
  surfaceEncodingUniforms,
  surfaceShaderMaterial,
  updateSurfaceMaterialLevels,
} from "./material.js";
import { SurfaceVolumeLevels } from "./volume-levels.js";

const LAYER_MAIN = `
  uniform int layerLabels;
  ${nearestShader(MATERIAL_LEVELS)}
  ${LABEL_COLOR_SHADER}
  ${LAYER_COLOR_SHADER}
  void main() {
    if (drawn == 0) discard;
    vec2 screen_UV = (vUv - vec2(0.5)) * surfaceVisibleSpan_UV;
    vec2 surface_UV = surfaceCenter_UV + surfaceCameraScreenToPoint(screen_UV, 0.0, surfaceVisibleSpan_UV);
    vec3 point_ZYX; vec3 tangentU_ZYX; vec3 tangentV_ZYX; vec3 normal_ZYX;
    if (!sampleSurfaceFrame(surface_UV, point_ZYX, tangentU_ZYX, tangentV_ZYX, normal_ZYX)) discard;
    vec3 sample_ZYX = point_ZYX + normal_ZYX * normalOffset;
    bool inside = all(greaterThanEqual(sample_ZYX, sourceMinimum_ZYX)) && all(lessThan(sample_ZYX, sourceMaximum_ZYX));
    float raw = 0.0;
    if (layerLabels == 1) {
      if (!inside || !sampleNearest(sample_ZYX, raw) || raw == 0.0) discard;
      outColor = vec4(labelColor(raw), layerOpacity);
      return;
    }
    if (!inside || !sampleWorld(sample_ZYX, raw)) discard;
    outColor = layerColor(clamp(encodeRaw(raw), 0.0, 1.0));
  }
`;

// The derivatives are taken before a pixel off the surface is discarded, since a mipmapped read needs its neighbours'.
const IMAGE_MAIN = `
  ${LAYER_COLOR_SHADER}
  uniform sampler2D surfaceImage;
  uniform sampler2D surfaceDetail;
  uniform vec2 detailStart_XY;
  uniform vec2 detailSpan_XY;
  void main() {
    if (drawn == 0) discard;
    vec2 screen_UV = (vUv - vec2(0.5)) * surfaceVisibleSpan_UV;
    vec2 surface_UV = surfaceCenter_UV + surfaceCameraScreenToPoint(screen_UV, 0.0, surfaceVisibleSpan_UV);
    vec2 image_UV = surface_UV / surfaceGrid_HW;
    vec2 acrossX = dFdx(image_UV);
    vec2 acrossY = dFdy(image_UV);
    vec2 detail_UV = (surface_UV / surfaceScale_UV - detailStart_XY) / max(detailSpan_XY, vec2(1.0));
    vec2 detailAcrossX = dFdx(detail_UV);
    vec2 detailAcrossY = dFdy(detail_UV);
    vec3 point_ZYX; vec3 tangentU_ZYX; vec3 tangentV_ZYX; vec3 normal_ZYX;
    if (!sampleSurfaceFrame(surface_UV, point_ZYX, tangentU_ZYX, tangentV_ZYX, normal_ZYX)) discard;
    if (any(lessThan(image_UV, vec2(0.0))) || any(greaterThanEqual(image_UV, vec2(1.0)))) discard;
    bool detailed = detailSpan_XY.x > 0.0 && all(greaterThanEqual(detail_UV, vec2(0.0))) &&
      all(lessThan(detail_UV, vec2(1.0)));
    float value = detailed
      ? textureGrad(surfaceDetail, detail_UV, detailAcrossX, detailAcrossY).r
      : textureGrad(surfaceImage, image_UV, acrossX, acrossY).r;
    outColor = layerColor(value);
  }
`;

function bytesPerVoxel(scale) {
  return atlasFormatForDtype(scale.numericType.dtype).bytesPerVoxel;
}

export function greyTexture(rendering, { data_HW, shape_HW }, freed) {
  const { maxTextureSize } = rendering.textureLimits;
  if (!shape_HW.every((size) => Number.isInteger(size) && size >= 1) || data_HW.length !== shape_HW[0] * shape_HW[1])
    throw new Error(`An image of ${data_HW.length} bytes is not ${shape_HW.join(" × ")} pixels.`);
  if (shape_HW.some((size) => size > maxTextureSize))
    throw new Error(`An image of ${shape_HW.join(" × ")} pixels passes this GPU's texture size, ${maxTextureSize}.`);
  const bytes = Math.ceil((data_HW.length * 4) / 3);
  if (bytes > rendering.availableBytes() + freed)
    throw new Error(
      `An image of ${Math.ceil(bytes / 2 ** 20)} MiB passes what the Live GPU budget has left. Increase it in Settings.`,
    );
  const texture = new THREE.DataTexture(data_HW, shape_HW[1], shape_HW[0], THREE.RedFormat, THREE.UnsignedByteType);
  Object.assign(texture, { minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter });
  Object.assign(texture, { generateMipmaps: true, needsUpdate: true });
  return { texture, bytes };
}

export class SurfaceLayers {
  constructor({ rendering, resources, source, volume, onChange, onError }) {
    this.rendering = rendering;
    this.resources = resources;
    this.source = source;
    this.volume = volume;
    this.onChange = onChange;
    this.onError = onError;
    this.layers = [];
    this.images = [];
  }

  add(volume, scales, { color_RGB, opacity, range }, renderOrder) {
    const layer = {
      volume,
      scales,
      byteShare: bytesPerVoxel(scales[0]) / bytesPerVoxel(this.volume.scales[0]),
      levels: null,
      material: null,
      mesh: null,
      stream: null,
    };
    layer.levels = new SurfaceVolumeLevels({
      brickPool: this.rendering.brickPool,
      namespace: this.rendering.brickPool.namespace(volume, 0),
      onChange: (evicted) => {
        updateSurfaceMaterialLevels(layer.material, layer.levels.shown);
        if (evicted === null) layer.stream.stream();
        this.onChange();
      },
      scales: this.shownScales(layer),
      byteLimit: Math.ceil(this.resources.levels.byteLimit * layer.byteShare),
      maxLayers: this.rendering.textureLimits.maxLayers,
      maxTextureSize: this.rendering.textureLimits.maxTextureSize,
    });
    layer.material = surfaceShaderMaterial(
      {
        ...this.resources.material.uniforms,
        ...surfaceEncodingUniforms(volume.metadata.displayEncoding),
        layerColor_RGB: { value: new THREE.Vector3(...(color_RGB ?? [0, 0, 0])) },
        layerOpacity: { value: opacity },
        layerLabels: { value: color_RGB === null ? 1 : 0 },
        layerRange: { value: new THREE.Vector2(...(range ?? [0, 255])) },
        drawn: { value: 1 },
      },
      layer.levels.shown,
      LAYER_MAIN,
    );
    layer.material.transparent = true;
    layer.mesh = new THREE.Mesh(this.resources.mesh.geometry, layer.material);
    layer.mesh.frustumCulled = false;
    layer.mesh.renderOrder = renderOrder;
    this.resources.pane.baseScene.add(layer.mesh);
    layer.stream = new SurfaceChunkStream({
      volume,
      selectedChannel: 0,
      advanceGeneration: () => volume.navigationGeneration + 1,
      levels: layer.levels,
      renderer: this.rendering.renderer,
      uploadBudget: this.rendering.uploadBudget,
      onUploaded: this.onChange,
      onError: this.onError,
    });
    this.layers.push(layer);
    try {
      this.follow();
    } catch (error) {
      this.remove(layer);
      throw error;
    }
    return layer;
  }

  addImage({ texture, shape_HW, detail, style, renderOrder }) {
    const { color_RGB, opacity, range } = style;
    const { grid_HW, scale_UV } = this.source;
    const full_HW = [grid_HW[0] / scale_UV[1], grid_HW[1] / scale_UV[0]];
    const downsample = full_HW[0] / shape_HW[0];
    if (Math.abs(full_HW[1] - downsample * shape_HW[1]) > downsample) {
      this.onError(
        new Error(
          `An image of ${shape_HW.join(" × ")} pixels is not drawn on a surface of ${grid_HW.join(" × ")} cells at ` +
            `scale ${scale_UV.map((scale) => scale.toPrecision(3)).join(" × ")}: its sides are not ` +
            `${full_HW.map(Math.round).join(" × ")} over one downsampling.`,
        ),
      );
      return null;
    }
    const own = {
      layerColor_RGB: { value: new THREE.Vector3(...color_RGB) },
      layerOpacity: { value: opacity },
      layerLabels: { value: 0 },
      layerRange: { value: new THREE.Vector2(...range) },
      surfaceImage: { value: texture },
      surfaceDetail: { value: null },
      detailStart_XY: { value: new THREE.Vector2() },
      detailSpan_XY: { value: new THREE.Vector2() },
      drawn: { value: 1 },
    };
    const material = surfaceShaderMaterial(
      { ...this.resources.material.uniforms, ...own },
      this.resources.levels.shown,
      IMAGE_MAIN,
    );
    Object.assign(material.uniforms, this.resources.material.uniforms, own);
    material.transparent = true;
    const mesh = new THREE.Mesh(this.resources.mesh.geometry, material);
    mesh.frustumCulled = false;
    mesh.renderOrder = renderOrder;
    this.resources.pane.baseScene.add(mesh);
    const image = { material, mesh };
    this.images.push(image);
    this.setDetail(image, detail);
    return image;
  }

  setDetail(image, detail) {
    const { surfaceDetail, detailStart_XY, detailSpan_XY } = image.material.uniforms;
    surfaceDetail.value = detail?.texture ?? null;
    if (detail === null) detailSpan_XY.value.set(0, 0);
    else {
      const { start_HW, shape_HW, downsample } = detail;
      detailStart_XY.value.set(start_HW[1], start_HW[0]);
      detailSpan_XY.value.set(shape_HW[1] * downsample, shape_HW[0] * downsample);
    }
    this.onChange();
  }

  removeImage(image) {
    this.images = this.images.filter((candidate) => candidate !== image);
    image.mesh.removeFromParent();
    image.material.dispose();
    this.onChange();
  }

  setStyle(layer, { color_RGB, opacity, range }) {
    const uniforms = layer.material.uniforms;
    uniforms.layerLabels.value = color_RGB === null ? 1 : 0;
    if (color_RGB !== null) {
      uniforms.layerColor_RGB.value.set(...color_RGB);
      uniforms.layerRange.value.set(...range);
    }
    uniforms.layerOpacity.value = opacity;
    this.onChange();
  }

  refresh(layer, changes) {
    layer.stream.refresh(changes);
  }

  remove(layer) {
    this.layers = this.layers.filter((candidate) => candidate !== layer);
    layer.stream.dispose();
    layer.mesh.removeFromParent();
    layer.material.dispose();
    layer.levels.dispose();
    this.onChange();
  }

  follow() {
    const shown = this.resources.levels.shown;
    if (shown.some((level) => level.coverage === null)) return;
    const plans = shown.map((level) => level.coverage);
    for (const layer of this.layers) {
      layer.levels.setByteLimit(Math.ceil(this.resources.levels.byteLimit * layer.byteShare));
      const stage = layer.levels.stage(this.shownScales(layer), shown.at(-1).capacity);
      if (stage !== null) {
        try {
          layer.levels.commit(stage, plans, this.rendering.renderer);
        } finally {
          updateSurfaceMaterialLevels(layer.material, layer.levels.shown);
          layer.stream.levelsReplaced();
        }
      }
      layer.stream.replace(layer.levels.admitPlans(plans));
    }
  }

  pending() {
    return this.layers.reduce((total, layer) => total + layer.stream.pending(), 0);
  }

  uploads() {
    return this.layers.reduce((total, layer) => total + layer.stream.uploads.length, 0);
  }

  dispose() {
    for (const layer of this.layers) this.remove(layer);
    for (const image of this.images) this.removeImage(image);
  }

  shownScales(layer) {
    return this.resources.levels.shown.map(
      (level) => layer.scales[this.volume.scales.indexOf(level.scale.sourceScale ?? level.scale)],
    );
  }
}
