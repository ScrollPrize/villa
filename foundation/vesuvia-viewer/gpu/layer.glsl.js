import { DTYPE_MAX } from '../config.js'

export const LAYER_UNIFORMS = ['level', 'mode', 'opacity', 'threshold', 'scale', 'color', 'min', 'max']
const LABEL_SATURATION = 0.7

export function labelColor(id) {
  const hue = (Math.imul(id, 2654435761) >>> 8) / 16777216
  return [0, 2 / 3, 1 / 3].map(shift => 1 + LABEL_SATURATION * (Math.min(1, Math.max(0, Math.abs(((hue + shift) % 1) * 6 - 3) - 1)) - 1))
}

const TEMPLATE = `
uniform int NAME_level;
uniform int NAME_mode;             // 0 probability, 1 labels
uniform float NAME_opacity;
uniform float NAME_threshold;
uniform float NAME_scale;
uniform vec3 NAME_color;
uniform vec3 NAME_min, NAME_max;

bool NAME_nearest(vec3 world_zyx, int fromLevel, out float value) {
  for (int level = fromLevel; level < NAME_levelCount; level++) {
    vec3 d_zyx = NAME_downsample[level];
    vec3 p_zyx = clamp((world_zyx - NAME_offset[level] - (d_zyx - 1.0) / 2.0) / d_zyx, vec3(0.0), NAME_shape[level] - 1.0);
    if (NAME_voxel(level, ivec3(floor(p_zyx + 0.5)), value)) return true;
  }
  return false;
}

vec3 NAME_labelColor(float id) {
  uint h = uint(id) * 2654435761u;              // Knuth multiplicative hash
  float hue = float(h >> 8u) / 16777216.0;
  vec3 rgb = clamp(abs(fract(hue + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
  return mix(vec3(1.0), rgb, ${LABEL_SATURATION});
}

vec3 NAME_blend(vec3 color, vec3 world_zyx) {
  if (NAME_opacity == 0.0) return color;
  if (any(lessThan(world_zyx, NAME_min)) || any(greaterThanEqual(world_zyx, NAME_max))) return color;
  float value;
  if (NAME_mode == 0) {
    if (!NAME_sample(world_zyx, NAME_level, value)) return color;
    return mix(color, NAME_color, NAME_opacity * smoothstep(NAME_threshold, 1.0, value * NAME_scale));
  }
  if (!NAME_nearest(world_zyx, NAME_level, value) || value == 0.0) return color;
  return mix(color, NAME_labelColor(value), NAME_opacity);
}
`

export function layerGlsl(name) {
  return TEMPLATE.replaceAll('NAME', name)
}

export function bindLayer(gl, program, loc, name, layer, level, unit) {
  const { style, residency } = layer
  const { offset_zyx, shape_zyx, downsample_zyx } = residency.levels[0]
  gl.uniform1i(loc[`${name}_level`], level)
  gl.uniform1i(loc[`${name}_mode`], style.mode === 'labels' ? 1 : 0)
  gl.uniform1f(loc[`${name}_opacity`], style.visible ? style.opacity : 0)
  gl.uniform1f(loc[`${name}_threshold`], style.threshold)
  gl.uniform1f(loc[`${name}_scale`], 1 / DTYPE_MAX[layer.volume.dtype])
  gl.uniform3fv(loc[`${name}_color`], style.color)
  gl.uniform3fv(loc[`${name}_min`], offset_zyx)
  gl.uniform3fv(loc[`${name}_max`], offset_zyx.map((o, a) => o + shape_zyx[a] * downsample_zyx[a]))
  return residency.bind(program, name, unit)
}
