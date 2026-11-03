// Every vec3/ivec3 of coordinates is ZYX, indexed [0]=z [1]=y [2]=x; the .zyx swizzle appears only inside texelFetch, where GL wants (x, y, z).
import { WINDOW_CHUNKS } from '../config.js'

export const MAX_LEVELS = 8
export const PAGE_FILL = 65535   // page entry of a fill chunk; 0 is absent, else slot + 1

const SAMPLERS = { uint8: 'usampler3D', uint16: 'usampler3D', float32: 'sampler3D' }

const TEMPLATE = `
uniform highp SAMPLER NAME_atlas;
uniform highp usampler2DArray NAME_pages;
uniform int   NAME_levelCount;
uniform vec3  NAME_shape[MAX_LEVELS];
uniform vec3  NAME_chunk[MAX_LEVELS];
uniform vec3  NAME_downsample[MAX_LEVELS];
uniform vec3  NAME_offset[MAX_LEVELS];
uniform ivec3 NAME_window[MAX_LEVELS];
uniform ivec3 NAME_slots;
uniform ivec3 NAME_slotSize;
uniform float NAME_fill;

uint NAME_page(int level, ivec3 chunk_zyx, out ivec3 origin_zyx) {
  ivec3 local_zyx = chunk_zyx - NAME_window[level];
  if (any(lessThan(local_zyx, ivec3(0))) || any(greaterThanEqual(local_zyx, ivec3(WINDOW_CHUNKS)))) return 0u;
  uint entry = texelFetch(NAME_pages, ivec3(local_zyx[2], local_zyx[0] * WINDOW_CHUNKS + local_zyx[1], level), 0).r;
  if (entry == 0u || entry == PAGE_FILLu) return entry;
  int slot = int(entry) - 1;
  origin_zyx = ivec3(slot / (NAME_slots[1] * NAME_slots[2]), (slot / NAME_slots[2]) % NAME_slots[1], slot % NAME_slots[2]) * NAME_slotSize;
  return entry;
}

bool NAME_voxel(int level, ivec3 v_zyx, out float value) {
  ivec3 chunk_zyx = ivec3(NAME_chunk[level]);
  ivec3 origin_zyx;
  uint entry = NAME_page(level, v_zyx / chunk_zyx, origin_zyx);
  if (entry == 0u) return false;
  if (entry == PAGE_FILLu) {
    value = NAME_fill;
    return true;
  }
  ivec3 texel_zyx = origin_zyx + v_zyx % chunk_zyx;
  value = float(texelFetch(NAME_atlas, texel_zyx.zyx, 0).r);
  return true;
}

bool NAME_at(int level, vec3 world_zyx, out float value) {
  if (any(lessThan(world_zyx, NAME_offset[0])) || any(greaterThanEqual(world_zyx, NAME_offset[0] + NAME_shape[0] * NAME_downsample[0]))) return false;
  vec3 d_zyx = NAME_downsample[level];
  vec3 last_zyx = NAME_shape[level] - 1.0;
  vec3 p_zyx = clamp((world_zyx - NAME_offset[level] - (d_zyx - 1.0) / 2.0) / d_zyx, vec3(0.0), last_zyx);
  vec3 f_zyx = floor(p_zyx);
  vec3 t_zyx = p_zyx - f_zyx;
  ivec3 v0_zyx = ivec3(f_zyx);
  ivec3 v1_zyx = min(v0_zyx + 1, ivec3(last_zyx));
  ivec3 chunk_zyx = ivec3(NAME_chunk[level]);
  ivec3 c_zyx = v0_zyx / chunk_zyx;
  float c[8];
  if (all(equal(c_zyx, v1_zyx / chunk_zyx))) {
    ivec3 origin_zyx;
    uint entry = NAME_page(level, c_zyx, origin_zyx);
    if (entry == 0u) return false;
    if (entry == PAGE_FILLu) { value = NAME_fill; return true; }
    ivec3 a_zyx = origin_zyx + v0_zyx % chunk_zyx;
    ivec3 b_zyx = origin_zyx + v1_zyx % chunk_zyx;
    c[0] = float(texelFetch(NAME_atlas, ivec3(a_zyx[2], a_zyx[1], a_zyx[0]), 0).r);
    c[1] = float(texelFetch(NAME_atlas, ivec3(b_zyx[2], a_zyx[1], a_zyx[0]), 0).r);
    c[2] = float(texelFetch(NAME_atlas, ivec3(a_zyx[2], b_zyx[1], a_zyx[0]), 0).r);
    c[3] = float(texelFetch(NAME_atlas, ivec3(b_zyx[2], b_zyx[1], a_zyx[0]), 0).r);
    c[4] = float(texelFetch(NAME_atlas, ivec3(a_zyx[2], a_zyx[1], b_zyx[0]), 0).r);
    c[5] = float(texelFetch(NAME_atlas, ivec3(b_zyx[2], a_zyx[1], b_zyx[0]), 0).r);
    c[6] = float(texelFetch(NAME_atlas, ivec3(a_zyx[2], b_zyx[1], b_zyx[0]), 0).r);
    c[7] = float(texelFetch(NAME_atlas, ivec3(b_zyx[2], b_zyx[1], b_zyx[0]), 0).r);
  } else {
    for (int i = 0; i < 8; i++) {
      ivec3 corner_zyx = ivec3(i >> 2, (i >> 1) & 1, i & 1);
      if (!NAME_voxel(level, v0_zyx + corner_zyx * (v1_zyx - v0_zyx), c[i])) return false;
    }
  }
  value = mix(
    mix(mix(c[0], c[1], t_zyx[2]), mix(c[2], c[3], t_zyx[2]), t_zyx[1]),
    mix(mix(c[4], c[5], t_zyx[2]), mix(c[6], c[7], t_zyx[2]), t_zyx[1]),
    t_zyx[0]);
  return true;
}

bool NAME_sample(vec3 world_zyx, int fromLevel, out float value) {
  for (int level = fromLevel; level < NAME_levelCount; level++) {
    if (NAME_at(level, world_zyx, value)) return true;
  }
  return false;
}
`

export function sampleGlsl(name, dtype) {
  return TEMPLATE
    .replaceAll('NAME', name)
    .replace('SAMPLER', SAMPLERS[dtype])
    .replaceAll('MAX_LEVELS', String(MAX_LEVELS))
    .replaceAll('WINDOW_CHUNKS', String(WINDOW_CHUNKS))
    .replaceAll('PAGE_FILL', String(PAGE_FILL))
}
