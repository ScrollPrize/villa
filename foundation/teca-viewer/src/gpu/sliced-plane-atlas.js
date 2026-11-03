import { CT_ATLAS_BYTES, CT_LEVEL_LIMIT, ORTHOGONAL_PLANES } from "../config.js";
import { chunkKey } from "./addresses.js";
import { atlasFormatForDtype } from "./atlas-format.js";
import { chunkAtlasLayout } from "./chunk-atlas.js";
import { pageTableBytes } from "./page-table.js";

const PAGE_TABLE_COUNT = CT_LEVEL_LIMIT * ORTHOGONAL_PLANES.length;

export function atlasSlotIdentity(scale) {
  const format = atlasFormatForDtype(scale.numericType.dtype);
  return JSON.stringify({
    chunks_ZYX: scale.chunks_ZYX,
    dtype: scale.numericType.dtype,
    normalizedDtype: format.normalizedDtype,
    samplerClass: format.samplerClass,
  });
}

export function scaleIdentity(scale) {
  const format = atlasFormatForDtype(scale.numericType.dtype);
  return JSON.stringify({
    path: scale.path,
    shape_ZYX: scale.shape_ZYX,
    chunks_ZYX: scale.chunks_ZYX,
    downsample_ZYX: scale.downsample_ZYX,
    offset_ZYX: scale.offset_ZYX,
    dtype: scale.numericType.dtype,
    normalizedDtype: format.normalizedDtype,
    samplerClass: format.samplerClass,
  });
}

export function atlasSlotGroups(levels) {
  const groups = new Map();
  for (const level of levels) {
    const slotKey = atlasSlotIdentity(level.scale);
    const group = groups.get(slotKey) ?? [];
    group.push(level);
    groups.set(slotKey, group);
  }
  return groups;
}

export function atlasCapacity(scale, groupCount, limits, bytes) {
  const format = atlasFormatForDtype(scale.numericType.dtype);
  const voxelCount = scale.chunks_ZYX.reduce((product, value) => product * value, 1);
  const layout = chunkAtlasLayout(voxelCount, limits.maxTextureSize);
  const slotBytes = layout.width * layout.height * format.bytesPerVoxel;
  const byteLimit = Math.floor(bytes / groupCount);
  for (let capacity = Math.min(limits.maxLayers, Math.floor(byteLimit / slotBytes)); capacity >= 1; capacity--) {
    if (capacity * slotBytes + PAGE_TABLE_COUNT * pageTableBytes(capacity, limits.maxTextureSize) <= byteLimit)
      return { capacity, slotBytes };
  }
  throw new Error("CT pane budget cannot hold one chunk and its page tables.");
}

export function atlasAdmission(scales, limits) {
  const groups = new Map(scales.map((scale) => [atlasSlotIdentity(scale), scale]));
  try {
    for (const scale of groups.values()) atlasCapacity(scale, groups.size, limits, CT_ATLAS_BYTES);
  } catch (error) {
    return { supported: false, reason: error.message };
  }
  return { supported: true, reason: null };
}

export function settledLevelStack({
  scales,
  scaleIndex,
  coarsestScaleIndex,
  planes,
  planeScaleIndex,
  addressesFor,
  limits,
}) {
  for (let displayScaleIndex = scaleIndex; ; displayScaleIndex++) {
    const lastScaleIndex = Math.min(displayScaleIndex + CT_LEVEL_LIMIT, coarsestScaleIndex + 1);
    const planePlans = planes.map((plane) => {
      const firstScaleIndex = Math.min(Math.max(displayScaleIndex, planeScaleIndex(plane)), lastScaleIndex - 1);
      return {
        plane,
        levels: Array.from({ length: lastScaleIndex - firstScaleIndex }, (_, offset) => {
          const scale = scales[firstScaleIndex + offset];
          return { scaleIndex: firstScaleIndex + offset, scale, addresses: addressesFor(scale, plane) };
        }),
      };
    });
    const groups = atlasSlotGroups(planePlans.flatMap((plan) => plan.levels));
    const fits = [...groups.values()].every((levels) => {
      const keys = new Set();
      for (const level of levels) {
        for (const { chunk_ZYX } of level.addresses) keys.add(chunkKey(level.scale, chunk_ZYX));
      }
      return keys.size <= atlasCapacity(levels[0].scale, groups.size, limits, CT_ATLAS_BYTES).capacity;
    });
    if (fits || displayScaleIndex === coarsestScaleIndex) {
      return { planePlans, levelSelection: { requestedScaleIndex: scaleIndex, scaleIndex: displayScaleIndex } };
    }
  }
}
