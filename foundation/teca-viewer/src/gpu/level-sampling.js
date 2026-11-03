import * as THREE from "../../vendor/three.module.js";
import { FILL_PAGE, PAGE_HASH_SHADER } from "./page-table.js";

export function commonAtlasFormat(levels) {
  const first = levels[0].atlas.formatDescriptor;
  for (const { atlas } of levels.slice(1)) {
    const format = atlas.formatDescriptor;
    if (format.normalizedDtype !== first.normalizedDtype || format.samplerClass !== first.samplerClass) {
      throw new Error("Levels drawn together must share one atlas format.");
    }
  }
  return first;
}

export function levelUniforms(levels, count) {
  const uniforms = { levelCount: { value: levels.length } };
  for (let index = 0; index < count; index++) {
    const { atlas, pageTable, scale } = levels[Math.min(index, levels.length - 1)];
    uniforms[`atlas${index}`] = { value: atlas.texture };
    uniforms[`pages${index}`] = { value: pageTable.texture };
    uniforms[`shape${index}`] = { value: new THREE.Vector3(...scale.shape_ZYX) };
    uniforms[`chunks${index}`] = { value: new THREE.Vector3(...scale.chunks_ZYX) };
    uniforms[`downsample${index}`] = { value: new THREE.Vector3(...scale.downsample_ZYX) };
    uniforms[`offset${index}`] = { value: new THREE.Vector3(...scale.offset_ZYX) };
    uniforms[`fill${index}`] = { value: atlas.formatDescriptor.ArrayType.of(scale.fillValue)[0] };
    uniforms[`atlasWidth${index}`] = { value: atlas.width };
    uniforms[`atlasHeight${index}`] = { value: atlas.height };
    uniforms[`atlasSlotsPerLayer${index}`] = { value: atlas.slotsPerLayer };
  }
  return uniforms;
}

export function updateLevelUniforms(uniforms, levels, count) {
  uniforms.levelCount.value = levels.length;
  for (let index = 0; index < count; index++) {
    const { atlas, pageTable, scale } = levels[Math.min(index, levels.length - 1)];
    uniforms[`atlas${index}`].value = atlas.texture;
    uniforms[`pages${index}`].value = pageTable.texture;
    uniforms[`shape${index}`].value.set(...scale.shape_ZYX);
    uniforms[`chunks${index}`].value.set(...scale.chunks_ZYX);
    uniforms[`downsample${index}`].value.set(...scale.downsample_ZYX);
    uniforms[`offset${index}`].value.set(...scale.offset_ZYX);
    uniforms[`fill${index}`].value = atlas.formatDescriptor.ArrayType.of(scale.fillValue)[0];
    uniforms[`atlasWidth${index}`].value = atlas.width;
    uniforms[`atlasHeight${index}`].value = atlas.height;
    uniforms[`atlasSlotsPerLayer${index}`].value = atlas.slotsPerLayer;
  }
}

// Labels are read this way, since a blend of two ids is neither.
export function nearestShader(count) {
  const levels = Array.from(
    { length: count },
    (_, index) => `
      if (${index} < levelCount) {
        vec3 position_ZYX = clamp(floor((world_ZYX - offset${index} + 0.5) / downsample${index}), vec3(0.0), shape${index} - vec3(1.0));
        int slot = -1;
        if (sampleVoxel${index}(ivec3(position_ZYX), cachedChunk_ZYX, slot, raw)) return true;
      }`,
  );
  return `
    bool sampleNearest(vec3 world_ZYX, out float raw) {
      ivec3 cachedChunk_ZYX = ivec3(0);
      ${levels.join("")}
      return false;
    }
  `;
}

export function levelShader(count, samplerClass) {
  const sampler =
    samplerClass === "uint" ? "usampler2DArray" : samplerClass === "int" ? "isampler2DArray" : "sampler2DArray";
  const levels = Array.from({ length: count }, (_, index) => {
    const texel = `texelFetch(atlas${index}, ivec3(atlasCoordinate, slot / atlasSlotsPerLayer${index}), 0).r`;
    return `
      uniform ${sampler} atlas${index};
      uniform usampler2D pages${index};
      uniform vec3 shape${index};
      uniform vec3 chunks${index};
      uniform vec3 downsample${index};
      uniform vec3 offset${index};
      uniform float fill${index};
      uniform int atlasWidth${index};
      uniform int atlasHeight${index};
      uniform int atlasSlotsPerLayer${index};

      bool sampleVoxel${index}(ivec3 voxel_ZYX, inout ivec3 cachedChunk_ZYX, inout int slot, out float raw) {
        ivec3 chunk_ZYX = voxel_ZYX / ivec3(chunks${index});
        if (slot < 0 || any(notEqual(chunk_ZYX, cachedChunk_ZYX))) {
          uvec3 encodedChunk_ZYX = uvec3(chunk_ZYX) + uvec3(1u);
          ivec2 pageSize = textureSize(pages${index}, 0);
          uint pageMask = uint(pageSize.x * pageSize.y - 1);
          uint pageStart = pageHash(uvec3(chunk_ZYX)) & pageMask;
          slot = -1;
          for (uint probe = 0u; probe <= pageMask; probe++) {
            uint pageIndex = (pageStart + probe) & pageMask;
            uvec4 page = texelFetch(pages${index}, ivec2(int(pageIndex % uint(pageSize.x)), int(pageIndex / uint(pageSize.x))), 0);
            if (page.a == 0u) return false;
            if (all(equal(page.rgb, encodedChunk_ZYX))) {
              slot = page.a == ${FILL_PAGE}u ? FILL_SLOT : int(page.a) - 1;
              break;
            }
          }
          if (slot < 0) return false;
          cachedChunk_ZYX = chunk_ZYX;
        }
        if (slot == FILL_SLOT) {
          raw = fill${index};
          return true;
        }
        ivec3 local_ZYX = voxel_ZYX - chunk_ZYX * ivec3(chunks${index});
        int linearIndex = (local_ZYX.x * int(chunks${index}.y) + local_ZYX.y) * int(chunks${index}.z) + local_ZYX.z;
        ivec2 atlasCoordinate = ivec2(
          linearIndex % atlasWidth${index},
          linearIndex / atlasWidth${index} + (slot % atlasSlotsPerLayer${index}) * atlasHeight${index}
        );
        raw = ${samplerClass === "float" ? texel : `float(${texel})`};
        return true;
      }
    `;
  });
  return `
    uniform int levelCount;
    const int FILL_SLOT = 0x7fffffff;
    ${PAGE_HASH_SHADER}
    ${levels.join("\n")}
  `;
}
