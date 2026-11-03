export const AXIS_TOLERANCE = 1e-9;
const PAGE = 64;

function pairs() {
  let data = new Int32Array(1024);
  let length = 0;
  return {
    get length() {
      return length;
    },
    push(first, second) {
      if (length + 2 > data.length) {
        const grown = new Int32Array(data.length * 2);
        grown.set(data);
        data = grown;
      }
      data[length++] = first;
      data[length++] = second;
    },
    first: () => data[length - 2],
    second: () => data[length - 1],
    pop() {
      length -= 2;
    },
    take() {
      const out = data.subarray(0, length).slice();
      length = 0;
      return out;
    },
  };
}

export async function planeFill(start_ZYX, normal_ZYX, shape_ZYX, open, load, limit, pause) {
  const along = [0, 1, 2].reduce((best, axis) =>
    Math.abs(normal_ZYX[axis]) > Math.abs(normal_ZYX[best]) ? axis : best,
  );
  const [first, second] = [0, 1, 2].filter((axis) => axis !== along);
  const flat = Math.abs(normal_ZYX[along]) > 1 - AXIS_TOLERANCE;
  const slopeA = flat ? 0 : -normal_ZYX[first] / normal_ZYX[along];
  const slopeB = flat ? 0 : -normal_ZYX[second] / normal_ZYX[along];
  const [startA, startB, startC] = [start_ZYX[first], start_ZYX[second], start_ZYX[along]];
  const [sizeA, sizeB, depth] = [shape_ZYX[first], shape_ZYX[second], shape_ZYX[along]];
  const voxel_ZYX = new Int32Array(3);
  const depthAt = (a, b) => Math.round(startC + slopeA * (a - startA) + slopeB * (b - startB));

  const pagesB = Math.ceil(sizeB / PAGE);
  const pages = new Map();
  let pageKey = -1;
  let page = null;
  const pageOf = (a, b) => {
    const key = Math.floor(a / PAGE) * pagesB + Math.floor(b / PAGE);
    if (key !== pageKey) {
      pageKey = key;
      page = pages.get(key) ?? null;
      if (page === null) {
        page = new Uint32Array((PAGE * PAGE) / 32);
        pages.set(key, page);
      }
    }
    return page;
  };
  const taken = (a, b) => {
    const bit = (a % PAGE) * PAGE + (b % PAGE);
    return (pageOf(a, b)[bit >>> 5] & (1 << (bit & 31))) !== 0;
  };
  const take = (a, b) => {
    const bit = (a % PAGE) * PAGE + (b % PAGE);
    pageOf(a, b)[bit >>> 5] |= 1 << (bit & 31);
  };

  const seeds = pairs();
  const waiting = pairs();
  const runs = [];
  let count = 0;
  const may = (a, b) => {
    const c = depthAt(a, b);
    if (c < 0 || c >= depth) return false;
    voxel_ZYX[first] = a;
    voxel_ZYX[second] = b;
    voxel_ZYX[along] = c;
    const verdict = open(voxel_ZYX[0], voxel_ZYX[1], voxel_ZYX[2]);
    if (verdict === null) waiting.push(a, b);
    return verdict === true;
  };

  seeds.push(startA, startB);
  while (seeds.length > 0 || waiting.length > 0) {
    if (seeds.length === 0) {
      const cells = waiting.take();
      const voxels_N3 = new Int32Array((cells.length / 2) * 3);
      for (let index = 0; index < cells.length; index += 2) {
        voxels_N3[(index / 2) * 3 + first] = cells[index];
        voxels_N3[(index / 2) * 3 + second] = cells[index + 1];
        voxels_N3[(index / 2) * 3 + along] = depthAt(cells[index], cells[index + 1]);
      }
      await load(voxels_N3);
      for (let index = 0; index < cells.length; index += 2) seeds.push(cells[index], cells[index + 1]);
    }
    const a = seeds.first();
    const b = seeds.second();
    seeds.pop();
    if (taken(a, b) || !may(a, b)) continue;
    let from = b;
    let to = b;
    while (from > 0 && !taken(a, from - 1) && may(a, from - 1)) from--;
    while (to < sizeB - 1 && !taken(a, to + 1) && may(a, to + 1)) to++;
    for (let cell = from; cell <= to; cell++) take(a, cell);
    count += to - from + 1;
    if (count > limit) return null;
    runs.push(a, from, to);
    for (const next of [a - 1, a + 1]) {
      if (next < 0 || next >= sizeA) continue;
      let stretch = false;
      for (let cell = from; cell <= to; cell++) {
        const opens = !taken(next, cell) && may(next, cell);
        if (opens && !stretch) seeds.push(next, cell);
        stretch = opens;
      }
    }
    await pause();
  }
  const voxels_N3 = new Int32Array(count * 3);
  let index = 0;
  for (let run = 0; run < runs.length; run += 3) {
    await pause();
    for (let b = runs[run + 1]; b <= runs[run + 2]; b++, index += 3) {
      voxels_N3[index + first] = runs[run];
      voxels_N3[index + second] = b;
      voxels_N3[index + along] = depthAt(runs[run], b);
    }
  }
  return voxels_N3;
}
