import { Residency } from '../gpu/residency.js'
import { createMemoryVolume } from '../volume/memory.js'
import { createContext, compileProgram, FULLSCREEN_VERTEX } from '../gpu/gl.js'
import { UPLOAD_BYTES_PER_FRAME as BYTES, UPLOAD_CALLS_PER_FRAME as CALLS } from '../config.js'

export default async function () {
  const canvas = document.createElement('canvas');
  canvas.width = 1;
  canvas.height = 1;
  const gl = createContext(canvas), errors = [], owned = [];
  const original = gl.texSubImage3D;
  let calls = [];
  gl.texSubImage3D = function(...a) {
    calls.push({
      target: a[0] === gl.TEXTURE_3D ? 'atlas' : 'page',
      bytes: a[5] * a[6] * a[7] * (a[9] === gl.UNSIGNED_BYTE ? 1 : a[9] === gl.UNSIGNED_SHORT ? 2 : 4)
    });
    return original.apply(gl, a);
  };
  const check = (ok, message) => { if (!ok) errors.push(message) };
  const volume = (name, shape_zyx, chunk_zyx, dtype = 'uint8') => createMemoryVolume({
    name, dtype, fill: 0, labels: false,
    levels: [{ shape_zyx, chunk_zyx, downsample_zyx: [1, 1, 1], offset_zyx: [0, 0, 0] }]
  });
  const residency = (v, atlasBytes) => {
    const r = new Residency(gl, v, { atlasBytes });
    owned.push(r);
    return r
  };
  const plan = (r, coords, origin = [0, 0, 0]) => {
    const before = calls.length;
    r.ensure({
      windows: [origin],
      wanted: coords.map(([cz, cy, cx]) => ({ level: 0, cz, cy, cx, priority: 0, views: [0] })),
      ahead: []
    });
    check(calls.length === before, 'ensure uploaded ' + (calls.length - before) + ' calls');
  };
  const recount = (name, r) => {
    const stats = r.stats(), key = c => r.volume.key(c.level, c.cz, c.cy, c.cx);
    const missing = r.plan.wanted.filter(c => !r.slotByKey.has(key(c)) && !r.fills.has(key(c))).length;
    const pending = r.pendingWindows.size + r.admission.length + (r.upload === null ? 0 : 1);
    check(stats.missing === missing && stats.pending === pending && stats.wanted === r.plan.wanted.length,
      name + ' stale stats ' + JSON.stringify({ stats, missing, pending, wanted: r.plan.wanted.length }));
  };
  const frame = (name, rs, bytes = BYTES, count = CALLS) => {
    calls = [];
    const budget = { bytes, calls: count };
    for (const r of rs) r.tick(budget);
    const actualBytes = calls.reduce((n, c) => n + c.bytes, 0);
    check(calls.length <= count && actualBytes <= bytes,
      name + ' exceeded budget ' + JSON.stringify({ budget, calls }));
    check(bytes - budget.bytes === actualBytes && count - budget.calls === calls.length,
      name + ' debit mismatch ' + JSON.stringify({ budget, calls }));
    const error = gl.getError();
    check(error === 0, name + ' GL error ' + error);
    for (const r of rs) recount(name, r);
  };
  const finish = (name, r) => {
    let n = 0;
    while (r.stats().pending > 0 && n < 100) {
      frame(name, [r]);
      n++
    }
    check(n < 100, name + ' did not finish');
    return n
  };
  const programs = new Map();
  const read = (r, point) => {
    let p = programs.get(r.volume.dtype);
    if (p === undefined) {
      p = compileProgram(gl, FULLSCREEN_VERTEX, `#version 300 es
precision highp float;
precision highp int;
${r.glsl('data')}
uniform vec3 point;
out vec4 color;
void main() {
  float value;
  bool found = data_sample(point, 0, value);
  color = found ? vec4(value / 255.0, 0, 0, 1) : vec4(0, 1, 0, 1);
}
`);
      programs.set(r.volume.dtype, p)
    }
    gl.useProgram(p);
    r.bind(p, 'data', 0);
    gl.uniform3fv(gl.getUniformLocation(p, 'point'), point);
    gl.viewport(0, 0, 1, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const pixel = new Uint8Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
    check(gl.getError() === 0, 'read GL error');
    return pixel[1] === 255 ? null : pixel[0];
  };
  {
    const vs = [volume('shared-a', [16, 16, 256], [16, 16, 16]), volume('shared-b', [16, 16, 256], [16, 16, 16])];
    const rs = vs.map(v => residency(v, 16 * 16 * 256));
    vs.forEach((v, j) => {
      for (let x = 0; x < 16; x += 2)
        v.write(0, [0, 0, x * 16], [16, 16, 16], new Uint8Array(4096).fill(20 + j + x));
    });
    rs.forEach(r => plan(r, Array.from({ length: 16 }, (_, x) => [0, 0, x])));
    let n = 0;
    while (rs.some(r => r.stats().pending > 0) && n < 40) {
      frame('shared', n % 2 === 0 ? rs : [...rs].reverse());
      n++
    }
    check(n < 40, 'shared did not finish');
    rs.forEach((r, j) => {
      for (let x = 0; x < 16; x++)
        check(read(r, [8, 8, x * 16 + 8]) === (x % 2 === 0 ? 20 + j + x : 0), 'shared sample ' + j + '/' + x)
    });
  }
  {
    const v = volume('many-fills', [128, 128, 1024], [16, 16, 16]), r = residency(v, 4096);
    const coords = Array.from({ length: 4096 }, (_, i) => [Math.floor(i / 512), Math.floor(i / 64) % 8, i % 64]);
    plan(r, coords);
    check(read(r, [8, 8, 8]) === null, 'queued fills were visible before upload');
    frame('many-fills', [r]);
    check(calls.length === 1 && calls[0].bytes === 524288, 'known fills were not batched into one window');
    check(r.stats().pending === 0 && r.stats().missing === 0 && read(r, [120, 120, 1016]) === 0,
      'batched fill window incomplete');
  }
  {
    const v = volume('evict', [16, 16, 32], [16, 16, 16]), r = residency(v, 4096);
    v.write(0, [0, 0, 0], [16, 16, 16], new Uint8Array(4096).fill(37));
    v.write(0, [0, 0, 16], [16, 16, 16], new Uint8Array(4096).fill(91));
    plan(r, [[0, 0, 0]]);
    finish('evict-initial', r);
    check(read(r, [8, 8, 8]) === 37, 'initial evict sample');
    plan(r, [[0, 0, 1]]);
    frame('evict-clear', [r], BYTES, 1);
    const cleared = [read(r, [8, 8, 8]), read(r, [8, 8, 24])];
    check(cleared.every(x => x === null), 'eviction did not clear ' + cleared);
    frame('evict-part', [r], 2048, 1);
    check(read(r, [8, 8, 24]) === null, 'partial eviction replacement visible');
    finish('evict-finish', r);
    const after = [read(r, [8, 8, 8]), read(r, [8, 8, 24])];
    check(after[0] === null && after[1] === 91, 'eviction aliased ' + after);
  }
  {
    const v = volume('window', [4, 4, 512], [4, 4, 4]), r = residency(v, 64);
    v.write(0, [0, 0, 0], [4, 4, 4], new Uint8Array(64).fill(61));
    v.write(0, [0, 0, 256], [4, 4, 4], new Uint8Array(64).fill(149));
    plan(r, [[0, 0, 0]]);
    finish('window-initial', r);
    plan(r, [[0, 0, 64]], [0, 0, 64]);
    frame('window-postponed', [r], 524287, 8);
    const delayed = [read(r, [2, 2, 2]), read(r, [2, 2, 258])];
    check(delayed[0] === 61 && delayed[1] === null, 'unpublished window changed ' + delayed);
    finish('window-moved', r);
    const moved = [read(r, [2, 2, 2]), read(r, [2, 2, 258])];
    check(moved[0] === null && moved[1] === 149, 'moved window alias ' + moved);
  }
  {
    const v = volume('large', [256, 256, 128], [256, 256, 128]), r = residency(v, 8 * 2**20);
    v.write(0, [0, 0, 0], [256, 256, 128], new Uint8Array(8 * 2**20).fill(173));
    plan(r, [[0, 0, 0]]);
    frame('large-first', [r], 4 * 2**20);
    const initial = { z: r.upload.z, y: r.upload.y, value: read(r, [20, 20, 20]), missing: r.stats().missing };
    check(initial.value === null && initial.missing === 1, 'large partial was published');
    plan(r, [[0, 0, 0]], [-1, -1, -1]);
    frame('large-replan', [r], 4 * 2**20);
    check(read(r, [20, 20, 20]) === null, 'large partial replan was published');
    finish('large-finish', r);
    check(read(r, [20, 20, 20]) === 173 && read(r, [255, 255, 127]) === 173, 'large read failed');
    const slot = r.slotByKey.get(v.key(0, 0, 0, 0)).slot, revision = r.revision, rewrite = r.revisionAt(0);
    v.write(0, [3, 0, 0], [2, 256, 128], new Uint8Array(2 * 256 * 128).fill(211));
    frame('stale-part', [r], 1280, 1);
    const part = {
      slot: r.slotByKey.get(v.key(0, 0, 0, 0)).slot, revision: r.revision,
      rewritten: r.revisionAt(0) - rewrite, new: read(r, [3, 2, 2]), old: read(r, [3, 20, 2])
    };
    check(part.slot === slot && part.revision === revision && part.rewritten > 0 && part.new === 211 && part.old === 173,
      'stale partial incorrect ' + JSON.stringify(part));
    plan(r, []);
    check(r.upload === null && r.slotByKey.get(v.key(0, 0, 0, 0)).slot === slot, 'stale cancel lost slot');
    plan(r, [[0, 0, 0]]);
    finish('stale-finish', r);
    check(read(r, [3, 200, 2]) === 211 && read(r, [2, 200, 2]) === 173, 'stale replan samples failed');
  }
  {
    const v = volume('row', [1, 2048, 2048], [1, 2048, 2048], 'uint16'), r = residency(v, 8 * 2**20);
    v.write(0, [0, 0, 0], [1, 2048, 2048], new Uint16Array(2048 * 2048).fill(151));
    plan(r, [[0, 0, 0]]);
    frame('wide-first', [r], 4 * 2**20);
    const partial = { z: r.upload.z, y: r.upload.y };
    check(partial.z === 0 && partial.y > 0, 'wide plane was not split');
    finish('wide-finish', r);
    check(read(r, [0, 100, 100]) === 151 && read(r, [0, 2047, 2047]) === 151, 'wide plane samples');
  }
  {
    const v = volume('cancel', [16, 16, 32], [16, 16, 16]), r = residency(v, 4096);
    v.write(0, [0, 0, 0], [16, 16, 16], new Uint8Array(4096).fill(72));
    v.write(0, [0, 0, 16], [16, 16, 16], new Uint8Array(4096).fill(93));
    plan(r, [[0, 0, 0]]);
    frame('cancel-part', [r], 524288 + 2048, 2);
    check(r.upload !== null && r.upload.z < r.upload.zEnd, 'cancel no partial upload');
    plan(r, [[0, 0, 1]]);
    check(r.freeSlots.length === 1 && r.upload === null, 'cancel did not release slot');
    finish('cancel-finish', r);
    const after = [read(r, [8, 8, 8]), read(r, [8, 8, 24])];
    check(after[0] === null && after[1] === 93, 'cancel aliased ' + after);
  }
  {
    const v = volume('fill-write-race', [16, 16, 16], [16, 16, 16]), r = residency(v, 4096);
    plan(r, [[0, 0, 0]]);
    v.write(0, [0, 0, 0], [16, 16, 16], new Uint8Array(4096).fill(177));
    finish('fill-write-race', r);
    const first = read(r, [8, 8, 8]);
    plan(r, [[0, 0, 0]]);
    finish('fill-write-replan', r);
    const after = read(r, [8, 8, 8]);
    check(first === 177 && after === 177, 'queued fill swallowed write: first=' + first + ' after replan=' + after);
  }
  check(gl.getError() === 0, 'final GL error');
  gl.texSubImage3D = original;
  for (const r of owned) r.close();
  for (const p of programs.values()) gl.deleteProgram(p);
  return errors;
}
