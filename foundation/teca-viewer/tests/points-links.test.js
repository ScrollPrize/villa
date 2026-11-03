import assert from "node:assert/strict";
import { test } from "node:test";
import { treeLinks } from "../src/extensions/points/links.js";

function line(xs, ys = xs.map(() => 0)) {
  return Float32Array.from(xs.flatMap((x, at) => [0, ys[at], x]));
}

test("three scribbles along x are linked each to the next, between their nearest ends", () => {
  const points_zyx_N3 = line([0, 1, 10, 11, 30]);
  const edges_E2 = Uint32Array.from([0, 1, 2, 3]);
  assert.deepEqual([...treeLinks({ points_zyx_N3, edges_E2 })], [1, 2, 3, 4]);
});

test("a link joins the nearest nodes of its scribbles, inside a scribble as at its ends", () => {
  const xs = [...Array.from({ length: 20 }, (_, x) => x), 9.4];
  const points_zyx_N3 = line(xs, [...Array(20).fill(0), 5]);
  const edges_E2 = Uint32Array.from(Array.from({ length: 19 }, (_, at) => [at, at + 1]).flat());
  assert.deepEqual([...treeLinks({ points_zyx_N3, edges_E2 })], [9, 20]);
});

test("one scribble, or none, has no links", () => {
  assert.deepEqual([...treeLinks({ points_zyx_N3: line([0, 1]), edges_E2: Uint32Array.from([0, 1]) })], []);
  assert.deepEqual([...treeLinks({ points_zyx_N3: new Float32Array(0), edges_E2: new Uint32Array(0) })], []);
});

test("120 scribbles of 1 to 40 nodes are linked by the least total length, as Prim's over every pair of nodes finds", () => {
  let seed = 1;
  const random = () => {
    seed = (seed * 16807) % 2147483647;
    return seed / 2147483647;
  };
  const [points, edges, scribbleOf] = [[], [], []];
  for (let scribble = 0; scribble < 120; scribble++) {
    let zyx = [random() * 500, random() * 500, random() * 500];
    const nodes = 1 + Math.floor(random() * 40);
    for (let node = 0; node < nodes; node++) {
      if (node > 0) edges.push(scribbleOf.length - 1, scribbleOf.length);
      points.push(...zyx);
      scribbleOf.push(scribble);
      zyx = zyx.map((value) => value + random() * 6 - 3);
    }
  }
  const points_zyx_N3 = Float32Array.from(points);
  const distance = (a, b) =>
    Math.hypot(...[0, 1, 2].map((axis) => points_zyx_N3[3 * a + axis] - points_zyx_N3[3 * b + axis]));
  // Prim's over every pair of nodes, a scribble's own costing nothing.
  const count = scribbleOf.length;
  const [inTree, nearest] = [new Uint8Array(count), new Float64Array(count).fill(Infinity)];
  nearest[0] = 0;
  let least = 0;
  for (let step = 0; step < count; step++) {
    let next = -1;
    for (let point = 0; point < count; point++)
      if (!inTree[point] && (next === -1 || nearest[point] < nearest[next])) next = point;
    inTree[next] = 1;
    least += nearest[next];
    for (let point = 0; point < count; point++)
      if (!inTree[point])
        nearest[point] = Math.min(nearest[point], scribbleOf[point] === scribbleOf[next] ? 0 : distance(next, point));
  }
  const links = treeLinks({ points_zyx_N3, edges_E2: Uint32Array.from(edges) });
  let total = 0;
  for (let at = 0; at < links.length; at += 2) total += distance(links[at], links[at + 1]);
  assert.equal(links.length, 2 * 119);
  assert.ok(Math.abs(total - least) < 1e-9 * least, `links of ${total} where the least is ${least}`);
});
