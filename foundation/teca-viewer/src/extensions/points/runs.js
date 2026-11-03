const TAG = /\[kollesis\]\s*/gi;
const BAND_COLORS = { kollesis: [1, 0.72, 0.1], picked: [1, 1, 1] };

export function commentTags(comment) {
  const tags = { kollesis: false, text: "" };
  tags.text = (comment ?? "")
    .replace(TAG, () => {
      tags.kollesis = true;
      return "";
    })
    .trim();
  return tags;
}

export function tagsComment(tags) {
  const text = [tags.kollesis ? "[KOLLESIS]" : "", tags.text.trim()].filter(Boolean).join(" ");
  return text === "" ? null : text;
}

export function commentMark(comment) {
  return commentTags(comment).kollesis ? "kollesis" : null;
}

export function markComment(mark) {
  return tagsComment({ kollesis: mark === "kollesis", text: "" });
}

export function treeRuns(tree, ids_N, marked) {
  const runs = [];
  let run = null;
  for (const id of ids_N) {
    const mark = marked.get(id) ?? null;
    if (mark === null) run = null;
    else if (run?.mark === mark) run.ids.push(id);
    else {
      run = { tree, mark, ids: [id] };
      runs.push(run);
    }
  }
  return runs;
}

export function stepRun(runs, skeleton, selected, step) {
  if (selected === null) return (step > 0 ? runs[0] : runs.at(-1)) ?? null;
  const tree = skeleton.nodes.get(selected).tree;
  const order = (run) => (run.tree === tree ? run.ids[0] - selected : run.tree - tree);
  return (step > 0 ? runs.find((run) => order(run) > 0) : runs.findLast((run) => order(run) < 0)) ?? null;
}

export function runOf(runs, skeleton, id) {
  const tree = skeleton.nodes.get(id).tree;
  return runs.findIndex((run) => run.tree === tree && run.ids.includes(id));
}

export function bandMarks(skeleton, marked, depth, lines) {
  const index = new Map();
  const points_zyx_N3 = new Float32Array(marked.size * 3);
  const colors_N3 = new Float32Array(marked.size * 3);
  const ids_N = new Float64Array(marked.size);
  const marks_N = [];
  for (const [id, mark] of marked) {
    const at = index.size;
    const point_ZYX = skeleton.nodes.get(id).point_ZYX;
    const color = BAND_COLORS[mark];
    for (let axis = 0; axis < 3; axis++) {
      points_zyx_N3[3 * at + axis] = point_ZYX[axis];
      colors_N3[3 * at + axis] = color[axis];
    }
    ids_N[at] = id;
    marks_N.push(mark);
    index.set(id, at);
  }
  const edges = [];
  const joined = new Uint8Array(marked.size);
  for (const { marks, ids_N: drawn_N } of lines) {
    const band_N = new Int32Array(drawn_N.length);
    for (let at = 0; at < drawn_N.length; at++) band_N[at] = index.get(drawn_N[at]) ?? -1;
    for (let edge = 0; edge < marks.edges_E2.length; edge += 2) {
      const [a, b] = [band_N[marks.edges_E2[edge]], band_N[marks.edges_E2[edge + 1]]];
      if (a < 0 || b < 0 || marks_N[a] !== marks_N[b]) continue;
      edges.push(a, b);
      joined[a] = joined[b] = 1;
    }
  }
  for (let at = 0; at < joined.length; at++) if (joined[at] === 0) edges.push(at, at);
  const marks = { points_zyx_N3, colors_N3, edges_E2: Uint32Array.from(edges), selected: -1, depth };
  return { marks, ids_N };
}
