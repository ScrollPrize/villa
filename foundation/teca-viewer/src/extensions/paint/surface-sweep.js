import { MARGIN, Runs, SLAB, slabFirst, slabLast, sweptSpan } from "./sweep.js";

// A triangle's prisms run along each axis its normal leans on by at least this share of the most it leans on any, so
// two triangles that meet share an axis unless their normals are 12° apart, and one axis's prisms leave no gap.
const SHARE = 0.8;
const EDGE = 1e-6;
const CELLS = 2 ** 16;

function meshMetric(radii_ZYX, from_ZYX, to_ZYX) {
  const q_ZYX = radii_ZYX.map((radius) => 1 / radius ** 2);
  const d_ZYX = to_ZYX.map((to, axis) => to - from_ZYX[axis]);
  const dd = d_ZYX[0] ** 2 * q_ZYX[0] + d_ZYX[1] ** 2 * q_ZYX[1] + d_ZYX[2] ** 2 * q_ZYX[2];
  const widest = Math.sqrt(Math.max(...q_ZYX));
  const deepest = (SLAB * Math.sqrt(3) * widest) / SHARE;
  const [qZ, qY, qX] = q_ZYX;
  const apart = (a_ZYX, b_ZYX) =>
    Math.sqrt((a_ZYX[0] - b_ZYX[0]) ** 2 * qZ + (a_ZYX[1] - b_ZYX[1]) ** 2 * qY + (a_ZYX[2] - b_ZYX[2]) ** 2 * qX);
  const fromSegment = (p_ZYX) => {
    const [wZ, wY, wX] = [p_ZYX[0] - from_ZYX[0], p_ZYX[1] - from_ZYX[1], p_ZYX[2] - from_ZYX[2]];
    const t =
      dd === 0 ? 0 : Math.min(1, Math.max(0, (wZ * d_ZYX[0] * qZ + wY * d_ZYX[1] * qY + wX * d_ZYX[2] * qX) / dd));
    return Math.sqrt((wZ - t * d_ZYX[0]) ** 2 * qZ + (wY - t * d_ZYX[1]) ** 2 * qY + (wX - t * d_ZYX[2]) ** 2 * qX);
  };
  return { q_ZYX, d_ZYX, dd, widest, deepest, apart, fromSegment };
}

export async function meshTriangles(radii_ZYX, vertexAt, cell_HW, from_ZYX, to_ZYX, before, pause) {
  const { deepest, apart, fromSegment } = meshMetric(radii_ZYX, from_ZYX, to_ZYX);
  const points = [];
  const [h0, w0] = cell_HW.map(Math.round);
  const seen = new Set();
  const cells = [];
  for (const [h, w] of [
    [h0, w0],
    [h0 - 1, w0],
    [h0, w0 - 1],
    [h0 - 1, w0 - 1],
  ]) {
    seen.add(h * CELLS + w);
    cells.push(h, w);
  }
  while (cells.length > 0) {
    const w = cells.pop();
    const h = cells.pop();
    const corners_ZYX = [vertexAt(h, w), vertexAt(h, w + 1), vertexAt(h + 1, w + 1), vertexAt(h + 1, w)];
    if (corners_ZYX.some((corner_ZYX) => corner_ZYX === null)) continue;
    let across = 0;
    for (let i = 0; i < 4; i++)
      for (let j = i + 1; j < 4; j++) across = Math.max(across, apart(corners_ZYX[i], corners_ZYX[j]));
    const nearest = Math.min(...corners_ZYX.map(fromSegment));
    if (nearest > 1 + across + deepest) continue;
    if (!before || corners_ZYX.some((corner_ZYX) => apart(corner_ZYX, from_ZYX) + deepest >= Math.sqrt(1 - MARGIN))) {
      for (const corner of [0, 1, 2, 0, 2, 3]) points.push(...corners_ZYX[corner]);
    }
    for (const [nextH, nextW] of [
      [h - 1, w],
      [h + 1, w],
      [h, w - 1],
      [h, w + 1],
    ]) {
      const key = nextH * CELLS + nextW;
      if (seen.has(key)) continue;
      seen.add(key);
      cells.push(nextH, nextW);
    }
    await pause();
  }
  return Float64Array.from(points);
}

export function meshSwept({ radii_ZYX, points_N9, from_ZYX, to_ZYX, before, shape_ZYX }) {
  const { q_ZYX, d_ZYX, dd, widest, apart, fromSegment } = meshMetric(radii_ZYX, from_ZYX, to_ZYX);
  const span = new Float64Array(2);
  const runs = new Runs();
  const voxel_ZYX = [0, 0, 0];
  const triangle = (points_ZYX) => {
    const [p0, p1, p2] = points_ZYX;
    const e1_ZYX = p1.map((value, axis) => value - p0[axis]);
    const e2_ZYX = p2.map((value, axis) => value - p0[axis]);
    const cross_ZYX = [
      e1_ZYX[1] * e2_ZYX[2] - e1_ZYX[2] * e2_ZYX[1],
      e1_ZYX[2] * e2_ZYX[0] - e1_ZYX[0] * e2_ZYX[2],
      e1_ZYX[0] * e2_ZYX[1] - e1_ZYX[1] * e2_ZYX[0],
    ];
    const size = Math.hypot(...cross_ZYX);
    if (size === 0) return;
    const normal_ZYX = cross_ZYX.map((value) => value / size);
    const level = normal_ZYX[0] * p0[0] + normal_ZYX[1] * p0[1] + normal_ZYX[2] * p0[2];
    const lean_ZYX = normal_ZYX.map(Math.abs);
    const most = Math.max(...lean_ZYX);
    const farthest = Math.max(fromSegment(p0), fromSegment(p1), fromSegment(p2));
    for (let prism = 0; prism < 3; prism++) {
      const nPrism = normal_ZYX[prism];
      if (lean_ZYX[prism] < SHARE * most) continue;
      const depth = (SLAB / lean_ZYX[prism]) * widest;
      if (before && points_ZYX.every((p_ZYX) => apart(p_ZYX, from_ZYX) + depth < Math.sqrt(1 - MARGIN))) continue;
      const whole = farthest + depth < 1 - MARGIN;
      const [one, other] = prism === 0 ? [1, 2] : prism === 1 ? [0, 2] : [0, 1];
      const row = lean_ZYX[other] <= lean_ZYX[one] ? other : one;
      const side = 3 - prism - row;
      const [qSide, qRow, qPrism] = [q_ZYX[side], q_ZYX[row], q_ZYX[prism]];
      const [dSide, dRow, dPrism] = [d_ZYX[side], d_ZYX[row], d_ZYX[prism]];
      const [nSide, nRow] = [normal_ZYX[side], normal_ZYX[row]];
      const [fromSide, fromRow, fromPrism] = [from_ZYX[side], from_ZYX[row], from_ZYX[prism]];
      const s_N = points_ZYX.map((p_ZYX) => p_ZYX[side]);
      const r_N = points_ZYX.map((p_ZYX) => p_ZYX[row]);
      const sign = Math.sign((s_N[1] - s_N[0]) * (r_N[2] - r_N[0]) - (s_N[2] - s_N[0]) * (r_N[1] - r_N[0]));
      if (sign === 0) continue;
      const lowSide = Math.max(0, Math.ceil(Math.min(...s_N)));
      const highSide = Math.min(shape_ZYX[side] - 1, Math.floor(Math.max(...s_N)));
      for (let at = lowSide; at <= highSide; at++) {
        let low = -Infinity;
        let high = Infinity;
        for (let i = 0; i < 3; i++) {
          const j = (i + 1) % 3;
          const k = sign * (s_N[j] - s_N[i]);
          const c = -sign * ((r_N[j] - r_N[i]) * (at - s_N[i]) + (s_N[j] - s_N[i]) * r_N[i]);
          if (k > 0) low = Math.max(low, (-EDGE - c) / k);
          else if (k < 0) high = Math.min(high, (-EDGE - c) / k);
          else if (c < -EDGE) high = -Infinity;
        }
        let first = Math.max(0, Math.ceil(low));
        let last = Math.min(shape_ZYX[row] - 1, Math.floor(high));
        const wSide = at - fromSide;
        if (!whole) {
          const wLine = (level - nSide * at - nRow * fromRow) / nPrism - fromPrism;
          const eLine = -nRow / nPrism;
          sweptSpan(
            span,
            wSide ** 2 * qSide + wLine ** 2 * qPrism,
            wLine * eLine * qPrism,
            qRow + eLine ** 2 * qPrism,
            (wSide - dSide) ** 2 * qSide + dRow ** 2 * qRow + (wLine - dPrism) ** 2 * qPrism,
            -dRow * qRow + (wLine - dPrism) * eLine * qPrism,
            wSide * dSide * qSide + wLine * dPrism * qPrism,
            dRow * qRow + eLine * dPrism * qPrism,
            dd,
            (1 + depth + MARGIN) ** 2,
          );
          first = Math.max(first, Math.ceil(fromRow + span[0]));
          last = Math.min(last, Math.floor(fromRow + span[1]));
        }
        if (first > last) continue;
        const [atFirst, atLast] = [nSide * at + nRow * first - level, nSide * at + nRow * last - level];
        const lowPrism = Math.max(0, Math.min(slabFirst(atFirst, nPrism), slabFirst(atLast, nPrism)));
        const highPrism = Math.min(shape_ZYX[prism] - 1, Math.max(slabLast(atFirst, nPrism), slabLast(atLast, nPrism)));
        for (let deep = lowPrism; deep <= highPrism; deep++) {
          const height = nSide * at + nPrism * deep - level;
          let begin = Math.max(first, slabFirst(height, nRow));
          let final = Math.min(last, slabLast(height, nRow));
          if (begin > final) continue;
          const wPrism = deep - fromPrism;
          const w0w0 = wSide ** 2 * qSide + wPrism ** 2 * qPrism;
          const held = 1 - MARGIN - w0w0;
          if (before && (begin - fromRow) ** 2 * qRow <= held && (final - fromRow) ** 2 * qRow <= held) continue;
          if (!whole) {
            const endEnd = (wSide - dSide) ** 2 * qSide + (wPrism - dPrism) ** 2 * qPrism + dRow ** 2 * qRow;
            const w0D = wSide * dSide * qSide + wPrism * dPrism * qPrism;
            sweptSpan(span, w0w0, 0, qRow, endEnd, -dRow * qRow, w0D, dRow * qRow, dd, 1);
            begin = Math.max(begin, Math.ceil(fromRow + span[0]));
            final = Math.min(final, Math.floor(fromRow + span[1]));
            if (begin > final) continue;
          }
          voxel_ZYX[side] = at;
          voxel_ZYX[prism] = deep;
          if (before && held >= 0) {
            const reach = Math.sqrt(held / qRow);
            const skip = Math.max(begin, Math.ceil(fromRow - reach));
            const stop = Math.min(final, Math.floor(fromRow + reach));
            if (skip <= stop) {
              if (begin < skip) runs.add(voxel_ZYX, row, begin, skip - 1);
              if (stop < final) runs.add(voxel_ZYX, row, stop + 1, final);
              continue;
            }
          }
          runs.add(voxel_ZYX, row, begin, final);
        }
      }
    }
  };
  for (let at = 0; at < points_N9.length; at += 9)
    triangle(
      [0, 3, 6].map((corner) => [points_N9[at + corner], points_N9[at + corner + 1], points_N9[at + corner + 2]]),
    );
  return runs.packed();
}
