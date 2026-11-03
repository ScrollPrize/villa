// Squared voxels under which two distances are equal.
export const DISTANCE_TOLERANCE = 1e-6;
// Cross product length, twice the area in square voxels, at or under which a triangle is degenerate.
const DEGENERATE_CROSS = 1e-9;

export function degenerateTriangle(points, a, b, c) {
  const ab = [points[b] - points[a], points[b + 1] - points[a + 1], points[b + 2] - points[a + 2]];
  const ac = [points[c] - points[a], points[c + 1] - points[a + 1], points[c + 2] - points[a + 2]];
  return (
    Math.hypot(ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]) <=
    DEGENERATE_CROSS
  );
}

export function nearerTriangle(nearest, query_ZYX, points, a, b, c, au, av, bu, bv, cu, cv) {
  const az = points[a];
  const ay = points[a + 1];
  const ax = points[a + 2];
  const abz = points[b] - az;
  const aby = points[b + 1] - ay;
  const abx = points[b + 2] - ax;
  const acz = points[c] - az;
  const acy = points[c + 1] - ay;
  const acx = points[c + 2] - ax;
  const nz = abx * acy - aby * acx;
  const ny = abz * acx - abx * acz;
  const nx = aby * acz - abz * acy;
  const normalLength = Math.hypot(nz, ny, nx);
  if (normalLength <= DEGENERATE_CROSS) return;

  const apz = query_ZYX[0] - az;
  const apy = query_ZYX[1] - ay;
  const apx = query_ZYX[2] - ax;
  const d1 = abz * apz + aby * apy + abx * apx;
  const d2 = acz * apz + acy * apy + acx * apx;
  let weightA;
  let weightB;
  let weightC;
  if (d1 <= 0 && d2 <= 0) {
    weightA = 1;
    weightB = 0;
    weightC = 0;
  } else {
    const bpz = query_ZYX[0] - points[b];
    const bpy = query_ZYX[1] - points[b + 1];
    const bpx = query_ZYX[2] - points[b + 2];
    const d3 = abz * bpz + aby * bpy + abx * bpx;
    const d4 = acz * bpz + acy * bpy + acx * bpx;
    if (d3 >= 0 && d4 <= d3) {
      weightA = 0;
      weightB = 1;
      weightC = 0;
    } else {
      const vc = d1 * d4 - d3 * d2;
      if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        weightB = d1 / (d1 - d3);
        weightA = 1 - weightB;
        weightC = 0;
      } else {
        const cpz = query_ZYX[0] - points[c];
        const cpy = query_ZYX[1] - points[c + 1];
        const cpx = query_ZYX[2] - points[c + 2];
        const d5 = abz * cpz + aby * cpy + abx * cpx;
        const d6 = acz * cpz + acy * cpy + acx * cpx;
        if (d6 >= 0 && d5 <= d6) {
          weightA = 0;
          weightB = 0;
          weightC = 1;
        } else {
          const vb = d5 * d2 - d1 * d6;
          if (vb <= 0 && d2 >= 0 && d6 <= 0) {
            weightC = d2 / (d2 - d6);
            weightA = 1 - weightC;
            weightB = 0;
          } else {
            const va = d3 * d6 - d5 * d4;
            if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
              weightC = (d4 - d3) / (d4 - d3 + d5 - d6);
              weightA = 0;
              weightB = 1 - weightC;
            } else {
              const denominator = 1 / (va + vb + vc);
              weightB = vb * denominator;
              weightC = vc * denominator;
              weightA = 1 - weightB - weightC;
            }
          }
        }
      }
    }
  }
  const pointZ = az * weightA + points[b] * weightB + points[c] * weightC;
  const pointY = ay * weightA + points[b + 1] * weightB + points[c + 1] * weightC;
  const pointX = ax * weightA + points[b + 2] * weightB + points[c + 2] * weightC;
  const dz = query_ZYX[0] - pointZ;
  const dy = query_ZYX[1] - pointY;
  const dx = query_ZYX[2] - pointX;
  const distanceSquared = dz * dz + dy * dy + dx * dx;
  const previewU = au * weightA + bu * weightB + cu * weightC;
  const previewV = av * weightA + bv * weightB + cv * weightC;
  if (
    Math.abs(distanceSquared - nearest.distanceSquared) <= DISTANCE_TOLERANCE &&
    Math.hypot(previewU - nearest.preview_UV[0], previewV - nearest.preview_UV[1]) > 1
  ) {
    nearest.ambiguous = true;
  } else if (distanceSquared < nearest.distanceSquared - DISTANCE_TOLERANCE) {
    nearest.ambiguous = false;
  }
  if (distanceSquared > nearest.distanceSquared) return;
  if (
    distanceSquared === nearest.distanceSquared &&
    (previewV > nearest.preview_UV[1] || (previewV === nearest.preview_UV[1] && previewU >= nearest.preview_UV[0]))
  ) {
    return;
  }
  nearest.point_ZYX[0] = pointZ;
  nearest.point_ZYX[1] = pointY;
  nearest.point_ZYX[2] = pointX;
  nearest.preview_UV[0] = previewU;
  nearest.preview_UV[1] = previewV;
  nearest.normal_ZYX[0] = nz / normalLength;
  nearest.normal_ZYX[1] = ny / normalLength;
  nearest.normal_ZYX[2] = nx / normalLength;
  nearest.distanceSquared = distanceSquared;
}

export function cellDistanceSquared(point_ZYX, points, p00, p10, p01, p11) {
  let total = 0;
  for (let axis = 0; axis < 3; axis++) {
    const a = points[p00 + axis];
    const b = points[p10 + axis];
    const c = points[p01 + axis];
    const d = points[p11 + axis];
    const distance = Math.max(Math.min(a, b, c, d) - point_ZYX[axis], 0, point_ZYX[axis] - Math.max(a, b, c, d));
    total += distance * distance;
  }
  return total;
}
