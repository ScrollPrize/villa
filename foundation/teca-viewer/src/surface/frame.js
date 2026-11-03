function surfacePoint(surface_UV, grid_HW, pixel) {
  const position_UV = [
    Math.min(grid_HW[1] - 1, Math.max(0, surface_UV[0])),
    Math.min(grid_HW[0] - 1, Math.max(0, surface_UV[1])),
  ];
  const low_UV = position_UV.map(Math.floor);
  const high_UV = [Math.min(grid_HW[1] - 1, low_UV[0] + 1), Math.min(grid_HW[0] - 1, low_UV[1] + 1)];
  const points_ZYX = [
    pixel([low_UV[0], low_UV[1]]),
    pixel([high_UV[0], low_UV[1]]),
    pixel([low_UV[0], high_UV[1]]),
    pixel([high_UV[0], high_UV[1]]),
  ];
  if (points_ZYX.some((point_ZYX) => point_ZYX === null)) return null;
  const weight_UV = [position_UV[0] - low_UV[0], position_UV[1] - low_UV[1]];
  return points_ZYX[0].map((value, axis) => {
    const low = value * (1 - weight_UV[0]) + points_ZYX[1][axis] * weight_UV[0];
    const high = points_ZYX[2][axis] * (1 - weight_UV[0]) + points_ZYX[3][axis] * weight_UV[0];
    return low * (1 - weight_UV[1]) + high * weight_UV[1];
  });
}

export function surfaceFrameAt(surface_UV, normalSign, grid_HW, pixel) {
  if (surface_UV.some((value, axis) => value < 0 || value > grid_HW[1 - axis] - 1)) return null;
  const point = (uv) => surfacePoint(uv, grid_HW, pixel);
  const point_ZYX = point(surface_UV);
  const left_ZYX = point([surface_UV[0] - 1, surface_UV[1]]);
  const right_ZYX = point([surface_UV[0] + 1, surface_UV[1]]);
  const down_ZYX = point([surface_UV[0], surface_UV[1] - 1]);
  const up_ZYX = point([surface_UV[0], surface_UV[1] + 1]);
  if ([point_ZYX, left_ZYX, right_ZYX, down_ZYX, up_ZYX].some((point) => point === null)) return null;
  const tangentU_XYZ = [2, 1, 0].map((axis) => right_ZYX[axis] - left_ZYX[axis]);
  const tangentV_XYZ = [2, 1, 0].map((axis) => up_ZYX[axis] - down_ZYX[axis]);
  const normal_XYZ = [
    tangentU_XYZ[1] * tangentV_XYZ[2] - tangentU_XYZ[2] * tangentV_XYZ[1],
    tangentU_XYZ[2] * tangentV_XYZ[0] - tangentU_XYZ[0] * tangentV_XYZ[2],
    tangentU_XYZ[0] * tangentV_XYZ[1] - tangentU_XYZ[1] * tangentV_XYZ[0],
  ];
  const length = Math.hypot(...normal_XYZ);
  if (length === 0) return null;
  const normal_ZYX = [normal_XYZ[2], normal_XYZ[1], normal_XYZ[0]].map((value) => (value / length) * normalSign);
  const uLength = Math.hypot(...tangentU_XYZ);
  const u_ZYX = [tangentU_XYZ[2], tangentU_XYZ[1], tangentU_XYZ[0]].map((value) => value / uLength);
  const v_ZYX = [
    normal_ZYX[1] * u_ZYX[2] - normal_ZYX[2] * u_ZYX[1],
    normal_ZYX[2] * u_ZYX[0] - normal_ZYX[0] * u_ZYX[2],
    normal_ZYX[0] * u_ZYX[1] - normal_ZYX[1] * u_ZYX[0],
  ];
  // axes_ZYX3 rows are the orthonormal normal, V, and U directions as ZYX vectors.
  return { point_ZYX, normal_ZYX, axes_ZYX3: [normal_ZYX, v_ZYX, u_ZYX] };
}
