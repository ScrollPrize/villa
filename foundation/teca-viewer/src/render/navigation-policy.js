export function sliceNavigationPolicy({ plane, rotated, heldSliceDirection, dirtyPlanes }) {
  const held = heldSliceDirection !== 0 && (dirtyPlanes === null || dirtyPlanes.includes(plane));
  return { held, axisAligned: held && !rotated };
}
