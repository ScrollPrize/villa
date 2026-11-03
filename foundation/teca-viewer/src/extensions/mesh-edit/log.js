// An edit is checked against the points bit for bit, NaN included, so the edits kept of files that changed since are
// refused.

export function gathered(points_zyx_HW3, indices) {
  const points_zyx_N3 = new Float32Array(3 * indices.length);
  for (const [position, index] of indices.entries())
    points_zyx_N3.set(points_zyx_HW3.subarray(3 * index, 3 * index + 3), 3 * position);
  return points_zyx_N3;
}

export function extent(indices, width) {
  const start_HW = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const stop_HW = [0, 0];
  for (const index of indices) {
    const cell_HW = [Math.floor(index / width), index % width];
    for (let axis = 0; axis < 2; axis++) {
      start_HW[axis] = Math.min(start_HW[axis], cell_HW[axis]);
      stop_HW[axis] = Math.max(stop_HW[axis], cell_HW[axis] + 1);
    }
  }
  return [start_HW, stop_HW];
}

export function stepped(points_zyx_HW3, edit, forward) {
  const [from, to] = forward ? [edit.before_N3, edit.after_N3] : [edit.after_N3, edit.before_N3];
  const bits = new Uint32Array(points_zyx_HW3.buffer, points_zyx_HW3.byteOffset, points_zyx_HW3.length);
  const fromBits = new Uint32Array(from.buffer, from.byteOffset, from.length);
  for (const [position, index] of edit.indices.entries())
    for (let axis = 0; axis < 3; axis++)
      if (bits[3 * index + axis] !== fromBits[3 * position + axis])
        throw new Error(
          `Vertex ${index} is at ${points_zyx_HW3.subarray(3 * index, 3 * index + 3).join(", ")}, not where the ` +
            `edit ${forward ? "found" : "left"} it: the surface changed since the edit was kept.`,
        );
  for (const [position, index] of edit.indices.entries())
    points_zyx_HW3.set(to.subarray(3 * position, 3 * position + 3), 3 * index);
}
