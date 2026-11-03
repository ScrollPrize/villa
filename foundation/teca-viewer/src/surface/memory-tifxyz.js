export class MemoryTifxyz {
  constructor({ grid_HW, scale_UV, points_zyx_HW3 }) {
    this.grid_HW = Object.freeze([...grid_HW]);
    this.scale_UV = Object.freeze([...scale_UV]);
    this.points_zyx_HW3 = points_zyx_HW3;
    this.cacheKey = null;
  }

  release() {}

  dispose() {}

  async readRows(axisIndex, rowStart, rowCount) {
    const width = this.grid_HW[1];
    const values = new Float32Array(rowCount * width);
    for (let cell = 0; cell < values.length; cell++)
      values[cell] = this.points_zyx_HW3[3 * (rowStart * width + cell) + 2 - axisIndex];
    return values;
  }
}
