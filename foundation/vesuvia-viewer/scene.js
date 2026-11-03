// Coordinates are ZYX world voxels (CT level 0). `zoom` is world voxels per CSS pixel.
const AXES_EPSILON = 1e-8

export function orthonormalizeAxes(axes_zyx) {
  if (!Array.isArray(axes_zyx) || axes_zyx.length !== 3
    || axes_zyx.some(axis => !Array.isArray(axis) || axis.length !== 3 || axis.some(v => !Number.isFinite(v)))) {
    throw new Error(`axes ${JSON.stringify(axes_zyx)}: expected three finite ZYX vectors`)
  }
  const basis_zyx = []
  for (const axis_zyx of axes_zyx) {
    let vector_zyx = [...axis_zyx]
    for (const previous_zyx of basis_zyx) {
      const dot = vector_zyx.reduce((sum, v, a) => sum + v * previous_zyx[a], 0)
      vector_zyx = vector_zyx.map((v, a) => v - dot * previous_zyx[a])
    }
    const length = Math.hypot(...vector_zyx)
    if (!Number.isFinite(length) || length <= AXES_EPSILON) throw new Error(`axes ${JSON.stringify(axes_zyx)}: degenerate basis`)
    basis_zyx.push(vector_zyx.map(v => v / length))
  }
  const [z, y, x] = basis_zyx
  const determinant = z[0] * (y[1] * x[2] - y[2] * x[1])
    - z[1] * (y[0] * x[2] - y[2] * x[0]) + z[2] * (y[0] * x[1] - y[1] * x[0])
  if (determinant < 0) throw new Error(`axes ${JSON.stringify(axes_zyx)}: reflected basis is not a rotation`)
  return basis_zyx
}

export function tilted(axes_zyx) {
  return axes_zyx.some((vector_zyx, a) => vector_zyx.some((v, b) => v !== Number(a === b)))
}

export function voxelStep(normal_zyx, spacing_zyx) {
  return Math.hypot(...normal_zyx.map((value, a) => value * spacing_zyx[a]))
}

export function createScene() {
  return {
    loading: false,
    ct: null,
    layers: [],
    surface: null,
    image: null,
    outlineIndex: null,
    focus_zyx: [0, 0, 0],
    axes_zyx: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    zoom: 1,
    levelCap: 0,
    window: { black: 0, white: 255 },
    surfaceView: { center_uv: [0, 0], zoom: 1, offset: 0, slab: { front: 0, back: 0, step: 1, mode: 'mean' }, imageOpacity: 1 },
    overlays: { outline: true, crosshair: true },
    stepDirection: 1,
    velocity_zyx: [0, 0, 0],
    stepSize: 2,
    tool: null,
    surfaceTool: null,
    panKey: false,
    dirty: true,
  }
}
