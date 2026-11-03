export const FULLSCREEN_VERTEX = `#version 300 es
uniform vec2 u_size;
out vec2 v_pixel;
void main() {
  vec2 corner = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
  v_pixel = vec2(corner.x, 1.0 - corner.y) * u_size;
}
`

export function createContext(canvas) {
  const gl = canvas.getContext('webgl2', {
    antialias: false,
    preserveDrawingBuffer: false,
    alpha: false,
    powerPreference: 'high-performance',
  })
  if (gl === null) throw new Error('WebGL2 is not available in this browser')
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)
  return gl
}

// How many layers one shader can blend over the CT and the surface image on this device, at most `wish`. A volume
// is two texture units and 46 uniform vectors, its blend 8 more, the surface three units and 14 vectors; the image
// is a volume with no more uniforms than a blend. WebGL2 promises 16 units and 224 vectors, which is two layers; a
// desktop GPU has several times that.
export function layerCapacity(gl, wish) {
  const byUnits = Math.floor((gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) - 3) / 2) - 2
  const byVectors = Math.floor((gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS) - 46 - 14) / (46 + 8)) - 1
  return Math.max(0, Math.min(wish, byUnits, byVectors))
}

function checkProgram(gl, program) {
  if (gl.getProgramParameter(program, gl.LINK_STATUS)) return
  const errors = [gl.getProgramInfoLog(program)]
  for (const shader of gl.getAttachedShaders(program)) {
    if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)) continue
    errors.push(gl.getShaderInfoLog(shader), gl.getShaderSource(shader).split('\n')
      .map((line, i) => `${String(i + 1).padStart(4)}  ${line}`).join('\n'))
  }
  gl.deleteProgram(program)
  throw new Error(`program failed to link:\n${errors.join('\n')}`)
}

export function startProgram(gl, vertexSource, fragmentSource) {
  gl.getExtension('KHR_parallel_shader_compile') // enable before linking, not at the first readiness poll
  const program = gl.createProgram()
  for (const [type, source] of [[gl.VERTEX_SHADER, vertexSource], [gl.FRAGMENT_SHADER, fragmentSource]]) {
    const shader = gl.createShader(type)
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    gl.attachShader(program, shader)
    gl.deleteShader(shader)
  }
  gl.linkProgram(program)
  return program
}

export function programReady(gl, program) {
  const parallel = gl.getExtension('KHR_parallel_shader_compile')
  if (parallel !== null && !gl.getProgramParameter(program, parallel.COMPLETION_STATUS_KHR)) return false
  checkProgram(gl, program)
  return true
}

export function compileProgram(gl, vertexSource, fragmentSource) {
  const program = startProgram(gl, vertexSource, fragmentSource)
  checkProgram(gl, program)
  return program
}

export function fullscreenTriangle(gl) {
  return gl.createVertexArray()
}

export function paneRect(canvas, r) {
  const dpr = devicePixelRatio
  return {
    x: Math.round(r.left * dpr),
    y: canvas.height - Math.round(r.bottom * dpr),
    width: Math.round(r.right * dpr) - Math.round(r.left * dpr),
    height: Math.round(r.bottom * dpr) - Math.round(r.top * dpr),
  }
}

export function uniformLocations(gl, program, names) {
  const locations = {}
  for (const name of names) locations[name] = gl.getUniformLocation(program, name)
  return locations
}
