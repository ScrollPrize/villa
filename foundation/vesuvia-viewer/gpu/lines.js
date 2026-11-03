import { compileProgram, uniformLocations } from './gl.js'

const VERTICES_PER_SEGMENT = 6

// One instance per segment; the six vertices of its quad come from gl_VertexID:
// triangles A B C and C B D with A = p0 + n, B = p0 - n, C = p1 + n, D = p1 - n.
const VERTEX = `#version 300 es
in vec4 a_segment;
uniform vec2 u_size;
uniform float u_halfWidth;
void main() {
  int v = gl_VertexID;
  bool atP1 = v == 2 || v == 3 || v == 5;
  float side = (v == 0 || v == 2 || v == 3) ? 1.0 : -1.0;
  vec2 d = a_segment.zw - a_segment.xy;
  float len = length(d);
  vec2 n = len > 0.0 ? vec2(-d.y, d.x) / len * u_halfWidth : vec2(0.0);
  vec2 p = (atP1 ? a_segment.zw : a_segment.xy) + n * side;
  gl_Position = vec4(p.x / u_size.x * 2.0 - 1.0, 1.0 - p.y / u_size.y * 2.0, 0.0, 1.0);
}
`
const FRAGMENT = `#version 300 es
precision mediump float;
uniform vec4 u_color;
out vec4 fragColor;
void main() {
  fragColor = u_color;
}
`

export class Lines {
  constructor(gl) {
    this.gl = gl
    this.program = compileProgram(gl, VERTEX, FRAGMENT)
    this.loc = uniformLocations(gl, this.program, ['u_size', 'u_halfWidth', 'u_color'])
    this.buffer = gl.createBuffer()
    this.segments = null
    this.vao = gl.createVertexArray()
    gl.bindVertexArray(this.vao)
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer)
    const index = gl.getAttribLocation(this.program, 'a_segment')
    gl.enableVertexAttribArray(index)
    gl.vertexAttribPointer(index, 4, gl.FLOAT, false, 0, 0)
    gl.vertexAttribDivisor(index, 1)
    gl.bindVertexArray(null)
  }

  draw(rect, segments, color, widthPx) {
    const count = segments.length / 4
    if (count === 0) return
    const gl = this.gl
    gl.useProgram(this.program)
    gl.bindVertexArray(this.vao)
    if (segments !== this.segments) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer)
      gl.bufferData(gl.ARRAY_BUFFER, segments, gl.DYNAMIC_DRAW)
      this.segments = segments
    }
    gl.uniform2f(this.loc.u_size, rect.width / devicePixelRatio, rect.height / devicePixelRatio)
    gl.uniform1f(this.loc.u_halfWidth, widthPx / 2)
    gl.uniform4f(this.loc.u_color, color[0], color[1], color[2], color[3])
    gl.enable(gl.BLEND)
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
    gl.drawArraysInstanced(gl.TRIANGLES, 0, VERTICES_PER_SEGMENT, count)
    gl.disable(gl.BLEND)
    gl.bindVertexArray(null)
  }

  points(rect, centers, radiusPx, color) {
    const count = centers.length / 2
    const segments = new Float32Array(count * 4)
    for (let p = 0; p < count; p++) {
      const x = centers[2 * p]
      const y = centers[2 * p + 1]
      segments.set([x - radiusPx, y, x + radiusPx, y], 4 * p)
    }
    this.draw(rect, segments, color, 2 * radiusPx)
  }
}
