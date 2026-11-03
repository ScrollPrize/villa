// A ray that is decided writes the depth of the cube's front, so the slice planes the pane draws after it are hidden
// inside and behind the cube.

import * as THREE from "../../../vendor/three.module.js";

// Half-texel steps along the diagonal of a 256-texel cube number 887.
const STEP_LIMIT = 2048;
const OPAQUE_TRANSMITTANCE = 0.004;
const AMBIENT = 0.3;

export function boxMaterial(texture, texels_XYZ, look) {
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      box: { value: texture },
      texels: { value: new THREE.Vector3(...texels_XYZ) },
      black: { value: look.black },
      white: { value: look.white },
      density: { value: look.density },
      maximum: { value: look.maximum },
    },
    vertexShader: `
      out vec3 vOrigin;
      out vec3 vDirection;
      void main() {
        vOrigin = (inverse(modelMatrix) * vec4(cameraPosition, 1.0)).xyz;
        vDirection = position - vOrigin;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;
      precision highp sampler3D;
      uniform mat4 projectionMatrix;
      uniform mat4 modelViewMatrix;
      uniform sampler3D box;
      uniform vec3 texels;
      uniform float black;
      uniform float white;
      uniform float density;
      uniform bool maximum;
      in vec3 vOrigin;
      in vec3 vDirection;
      out vec4 outColor;
      void main() {
        vec3 direction = normalize(vDirection);
        vec3 near = (vec3(-0.5) - vOrigin) / direction;
        vec3 far = (vec3(0.5) - vOrigin) / direction;
        vec3 entering = min(near, far);
        vec3 leaving = max(near, far);
        float enter = max(max(max(entering.x, entering.y), entering.z), 0.0);
        float leave = min(min(leaving.x, leaving.y), leaving.z);
        if (enter >= leave) discard;
        vec3 across = abs(direction) * texels;
        float stride = 0.5 / max(max(across.x, across.y), across.z);
        float texelsPerStride = length(direction * texels) * stride;
        float brightest = 0.0;
        float emitted = 0.0;
        float transmitted = 1.0;
        vec3 texel = 1.0 / texels;
        for (int index = 0; index < ${STEP_LIMIT}; index++) {
          float t = enter + (float(index) + 0.5) * stride;
          if (t > leave) break;
          vec3 at = vOrigin + t * direction + 0.5;
          float grey = clamp((texture(box, at).r - black) / (white - black), 0.0, 1.0);
          if (maximum) {
            brightest = max(brightest, grey);
          } else if (grey > 0.0) {
            float alpha = 1.0 - exp(-density * grey * texelsPerStride);
            vec3 gradient = vec3(
              texture(box, at + vec3(texel.x, 0.0, 0.0)).r - texture(box, at - vec3(texel.x, 0.0, 0.0)).r,
              texture(box, at + vec3(0.0, texel.y, 0.0)).r - texture(box, at - vec3(0.0, texel.y, 0.0)).r,
              texture(box, at + vec3(0.0, 0.0, texel.z)).r - texture(box, at - vec3(0.0, 0.0, texel.z)).r
            );
            float facing = length(gradient) > 0.0 ? abs(dot(normalize(gradient), normalize(direction * texels))) : 1.0;
            emitted += transmitted * alpha * (${AMBIENT} + ${1 - AMBIENT} * facing);
            transmitted *= 1.0 - alpha;
            if (transmitted < ${OPAQUE_TRANSMITTANCE}) break;
          }
        }
        outColor = vec4(vec3(maximum ? brightest : emitted), 1.0);
        vec4 front = projectionMatrix * modelViewMatrix * vec4(vOrigin + enter * direction, 1.0);
        bool decided = maximum ? brightest > 0.0 : transmitted < 0.5;
        gl_FragDepth = !decided ? 1.0 : enter == 0.0 ? 0.0 : 0.5 + 0.5 * front.z / front.w;
      }
    `,
    side: THREE.BackSide,
  });
}
