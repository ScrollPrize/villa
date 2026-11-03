const SATURATION = 0.7;

export function labelColor(id) {
  const hue = (Math.imul(id, 2654435761) >>> 8) / 16777216;
  return [0, 2 / 3, 1 / 3].map(
    (shift) => 1 + SATURATION * (Math.min(1, Math.max(0, Math.abs(((hue + shift) % 1) * 6 - 3) - 1)) - 1),
  );
}

export const LABEL_COLOR_SHADER = `
  vec3 labelColor(float id) {
    float hue = float((uint(id) * 2654435761u) >> 8u) / 16777216.0;
    vec3 hue_RGB = clamp(abs(fract(hue + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return mix(vec3(1.0), hue_RGB, ${SATURATION});
  }
`;
