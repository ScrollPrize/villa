export function surfaceGeometryShader() {
  return `
      bool fetchSurfaceStripSlot(int row, out int slot) {
        int strip = row / surfaceStripRows;
        ivec2 pageSize = textureSize(surfacePages, 0);
        slot = int(texelFetch(surfacePages, ivec2(strip % pageSize.x, strip / pageSize.x), 0).r) - 1;
        return slot >= 0;
      }

      bool fetchSurfaceAtlasPixel(ivec2 pixel_UV, int slot, out vec3 point_ZYX) {
        int linearSurfacePixel = (pixel_UV.y % surfaceStripRows) * int(surfaceGrid_HW.x) + pixel_UV.x;
        vec4 point_ZYXV = texelFetch(
          surfaceAtlas,
          ivec3(linearSurfacePixel % surfaceAtlasWidth, linearSurfacePixel / surfaceAtlasWidth, slot),
          0
        );
        if (point_ZYXV.a < 0.5) return false;
        point_ZYX = point_ZYXV.rgb;
        return true;
      }

      bool fetchSurfacePixel(ivec2 pixel_UV, out vec3 point_ZYX) {
        ivec2 grid_WH = ivec2(surfaceGrid_HW.x, surfaceGrid_HW.y);
        if (any(lessThan(pixel_UV, ivec2(0))) || any(greaterThanEqual(pixel_UV, grid_WH))) return false;
        int slot;
        return fetchSurfaceStripSlot(pixel_UV.y, slot) && fetchSurfaceAtlasPixel(pixel_UV, slot, point_ZYX);
      }

      bool sampleSurfacePoint(vec2 surface_UV, out vec3 point_ZYX) {
        vec2 position = clamp(surface_UV, vec2(0.0), surfaceGrid_HW - vec2(1.0));
        ivec2 low = ivec2(floor(position));
        ivec2 high = min(low + ivec2(1), ivec2(surfaceGrid_HW) - ivec2(1));
        vec3 p00; vec3 p10; vec3 p01; vec3 p11;
        if (!fetchSurfacePixel(ivec2(low.x, low.y), p00)
          || !fetchSurfacePixel(ivec2(high.x, low.y), p10)
          || !fetchSurfacePixel(ivec2(low.x, high.y), p01)
          || !fetchSurfacePixel(ivec2(high.x, high.y), p11)) return false;
        vec2 weight = fract(position);
        point_ZYX = mix(mix(p00, p10, weight.x), mix(p01, p11, weight.x), weight.y);
        return true;
      }

      // The five bilinear stencil points share 12 grid texels whenever each neighbor's own clamped floor lands
      // one texel from the center's. Positions, floors, and weights are computed per point exactly as
      // sampleSurfacePoint does; any other case (grid edges, float rounding across an integer) takes that path.
      bool sampleSurfaceStencil(
        vec2 surface_UV,
        out vec3 point_ZYX,
        out vec3 left_ZYX,
        out vec3 right_ZYX,
        out vec3 down_ZYX,
        out vec3 up_ZYX
      ) {
        vec2 limit = surfaceGrid_HW - vec2(1.0);
        vec2 position = clamp(surface_UV, vec2(0.0), limit);
        float leftX = clamp((surface_UV + vec2(-1.0, 0.0)).x, 0.0, limit.x);
        float rightX = clamp((surface_UV + vec2(1.0, 0.0)).x, 0.0, limit.x);
        float downY = clamp((surface_UV + vec2(0.0, -1.0)).y, 0.0, limit.y);
        float upY = clamp((surface_UV + vec2(0.0, 1.0)).y, 0.0, limit.y);
        ivec2 low = ivec2(floor(position));
        if (surfaceStripRows < 3 || int(floor(leftX)) != low.x - 1 || int(floor(rightX)) != low.x + 1
          || int(floor(downY)) != low.y - 1 || int(floor(upY)) != low.y + 1) {
          return sampleSurfacePoint(surface_UV, point_ZYX)
            && sampleSurfacePoint(surface_UV + vec2(-1.0, 0.0), left_ZYX)
            && sampleSurfacePoint(surface_UV + vec2(1.0, 0.0), right_ZYX)
            && sampleSurfacePoint(surface_UV + vec2(0.0, -1.0), down_ZYX)
            && sampleSurfacePoint(surface_UV + vec2(0.0, 1.0), up_ZYX);
        }
        ivec2 grid_WH = ivec2(surfaceGrid_HW);
        int x0 = low.x - 1; int x1 = low.x; int x2 = low.x + 1; int x3 = min(low.x + 2, grid_WH.x - 1);
        int y0 = low.y - 1; int y1 = low.y; int y2 = low.y + 1; int y3 = min(low.y + 2, grid_WH.y - 1);
        int firstStrip = y0 / surfaceStripRows;
        int slotFirst; int slotLast;
        if (!fetchSurfaceStripSlot(y0, slotFirst)) return false;
        if (y3 / surfaceStripRows == firstStrip) slotLast = slotFirst;
        else if (!fetchSurfaceStripSlot(y3, slotLast)) return false;
        int slot0 = slotFirst;
        int slot1 = y1 / surfaceStripRows == firstStrip ? slotFirst : slotLast;
        int slot2 = y2 / surfaceStripRows == firstStrip ? slotFirst : slotLast;
        int slot3 = slotLast;
        vec2 weight = fract(position);
        float leftWeight = fract(leftX);
        float rightWeight = fract(rightX);
        float downWeight = fract(downY);
        float upWeight = fract(upY);
        vec3 a; vec3 b; vec3 c; vec3 d;
        if (!fetchSurfaceAtlasPixel(ivec2(x1, y0), slot0, a)
          || !fetchSurfaceAtlasPixel(ivec2(x2, y0), slot0, b)) return false;
        vec3 downRow = mix(a, b, weight.x);
        if (!fetchSurfaceAtlasPixel(ivec2(x0, y1), slot1, a)
          || !fetchSurfaceAtlasPixel(ivec2(x1, y1), slot1, b)
          || !fetchSurfaceAtlasPixel(ivec2(x2, y1), slot1, c)
          || !fetchSurfaceAtlasPixel(ivec2(x3, y1), slot1, d)) return false;
        vec3 lower = mix(b, c, weight.x);
        down_ZYX = mix(downRow, lower, downWeight);
        vec3 leftRow = mix(a, b, leftWeight);
        vec3 rightRow = mix(c, d, rightWeight);
        if (!fetchSurfaceAtlasPixel(ivec2(x0, y2), slot2, a)
          || !fetchSurfaceAtlasPixel(ivec2(x1, y2), slot2, b)
          || !fetchSurfaceAtlasPixel(ivec2(x2, y2), slot2, c)
          || !fetchSurfaceAtlasPixel(ivec2(x3, y2), slot2, d)) return false;
        vec3 upper = mix(b, c, weight.x);
        point_ZYX = mix(lower, upper, weight.y);
        left_ZYX = mix(leftRow, mix(a, b, leftWeight), weight.y);
        right_ZYX = mix(rightRow, mix(c, d, rightWeight), weight.y);
        if (!fetchSurfaceAtlasPixel(ivec2(x1, y3), slot3, a)
          || !fetchSurfaceAtlasPixel(ivec2(x2, y3), slot3, b)) return false;
        up_ZYX = mix(upper, mix(a, b, weight.x), upWeight);
        return true;
      }

      bool sampleSurfaceFrame(
        vec2 surface_UV,
        out vec3 point_ZYX,
        out vec3 tangentU_ZYX,
        out vec3 tangentV_ZYX,
        out vec3 normal_ZYX
      ) {
        vec3 left_ZYX; vec3 right_ZYX; vec3 down_ZYX; vec3 up_ZYX;
        if (!sampleSurfaceStencil(surface_UV, point_ZYX, left_ZYX, right_ZYX, down_ZYX, up_ZYX)) return false;
        vec3 rawTangentU_ZYX = right_ZYX - left_ZYX;
        vec3 rawTangentV_ZYX = up_ZYX - down_ZYX;
        float tangentULength = length(rawTangentU_ZYX);
        if (tangentULength < 0.000001) return false;
        tangentU_ZYX = rawTangentU_ZYX / tangentULength;
        vec3 orthogonalV_ZYX = rawTangentV_ZYX - tangentU_ZYX * dot(rawTangentV_ZYX, tangentU_ZYX);
        float tangentVLength = length(orthogonalV_ZYX);
        if (tangentVLength < 0.000001) return false;
        tangentV_ZYX = orthogonalV_ZYX / tangentVLength;
        normal_ZYX = normalize(cross(tangentU_ZYX.zyx, tangentV_ZYX.zyx)).zyx * normalSign;
        return all(greaterThanEqual(surface_UV, vec2(0.0)))
          && all(lessThanEqual(surface_UV, surfaceGrid_HW - vec2(1.0)));
      }
  `;
}
