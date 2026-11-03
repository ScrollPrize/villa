// It binds through three.js's state, so three's record of bound textures stays right.

export function uploadFloatRgbaArrayLayer({
  renderer,
  texture,
  layer,
  atlasWidth,
  sourcePixelCount,
  data_RGBA,
  pixelOffset = 0,
  pixelCount = sourcePixelCount - pixelOffset,
}) {
  if (
    !Number.isSafeInteger(layer) ||
    layer < 0 ||
    !Number.isSafeInteger(atlasWidth) ||
    atlasWidth < 1 ||
    !Number.isSafeInteger(sourcePixelCount) ||
    sourcePixelCount < 1 ||
    !(data_RGBA instanceof Float32Array) ||
    data_RGBA.length !== sourcePixelCount * 4 ||
    !Number.isSafeInteger(pixelOffset) ||
    pixelOffset < 0 ||
    !Number.isSafeInteger(pixelCount) ||
    pixelCount < 1 ||
    pixelOffset + pixelCount > sourcePixelCount
  ) {
    throw new Error("Surface array-layer upload inputs are malformed.");
  }
  const gl = renderer.getContext();
  renderer.state.activeTexture(gl.TEXTURE0);
  renderer.initTexture(texture);
  const webglTexture = renderer.properties.get(texture).__webglTexture;
  if (webglTexture === undefined) throw new Error("Surface strip texture allocation has no WebGL object.");
  renderer.state.bindTexture(gl.TEXTURE_2D_ARRAY, webglTexture, gl.TEXTURE0);
  try {
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
    gl.pixelStorei(gl.UNPACK_IMAGE_HEIGHT, 0);
    gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0);
    gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
    gl.pixelStorei(gl.UNPACK_SKIP_IMAGES, 0);
    const endPixel = pixelOffset + pixelCount;
    while (pixelOffset < endPixel) {
      const x = pixelOffset % atlasWidth;
      const y = Math.floor(pixelOffset / atlasWidth);
      const remaining = endPixel - pixelOffset;
      const width = Math.min(atlasWidth - x, remaining);
      const height = x === 0 && width === atlasWidth ? Math.floor(remaining / atlasWidth) : 1;
      gl.texSubImage3D(
        gl.TEXTURE_2D_ARRAY,
        0,
        x,
        y,
        layer,
        width,
        height,
        1,
        gl.RGBA,
        gl.FLOAT,
        data_RGBA,
        pixelOffset * 4,
      );
      pixelOffset += width * height;
    }
  } finally {
    renderer.state.unbindTexture();
  }
  return webglTexture;
}
