import assert from "node:assert/strict";
import { createServer } from "node:http";
import { test } from "node:test";
import { deflateSync } from "node:zlib";
import { boxMesh, chunkBand, prepared } from "../src/extensions/ink/band.js";
import { catalogInk } from "../src/extensions/ink/catalog.js";
import { detailRegion, downsampled } from "../src/extensions/ink/detail.js";
import { openTiff, readTile, tiffLayout, tilePixels } from "../src/extensions/ink/tiff.js";
import { savedImages } from "../src/extensions/ink.js";

// Paris 4's ink TIFF of the checks' segment: 31960 × 42060 pixels in tiles of 1024², over its ds8 JPEG, on a grid of
// scale 0.05.
const PARIS = { shape_HW: [31960, 42060], tile_HW: [1024, 1024], grid_HW: [32, 42] };
const SCALE_UV = [0.05, 0.05];

async function deflated(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new CompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function tiff({ shape_HW, tile_HW, pixels_HW, little, compression, predictor, bits = 8 }) {
  const grid_HW = shape_HW.map((size, axis) => Math.ceil(size / tile_HW[axis]));
  const tiles = [];
  for (let tileRow = 0; tileRow < grid_HW[0]; tileRow++)
    for (let tileColumn = 0; tileColumn < grid_HW[1]; tileColumn++) {
      const tile = new Uint8Array(tile_HW[0] * tile_HW[1]);
      for (let row = 0; row < tile_HW[0]; row++)
        for (let column = 0; column < tile_HW[1]; column++) {
          const [y, x] = [tileRow * tile_HW[0] + row, tileColumn * tile_HW[1] + column];
          if (y < shape_HW[0] && x < shape_HW[1]) tile[row * tile_HW[1] + column] = pixels_HW[y * shape_HW[1] + x];
        }
      if (predictor === 2)
        for (let row = 0; row < tile_HW[0]; row++)
          for (let at = (row + 1) * tile_HW[1] - 1; at > row * tile_HW[1]; at--) tile[at] -= tile[at - 1];
      tiles.push(compression === 1 ? tile : await deflated(tile));
    }
  const entries = [
    [256, 3, 1, shape_HW[1]],
    [257, 3, 1, shape_HW[0]],
    [258, 3, 1, bits],
    [259, 3, 1, compression],
    [262, 3, 1, 1],
    [277, 3, 1, 1],
    [317, 3, 1, predictor],
    [322, 3, 1, tile_HW[1]],
    [323, 3, 1, tile_HW[0]],
    [324, 4, tiles.length, 0],
    [325, 4, tiles.length, 0],
  ];
  const tables = 8 + 2 + entries.length * 12 + 4;
  entries[9][3] = tiles.length === 1 ? tables + 8 : tables;
  entries[10][3] = tiles.length === 1 ? tiles[0].length : tables + tiles.length * 4;
  let at = tables + tiles.length * 8;
  const bytes = new Uint8Array(at + tiles.reduce((total, tile) => total + tile.length, 0));
  const view = new DataView(bytes.buffer);
  bytes.set(little ? [73, 73] : [77, 77]);
  view.setUint16(2, 42, little);
  view.setUint32(4, 8, little);
  view.setUint16(8, entries.length, little);
  for (const [index, [tag, type, count, value]] of entries.entries()) {
    const entry = 10 + index * 12;
    view.setUint16(entry, tag, little);
    view.setUint16(entry + 2, type, little);
    view.setUint32(entry + 4, count, little);
    if (type === 3) view.setUint16(entry + 8, value, little);
    else view.setUint32(entry + 8, value, little);
  }
  for (const [index, tile] of tiles.entries()) {
    view.setUint32(tables + index * 4, at, little);
    view.setUint32(tables + tiles.length * 4 + index * 4, tile.length, little);
    bytes.set(tile, at);
    at += tile.length;
  }
  return bytes;
}

async function readBack(bytes) {
  const layout = await tiffLayout(bytes.subarray(0, 64), async (offset, length) =>
    bytes.slice(offset, offset + length),
  );
  const [height, width] = layout.shape_HW;
  const image_HW = new Uint8Array(height * width);
  for (let index = 0; index < layout.offsets_N.length; index++) {
    const tile_HW = await tilePixels(
      layout,
      bytes.slice(layout.offsets_N[index], layout.offsets_N[index] + layout.counts_N[index]),
    );
    const [top, left] = [Math.floor(index / layout.grid_HW[1]), index % layout.grid_HW[1]].map(
      (tile, axis) => tile * layout.tile_HW[axis],
    );
    for (let row = 0; row < layout.tile_HW[0] && top + row < height; row++)
      for (let column = 0; column < layout.tile_HW[1] && left + column < width; column++)
        image_HW[(top + row) * width + left + column] = tile_HW[row * layout.tile_HW[1] + column];
  }
  return { layout, image_HW };
}

async function serve(t, respond) {
  const server = createServer(respond);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });
  return `http://127.0.0.1:${server.address().port}/ink.tif`;
}

const SHAPE_HW = [40, 50];
const PIXELS_HW = Uint8Array.from(
  { length: 40 * 50 },
  (_, index) => (Math.floor(index / 50) * 7 + (index % 50) * 3) % 256,
);

test("a deflated little-endian TIFF with horizontal differencing reads back to its pixels, tile by tile", async () => {
  const bytes = await tiff({
    shape_HW: SHAPE_HW,
    tile_HW: [16, 32],
    pixels_HW: PIXELS_HW,
    little: true,
    compression: 32946,
    predictor: 2,
  });
  const { layout, image_HW } = await readBack(bytes);
  assert.deepEqual(
    [layout.shape_HW, layout.tile_HW, layout.grid_HW, layout.deflated, layout.differenced],
    [[40, 50], [16, 32], [3, 2], true, true],
  );
  assert.deepEqual(image_HW, PIXELS_HW);
});

test("a big-endian TIFF of raw tiles reads back to its pixels", async () => {
  const bytes = await tiff({
    shape_HW: SHAPE_HW,
    tile_HW: [32, 16],
    pixels_HW: PIXELS_HW,
    little: false,
    compression: 1,
    predictor: 1,
  });
  const { layout, image_HW } = await readBack(bytes);
  assert.deepEqual([layout.grid_HW, layout.deflated, layout.differenced], [[2, 4], false, false]);
  assert.deepEqual(image_HW, PIXELS_HW);
});

test("a TIFF of 16-bit pixels or LZW tiles is refused, naming what it has", async () => {
  const page = { shape_HW: SHAPE_HW, tile_HW: [16, 16], pixels_HW: PIXELS_HW, little: true, predictor: 1 };
  const read = (bytes) => tiffLayout(bytes, async () => assert.fail("the whole file is the head"));
  await assert.rejects(read(await tiff({ ...page, compression: 1, bits: 16 })), {
    message: "The TIFF is not one grey 8-bit sample a pixel, black at 0.",
  });
  await assert.rejects(read(await tiff({ ...page, compression: 5 })), {
    message: "TIFF compression 5 with predictor 1 is not read; 1 or deflate, 1 or 2.",
  });
});

test("a deflated tile stops at its pixel count before a missing zlib trailer is read", async () => {
  const layout = { tile_HW: [16, 16], deflated: true, differenced: false };
  const bomb = deflateSync(new Uint8Array(1024 * 1024));
  for (const bytes of [bomb, bomb.subarray(0, -4)])
    await assert.rejects(tilePixels(layout, bytes), { message: "A TIFF tile of 16 × 16 pixels exceeds 256 bytes." });
  await assert.rejects(tilePixels(layout, deflateSync(new Uint8Array(255))), {
    message: "A TIFF tile of 16 × 16 pixels decoded to 255 bytes.",
  });
});

test("TIFF tile tables refuse excessive, empty or short raw counts and ranges beyond classic TIFF", async () => {
  const page = { shape_HW: [16, 16], tile_HW: [16, 16], pixels_HW: PIXELS_HW, little: true, predictor: 1 };
  for (const [compression, count, offset, expected] of [
    [8, 0xffffffff, 154, /tile 0 has 4294967295 encoded bytes.*limit 1536/],
    [8, 0, 154, /tile 0 has 0 encoded bytes/],
    [1, 255, 154, /tile 0 has 255 encoded bytes/],
    [8, 30, 0xfffffff0, /range of 30 bytes at 4294967280 is refused/],
    [8, 30, 0, /range of 30 bytes at 0 is refused/],
  ]) {
    const bytes = await tiff({ ...page, compression });
    const view = new DataView(bytes.buffer);
    view.setUint32(10 + 9 * 12 + 8, offset, true);
    view.setUint32(10 + 10 * 12 + 8, count, true);
    await assert.rejects(
      tiffLayout(bytes, async () => assert.fail("invalid tile tables must not request tile bytes")),
      expected,
    );
  }
});

test("a short TIFF header range and exact tile range read incompressible deflate with inline tile tables", async (t) => {
  let state = 12345;
  const pixels_HW = Uint8Array.from({ length: 16 * 16 }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state >>> 24;
  });
  const bytes = await tiff({
    shape_HW: [16, 16],
    tile_HW: [16, 16],
    pixels_HW,
    little: true,
    compression: 8,
    predictor: 1,
  });
  const asked = [];
  const url = await serve(t, (request, response) => {
    asked.push(request.headers.range);
    const [start, end] = request.headers.range.slice(6).split("-").map(Number);
    response.writeHead(206).end(bytes.subarray(start, end + 1));
  });
  const layout = await openTiff(url);
  assert.ok(layout.counts_N[0] > pixels_HW.length, "incompressible deflate may exceed its decoded byte count");
  assert.deepEqual(await readTile(url, layout, 0), pixels_HW);
  assert.deepEqual(asked, [
    "bytes=0-65535",
    `bytes=${layout.offsets_N[0]}-${layout.offsets_N[0] + layout.counts_N[0] - 1}`,
  ]);
});

test("rejected HTTP statuses and excessive fixed or chunked range bodies cancel without waiting for EOF", async (t) => {
  for (const mode of ["status200", "status404", "declared", "chunked"])
    await t.test(mode, { timeout: 5000 }, async (t) => {
      let closed;
      const url = await serve(t, (request, response) => {
        assert.equal(request.headers.range, "bytes=0-65535");
        closed = new Promise((resolve) => response.once("close", resolve));
        const status = mode.startsWith("status") ? Number(mode.slice(6)) : 206;
        response.writeHead(status, mode === "declared" ? { "Content-Length": "4294967295" } : {});
        response.flushHeaders();
        if (mode === "chunked") response.write(Buffer.alloc(65537));
      });
      const signal = AbortSignal.timeout(3000);
      const expected = mode.startsWith("status")
        ? /answered (200|404) to a request for bytes=0-65535/
        : mode === "declared"
          ? /declared 4294967295 bytes for bytes=0-65535, more than 65536/
          : /bytes=0-65535 exceeds 65536 bytes/;
      await assert.rejects(openTiff(url, signal), expected);
      await closed;
      assert.equal(signal.aborted, false, "the reader must cancel the response before the test's abort");
    });
});

test("a tile range still requires its exact byte count", async (t) => {
  const layout = {
    tile_HW: [16, 16],
    offsets_N: [154],
    counts_N: [256],
    deflated: false,
    differenced: false,
  };
  const url = await serve(t, (request, response) => {
    assert.equal(request.headers.range, "bytes=154-409");
    response.writeHead(206).end(new Uint8Array(255));
  });
  await assert.rejects(readTile(url, layout, 0), /gave 255 bytes at 154, not 256/);
});

test("the detail of a view zoomed to 1.5 full pixels a device pixel is the whole tiles it touches, nearest first", () => {
  const view = { start_HW: [100, 200], stop_HW: [150, 260], scale_UV: SCALE_UV, pixel: 1.5 };
  assert.deepEqual(detailRegion(PARIS, 8, view), {
    downsample: 1,
    start_HW: [1024, 3072],
    shape_HW: [2048, 3072],
    tiles: [88, 46, 87, 89, 45, 47],
  });
});

test("the detail of the whole segment at 5 full pixels a device pixel is 16 tiles a side about its centre, at 4", () => {
  const view = { start_HW: [0, 0], stop_HW: [1598, 2103], scale_UV: SCALE_UV, pixel: 5 };
  const region = detailRegion(PARIS, 8, view);
  assert.deepEqual(
    [region.downsample, region.start_HW, region.shape_HW, region.tiles.length, region.tiles[0]],
    [4, [8192, 13312], [4096, 4096], 256, 15 * 42 + 20],
  );
});

test("the detail at the image's corner stops at its edge, and there is none off it or zoomed out to its own pixels", () => {
  const corner = { start_HW: [1590, 2100], stop_HW: [1600, 2110], scale_UV: SCALE_UV, pixel: 1 };
  assert.deepEqual(detailRegion(PARIS, 8, corner), {
    downsample: 1,
    start_HW: [31744, 41984],
    shape_HW: [216, 76],
    tiles: [1343],
  });
  assert.equal(detailRegion(PARIS, 8, { ...corner, start_HW: [-50, 0], stop_HW: [-10, 40] }), null);
  assert.equal(detailRegion(PARIS, 8, { ...corner, pixel: 8 }), null);
});

test("a tile downsampled by 4 is the mean of each 4 × 4 of its pixels", () => {
  const tile_HW = Uint8Array.from({ length: 32 * 32 }, (_, index) => Math.floor(index / 32));
  const kept_HW = downsampled(tile_HW, [32, 32], 4);
  assert.deepEqual(
    kept_HW,
    Uint8Array.from({ length: 8 * 8 }, (_, index) => 4 * Math.floor(index / 8) + 2),
  );
  assert.equal(downsampled(tile_HW, [32, 32], 1), tile_HW);
});

function sheet(z = 10) {
  const points_zyx_HW3 = new Float32Array(5 * 5 * 3);
  for (let row = 0; row < 5; row++)
    for (let column = 0; column < 5; column++) points_zyx_HW3.set([z, 4 * row, 4 * column], 3 * (row * 5 + column));
  return prepared({ grid_HW: [5, 5], points_zyx_HW3 });
}
const FINE = { shape_ZYX: [32, 32, 32], chunks_ZYX: [8, 8, 8], downsample_ZYX: [1, 1, 1], offset_ZYX: [0, 0, 0] };
const COARSE = {
  shape_ZYX: [16, 16, 16],
  chunks_ZYX: [8, 8, 8],
  downsample_ZYX: [2, 2, 2],
  offset_ZYX: [0.5, 0.5, 0.5],
};
const IMAGE = { data_HW: new Uint8Array(25).fill(0).fill(200, 7, 8), shape_HW: [5, 5] };

function voxels(data_DHW, chunk_ZYX) {
  const found = [];
  data_DHW.forEach((value, index) => {
    if (value === 0) return;
    const local = [Math.floor(index / 64), Math.floor(index / 8) % 8, index % 8];
    found.push(`${local.map((at, axis) => at + chunk_ZYX[axis] * 8).join()}=${value}`);
  });
  return found.sort();
}

function block(z, depth, y, x, side, value) {
  const found = [];
  for (let dz = 0; dz < depth; dz++)
    for (let dy = 0; dy < side; dy++)
      for (let dx = 0; dx < side; dx++) found.push(`${z + dz},${y + dy},${x + dx}=${value}`);
  return found.sort();
}

test("the band of a flat sheet reaches the steps to each side of it, under the one inked pixel, in the chunk", () => {
  assert.deepEqual(voxels(chunkBand(sheet(), IMAGE, FINE, [1, 0, 1], 2), [1, 0, 1]), block(8, 5, 4, 8, 4, 200));
  assert.deepEqual(voxels(chunkBand(sheet(), IMAGE, FINE, [1, 0, 1], 0), [1, 0, 1]), block(10, 1, 4, 8, 4, 200));
  assert.equal(chunkBand(sheet(), IMAGE, FINE, [1, 0, 0], 2), null);
  assert.equal(chunkBand(sheet(), IMAGE, FINE, [0, 0, 1], 2), null);
});

test("each chunk of a band across a chunk's face holds its own share of it", () => {
  assert.deepEqual(voxels(chunkBand(sheet(8), IMAGE, FINE, [0, 0, 1], 2), [0, 0, 1]), block(6, 2, 4, 8, 4, 200));
  assert.deepEqual(voxels(chunkBand(sheet(8), IMAGE, FINE, [1, 0, 1], 2), [1, 0, 1]), block(8, 3, 4, 8, 4, 200));
});

test("a coarser level's band covers the voxels whose span of finest voxels it crosses", () => {
  assert.deepEqual(
    voxels(chunkBand(sheet(), IMAGE, COARSE, [0, 0, 0], 2), [0, 0, 0]),
    [4, 5, 6].flatMap((z) => block(z, 1, 2, 4, 2, 200)).sort(),
  );
});

test("a chunk at the level's edge is whole, 0 past the edge", () => {
  const narrow = { ...FINE, shape_ZYX: [32, 32, 10] };
  const data_DHW = chunkBand(sheet(), IMAGE, narrow, [1, 0, 1], 2);
  assert.equal(data_DHW.length, 8 ** 3);
  assert.deepEqual(
    voxels(data_DHW, [1, 0, 1]),
    block(8, 5, 4, 8, 4, 200).filter((voxel) => Number(voxel.split(/[,=]/)[2]) < 10),
  );
});

test("the mesh of a box has the cells with a corner in it, each corner once, where the image lies at it", () => {
  const surface = sheet();
  const mesh = boxMesh(surface, [0, 0, 0], [20, 5, 5]);
  assert.equal(mesh.triangles_T3.length, 4 * 6);
  assert.equal(mesh.positions_xyz_N3.length, 9 * 3);
  const corners = [];
  for (let vertex = 0; vertex < 9; vertex++)
    corners.push([
      ...mesh.positions_xyz_N3.subarray(3 * vertex, 3 * vertex + 3),
      ...mesh.uvs_N2.subarray(2 * vertex, 2 * vertex + 2),
    ]);
  const expected = [];
  for (let row = 0; row < 3; row++)
    for (let column = 0; column < 3; column++) expected.push([4 * column, 4 * row, 10, column / 5, row / 5]);
  assert.deepEqual(corners.sort(), expected.map((corner) => corner.map((value) => Math.fround(value))).sort());
  surface.valid_HW[0] = 0;
  assert.equal(boxMesh(surface, [0, 0, 0], [20, 5, 5]).triangles_T3.length, 3 * 6);
});

const BUCKET_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";
function asset(type, path, parameters = null) {
  return {
    type,
    origins: [{ path, access_roots: [{ type: "s3", url: "s3://vesuvius-challenge-open-data" }] }],
    parameters,
  };
}

const SEGMENT = "P4/segments/20230702185753";
const CATALOG = {
  models: {
    20260417190342: { long_id: "20260417190342-new_canon_autoresearch_recipe" },
    20260709123958: { long_id: "20260709123958-mrg20736-1um-s1z2" },
  },
  samples: {
    PHercParis4: {
      volumes: {},
      segments: {
        20230702185753: {
          data: [
            asset("tifxyz-transformed", `${SEGMENT}/mesh/on-a.tifxyz/`, { target_volume: "20260411134726" }),
            asset("tifxyz-transformed", `${SEGMENT}/mesh/on-b.tifxyz/`, { target_volume: "20260608103018" }),
            asset("ink-detection", `${SEGMENT}/ink/a-new_canon.tif`, {
              target_volume: "20260411134726",
              model_id: "20260417190342",
            }),
            asset("ink-detection-downsampled", `${SEGMENT}/ink/a-new_canon-ds8.jpg`, {
              target_volume: "20260411134726",
              model_id: "20260417190342",
              downscale: 8,
              format: "jpg",
            }),
            asset("ink-detection", `${SEGMENT}/ink/b-new_canon.tif`, {
              target_volume: "20260608103018",
              model_id: "20260417190342",
            }),
            asset("ink-detection", `${SEGMENT}/ink/b-L1-mrg.tif`, {
              target_volume: "20260608103018",
              model_id: "20260709123958",
              render_level: "1",
            }),
            asset("ink-detection-downsampled", `${SEGMENT}/ink/b-L1-mrg-ds8.jpg`, {
              target_volume: "20260608103018",
              model_id: "20260709123958",
              render_level: "1",
              downscale: 8,
              format: "jpg",
            }),
            asset("3d-ink-prediction", `${SEGMENT}/ink/a-L2-3d-ink-max.zarr/`, {
              target_volume: "20260411134726",
              render_level: "2",
            }),
          ],
        },
      },
    },
  },
};

test("a surface's ink images in the catalog are those on its volume, each with the TIFF of its model and level", () => {
  const images = catalogInk(CATALOG);
  assert.deepEqual(images.get(`${BUCKET_URL}/${SEGMENT}/mesh/on-a.tifxyz`), [
    {
      name: "20260417190342-new_canon_autoresearch_recipe",
      image: `${BUCKET_URL}/${SEGMENT}/ink/a-new_canon-ds8.jpg`,
      tiff: `${BUCKET_URL}/${SEGMENT}/ink/a-new_canon.tif`,
    },
  ]);
  assert.deepEqual(images.get(`${BUCKET_URL}/${SEGMENT}/mesh/on-b.tifxyz`), [
    {
      name: "20260709123958-mrg20736-1um-s1z2",
      image: `${BUCKET_URL}/${SEGMENT}/ink/b-L1-mrg-ds8.jpg`,
      tiff: `${BUCKET_URL}/${SEGMENT}/ink/b-L1-mrg.tif`,
    },
  ]);
});

test("a surface the catalog does not have has no ink images there", () => {
  assert.equal(catalogInk(CATALOG).get("https://example.test/elsewhere.tifxyz"), undefined);
});

test("saved ink images pass as they are, and one whose range is not 0 to 255 is refused", () => {
  const image = {
    location: "https://example.test/ink.jpg",
    tiff: null,
    surface: "https://example.test/segment.tifxyz",
    color: "#00ff00",
    opacity: 1,
    visible: true,
    volume: true,
    steps: 4,
    range: [100, 255],
  };
  assert.deepEqual(savedImages([image]), [image]);
  for (const range of [[255, 0], [0, 300], [100, 255.5], [100, 200, 255], { low: 100 }])
    assert.throws(() => savedImages([{ ...image, range }]), {
      message: `Saved ink image's range is not [low, high] with 0 ≤ low ≤ high ≤ 255: ${JSON.stringify(range)}.`,
    });
  assert.throws(() => savedImages([{ ...image, cut: [100, 255] }]), /^Error: Saved ink image is malformed: .*"cut"/);
});
