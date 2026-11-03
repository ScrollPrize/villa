import assert from "node:assert/strict";
import { test } from "node:test";
import { catalogLayers, layerStyle, savedLayers } from "../src/extensions/layers.js";

const INK = "https://example.test/ink.zarr/";
const BUCKET_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";

function asset(type, path, parameters = null) {
  return {
    type,
    origins: [{ path, access_roots: [{ type: "s3", url: "s3://vesuvius-challenge-open-data" }] }],
    parameters,
  };
}

const METADATA = {
  models: {
    20260428123845: { long_id: "20260428123845-v3-78k-fullsup" },
    20260413141734: { long_id: "20260413141734-surface-recto-2um-ps256" },
    20260413222639: { long_id: "20260413222639-surface-m7" },
  },
  samples: {
    PHercParis4: {
      volumes: {
        20260411134726: {
          data: [
            asset("ome-zarr", "P4/volumes/v.zarr/"),
            asset("ink-detection-3d-zarr", "P4/ink3d.zarr/", { model_id: "20260428123845" }),
            asset("surface-prediction-zarr", "P4/m7-L2.zarr/", { model_id: "20260413222639", level: 2 }),
            asset("surface-prediction-zarr", "P4/recto-L0.zarr/", { model_id: "20260413141734", level: 0 }),
            asset("normal-grids", "P4/recto-L0.normal-grids/", { model_id: "20260413141734", level: 0 }),
            asset("lasagna", "P4/lasagna/", { model_id: "20260413141734", level: 2 }),
          ],
        },
        20230205180739: { data: [asset("ome-zarr", "P4/volumes/old.zarr")] },
      },
      segments: {},
    },
  },
};

test("a layer's #ff8000 at half opacity from 100 to 255 is drawn in red 1, green 128/255, blue 0 at 0.5 there", () => {
  assert.deepEqual(layerStyle({ color: "#ff8000", opacity: 0.5, range: [100, 255] }), {
    color_RGB: [1, 128 / 255, 0],
    opacity: 0.5,
    range: [100, 255],
  });
});

test("saved layers pass as they are, and one with an opacity over 1, a named colour or a repeated URL is refused", () => {
  const ink = { location: INK, color: "#ff00ff", opacity: 0.65, visible: false, range: [100, 255] };
  assert.deepEqual(savedLayers([ink]), [ink]);
  assert.throws(() => savedLayers([{ ...ink, opacity: 2 }]), /^Error: Saved layer is malformed: .*"opacity":2/);
  assert.throws(() => savedLayers([{ ...ink, color: "red" }]), /Saved layer is malformed/);
  assert.throws(() => savedLayers([ink, ink]), { message: "Saved layer locations must be unique." });
  assert.throws(() => savedLayers({}), { message: "Saved layers must be an array: {}." });
});

test("a saved layer without a range draws every value, and one whose range is not 0 ≤ low ≤ high ≤ 255 is refused", () => {
  const ink = { location: INK, color: "#ff00ff", opacity: 0.65, visible: true };
  assert.deepEqual(savedLayers([ink]), [{ ...ink, range: [0, 255] }]);
  for (const range of [[200, 100], [0, 256], [-1, 255], [0.5, 255], [100], "100-255", null])
    assert.throws(() => savedLayers([{ ...ink, range }]), {
      message: `Saved layer's range is not [low, high] with 0 ≤ low ≤ high ≤ 255: ${JSON.stringify(range)}.`,
    });
});

test("a CT's layers in the catalog are its 3D ink and its level 0 surface prediction, by kind and model", () => {
  assert.deepEqual(catalogLayers(METADATA).get(`${BUCKET_URL}/P4/volumes/v.zarr/`), [
    { name: "3D ink 20260428123845-v3-78k-fullsup", url: `${BUCKET_URL}/P4/ink3d.zarr/` },
    { name: "Surface 20260413141734-surface-recto-2um-ps256", url: `${BUCKET_URL}/P4/recto-L0.zarr/` },
  ]);
});

test("a CT with no predictions, or one the catalog does not have, has no layers there", () => {
  const layers = catalogLayers(METADATA);
  assert.deepEqual([...layers.keys()], [`${BUCKET_URL}/P4/volumes/v.zarr/`]);
  assert.equal(layers.get(`${BUCKET_URL}/P4/volumes/old.zarr/`), undefined);
  assert.equal(layers.get("https://example.test/elsewhere.zarr/"), undefined);
});
