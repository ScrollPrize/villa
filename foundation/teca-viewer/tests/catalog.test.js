import assert from "node:assert/strict";
import { test } from "node:test";
import { catalogVolumes, everySurface, matching } from "../src/extensions/catalog.js";

const BUCKET_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com";

function asset(type, roots, path, parameters = null) {
  return {
    type,
    origins: roots.map((url) => ({ path, access_roots: [{ type: url.split(":")[0], url }] })),
    parameters,
  };
}

const METADATA = {
  samples: {
    PHercParis4: {
      volumes: {
        20260411134726: {
          id: "20260411134726",
          sample_id: "PHercParis4",
          long_id: "20260411134726-2.400um-0.2m-78keV-masked.zarr",
          properties: { pixel_size_um: 2.4, energy_keV: 78.0 },
          data: [
            asset("ome-zarr", ["https://data.aws.ash2txt.org", "s3://vesuvius-challenge-open-data"], "P4/v.zarr/"),
            asset("ink-detection-3d-zarr", ["s3://vesuvius-challenge-open-data"], "P4/ink.zarr/"),
          ],
        },
        20230205180739: {
          id: "20230205180739",
          sample_id: "PHercParis4",
          long_id: "20230205180739-7.910um-54keV-masked.zarr",
          properties: { pixel_size_um: 7.91, energy_keV: 54.0 },
          data: [asset("ome-zarr", ["https://data.aws.ash2txt.org"], "samples/P4/old.zarr/")],
        },
        20200101000000: {
          id: "20200101000000",
          sample_id: "PHercParis4",
          long_id: "20200101000000-private.zarr",
          data: [asset("ome-zarr", ["s3://another-bucket"], "private.zarr/")],
        },
      },
      segments: {
        20231221180251: {
          id: "20231221180251",
          data: [
            asset("obj", ["s3://vesuvius-challenge-open-data"], "P4/s/a.obj"),
            asset("tifxyz-transformed", ["s3://vesuvius-challenge-open-data"], "P4/s/a-on-new.tifxyz/", {
              target_volume: "20260411134726",
            }),
            asset("tifxyz-transformed", ["s3://vesuvius-challenge-open-data"], "P4/s/a-on-private.tifxyz/", {
              target_volume: "20200101000000",
            }),
            asset("tifxyz-transformed", ["s3://vesuvius-challenge-open-data"], "P4/s/a-on-old.tifxyz/", {
              target_volume: "20230205180739",
            }),
          ],
        },
        20231106155351: {
          id: "20231106155351",
          data: [
            asset("tifxyz-transformed", ["s3://vesuvius-challenge-open-data"], "P4/s/b-on-new.tifxyz/", {
              target_volume: "20260411134726",
            }),
          ],
        },
      },
    },
  },
};

const OLD = "https://data.aws.ash2txt.org/samples/P4/old.zarr/";
const NEW = `${BUCKET_URL}/P4/v.zarr/`;
const OLD_TITLE = "PHercParis4 20230205180739-7.910um-54keV-masked";
const NEW_TITLE = "PHercParis4 20260411134726-2.400um-0.2m-78keV-masked";
const OLD_FIELDS = ["7.91 µm", "54 keV", "2023-02-05"];
const NEW_FIELDS = ["2.4 µm", "78 keV", "2026-04-11"];

function surface(segment, url, title, fields, volume) {
  const name = `${segment} ${title} ${fields.join(" ")}`;
  return { name, title, lines: [[segment, "PHercParis4"], fields], url, volume };
}

test("the catalog's volumes are those a URL reaches, in the bucket first, each a row of its sample and its fields", () => {
  assert.deepEqual(catalogVolumes(METADATA), [
    {
      name: `${OLD_TITLE} ${OLD_FIELDS.join(" ")}`,
      title: OLD_TITLE,
      lines: [["PHercParis4"], OLD_FIELDS],
      url: OLD,
      surfaces: [surface("20231221180251", `${BUCKET_URL}/P4/s/a-on-old.tifxyz/`, OLD_TITLE, OLD_FIELDS, OLD)],
    },
    {
      name: `${NEW_TITLE} ${NEW_FIELDS.join(" ")}`,
      title: NEW_TITLE,
      lines: [["PHercParis4"], NEW_FIELDS],
      url: NEW,
      surfaces: [
        surface("20231106155351", `${BUCKET_URL}/P4/s/b-on-new.tifxyz/`, NEW_TITLE, NEW_FIELDS, NEW),
        surface("20231221180251", `${BUCKET_URL}/P4/s/a-on-new.tifxyz/`, NEW_TITLE, NEW_FIELDS, NEW),
      ],
    },
  ]);
});

test("two volumes of a sample alike in voxel size, energy and date are told apart by the time of their scans", () => {
  const metadata = structuredClone(METADATA);
  metadata.samples.PHercParis4.volumes[20260411150000] = {
    id: "20260411150000",
    sample_id: "PHercParis4",
    long_id: "20260411150000-2.400um-0.2m-78keV-masked.zarr",
    properties: { pixel_size_um: 2.4, energy_keV: 78.0 },
    data: [asset("ome-zarr", ["s3://vesuvius-challenge-open-data"], "P4/w.zarr/")],
  };
  assert.deepEqual(
    catalogVolumes(metadata).map((volume) => volume.lines[1]),
    [OLD_FIELDS, ["2.4 µm", "78 keV", "2026-04-11 13:47:26"], ["2.4 µm", "78 keV", "2026-04-11 15:00:00"]],
  );
});

test("a volume with no energy, and one with no scan time for an id nor a number for a size, are listed", () => {
  const metadata = structuredClone(METADATA);
  delete metadata.samples.PHercParis4.volumes[20260411134726].properties.energy_keV;
  metadata.samples.PHercParis4.volumes.v2 = {
    id: "v2",
    sample_id: "PHercParis4",
    long_id: "v2-masked.zarr",
    properties: { pixel_size_um: "2.4" },
    data: [asset("ome-zarr", ["s3://vesuvius-challenge-open-data"], "P4/v2.zarr/")],
  };
  const volumes = catalogVolumes(metadata);
  assert.deepEqual(
    volumes.map((volume) => [volume.title, volume.lines, volume.url]),
    [
      [OLD_TITLE, [["PHercParis4"], OLD_FIELDS], OLD],
      [NEW_TITLE, [["PHercParis4"], ["2.4 µm", "2026-04-11"]], NEW],
      ["PHercParis4 v2-masked", [["PHercParis4"], ["v2"]], `${BUCKET_URL}/P4/v2.zarr/`],
    ],
  );
  assert.deepEqual(volumes[1].surfaces[0].lines, [
    ["20231106155351", "PHercParis4"],
    ["2.4 µm", "2026-04-11"],
  ]);
});

test("two volumes of a sample alike in their fields, their ids not times, are told apart by their ids", () => {
  const metadata = structuredClone(METADATA);
  for (const id of ["v2", "v3"])
    metadata.samples.PHercParis4.volumes[id] = {
      id,
      sample_id: "PHercParis4",
      long_id: `${id}.zarr`,
      properties: { pixel_size_um: 2.4, energy_keV: 78.0 },
      data: [asset("ome-zarr", ["s3://vesuvius-challenge-open-data"], `P4/${id}.zarr/`)],
    };
  assert.deepEqual(
    catalogVolumes(metadata).map((volume) => volume.lines[1]),
    [OLD_FIELDS, NEW_FIELDS, ["2.4 µm", "78 keV", "v2"], ["2.4 µm", "78 keV", "v3"]],
  );
});

test("a search keeps the items whose name holds every word, ignoring case", () => {
  const items = catalogVolumes(METADATA);
  assert.deepEqual(
    matching(items, " paris4  78KEV ").map((item) => item.url),
    [NEW],
  );
  assert.deepEqual(
    matching(items, "2.4 µm 2026-04").map((item) => item.url),
    [NEW],
  );
  assert.equal(matching(items, "").length, 2);
  assert.equal(matching(items, "paris4 7.91 78kev").length, 0);
});

test("every volume's surfaces are listed by segment and then by volume, each naming the volume it is registered to", () => {
  assert.deepEqual(everySurface(catalogVolumes(METADATA)), [
    surface("20231106155351", `${BUCKET_URL}/P4/s/b-on-new.tifxyz/`, NEW_TITLE, NEW_FIELDS, NEW),
    surface("20231221180251", `${BUCKET_URL}/P4/s/a-on-old.tifxyz/`, OLD_TITLE, OLD_FIELDS, OLD),
    surface("20231221180251", `${BUCKET_URL}/P4/s/a-on-new.tifxyz/`, NEW_TITLE, NEW_FIELDS, NEW),
  ]);
});

test("a volume whose path has no slash at its end is listed with one, as the open volume's URL has it", () => {
  const metadata = structuredClone(METADATA);
  metadata.samples.PHercParis4.volumes[20260411134726].data[0].origins[1].path = "P4/v.zarr";
  const volume = catalogVolumes(metadata).find((each) => each.title === NEW_TITLE);
  assert.deepEqual([volume.url, volume.surfaces.map((each) => each.volume)], [NEW, [NEW, NEW]]);
});
