import assert from "node:assert/strict";
import { test } from "node:test";
import { mirror, mirrors } from "../src/volume/s3-mirrors.js";

const HOSTED = "https://scrolls.s3.us-east-1.amazonaws.com";
const SHARED = "https://s3.us-east-1.amazonaws.com";
const HOSTED_DUALSTACK = "https://scrolls.s3.dualstack.us-east-1.amazonaws.com";
const SHARED_DUALSTACK = "https://s3.dualstack.us-east-1.amazonaws.com";
const PATH = "/volume.zarr/0/12/29/30";
const FOUR = [
  { url: `${HOSTED}${PATH}`, origin: HOSTED },
  { url: `${SHARED}/scrolls${PATH}`, origin: SHARED },
  { url: `${HOSTED_DUALSTACK}${PATH}`, origin: HOSTED_DUALSTACK },
  { url: `${SHARED_DUALSTACK}/scrolls${PATH}`, origin: SHARED_DUALSTACK },
];

test("an S3 URL that names its region has the same four mirrors in each of its four forms", () => {
  for (const { url } of FOUR) assert.deepEqual(mirrors(url), FOUR);
});

test("regionless, dotted-bucket, signed, non-default-port and non-S3 URLs are their own only mirror", () => {
  for (const [url, origin] of [
    ["https://scrolls.s3.amazonaws.com/volume.zarr/0/12/29/30", "https://scrolls.s3.amazonaws.com"],
    ["https://s3.us-east-1.amazonaws.com/scrolls.example.org/volume.zarr/0/1", SHARED],
    [`${HOSTED}${PATH}?X-Amz-Signature=abc`, HOSTED],
    [
      "https://scrolls.s3.us-east-1.amazonaws.com:8443/volume.zarr/0/1",
      "https://scrolls.s3.us-east-1.amazonaws.com:8443",
    ],
    ["https://data.example.org/volume.zarr/0/12/29/30", "https://data.example.org"],
    ["local://handle/volume.zarr/0/12/29/30", "null"],
  ]) {
    assert.deepEqual(mirrors(url), [{ url, origin }]);
    assert.deepEqual(mirror(url), { url, origin });
  }
});

test("an object keeps one mirror whichever form names it, and X chunks 31, 32 and 33 go to three hostnames", () => {
  for (const { url } of FOUR) assert.deepEqual(mirror(url), FOUR[0]);
  assert.deepEqual(
    [31, 32, 33].map((x) => mirror(`${HOSTED}/volume.zarr/0/12/29/${x}`).url),
    [
      `${SHARED_DUALSTACK}/scrolls/volume.zarr/0/12/29/31`,
      `${HOSTED_DUALSTACK}/volume.zarr/0/12/29/32`,
      `${SHARED}/scrolls/volume.zarr/0/12/29/33`,
    ],
  );
});

test("64 chunks of the plane z + y + x = 155 spread 13, 21, 15 and 15 over the four hostnames", () => {
  const counts = new Map(FOUR.map(({ origin }) => [origin, 0]));
  for (let a = 0; a < 8; a++) {
    for (let b = 0; b < 8; b++) {
      const { origin } = mirror(`${HOSTED}/volume.zarr/0/${40 + a}/${20 + b}/${95 - a - b}`);
      counts.set(origin, counts.get(origin) + 1);
    }
  }
  assert.deepEqual([...counts.values()], [13, 21, 15, 15]);
});
