import assert from "node:assert/strict";
import { mock, test } from "node:test";
import { HEDGE_AFTER_MS, ZARR_METADATA_TIME_LIMIT_MS } from "../src/config.js";
import { QueuedZarrStore } from "../src/volume/queued-zarr-store.js";
import { MirrorRequestQueue } from "../src/volume/request-queue.js";

const HOSTED = "https://scrolls.s3.us-east-1.amazonaws.com/volume.zarr/";
const SHARED = "https://s3.us-east-1.amazonaws.com/scrolls/volume.zarr/";
const PLAIN = "https://data.example.org/volume.zarr/";

function heldStore(root) {
  const asked = [];
  return {
    asked,
    metadata: new Map(),
    resolve: (key) => root + key.slice(1),
    get: (key, { url = root + key.slice(1), signal = null }) =>
      new Promise((answer, reject) => {
        asked.push({ url, signal, answer });
        signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
      }),
  };
}

async function tick(ms) {
  mock.timers.tick(ms);
  await new Promise(setImmediate);
}

test("a metadata read unanswered for 3 s is read from another hostname too, and the first answer wins", async () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const store = heldStore(HOSTED);
    const read = new QueuedZarrStore(store, new MirrorRequestQueue(HOSTED)).get("/.zattrs");
    await tick(HEDGE_AFTER_MS - 1);
    assert.deepEqual(
      store.asked.map(({ url }) => url),
      [`${HOSTED}.zattrs`],
    );
    await tick(1);
    assert.deepEqual(
      store.asked.map(({ url }) => url),
      [`${HOSTED}.zattrs`, `${SHARED}.zattrs`],
    );
    const bytes = new Uint8Array([123, 125]);
    store.asked[1].answer(bytes);
    assert.equal(await read, bytes);
    assert.equal(store.asked[0].signal.aborted, true);
  } finally {
    mock.timers.reset();
  }
});

test("one no hostname answers fails 20 s after it was asked, naming its URL; a host with no mirror is asked once", async () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    for (const [root, urls] of [
      [HOSTED, [`${HOSTED}0/.zarray`, `${SHARED}0/.zarray`]],
      [PLAIN, [`${PLAIN}0/.zarray`]],
    ]) {
      const store = heldStore(root);
      let failure = null;
      new QueuedZarrStore(store, new MirrorRequestQueue(root)).get("/0/.zarray").catch((error) => {
        failure = error;
      });
      await tick(ZARR_METADATA_TIME_LIMIT_MS - 1);
      assert.equal(failure, null);
      await tick(1);
      assert.equal(failure?.message, `No answer for ${root}0/.zarray in ${ZARR_METADATA_TIME_LIMIT_MS / 1000} s.`);
      assert.deepEqual(
        store.asked.map(({ url, signal }) => [url, signal.aborted]),
        urls.map((url) => [url, true]),
      );
    }
  } finally {
    mock.timers.reset();
  }
});
