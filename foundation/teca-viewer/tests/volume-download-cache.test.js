import assert from "node:assert/strict";
import { createServer } from "node:http";
import { test } from "node:test";
import { CachedDownload } from "../src/volume/cached-download.js";
import { freshUntil, storableEtag, strongEtag } from "../src/volume/download-cache.js";

const DAY = 24 * 3600 * 1000;

test("no ETag, a weak one, an unquoted one and one with an inner quote are not strong, and '\"multipart-2\"' is", () => {
  for (const etag of [null, 'W/"a"', "a", '"a"b"']) {
    assert.equal(strongEtag(new Headers(etag === null ? {} : { ETag: etag })), null);
  }
  assert.equal(strongEtag(new Headers({ ETag: '"multipart-2"' })), '"multipart-2"');
});

test("no-store in any case or position keeps a strong ETag from being stored, and max-age=60 does not", () => {
  for (const control of ["no-store", "private, no-store", "No-Store, max-age=0"]) {
    assert.equal(storableEtag(new Headers({ ETag: '"a"', "Cache-Control": control })), null);
  }
  assert.equal(storableEtag(new Headers({ ETag: '"a"', "Cache-Control": "max-age=60" })), '"a"');
});

test("a body stored 169 days after its Last-Modified is fresh 16.9 days, and max-age, Expires and no-cache overrule that", () => {
  const at = Date.parse("2026-09-30T00:00:00Z");
  const modified = "Tue, 14 Apr 2026 00:00:00 GMT";
  const fresh = (headers) => new Date(freshUntil(new Headers(headers), at)).toISOString();
  assert.equal(fresh({ "Last-Modified": modified }), "2026-10-16T21:36:00.000Z");
  assert.equal(fresh({ "Last-Modified": modified, "Cache-Control": "s-maxage=600" }), "2026-10-16T21:36:00.000Z");
  assert.equal(fresh({ "Last-Modified": modified, "Cache-Control": "public, max-age=60" }), "2026-09-30T00:01:00.000Z");
  assert.equal(
    fresh({ "Last-Modified": modified, Expires: "Thu, 01 Oct 2026 00:00:00 GMT" }),
    "2026-10-01T00:00:00.000Z",
  );
  for (const headers of [
    { "Last-Modified": modified, "Cache-Control": "max-age=60, no-cache" },
    { "Last-Modified": modified, Expires: "0" },
    { "Last-Modified": "Thu, 01 Oct 2026 00:00:00 GMT" },
    {},
  ])
    assert.equal(fresh(headers), "2026-09-30T00:00:00.000Z", JSON.stringify(headers));
});

async function serve() {
  const asked = [];
  const server = createServer((request, response) => {
    asked.push(`${request.method} ${request.url}`);
    response
      .writeHead(200, { ETag: '"v1"', "Last-Modified": new Date(Date.now() - 100 * DAY).toUTCString() })
      .end("body");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return { url: `http://127.0.0.1:${server.address().port}/0/0/0`, asked, close: () => server.close() };
}

function storing(entry) {
  const refreshed = [];
  return {
    hitBytes: 0,
    refreshed,
    lookup: async () => ({ key: "key", epoch: 0, entry }),
    refresh: async (_lease, until) => refreshed.push(until),
  };
}

test("a fresh stored body is served asking nothing; a stale one is asked once with a HEAD and is fresh 10 days on", async () => {
  const server = await serve();
  try {
    const entry = { etag: '"v1"', status: 200, headers: [], body: new Blob(["stored"]), bytes: 6 };
    const fresh = storing({ ...entry, freshUntil: Date.now() + DAY });
    const read = await new CachedDownload("group", fresh).fetch(server.url, {});
    assert.equal(await read.response.text(), "stored");
    assert.deepEqual(server.asked, []);
    const stale = storing({ ...entry, freshUntil: Date.now() - 1 });
    const download = new CachedDownload("group", stale);
    const before = Date.now();
    const reads = await Promise.all([download.fetch(server.url, {}), download.fetch(server.url, {})]);
    assert.deepEqual(await Promise.all(reads.map(({ response }) => response.text())), ["stored", "stored"]);
    assert.deepEqual(server.asked, ["HEAD /0/0/0"]);
    assert.equal(stale.refreshed.length, 2);
    for (const until of stale.refreshed) assert.ok(Math.abs(until - before - 10 * DAY) < 5000, `fresh until ${until}`);
  } finally {
    server.close();
  }
});

async function stalling() {
  const asked = [];
  const closed = [];
  const server = createServer((request, response) => {
    asked.push(`${request.method} ${request.url}`);
    if (request.method === "HEAD") response.on("close", () => closed.push(request.url));
    else response.writeHead(200, { ETag: '"v1"' }).end("body");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return {
    url: `http://127.0.0.1:${server.address().port}/0/0/0`,
    asked,
    closed,
    close: () => server.close(),
    stop: () => server.closeAllConnections(),
  };
}

const settled = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

test("a stale body's HEAD lasts while one read waits on it, is cancelled with the last, and is asked again after", async () => {
  const server = await stalling();
  try {
    const entry = { etag: '"v1"', status: 200, headers: [], body: new Blob(["stored"]), bytes: 6, freshUntil: 0 };
    const download = new CachedDownload("group", storing(entry));
    const [first, second] = [new AbortController(), new AbortController()];
    const reads = [first, second].map(({ signal }) => download.fetch(server.url, { signal }));
    await settled(100);
    first.abort();
    await assert.rejects(reads[0], { name: "AbortError" });
    await settled(100);
    assert.deepEqual([server.asked, server.closed], [["HEAD /0/0/0"], []]);
    second.abort();
    await assert.rejects(reads[1], { name: "AbortError" });
    await settled(100);
    assert.deepEqual(server.closed, ["/0/0/0"]);
    const third = new AbortController();
    const again = download.fetch(server.url, { signal: third.signal });
    await settled(100);
    assert.deepEqual(server.asked, ["HEAD /0/0/0", "HEAD /0/0/0"]);
    third.abort();
    await assert.rejects(again, { name: "AbortError" });
  } finally {
    server.stop();
    server.close();
  }
});
