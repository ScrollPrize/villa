import assert from "node:assert/strict";
import { createServer } from "node:http";
import { test } from "node:test";
import { gzipSync } from "node:zlib";
import { responseDownloads } from "../src/volume/download-stats.js";
import { readResponseBytes } from "../src/volume/response-bytes.js";

async function serve(t, answer) {
  const server = createServer(answer);
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });
  return `http://127.0.0.1:${server.address().port}`;
}

async function holdTail(t, body, tailLength, headers) {
  let tailSent = false;
  let close;
  const closed = new Promise((resolve) => {
    close = resolve;
  });
  const url = await serve(t, (_request, response) => {
    response.writeHead(200, headers);
    response.write(body.subarray(0, -tailLength));
    const deadline = setTimeout(() => {
      tailSent = true;
      response.end(body.subarray(-tailLength));
    }, 1000);
    response.on("close", () => {
      clearTimeout(deadline);
      close();
    });
  });
  return { url, closed, tailSent: () => tailSent };
}

test("a decoded body is read whole whether its Content-Length says 3 or 30", async () => {
  for (const length of [3, 30]) {
    const bytes = await readResponseBytes(
      new Response("decoded body", { headers: { "Content-Length": String(length) } }),
      64,
    );
    assert.equal(new TextDecoder().decode(bytes), "decoded body");
  }
});

test("a 20-byte body over a limit of 8 throws, and so does a declared length of 30", async () => {
  await assert.rejects(
    readResponseBytes(new Response("too much decoded data", { headers: { "Content-Length": "3" } }), 8),
    /exceeds the 8-byte limit/,
  );
  await assert.rejects(
    readResponseBytes(new Response("x", { headers: { "Content-Length": "30" } }), 8),
    /declares 30 bytes, exceeding the 8-byte limit/,
  );
});

test("HTTP gzip output beyond 2048 bytes is refused before its held trailer, with or without Content-Length", async (t) => {
  const compressed = gzipSync(Buffer.alloc(1024 * 1024));
  assert.ok(compressed.byteLength < 2048);
  for (const declared of [true, false]) {
    const headers = { "Content-Encoding": "gzip", ...(declared ? { "Content-Length": compressed.byteLength } : {}) };
    const server = await holdTail(t, compressed, 8, headers);
    const response = await fetch(server.url);
    const before = responseDownloads.snapshot();
    await assert.rejects(readResponseBytes(response, 2048), /exceeds the 2048-byte limit/);
    await server.closed;
    assert.equal(server.tailSent(), false, "the byte limit must cancel HTTP before the gzip trailer is sent");
    assert.equal(response.body.locked, false);
    assert.equal(responseDownloads.downloadRequests, before.downloadRequests);
    assert.equal(responseDownloads.downloadBytes, before.downloadBytes);
  }
});

test("HTTP without Content-Length is cancelled at 2049 bytes while the body remains open", async (t) => {
  const server = await holdTail(t, Buffer.alloc(2050, 7), 1, {});
  const response = await fetch(server.url);
  assert.equal(response.headers.get("Content-Length"), null);
  await assert.rejects(readResponseBytes(response, 2048), /exceeds the 2048-byte limit/);
  await server.closed;
  assert.equal(server.tailSent(), false, "the byte limit must cancel HTTP before the final byte is sent");
});

test("raw and HTTP gzip bodies preserve all 150000 bytes, at the exact limit, with and without Content-Length", async (t) => {
  const expected = Uint8Array.from({ length: 150000 }, (_, index) => index % 251);
  const compressed = gzipSync(expected);
  const url = await serve(t, (request, response) => {
    const gzip = request.url.includes("gzip");
    const body = gzip ? compressed : expected;
    const headers = gzip ? { "Content-Encoding": "gzip" } : {};
    if (request.url.includes("declared")) headers["Content-Length"] = body.byteLength;
    response.writeHead(200, headers);
    response.write(body.subarray(0, 17));
    response.end(body.subarray(17));
  });
  const before = responseDownloads.snapshot();
  for (const path of ["/raw-declared", "/raw-chunked", "/gzip-declared", "/gzip-chunked"]) {
    const response = await fetch(`${url}${path}`);
    const bytes = await readResponseBytes(response, expected.byteLength);
    assert.deepEqual(bytes, expected, path);
    assert.equal(response.body.locked, false);
  }
  assert.equal(responseDownloads.downloadRequests - before.downloadRequests, 4);
  assert.equal(responseDownloads.downloadBytes - before.downloadBytes, 4 * expected.byteLength);
});

test("zero-byte limits accept empty HTTP and absent bodies and refuse the first incoming byte", async (t) => {
  const url = await serve(t, (request, response) => {
    if (request.url === "/absent") response.writeHead(204).end();
    else response.writeHead(200, { "Content-Length": 0 }).end();
  });
  for (const path of ["/empty", "/absent"]) {
    assert.deepEqual(await readResponseBytes(await fetch(`${url}${path}`), 0), new Uint8Array(0));
  }
  const server = await holdTail(t, Buffer.from("xy"), 1, {});
  await assert.rejects(readResponseBytes(await fetch(server.url), 0), /exceeds the 0-byte limit/);
  await server.closed;
  assert.equal(server.tailSent(), false);
});

test("aborting a response reader cancels its pending HTTP read and releases the lock before the held tail", async (t) => {
  for (const abortFetch of [false, true]) {
    const controller = new AbortController();
    const server = await holdTail(t, Buffer.alloc(1024, 3), 1000, { "Content-Length": 1024 });
    const response = await fetch(server.url, abortFetch ? { signal: controller.signal } : {});
    const before = responseDownloads.snapshot();
    const reading = readResponseBytes(response, 1024, controller.signal);
    await new Promise(setImmediate);
    controller.abort();
    await assert.rejects(reading, { name: "AbortError" });
    await server.closed;
    assert.equal(server.tailSent(), false, "abort must close HTTP before the held tail is sent");
    assert.equal(response.body.locked, false);
    assert.equal(responseDownloads.downloadRequests, before.downloadRequests);
    assert.equal(responseDownloads.downloadBytes, before.downloadBytes);
  }
});
