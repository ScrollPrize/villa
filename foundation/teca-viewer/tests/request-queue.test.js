import assert from "node:assert/strict";
import { createServer } from "node:http";
import { test } from "node:test";
import { FETCH_ATTEMPTS, FETCH_RETRY_DELAY_MS } from "../src/config.js";
import { fetchWithRetry } from "../src/volume/request-queue.js";

async function serve() {
  const asked = new Map();
  const server = createServer((request, response) => {
    const statuses = request.url.slice(1).split(",").map(Number);
    const count = asked.get(request.url) ?? 0;
    asked.set(request.url, count + 1);
    response.writeHead(statuses[Math.min(count, statuses.length - 1)]).end("body");
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return {
    url: (path) => `http://127.0.0.1:${server.address().port}${path}`,
    asked,
    close: () => {
      server.closeAllConnections();
      server.close();
    },
  };
}

test("a 503 twice, then a 200, is a 200 after two waits; a 503 every time is handed back after the last try", async () => {
  const server = await serve();
  try {
    const started = performance.now();
    const [answered, refused] = await Promise.all([
      fetchWithRetry(server.url("/503,503,200")),
      fetchWithRetry(server.url("/503")),
    ]);
    assert.equal(answered.status, 200);
    assert.equal(refused.status, 503);
    assert.ok(performance.now() - started >= FETCH_RETRY_DELAY_MS * 3);
    assert.deepEqual(
      [...server.asked],
      [
        ["/503,503,200", 3],
        ["/503", FETCH_ATTEMPTS],
      ],
    );
  } finally {
    server.close();
  }
});

test("a 500, a 502 and a 504 are asked again; a 404 and a 403 are handed back at once", async () => {
  const server = await serve();
  try {
    const paths = ["/500,200", "/502,200", "/504,200", "/404,200", "/403,200"];
    const answers = await Promise.all(paths.map((path) => fetchWithRetry(server.url(path))));
    assert.deepEqual(
      answers.map((response) => response.status),
      [200, 200, 200, 404, 403],
    );
    assert.deepEqual(
      paths.map((path) => server.asked.get(path)),
      [2, 2, 2, 1, 1],
    );
  } finally {
    server.close();
  }
});

test("an abort while a read waits to ask again ends it at once, asked no more", async () => {
  const server = await serve();
  try {
    const controller = new AbortController();
    const started = performance.now();
    const read = fetchWithRetry(server.url("/503"), controller.signal);
    while (!server.asked.has("/503")) await new Promise(setImmediate);
    await new Promise((resolve) => setTimeout(resolve, FETCH_RETRY_DELAY_MS / 5));
    controller.abort();
    await assert.rejects(read, { name: "AbortError" });
    assert.ok(performance.now() - started < FETCH_RETRY_DELAY_MS);
    assert.equal(server.asked.get("/503"), 1);
  } finally {
    server.close();
  }
});
