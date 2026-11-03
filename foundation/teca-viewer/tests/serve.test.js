import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { copyFileSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { createInterface } from "node:readline";
import { test } from "node:test";

const TEST_TIMEOUT_MS = 10000;

async function serve(t) {
  const directory = mkdtempSync(join(tmpdir(), "viewer-serve-"));
  const root = join(directory, "viewer");
  const files = {
    "viewer/index.html": "<title>Viewer</title>",
    "viewer/styles.css": "body { margin: 0; }",
    "viewer/src/main.js": "export const main = true;",
    "viewer/src/nested/worker.js": "postMessage('ready');",
    "viewer/vendor/codec.wasm": "runtime binary",
    "viewer/.env": "private root secret",
    "viewer/.git/config": "private git secret",
    "viewer/README.md": "private repository file",
    "viewer/tests/fixture.js": "private test file",
    "viewer/src/.env": "private runtime secret",
    "viewer/src/.private/key.js": "private hidden directory secret",
    "outside/key.js": "private outside secret",
  };
  for (const [path, body] of Object.entries(files)) {
    mkdirSync(dirname(join(directory, path)), { recursive: true });
    writeFileSync(join(directory, path), body);
  }
  symlinkSync("nested/worker.js", join(root, "src/safe.js"));
  symlinkSync("../README.md", join(root, "src/repository.js"));
  symlinkSync(".private/key.js", join(root, "src/hidden.js"));
  symlinkSync("../../outside/key.js", join(root, "src/outside.js"));
  symlinkSync("../../outside", join(root, "src/outside"));
  symlinkSync("loop.js", join(root, "src/loop.js"));
  copyFileSync(new URL("../serve.py", import.meta.url), join(root, "serve.py"));
  const child = spawn("python3", [join(root, "serve.py"), "--port", "0"], { stdio: ["ignore", "pipe", "pipe"] });
  const exited = once(child, "exit");
  const lines = createInterface({ input: child.stdout });
  let errors = "";
  child.stderr.on("data", (chunk) => {
    errors += chunk;
  });
  t.after(async () => {
    child.kill();
    await exited;
    lines.close();
    rmSync(directory, { recursive: true, force: true });
  });
  const [line] = await Promise.race([
    once(lines, "line"),
    exited.then(([code]) => {
      throw new Error(`serve.py exited ${code}: ${errors}`);
    }),
  ]);
  const match = /^Viewer: http:\/\/127\.0\.0\.1:(\d+)\/$/.exec(line);
  assert.ok(match, `serve.py must print its loopback address, got ${line}`);
  return Number(match[1]);
}

function read(port, path, { method = "GET", host = `127.0.0.1:${port}`, headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const hosts = host === null ? [] : Array.isArray(host) ? host : [host];
    const outgoing = request(
      {
        hostname: "127.0.0.1",
        port,
        path,
        method,
        headers: [...hosts.flatMap((value) => ["Host", value]), ...Object.entries(headers).flat()],
        setHost: false,
        agent: false,
      },
      (response) => {
        let body = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => {
          body += chunk;
        });
        response.on("error", reject);
        response.on("end", () =>
          resolve({ status: response.statusCode, headers: response.headers, version: response.httpVersion, body }),
        );
      },
    );
    outgoing.on("error", reject);
    outgoing.end();
  });
}

test("the local static host serves runtime files and refuses private paths and foreign Host headers", {
  timeout: TEST_TIMEOUT_MS,
}, async (t) => {
  const port = await serve(t);
  await t.test(
    "GET and HEAD serve the entry page, CSS, nested modules and binary assets with no-store over HTTP/1.1",
    async () => {
      const files = {
        "/": "<title>Viewer</title>",
        "/?volume=https://example.test/volume": "<title>Viewer</title>",
        "/index.html": "<title>Viewer</title>",
        "/styles.css": "body { margin: 0; }",
        "/src/main.js": "export const main = true;",
        "/src/nested/worker.js?version=1": "postMessage('ready');",
        "/src/safe.js": "postMessage('ready');",
        "/vendor/codec.wasm": "runtime binary",
      };
      for (const [path, body] of Object.entries(files)) {
        for (const method of ["GET", "HEAD"]) {
          const response = await read(port, path, { method });
          assert.equal(response.status, 200, `${method} ${path}`);
          assert.equal(response.body, method === "HEAD" ? "" : body, `${method} ${path}`);
          assert.equal(Number(response.headers["content-length"]), Buffer.byteLength(body));
          assert.equal(response.headers["cache-control"], "no-store");
          assert.equal(response.version, "1.1");
        }
      }
      const conditional = await read(port, "/src/main.js", {
        headers: { "If-None-Match": "*", "If-Modified-Since": "Wed, 01 Jan 2100 00:00:00 GMT" },
      });
      assert.equal(conditional.status, 200, "the development host must send the current file on conditional reads");
      assert.equal(conditional.body, files["/src/main.js"]);
    },
  );
  await t.test(
    "root repository files, hidden paths, encoded traversal and directories have no readable body or listing",
    async () => {
      const paths = [
        "/.env",
        "/%2eenv",
        "/.git/config",
        "/README.md",
        "/serve.py",
        "/tests/fixture.js",
        "/src/.env",
        "/src/%2eprivate/key.js",
        "/src/../.env",
        "/src/%2e%2e/%2eenv",
        "/src%2f..%2f.env",
        "/src/nested/%2e%2e/main.js",
        "/src/%5c../README.md",
        "/src/%00.js",
        "/src/%ff.js",
        "/src",
        "/src/",
        "/src/nested/",
        "/vendor/",
        "/src/missing.js",
      ];
      for (const path of paths) {
        for (const method of ["GET", "HEAD"]) {
          const response = await read(port, path, { method });
          assert.equal(response.status, 404, `${method} ${path}`);
          assert.equal(response.headers["cache-control"], "no-store");
          assert.doesNotMatch(response.body, /private .* secret|Directory listing/);
          if (method === "HEAD") assert.equal(response.body, "");
        }
      }
    },
  );
  await t.test(
    "symlinks cannot expose repository files, hidden files or a file or directory outside the runtime tree",
    async () => {
      for (const path of [
        "/src/repository.js",
        "/src/hidden.js",
        "/src/outside.js",
        "/src/outside/key.js",
        "/src/loop.js",
      ]) {
        const response = await read(port, path);
        assert.equal(response.status, 404, path);
        assert.doesNotMatch(response.body, /private .* secret/);
      }
    },
  );
  await t.test("one localhost or 127.0.0.1 Host is required, with no port or the bound port", async () => {
    for (const host of ["localhost", `localhost:${port}`, `LOCALHOST:${port}`, "127.0.0.1", `127.0.0.1:${port}`]) {
      assert.equal((await read(port, "/", { host })).status, 200, host);
    }
    for (const host of [
      null,
      "attacker.example",
      `attacker.example:${port}`,
      "localhost.attacker.example",
      "localhost:1",
      "127.0.0.1:1",
      `user@localhost:${port}`,
      ["localhost", "attacker.example"],
    ]) {
      for (const method of ["GET", "HEAD"]) {
        const response = await read(port, "/", { host, method });
        assert.equal(response.status, 403, `${method} Host: ${host}`);
        assert.doesNotMatch(response.body, /<title>Viewer<\/title>/);
      }
    }
  });
});
