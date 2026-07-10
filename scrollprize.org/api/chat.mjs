// Vercel Function: POST /api/chat ("Ask the Scrolls").
//
// Vercel's Node.js runtime invokes the default export classic-style —
// handler(req, res) with an IncomingMessage/ServerResponse pair — so this
// entry bridges Node <-> Web and delegates to the runtime-agnostic
// handleChat (Request -> Response), the same core the local dev shim uses.
// If a runtime passes a Web Request instead, it is handled as-is.
//
// .mjs because ai@7 / @ai-sdk/gateway@4 are ESM-only while the root
// package.json is CommonJS. maxDuration for the ~60s per-request abort
// lives in vercel.json.

import { Readable } from 'node:stream';
import { handleChat } from './_lib/handler.mjs';

// Build a Web Request from an incoming Node request.
async function toWebRequest(req) {
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const body = chunks.length ? Buffer.concat(chunks) : undefined;
  const headers = new Headers();
  for (const [k, v] of Object.entries(req.headers)) {
    if (Array.isArray(v)) v.forEach((val) => headers.append(k, val));
    else if (v != null) headers.set(k, v);
  }
  const proto = req.headers['x-forwarded-proto'] || 'https';
  const host = req.headers.host || 'localhost';
  const init = { method: req.method, headers };
  if (body && req.method !== 'GET' && req.method !== 'HEAD') init.body = body;
  return new Request(`${proto}://${host}${req.url}`, init);
}

// Pipe a Web Response back out through the Node response (streams chunks).
async function sendWebResponse(res, webRes) {
  const headers = {};
  webRes.headers.forEach((v, k) => {
    headers[k] = v;
  });
  res.writeHead(webRes.status, headers);
  if (!webRes.body) {
    res.end();
    return;
  }
  Readable.fromWeb(webRes.body).pipe(res);
}

export default async function handler(req, res) {
  // Web-style invocation (req is a Fetch API Request): delegate directly.
  if (typeof req?.headers?.get === 'function') {
    return handleChat(req);
  }
  try {
    const request = await toWebRequest(req);
    const webRes = await handleChat(request);
    await sendWebResponse(res, webRes);
  } catch (err) {
    console.error(
      JSON.stringify({ route: '/api/chat', event: 'entry_error', error: String(err) })
    );
    if (!res.headersSent) {
      res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: 'Server error.' }));
    } else {
      res.end();
    }
  }
}
