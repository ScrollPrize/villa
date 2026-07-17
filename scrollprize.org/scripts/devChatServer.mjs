// Local dev shim for POST /api/chat. No extra deps — plain node:http.
//
//   node scripts/genChatCorpus.js      # produce api/_lib/corpus.json first
//   node scripts/devChatServer.mjs     # serve on PORT (default 3901)
//
// Reuses the exact production handler in api/_lib/handler.mjs (Web Request ->
// Web Response), bridging Node's http req/res to/from the Web standard so behavior
// matches the deployed function. Permissive CORS for localhost origins.
//
// Modes:
//   CHAT_MOCK=1                 -> streams a canned answer word-by-word, no gateway.
//   AI_GATEWAY_API_KEY unset    -> serves 503s (does not crash; the widget hides).
//   AI_GATEWAY_API_KEY set      -> real gateway calls.

import http from 'node:http';
import { Readable } from 'node:stream';
import { handleChat } from '../api/_lib/handler.mjs';
import { rateLimitMode } from '../api/_lib/rateLimit.mjs';

const PORT = Number.parseInt(process.env.PORT ?? '', 10) || 3901;
const DEFAULT_MODEL = 'openai/gpt-5-mini';

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
  const host = req.headers.host || `localhost:${PORT}`;
  const url = `http://${host}${req.url}`;
  const init = { method: req.method, headers };
  if (body && req.method !== 'GET' && req.method !== 'HEAD') init.body = body;
  return new Request(url, init);
}

// Pipe a Web Response back out through the Node response.
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

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    if (url.pathname !== '/api/chat') {
      res.writeHead(404, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: 'Not found. Try POST /api/chat.' }));
      return;
    }
    const request = await toWebRequest(req);
    // allowLocalhostOrigins: reflect any http://localhost:* / 127.0.0.1 origin so
    // the Docusaurus dev server (3000/3333) can call this shim cross-origin.
    const webRes = await handleChat(request, { allowLocalhostOrigins: true });
    await sendWebResponse(res, webRes);
  } catch (err) {
    if (!res.headersSent) {
      res.writeHead(500, { 'Content-Type': 'application/json; charset=utf-8' });
      res.end(JSON.stringify({ error: 'Dev server error.' }));
    } else {
      res.end();
    }
    console.error('[devChatServer] error:', err);
  }
});

server.listen(PORT, async () => {
  const model = process.env.CHAT_MODEL || DEFAULT_MODEL;
  const mock = process.env.CHAT_MOCK === '1';
  const hasKey = Boolean(process.env.AI_GATEWAY_API_KEY);

  let corpusChars = 0;
  let corpusDocs = 0;
  let corpusOk = false;
  try {
    // Same file the handler loads; read here only to report size at startup.
    const { createRequire } = await import('node:module');
    const require = createRequire(import.meta.url);
    const c = require('../api/_lib/corpus.json');
    corpusChars = c.chars ?? (c.corpus ? c.corpus.length : 0);
    corpusDocs = c.docCount ?? (Array.isArray(c.pages) ? c.pages.length : 0);
    corpusOk = typeof c.corpus === 'string' && c.corpus.length > 0;
  } catch {
    corpusOk = false;
  }

  console.log(`[devChatServer] listening on http://localhost:${PORT}/api/chat`);
  console.log(`[devChatServer] model:        ${model}`);
  console.log(`[devChatServer] mode:         ${mock ? 'MOCK (canned stream, no gateway)' : hasKey ? 'LIVE (gateway)' : 'NO KEY (serves 503)'}`);
  console.log(`[devChatServer] AI_GATEWAY_API_KEY: ${hasKey ? 'set' : 'UNSET'}`);
  console.log(`[devChatServer] rate limit:   ${rateLimitMode(process.env)}`);
  console.log(
    `[devChatServer] corpus:       ${corpusOk ? `ok (${corpusDocs} docs, ${corpusChars} chars)` : 'MISSING — run: node scripts/genChatCorpus.js (endpoint will 503)'}`
  );
  if (!mock && !hasKey) {
    console.log('[devChatServer] WARN: no key and not mock — POST /api/chat returns 503. Use CHAT_MOCK=1 for UI/QA.');
  }
});
