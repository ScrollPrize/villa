// Vercel Function: POST /api/chat ("Ask the Scrolls").
//
// Modern Web-standard handler: (Request) => Response. .mjs so it is ESM (the root
// package.json is CommonJS, and ai@7 / @ai-sdk/gateway@4 are ESM-only). Zero-config
// — any file under /api is deployed as a function; no vercel.json routing needed.
// maxDuration is raised in vercel.json so the ~60s per-request abort can run.

import { handleChat } from './_lib/handler.mjs';

export default function handler(request) {
  return handleChat(request);
}
