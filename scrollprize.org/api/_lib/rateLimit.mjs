// Rate limiting for the "Ask the Scrolls" chat endpoint.
//
// Two modes, chosen at runtime:
//   1. Upstash (durable, cross-instance) when UPSTASH_REDIS_REST_URL + _TOKEN are
//      set: a sliding-window burst limit plus a coarse daily cap, both keyed by IP.
//   2. In-memory fallback (best-effort, per warm instance) otherwise: just the
//      burst sliding window. Good enough to blunt abuse on a single instance.
//
// checkRateLimit() returns { ok: true } or { ok: false, retryAfter: <seconds> }.

import { Ratelimit } from '@upstash/ratelimit';
import { Redis } from '@upstash/redis';

// Defaults; MAX is overridable via env so the threshold can be tuned (and lowered
// for tests) without a code change. Window is fixed at 10 minutes.
const BURST_WINDOW_MS = 10 * 60 * 1000;
const BURST_WINDOW = '10 m'; // Upstash Duration form of BURST_WINDOW_MS
const DAILY_MAX = 150;

function burstMax(env) {
  const n = Number.parseInt(env.CHAT_RATELIMIT_MAX ?? '', 10);
  return Number.isFinite(n) && n > 0 ? n : 30;
}

// ---- IP extraction -------------------------------------------------------

// First hop of x-forwarded-for is the client; x-real-ip is the fallback.
export function getIp(request) {
  const xff = request.headers.get('x-forwarded-for');
  if (xff) {
    const first = xff.split(',')[0].trim();
    if (first) return first;
  }
  const real = request.headers.get('x-real-ip');
  return (real && real.trim()) || 'unknown';
}

// ---- Upstash limiters (lazy, memoized) -----------------------------------

let upstash; // undefined = not initialized, null = unavailable, else { burst, daily }
function getUpstash(env) {
  if (upstash !== undefined) return upstash;
  const url = env.UPSTASH_REDIS_REST_URL;
  const token = env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) {
    upstash = null;
    return upstash;
  }
  const redis = new Redis({ url, token });
  const ephemeralCache = new Map(); // absorbs bursts without a round-trip
  upstash = {
    burst: new Ratelimit({
      redis,
      limiter: Ratelimit.slidingWindow(burstMax(env), BURST_WINDOW),
      prefix: 'chat:rl:burst',
      ephemeralCache,
      analytics: false,
    }),
    daily: new Ratelimit({
      redis,
      limiter: Ratelimit.slidingWindow(DAILY_MAX, '1 d'),
      prefix: 'chat:rl:day',
      ephemeralCache,
      analytics: false,
    }),
  };
  return upstash;
}

function retryAfterFrom(resetMs) {
  return Math.max(1, Math.ceil((resetMs - Date.now()) / 1000));
}

// ---- In-memory sliding window (fallback) ---------------------------------

const hits = new Map(); // ip -> number[] of request timestamps (ms)

function memoryLimit(ip, env) {
  const max = burstMax(env);
  const now = Date.now();
  const cutoff = now - BURST_WINDOW_MS;
  const arr = (hits.get(ip) || []).filter((t) => t > cutoff);
  if (arr.length >= max) {
    const retryAfter = Math.max(1, Math.ceil((arr[0] + BURST_WINDOW_MS - now) / 1000));
    hits.set(ip, arr);
    return { ok: false, retryAfter };
  }
  arr.push(now);
  hits.set(ip, arr);
  // Opportunistic cleanup so the map doesn't grow unbounded on a warm instance.
  if (hits.size > 5000) {
    for (const [k, v] of hits) {
      const kept = v.filter((t) => t > cutoff);
      if (kept.length === 0) hits.delete(k);
      else hits.set(k, kept);
    }
  }
  return { ok: true };
}

// ---- Public API ----------------------------------------------------------

export async function checkRateLimit(ip, env) {
  const limiters = getUpstash(env);
  if (!limiters) return memoryLimit(ip, env);

  // Burst first (most common trip); only charge the daily cap if it passes.
  const burst = await limiters.burst.limit(ip);
  if (!burst.success) return { ok: false, retryAfter: retryAfterFrom(burst.reset) };
  const daily = await limiters.daily.limit(ip);
  if (!daily.success) return { ok: false, retryAfter: retryAfterFrom(daily.reset) };
  return { ok: true };
}

// Exposed for the dev shim / tests to report which mode is active.
export function rateLimitMode(env) {
  return env.UPSTASH_REDIS_REST_URL && env.UPSTASH_REDIS_REST_TOKEN ? 'upstash' : 'memory';
}
