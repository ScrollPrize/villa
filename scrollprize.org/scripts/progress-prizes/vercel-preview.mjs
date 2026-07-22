import { getCycleDeadline } from './dates.mjs';
import {
  assertPublicResponderUri,
  parseProgressPrizeMarkdown,
} from './markdown.mjs';

const GITHUB_API_ORIGIN = 'https://api.github.com';
const DEFAULT_PAGE_PATH = 'scrollprize.org/docs/34_prizes.md';
const DEFAULT_PREVIEW_PATH = '/prizes';
const DEFAULT_MAX_BODY_BYTES = 2 * 1024 * 1024;
const ALLOWED_DISPATCH_ACTIONS = new Set([
  'vercel.deployment.ready',
  'vercel.deployment.success',
]);

const SAFE_MESSAGES = Object.freeze({
  EVENT: 'The Vercel dispatch payload is invalid.',
  ASSOCIATION: 'The Vercel deployment is not associated with the expected commit and ref.',
  HOST: 'The Vercel preview URL is not an allowed HTTPS preview host.',
  GITHUB: 'The trusted prize-page state could not be read from GitHub.',
  EXPECTED: 'The expected Progress Prize page state is invalid.',
  REQUEST: 'The Vercel preview request failed.',
  REDIRECT: 'The Vercel preview returned a redirect.',
  STATUS: 'The Vercel preview did not return HTTP 200.',
  CONTENT_TYPE: 'The Vercel preview did not return HTML.',
  BODY: 'The Vercel preview response is too large or unreadable.',
  DEADLINE: 'The Vercel preview does not contain the expected Progress Prize deadline.',
  RESPONDER: 'The Vercel preview does not link to the exact expected responder URL.',
});

export class PreviewVerificationError extends Error {
  constructor(code) {
    super(SAFE_MESSAGES[code] ?? 'The Vercel preview verification failed.');
    this.name = 'PreviewVerificationError';
    this.code = code;
  }
}

function fail(code) {
  throw new PreviewVerificationError(code);
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function valuesAtPaths(root, paths) {
  const values = [];
  for (const path of paths) {
    let current = root;
    for (const segment of path) {
      if (!isRecord(current) || !(segment in current)) {
        current = undefined;
        break;
      }
      current = current[segment];
    }
    if (current !== undefined && current !== null && current !== '') {
      values.push(current);
    }
  }
  return values;
}

function oneConsistentString(root, paths) {
  const values = valuesAtPaths(root, paths);
  if (values.length === 0 || values.some((value) => typeof value !== 'string')) {
    fail('EVENT');
  }
  const unique = [...new Set(values)];
  if (unique.length !== 1) fail('EVENT');
  return unique[0];
}

function assertOpaqueIdentifier(value) {
  if (
    typeof value !== 'string'
    || value.length < 1
    || value.length > 1024
    || /[\u0000-\u001f\u007f]/.test(value)
  ) {
    fail('EVENT');
  }
  return value;
}

function assertGitSha(value) {
  if (typeof value !== 'string' || !/^[a-fA-F0-9]{40}$/.test(value)) fail('EVENT');
  return value.toLowerCase();
}

function assertGitRef(value) {
  if (
    typeof value !== 'string'
    || value.length < 1
    || value.length > 1024
    || /[\u0000-\u0020\u007f~^:?*[\\]/.test(value)
    || value.includes('..')
    || value.startsWith('/')
    || value.endsWith('/')
  ) {
    fail('EVENT');
  }
  return value;
}

function isVercelPreviewHostname(hostname) {
  if (!hostname.endsWith('.vercel.app') || hostname === 'vercel.app') return false;
  const prefix = hostname.slice(0, -'.vercel.app'.length);
  if (!prefix) return false;
  return prefix.split('.').every((label) => (
    label.length >= 1
    && label.length <= 63
    && /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/.test(label)
  ));
}

export function assertVercelPreviewUrl(value) {
  if (typeof value !== 'string') fail('HOST');

  let url;
  try {
    url = new URL(value);
  } catch {
    fail('HOST');
  }

  if (
    url.protocol !== 'https:'
    || url.username
    || url.password
    || url.port
    || url.pathname !== '/'
    || url.search
    || url.hash
    || (value !== url.origin && value !== `${url.origin}/`)
    || !isVercelPreviewHostname(url.hostname)
  ) {
    fail('HOST');
  }
  return url.origin;
}

/**
 * Validate a repository_dispatch event without trusting any deployment metadata.
 * Repeated aliases are accepted only when every supplied value is identical.
 */
export function validateVercelDispatch({
  event,
  expectedProjectId,
  expectedSha,
  expectedRef,
} = {}) {
  if (!isRecord(event) || !isRecord(event.client_payload)) fail('EVENT');
  const action = oneConsistentString(event, [
    ['action'],
    ['event_type'],
    ['eventType'],
  ]);
  if (!ALLOWED_DISPATCH_ACTIONS.has(action)) fail('EVENT');

  const payload = event.client_payload;
  const environment = oneConsistentString(payload, [
    ['environment'],
    ['target'],
    ['deployment', 'environment'],
    ['deployment', 'target'],
  ]);
  if (environment !== 'preview') fail('EVENT');

  const projectId = assertOpaqueIdentifier(oneConsistentString(payload, [
    ['projectId'],
    ['project_id'],
    ['project', 'id'],
    ['deployment', 'projectId'],
    ['deployment', 'project', 'id'],
  ]));
  if (projectId !== assertOpaqueIdentifier(expectedProjectId)) fail('EVENT');

  const deploymentOrigin = assertVercelPreviewUrl(oneConsistentString(payload, [
    ['url'],
    ['deploymentUrl'],
    ['deployment', 'url'],
  ]));

  const sha = assertGitSha(oneConsistentString(payload, [
    ['sha'],
    ['commitSha'],
    ['git', 'sha'],
    ['gitSource', 'sha'],
    ['meta', 'githubCommitSha'],
    ['deployment', 'sha'],
    ['deployment', 'gitSource', 'sha'],
    ['deployment', 'meta', 'githubCommitSha'],
  ]));
  const ref = assertGitRef(oneConsistentString(payload, [
    ['ref'],
    ['gitRef'],
    ['git', 'ref'],
    ['gitSource', 'ref'],
    ['meta', 'githubCommitRef'],
    ['deployment', 'ref'],
    ['deployment', 'gitSource', 'ref'],
    ['deployment', 'meta', 'githubCommitRef'],
  ]));

  if (sha !== assertGitSha(expectedSha) || ref !== assertGitRef(expectedRef)) {
    fail('ASSOCIATION');
  }

  return Object.freeze({
    action,
    environment,
    deploymentOrigin,
    sha,
    ref,
  });
}

function assertRepoSlug(value) {
  if (typeof value !== 'string' || !/^[A-Za-z0-9_.-]+$/.test(value)) fail('GITHUB');
  return value;
}

function assertRepositoryPath(value) {
  if (
    typeof value !== 'string'
    || !value
    || value.startsWith('/')
    || value.endsWith('/')
    || value.includes('\\')
    || value.split('/').some((segment) => !segment || segment === '.' || segment === '..')
  ) {
    fail('GITHUB');
  }
  return value;
}

async function readBodyLimited(response, maximumBytes, errorCode) {
  const declaredLength = response.headers.get('content-length');
  if (declaredLength !== null) {
    const parsedLength = Number(declaredLength);
    if (!Number.isSafeInteger(parsedLength) || parsedLength < 0 || parsedLength > maximumBytes) {
      fail(errorCode);
    }
  }

  if (!response.body || typeof response.body.getReader !== 'function') {
    try {
      const text = await response.text();
      if (Buffer.byteLength(text, 'utf8') > maximumBytes) fail(errorCode);
      return text;
    } catch (error) {
      if (error instanceof PreviewVerificationError) throw error;
      fail(errorCode);
    }
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let bytes = 0;
  let text = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > maximumBytes) {
        await reader.cancel();
        fail(errorCode);
      }
      text += decoder.decode(value, { stream: true });
    }
    text += decoder.decode();
    return text;
  } catch (error) {
    if (error instanceof PreviewVerificationError) throw error;
    fail(errorCode);
  }
}

/** Read the managed page state at an immutable Git commit through GitHub's API. */
export async function fetchExpectedProgressPrizeState({
  owner,
  repo,
  commitSha,
  githubToken,
  path = DEFAULT_PAGE_PATH,
  fetchImpl = globalThis.fetch,
  maximumBytes = DEFAULT_MAX_BODY_BYTES,
} = {}) {
  const safeOwner = assertRepoSlug(owner);
  const safeRepo = assertRepoSlug(repo);
  const safeSha = assertGitSha(commitSha);
  const safePath = assertRepositoryPath(path);
  if (typeof githubToken !== 'string' || !githubToken || typeof fetchImpl !== 'function') fail('GITHUB');

  const encodedPath = safePath.split('/').map(encodeURIComponent).join('/');
  const url = new URL(
    `/repos/${encodeURIComponent(safeOwner)}/${encodeURIComponent(safeRepo)}/contents/${encodedPath}`,
    GITHUB_API_ORIGIN,
  );
  url.searchParams.set('ref', safeSha);

  let response;
  try {
    response = await fetchImpl(url, {
      method: 'GET',
      redirect: 'error',
      headers: {
        accept: 'application/vnd.github.raw+json',
        authorization: `Bearer ${githubToken}`,
        'user-agent': 'scrollprize-progress-prize-preview-verifier',
        'x-github-api-version': '2022-11-28',
      },
    });
  } catch {
    fail('GITHUB');
  }
  if (!response || response.status !== 200) fail('GITHUB');

  let markdown;
  try {
    markdown = await readBodyLimited(response, maximumBytes, 'GITHUB');
  } catch {
    fail('GITHUB');
  }

  let parsed;
  try {
    parsed = parseProgressPrizeMarkdown(markdown);
  } catch {
    fail('GITHUB');
  }
  return Object.freeze({
    cycle: parsed.cycle,
    deadlineLabel: parsed.deadline.label,
    responderUri: parsed.responderUri,
  });
}

function validateExpectedState(expectedState, expectedCycle) {
  if (!isRecord(expectedState) || expectedState.cycle !== expectedCycle) fail('EXPECTED');
  let deadline;
  try {
    deadline = getCycleDeadline(expectedCycle);
  } catch {
    fail('EXPECTED');
  }
  if (expectedState.deadlineLabel !== deadline.label) fail('EXPECTED');

  let responderUri;
  try {
    responderUri = assertPublicResponderUri(expectedState.responderUri);
  } catch {
    fail('EXPECTED');
  }
  return Object.freeze({
    cycle: expectedCycle,
    deadlineLabel: deadline.label,
    responderUri,
  });
}

function decodeHtmlEntities(value) {
  return value.replace(/&(?:amp|quot|apos|#39|#x27|lt|gt);/gi, (entity) => {
    switch (entity.toLowerCase()) {
      case '&amp;': return '&';
      case '&quot;': return '"';
      case '&apos;':
      case '&#39;':
      case '&#x27;': return "'";
      case '&lt;': return '<';
      case '&gt;': return '>';
      default: return entity;
    }
  });
}

function stripNonRenderedBlocks(html) {
  return html
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<(script|style|template)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, ' ');
}

function extractAnchorHrefs(html) {
  const hrefs = [];
  const anchorPattern = /<a\b[^>]*\bhref\s*=\s*(?:"([^"]*)"|'([^']*)')[^>]*>/gi;
  for (const match of stripNonRenderedBlocks(html).matchAll(anchorPattern)) {
    hrefs.push(decodeHtmlEntities(match[1] ?? match[2]));
  }
  return hrefs;
}

function htmlText(html) {
  return decodeHtmlEntities(
    stripNonRenderedBlocks(html)
      .replace(/<[^>]*>/g, ' '),
  ).replace(/\s+/g, ' ').trim();
}

function buildPreviewPageUrl(origin, path) {
  if (
    typeof path !== 'string'
    || !path.startsWith('/')
    || path.startsWith('//')
    || path.includes('\\')
    || /[\u0000-\u001f\u007f]/.test(path)
  ) {
    fail('HOST');
  }
  const url = new URL(path, `${origin}/`);
  if (url.origin !== origin || url.protocol !== 'https:') fail('HOST');
  return url;
}

/**
 * Verify a Vercel preview against the exact page state stored at the deployed SHA.
 * No logging is performed; every thrown error has a fixed, non-sensitive message.
 */
export async function verifyVercelPreview({
  event,
  expectedProjectId,
  expectedSha,
  expectedRef,
  expectedCycle,
  expectedState,
  github,
  protectionBypassSecret,
  previewPath = DEFAULT_PREVIEW_PATH,
  fetchImpl = globalThis.fetch,
  maximumBytes = DEFAULT_MAX_BODY_BYTES,
} = {}) {
  if (typeof fetchImpl !== 'function') fail('REQUEST');
  const dispatch = validateVercelDispatch({
    event,
    expectedProjectId,
    expectedSha,
    expectedRef,
  });

  let sourceState = expectedState;
  let effectiveCycle = expectedCycle;
  if (sourceState === undefined) {
    if (!isRecord(github)) fail('EXPECTED');
    sourceState = await fetchExpectedProgressPrizeState({
      ...github,
      commitSha: expectedSha,
      fetchImpl,
      maximumBytes,
    });
    if (effectiveCycle === undefined) effectiveCycle = sourceState.cycle;
  } else if (effectiveCycle === undefined) {
    // Injected values are a test seam, not a trusted source for cycle inference.
    fail('EXPECTED');
  }
  const expected = validateExpectedState(sourceState, effectiveCycle);
  const previewUrl = buildPreviewPageUrl(dispatch.deploymentOrigin, previewPath);

  const headers = {
    accept: 'text/html',
    'user-agent': 'scrollprize-progress-prize-preview-verifier',
  };
  if (protectionBypassSecret !== undefined) {
    if (typeof protectionBypassSecret !== 'string' || !protectionBypassSecret) fail('REQUEST');
    headers['x-vercel-protection-bypass'] = protectionBypassSecret;
  }

  let response;
  try {
    response = await fetchImpl(previewUrl, {
      method: 'GET',
      redirect: 'manual',
      headers,
    });
  } catch {
    fail('REQUEST');
  }

  if (!response) fail('REQUEST');
  if (response.status >= 300 && response.status < 400) fail('REDIRECT');
  if (response.status !== 200) fail('STATUS');
  const contentType = response.headers.get('content-type') ?? '';
  if (!/^text\/html(?:;|$)/i.test(contentType.trim())) fail('CONTENT_TYPE');

  const html = await readBodyLimited(response, maximumBytes, 'BODY');
  if (!htmlText(html).includes(expected.deadlineLabel)) fail('DEADLINE');
  if (!extractAnchorHrefs(html).includes(expected.responderUri)) fail('RESPONDER');

  return Object.freeze({
    ok: true,
    cycle: expected.cycle,
    deadlineLabel: expected.deadlineLabel,
    responderUri: expected.responderUri,
    sha: dispatch.sha,
    ref: dispatch.ref,
  });
}
