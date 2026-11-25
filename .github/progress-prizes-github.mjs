#!/usr/bin/env node

import { appendFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

import {
  assertPublicResponderUri,
  parseProgressPrizeMarkdown,
  updateProgressPrizeMarkdown,
} from '../scrollprize.org/scripts/progress-prizes/markdown.mjs';

const OWNER = 'ScrollPrize';
const REPO = 'villa';
const API = 'https://api.github.com';
const PREVIEW_CONTEXT = 'progress-prizes/vercel-preview';
const PREVIEW_DESCRIPTION = 'Exact Progress Prize preview verified';
const PREVIEW_WORKFLOW_PATH = '.github/workflows/progress-prizes-vercel-preview.yml';
const TEST_CHECK = 'Public no-secret tests';
const REPOSITORY_ID = 890972577;
const OWNER_ID = 121906140;
const GITHUB_ACTIONS_BOT_ID = 41898282;
const PAGE_PATH = 'scrollprize.org/docs/34_prizes.md';
const SAFE_DIAGNOSTIC_CODES = new Set([
  'ambiguous-pr',
  'completed-state-missing',
  'invalid-main-ref',
  'invalid-pr-association',
  'unexpected-error',
]);
const READY_COMMENT = [
  'The deadline has passed, so this rollover page is ready to merge.',
  '',
  'Manual cutoff steps:',
  '',
  '1. In Google Forms, close the previous month\'s form (stop accepting responses).',
  '2. Open the new month\'s form (it is already published and closed).',
  '3. Merge this pull request.',
].join('\n');

class SafeCoordinationError extends Error {
  constructor(code, message) {
    super(message);
    this.diagnosticCode = code;
  }
}

function previewRunId(status) {
  const match = /^https:\/\/github\.com\/ScrollPrize\/villa\/actions\/runs\/([1-9]\d*)$/
    .exec(status?.target_url ?? '');
  return match?.[1];
}

function isTrustedPreviewStatus(status) {
  return status?.context === PREVIEW_CONTEXT
    && status.state === 'success'
    && status.creator?.login === 'github-actions[bot]'
    && status.creator?.id === GITHUB_ACTIONS_BOT_ID
    && status.creator?.type === 'Bot'
    && status.description === PREVIEW_DESCRIPTION
    && previewRunId(status) !== undefined;
}

function latestPreviewStatus(statuses) {
  return Array.isArray(statuses)
    ? statuses.find((status) => status.context === PREVIEW_CONTEXT)
    : undefined;
}

function fail(message) {
  throw new Error(message);
}

function failWithDiagnostic(code, message) {
  if (!SAFE_DIAGNOSTIC_CODES.has(code) || code === 'unexpected-error') {
    fail('Invalid safe diagnostic code.');
  }
  throw new SafeCoordinationError(code, message);
}

export function safeDiagnosticCode(error) {
  return SAFE_DIAGNOSTIC_CODES.has(error?.diagnosticCode)
    ? error.diagnosticCode
    : 'unexpected-error';
}

function parseArgs(argv) {
  const [command, ...tokens] = argv;
  const options = {};
  for (let index = 0; index < tokens.length; index += 2) {
    const name = tokens[index];
    const value = tokens[index + 1];
    if (!name?.startsWith('--') || value === undefined || value.startsWith('--')) {
      fail('Invalid trusted GitHub helper arguments.');
    }
    const key = name.slice(2);
    if (Object.hasOwn(options, key)) fail('Duplicate trusted GitHub helper option.');
    options[key] = value;
  }
  return { command, options };
}

function required(options, name) {
  const value = options[name];
  if (typeof value !== 'string' || value === '') fail(`Missing --${name}.`);
  return value;
}

function assertSha(value) {
  if (!/^[a-f0-9]{40}$/.test(value)) fail('Expected an exact lowercase commit SHA.');
  return value;
}

function assertCycle(value) {
  if (!/^\d{4}-(?:0[1-9]|1[0-2])$/.test(value)) fail('Cycle must use YYYY-MM.');
  return value;
}

export function assertAutomationBranch(head, base) {
  if (/^codex\/progress-prize-\d{4}-(?:0[1-9]|1[0-2])$/.test(head) && base === 'main') {
    return { kind: 'production', head, base };
  }
  fail('Head/base pair is outside the Progress Prize automation allowlist.');
}

function assertPullAssociation(pull, { head, base } = {}) {
  assertAutomationBranch(head, base);
  if (
    pull?.state !== 'open'
    || pull.head?.ref !== head
    || pull.base?.ref !== base
    || pull.head?.repo?.id !== REPOSITORY_ID
    || pull.base?.repo?.id !== REPOSITORY_ID
    || pull.head?.repo?.full_name !== `${OWNER}/${REPO}`
    || pull.base?.repo?.full_name !== `${OWNER}/${REPO}`
  ) {
    fail('Automation pull request association failed.');
  }
  return pull;
}

function pullShasMatch(pull, { headSha, baseSha }) {
  return pull?.head?.sha === headSha && pull?.base?.sha === baseSha;
}

export function assertPullBinding(pull, {
  head,
  base,
  headSha,
  baseSha,
} = {}) {
  assertPullAssociation(pull, { head, base });
  assertSha(headSha);
  assertSha(baseSha);
  if (!pullShasMatch(pull, { headSha, baseSha })) {
    fail('Automation pull request is not bound to the expected immutable refs.');
  }
  return pull;
}

export function assertDeterministicPageDelta({
  baseMarkdown,
  headMarkdown,
  sourceCycle,
  targetCycle,
  responderUri,
} = {}) {
  const source = assertCycle(sourceCycle);
  const target = assertCycle(targetCycle);
  const base = parseProgressPrizeMarkdown(baseMarkdown);
  const head = parseProgressPrizeMarkdown(headMarkdown);
  if (base.cycle !== source || head.cycle !== target) {
    fail('Managed page cycles do not match the immutable rollover contract.');
  }
  const expectedResponderUri = responderUri === undefined
    ? head.responderUri
    : assertPublicResponderUri(responderUri);
  if (head.responderUri !== expectedResponderUri) {
    fail('Managed page responder URL does not match the prepared public URL.');
  }
  const expected = updateProgressPrizeMarkdown(baseMarkdown, {
    targetCycle: target,
    responderUri: expectedResponderUri,
    expectedCurrentCycle: source,
    expectedCurrentResponderUri: base.responderUri,
  });
  if (!expected.changed || expected.content !== headMarkdown) {
    fail('Automation branch is not the exact deterministic marker-only page update.');
  }
  return Object.freeze({ source: base, target: head });
}

export function assertSinglePageCommit(commit, { headSha, baseSha } = {}) {
  assertSha(headSha);
  assertSha(baseSha);
  if (
    commit?.sha !== headSha
    || commit.parents?.length !== 1
    || commit.parents[0]?.sha !== baseSha
    || commit.files?.length !== 1
    || commit.files[0]?.filename !== PAGE_PATH
    || commit.files[0]?.status !== 'modified'
  ) {
    fail('Automation head must be one page-only commit directly above the immutable base.');
  }
  return commit;
}

/** A page-only commit's single parent, which may be an older main commit. */
export function pageCommitParent(commit, { headSha } = {}) {
  assertSha(headSha);
  if (
    commit?.sha !== headSha
    || commit.parents?.length !== 1
    || commit.files?.length !== 1
    || commit.files[0]?.filename !== PAGE_PATH
    || commit.files[0]?.status !== 'modified'
  ) {
    fail('Automation head must be one page-only commit with a single parent.');
  }
  return assertSha(commit.parents[0]?.sha ?? '');
}

export function isTrustedPreviewRun({ status, run, expectedSha } = {}) {
  const sha = assertSha(expectedSha);
  const runId = previewRunId(status);
  const expectedRunName = `Progress Prize Vercel preview ${sha}`;
  return isTrustedPreviewStatus(status)
    && String(run?.id ?? '') === runId
    && run?.html_url === status.target_url
    && run?.name === expectedRunName
    && run?.path === PREVIEW_WORKFLOW_PATH
    && run?.event === 'repository_dispatch'
    && run?.actor?.login === 'vercel[bot]'
    && run?.actor?.type === 'Bot'
    && run?.triggering_actor?.login === 'vercel[bot]'
    && run?.triggering_actor?.type === 'Bot'
    && run?.repository?.id === REPOSITORY_ID
    && run?.repository?.owner?.id === OWNER_ID
    && run?.repository?.full_name === `${OWNER}/${REPO}`
    && run?.head_repository?.id === REPOSITORY_ID
    && run?.head_branch === 'main'
    && run?.status === 'completed'
    && run?.conclusion === 'success'
    && run?.display_title === expectedRunName;
}

async function github(path, { method = 'GET', body, token = process.env.GITHUB_TOKEN } = {}) {
  if (!token) fail('GITHUB_TOKEN is required.');
  const response = await fetch(`${API}${path}`, {
    method,
    redirect: 'error',
    headers: {
      accept: 'application/vnd.github+json',
      authorization: `Bearer ${token}`,
      'content-type': 'application/json',
      'user-agent': 'progress-prize-github-orchestrator',
      'x-github-api-version': '2022-11-28',
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) fail('Trusted GitHub API operation failed.');
  return response.status === 204 ? undefined : response.json();
}

export async function waitForPullBinding({
  number,
  head,
  base,
  headSha,
  baseSha,
} = {}, {
  attempts = 12,
  delayMs = 5_000,
  readPull = (pullNumber) => github(`/repos/${OWNER}/${REPO}/pulls/${pullNumber}`),
  sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)),
} = {}) {
  if (!Number.isInteger(number) || number < 1) fail('Invalid pull request number.');
  assertAutomationBranch(head, base);
  assertSha(headSha);
  assertSha(baseSha);
  if (!Number.isInteger(attempts) || attempts < 1 || attempts > 12) {
    fail('Invalid pull request convergence attempts.');
  }
  if (!Number.isInteger(delayMs) || delayMs < 0 || delayMs > 5_000) {
    fail('Invalid pull request convergence delay.');
  }
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const pull = await readPull(number);
    assertPullAssociation(pull, { head, base });
    if (pullShasMatch(pull, { headSha, baseSha })) return pull;
    if (attempt < attempts) await sleep(delayMs);
  }
  fail('Automation pull request immutable refs did not converge.');
}

async function findPull(head, base) {
  assertAutomationBranch(head, base);
  const pulls = await github(
    `/repos/${OWNER}/${REPO}/pulls?state=open&head=${encodeURIComponent(`${OWNER}:${head}`)}&base=${encodeURIComponent(base)}&per_page=10`,
  );
  if (!Array.isArray(pulls) || pulls.length !== 1) fail('Expected exactly one open automation pull request.');
  const pull = pulls[0];
  return assertPullAssociation(pull, { head, base });
}

export async function resolveRolloverState({
  head,
  base,
  cycle,
} = {}, {
  listPulls = () => github(
    `/repos/${OWNER}/${REPO}/pulls?state=open&head=${encodeURIComponent(`${OWNER}:${head}`)}&base=${encodeURIComponent(base)}&per_page=10`,
  ),
  readMainRef = () => github(`/repos/${OWNER}/${REPO}/git/ref/heads/main`),
  readPage = (sha) => readPageAt(sha),
} = {}) {
  if (typeof head !== 'string' || head === '') fail('Missing rollover head.');
  if (typeof base !== 'string' || base === '') fail('Missing rollover base.');
  const targetCycle = assertCycle(cycle);
  assertAutomationBranch(head, base);
  const pulls = await listPulls();
  if (!Array.isArray(pulls) || pulls.length > 1) {
    failWithDiagnostic('ambiguous-pr', 'Rollover PR state is ambiguous.');
  }
  if (pulls.length === 1) {
    const pull = pulls[0];
    let sha;
    try {
      assertPullAssociation(pull, { head, base });
      sha = assertSha(pull.head?.sha ?? '');
      if (!Number.isInteger(pull.number) || pull.number < 1) fail('Invalid pull request number.');
    } catch {
      failWithDiagnostic(
        'invalid-pr-association',
        'The rollover pull request association is invalid.',
      );
    }
    return {
      state: 'pending',
      headSha: sha,
      pullNumber: String(pull.number),
      draft: pull.draft === true,
    };
  }
  const ref = await readMainRef();
  if (
    ref?.ref !== 'refs/heads/main'
    || ref?.object?.type !== 'commit'
    || !/^[a-f0-9]{40}$/.test(ref.object.sha ?? '')
  ) {
    failWithDiagnostic('invalid-main-ref', 'The authoritative production main ref is invalid.');
  }
  const page = await readPage(ref.object.sha);
  if (page.cycle !== targetCycle) {
    failWithDiagnostic(
      'completed-state-missing',
      'Neither a pending PR nor a completed production cycle exists.',
    );
  }
  return {
    state: 'completed',
    headSha: ref.object.sha,
    pullNumber: '',
    draft: false,
  };
}

async function rolloverState(options) {
  const result = await resolveRolloverState({
    head: required(options, 'head'),
    base: required(options, 'base'),
    cycle: required(options, 'cycle'),
  });
  if (process.env.GITHUB_OUTPUT) {
    await appendFile(
      process.env.GITHUB_OUTPUT,
      `state=${result.state}\nhead-sha=${result.headSha}\npr-number=${result.pullNumber}\ndraft=${result.draft}\n`,
    );
  }
  return result;
}

async function readPageMarkdownAt(sha) {
  assertSha(sha);
  const response = await fetch(
    `${API}/repos/${OWNER}/${REPO}/contents/${PAGE_PATH}?ref=${sha}`,
    {
      redirect: 'error',
      headers: {
        accept: 'application/vnd.github.raw+json',
        authorization: `Bearer ${process.env.GITHUB_TOKEN}`,
        'user-agent': 'progress-prize-github-orchestrator',
        'x-github-api-version': '2022-11-28',
      },
    },
  );
  if (response.status !== 200) fail('Could not read the managed page at the gated commit.');
  const markdown = await response.text();
  if (Buffer.byteLength(markdown, 'utf8') > 1024 * 1024) fail('Managed page response is too large.');
  return markdown;
}

async function readPageAt(sha) {
  return parseProgressPrizeMarkdown(await readPageMarkdownAt(sha));
}

function assertRef(ref, { branch, sha }) {
  if (ref?.ref !== `refs/heads/${branch}` || ref?.object?.type !== 'commit' || ref.object.sha !== sha) {
    fail('Automation branch moved away from its immutable expected commit.');
  }
  return ref;
}

async function verifyPageDelta({
  head,
  base,
  headSha,
  baseSha,
  sourceCycle,
  targetCycle,
  responderUri,
}) {
  assertAutomationBranch(head, base);
  assertSha(headSha);
  assertSha(baseSha);
  const [headRef, baseRef, commit, baseMarkdown, headMarkdown] = await Promise.all([
    github(`/repos/${OWNER}/${REPO}/git/ref/heads/${encodeURIComponent(head)}`),
    github(`/repos/${OWNER}/${REPO}/git/ref/heads/${encodeURIComponent(base)}`),
    github(`/repos/${OWNER}/${REPO}/commits/${headSha}`),
    readPageMarkdownAt(baseSha),
    readPageMarkdownAt(headSha),
  ]);
  assertRef(headRef, { branch: head, sha: headSha });
  assertRef(baseRef, { branch: base, sha: baseSha });
  assertSinglePageCommit(commit, { headSha, baseSha });
  return assertDeterministicPageDelta({
    baseMarkdown,
    headMarkdown,
    sourceCycle,
    targetCycle,
    responderUri,
  });
}

async function readPreviewRun(status, expectedSha) {
  if (!isTrustedPreviewStatus(status)) return undefined;
  const run = await github(`/repos/${OWNER}/${REPO}/actions/runs/${previewRunId(status)}`);
  return isTrustedPreviewRun({ status, run, expectedSha }) ? run : undefined;
}

/**
 * Only the two checks the rollover depends on gate the draft-to-ready flip: the
 * public no-secret tests and the exact Vercel preview. Unrelated repository
 * workflows may run on the same commit and be cancelled or fail without
 * blocking the cutoff.
 */
export function gateSnapshot({ statuses, checks, previewRun, expectedSha }) {
  const preview = latestPreviewStatus(statuses);
  const runs = checks?.check_runs;
  const publicTest = Array.isArray(runs)
    && checks.total_count <= 100
    && runs.find((run) => run.name === TEST_CHECK);
  return {
    ready: isTrustedPreviewRun({ status: preview, run: previewRun, expectedSha })
      && publicTest?.status === 'completed'
      && publicTest?.conclusion === 'success'
      && publicTest?.app?.slug === 'github-actions',
    previewState: preview?.state,
    publicTestConclusion: publicTest?.conclusion,
  };
}

async function waitForChecks({ sha, timeoutSeconds }) {
  const deadline = Date.now() + timeoutSeconds * 1000;
  while (true) {
    const [statuses, checks] = await Promise.all([
      github(`/repos/${OWNER}/${REPO}/commits/${sha}/statuses?per_page=100`),
      github(`/repos/${OWNER}/${REPO}/commits/${sha}/check-runs?per_page=100`),
    ]);
    const previewRun = await readPreviewRun(latestPreviewStatus(statuses), sha);
    if (gateSnapshot({ statuses, checks, previewRun, expectedSha: sha }).ready) return;
    if (Date.now() >= deadline) fail('Timed out waiting for the public tests and Vercel preview.');
    await new Promise((resolve) => setTimeout(resolve, 15_000));
  }
}

async function ensurePull(options) {
  const head = required(options, 'head');
  const base = required(options, 'base');
  assertAutomationBranch(head, base);
  const title = required(options, 'title');
  const headSha = assertSha(required(options, 'head-sha'));
  const baseSha = assertSha(required(options, 'base-sha'));
  const sourceCycle = assertCycle(required(options, 'source-cycle'));
  const targetCycle = assertCycle(required(options, 'target-cycle'));
  const responderUri = assertPublicResponderUri(required(options, 'responder-uri'));
  await verifyPageDelta({
    head,
    base,
    headSha,
    baseSha,
    sourceCycle,
    targetCycle,
    responderUri,
  });
  const pulls = await github(
    `/repos/${OWNER}/${REPO}/pulls?state=open&head=${encodeURIComponent(`${OWNER}:${head}`)}&per_page=10`,
  );
  if (!Array.isArray(pulls)) fail('Automation pull request listing failed.');
  let pull;
  if (pulls.length === 0) {
    pull = await github(`/repos/${OWNER}/${REPO}/pulls`, {
      method: 'POST',
      body: {
        title,
        head,
        base,
        body: 'Automated Progress Prize rollover. Private Google identifiers are intentionally omitted.',
        draft: true,
      },
    });
  } else if (pulls.length === 1) {
    pull = assertPullAssociation(pulls[0], { head, base });
  } else {
    fail('Multiple open pull requests use the fixed automation head.');
  }
  if (!Number.isInteger(pull?.number) || pull.number < 1) fail('Pull request creation failed.');
  pull = await waitForPullBinding({
    number: pull.number,
    head,
    base,
    headSha,
    baseSha,
  });
  await verifyPageDelta({
    head,
    base,
    headSha,
    baseSha,
    sourceCycle,
    targetCycle,
    responderUri,
  });
  if (process.env.GITHUB_OUTPUT) {
    await appendFile(
      process.env.GITHUB_OUTPUT,
      `pr-number=${pull.number}\nhead-sha=${headSha}\nbase-sha=${baseSha}\n`,
    );
  }
  return { pullNumber: pull.number, headSha, baseSha };
}

/**
 * Flip the rollover PR from draft to ready once its page-only commit passed
 * the public tests and the exact Vercel preview. Merging stays manual.
 */
async function ready(options) {
  const head = required(options, 'head');
  const base = required(options, 'base');
  const sourceCycle = assertCycle(required(options, 'source-cycle'));
  const targetCycle = assertCycle(required(options, 'cycle'));
  const timeoutSeconds = Number(options['timeout-seconds'] ?? '1800');
  if (!Number.isInteger(timeoutSeconds) || timeoutSeconds < 0 || timeoutSeconds > 3600) {
    fail('Ready timeout must be between zero and 3600 seconds.');
  }
  const pull = await findPull(head, base);
  const sha = assertSha(pull.head?.sha ?? '');
  const commit = await github(`/repos/${OWNER}/${REPO}/commits/${sha}`);
  const parentSha = pageCommitParent(commit, { headSha: sha });
  const [baseMarkdown, headMarkdown] = await Promise.all([
    readPageMarkdownAt(parentSha),
    readPageMarkdownAt(sha),
  ]);
  assertDeterministicPageDelta({ baseMarkdown, headMarkdown, sourceCycle, targetCycle });

  let flipped = false;
  if (pull.draft === true) {
    await waitForChecks({ sha, timeoutSeconds });
    const current = await findPull(head, base);
    if (current.number !== pull.number || current.head?.sha !== sha) {
      fail('Pull request changed while waiting for its checks.');
    }
    const result = await github('/graphql', {
      method: 'POST',
      body: {
        query: 'mutation($id:ID!){markPullRequestReadyForReview(input:{pullRequestId:$id}){pullRequest{isDraft}}}',
        variables: { id: pull.node_id },
      },
    });
    if (result?.errors?.length || result?.data?.markPullRequestReadyForReview?.pullRequest?.isDraft !== false) {
      fail('Could not mark the rollover pull request ready for review.');
    }
    await github(`/repos/${OWNER}/${REPO}/issues/${pull.number}/comments`, {
      method: 'POST',
      body: { body: READY_COMMENT },
    });
    flipped = true;
  }
  if (process.env.GITHUB_OUTPUT) {
    await appendFile(
      process.env.GITHUB_OUTPUT,
      `pr-number=${pull.number}\nhead-sha=${sha}\nflipped=${flipped}\n`,
    );
  }
  return { pullNumber: pull.number, headSha: sha, flipped };
}

async function main(argv) {
  const { command, options } = parseArgs(argv);
  if (command === 'rollover-state') return rolloverState(options);
  if (command === 'ensure-pr') return ensurePull(options);
  if (command === 'ready') return ready(options);
  fail('Unknown trusted GitHub helper command.');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await main(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(
      `Progress Prize GitHub coordination failed safely [${safeDiagnosticCode(error)}].\n`,
    );
    process.exitCode = 1;
  }
}
