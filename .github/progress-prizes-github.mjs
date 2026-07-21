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
const SMOKE_HEAD = 'codex/progress-prize-smoke-20260720';
const SMOKE_BASE = 'codex/progress-prize-smoke-base-20260720';
const PREVIEW_CONTEXT = 'progress-prizes/vercel-preview';
const PREVIEW_DESCRIPTION = 'Exact Progress Prize preview verified';
const PREVIEW_WORKFLOW_PATH = '.github/workflows/progress-prizes-vercel-preview.yml';
const TEST_CHECK = 'Public no-secret tests';
const SUCCESSFUL_CONCLUSIONS = new Set(['success', 'neutral', 'skipped']);
const REPOSITORY_ID = 890972577;
const OWNER_ID = 121906140;
const GITHUB_ACTIONS_BOT_ID = 41898282;
const PAGE_PATH = 'scrollprize.org/docs/34_prizes.md';

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
  if (head === SMOKE_HEAD && base === SMOKE_BASE) return { kind: 'smoke', head, base };
  if (/^codex\/progress-prize-\d{4}-(?:0[1-9]|1[0-2])$/.test(head) && base === 'main') {
    return { kind: 'production', head, base };
  }
  fail('Head/base pair is outside the Progress Prize automation allowlist.');
}

export function assertPullBinding(pull, {
  head,
  base,
  headSha,
  baseSha,
} = {}) {
  assertAutomationBranch(head, base);
  assertSha(headSha);
  assertSha(baseSha);
  if (
    pull?.state !== 'open'
    || pull.head?.ref !== head
    || pull.base?.ref !== base
    || pull.head?.sha !== headSha
    || pull.base?.sha !== baseSha
    || pull.head?.repo?.id !== REPOSITORY_ID
    || pull.base?.repo?.id !== REPOSITORY_ID
    || pull.head?.repo?.full_name !== `${OWNER}/${REPO}`
    || pull.base?.repo?.full_name !== `${OWNER}/${REPO}`
  ) {
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

async function findPull(head, base) {
  assertAutomationBranch(head, base);
  const pulls = await github(
    `/repos/${OWNER}/${REPO}/pulls?state=open&head=${encodeURIComponent(`${OWNER}:${head}`)}&base=${encodeURIComponent(base)}&per_page=10`,
  );
  if (!Array.isArray(pulls) || pulls.length !== 1) fail('Expected exactly one open automation pull request.');
  const pull = pulls[0];
  if (
    pull.head?.ref !== head
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

async function activationState(options) {
  const head = required(options, 'head');
  const base = required(options, 'base');
  const cycle = assertCycle(required(options, 'cycle'));
  const contract = assertAutomationBranch(head, base);
  if (contract.kind !== 'production') fail('Activation-state recovery is production-only.');
  const pulls = await github(
    `/repos/${OWNER}/${REPO}/pulls?state=open&head=${encodeURIComponent(`${OWNER}:${head}`)}&base=${encodeURIComponent(base)}&per_page=10`,
  );
  if (!Array.isArray(pulls) || pulls.length > 1) fail('Production activation PR state is ambiguous.');
  let state;
  let sha;
  let baseSha;
  let number = '';
  if (pulls.length === 1) {
    const pull = pulls[0];
    sha = assertSha(pull.head?.sha ?? '');
    baseSha = assertSha(pull.base?.sha ?? '');
    assertPullBinding(pull, { head, base, headSha: sha, baseSha });
    state = 'pending';
    number = String(pull.number);
  } else {
    const ref = await github(`/repos/${OWNER}/${REPO}/git/ref/heads/main`);
    sha = assertSha(ref?.object?.sha ?? '');
    baseSha = sha;
    const page = await readPageAt(sha);
    if (page.cycle !== cycle) fail('Neither a pending PR nor a completed production cycle exists.');
    state = 'completed';
  }
  if (process.env.GITHUB_OUTPUT) {
    await appendFile(
      process.env.GITHUB_OUTPUT,
      `state=${state}\nhead-sha=${sha}\nbase-sha=${baseSha}\npr-number=${number}\n`,
    );
  }
  return { state, headSha: sha, baseSha, pullNumber: number };
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

export function gateSnapshot({ statuses, checks, previewRun, expectedSha }) {
  const preview = latestPreviewStatus(statuses);
  const runs = checks?.check_runs;
  const publicTest = Array.isArray(runs) && runs.find((run) => run.name === TEST_CHECK);
  const allFinishedSuccessfully = Array.isArray(runs)
    && checks.total_count <= 100
    && runs.length > 0
    && runs.every((run) => run.status === 'completed' && SUCCESSFUL_CONCLUSIONS.has(run.conclusion));
  return {
    ready: isTrustedPreviewRun({ status: preview, run: previewRun, expectedSha })
      && publicTest?.status === 'completed'
      && publicTest?.conclusion === 'success'
      && publicTest?.app?.slug === 'github-actions'
      && allFinishedSuccessfully,
    previewState: preview?.state,
    publicTestConclusion: publicTest?.conclusion,
  };
}

async function gate(options) {
  const head = required(options, 'head');
  const base = required(options, 'base');
  const sourceCycle = assertCycle(required(options, 'source-cycle'));
  const targetCycle = assertCycle(required(options, 'cycle'));
  const expectedSha = assertSha(required(options, 'sha'));
  const expectedBaseSha = assertSha(required(options, 'base-sha'));
  assertAutomationBranch(head, base);
  const timeoutSeconds = Number(options['timeout-seconds'] ?? '1800');
  if (!Number.isInteger(timeoutSeconds) || timeoutSeconds < 0 || timeoutSeconds > 3600) {
    fail('Gate timeout must be between zero and 3600 seconds.');
  }
  const deadline = Date.now() + timeoutSeconds * 1000;

  while (true) {
    const pull = await findPull(head, base);
    const sha = assertSha(pull.head.sha);
    assertPullBinding(pull, {
      head,
      base,
      headSha: expectedSha,
      baseSha: expectedBaseSha,
    });
    if (sha !== expectedSha) fail('Pull request head changed after preparation.');
    const [statuses, checks] = await Promise.all([
      github(`/repos/${OWNER}/${REPO}/commits/${sha}/statuses?per_page=100`),
      github(`/repos/${OWNER}/${REPO}/commits/${sha}/check-runs?per_page=100`),
      verifyPageDelta({
        head,
        base,
        headSha: expectedSha,
        baseSha: expectedBaseSha,
        sourceCycle,
        targetCycle,
      }),
    ]);
    const preview = latestPreviewStatus(statuses);
    const previewRun = await readPreviewRun(preview, expectedSha);
    if (gateSnapshot({ statuses, checks, previewRun, expectedSha }).ready) {
      if (process.env.GITHUB_OUTPUT) {
        await appendFile(
          process.env.GITHUB_OUTPUT,
          `pr-number=${pull.number}\nhead-sha=${sha}\nbase-sha=${expectedBaseSha}\n`,
        );
      }
      return { pullNumber: pull.number, headSha: sha, baseSha: expectedBaseSha };
    }
    if (Date.now() >= deadline) fail('Timed out waiting for exact-commit code and Vercel checks.');
    await new Promise((resolve) => setTimeout(resolve, 15_000));
  }
}

async function waitPreview(options) {
  const branch = required(options, 'branch');
  if (branch !== SMOKE_BASE) fail('Post-merge preview is restricted to the fixed smoke base.');
  const sha = assertSha(required(options, 'sha'));
  const cycle = assertCycle(required(options, 'cycle'));
  const timeoutSeconds = Number(options['timeout-seconds'] ?? '1800');
  if (!Number.isInteger(timeoutSeconds) || timeoutSeconds < 0 || timeoutSeconds > 3600) {
    fail('Preview timeout must be between zero and 3600 seconds.');
  }
  const deadline = Date.now() + timeoutSeconds * 1000;
  while (true) {
    const [ref, statuses, page] = await Promise.all([
      github(`/repos/${OWNER}/${REPO}/git/ref/heads/${encodeURIComponent(branch)}`),
      github(`/repos/${OWNER}/${REPO}/commits/${sha}/statuses?per_page=100`),
      readPageAt(sha),
    ]);
    if (ref?.object?.type !== 'commit' || ref.object.sha !== sha) {
      fail('Fixed smoke base changed while waiting for its preview.');
    }
    if (page.cycle !== cycle) fail('Post-merge smoke base has the wrong page cycle.');
    const preview = latestPreviewStatus(statuses);
    const previewRun = await readPreviewRun(preview, sha);
    if (previewRun) return { headSha: sha };
    if (Date.now() >= deadline) fail('Timed out waiting for the post-merge smoke preview.');
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
    `/repos/${OWNER}/${REPO}/pulls?state=open&head=${encodeURIComponent(`${OWNER}:${head}`)}&base=${encodeURIComponent(base)}&per_page=10`,
  );
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
    pull = pulls[0];
  } else {
    fail('Multiple automation pull requests match the fixed head/base pair.');
  }
  if (!Number.isInteger(pull?.number) || pull.number < 1) fail('Pull request creation failed.');
  pull = await github(`/repos/${OWNER}/${REPO}/pulls/${pull.number}`);
  assertPullBinding(pull, { head, base, headSha, baseSha });
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

async function merge(options) {
  const head = required(options, 'head');
  const base = required(options, 'base');
  const sha = assertSha(required(options, 'sha'));
  const baseSha = assertSha(required(options, 'base-sha'));
  const sourceCycle = assertCycle(required(options, 'source-cycle'));
  const targetCycle = assertCycle(required(options, 'cycle'));
  const number = Number(required(options, 'pr-number'));
  if (!Number.isInteger(number) || number < 1) fail('Invalid pull request number.');
  const pull = await findPull(head, base);
  if (pull.number !== number) {
    fail('Pull request changed after the activation gate.');
  }
  assertPullBinding(pull, { head, base, headSha: sha, baseSha });
  await verifyPageDelta({
    head,
    base,
    headSha: sha,
    baseSha,
    sourceCycle,
    targetCycle,
  });
  if (pull.draft === true) {
    const ready = await github('/graphql', {
      method: 'POST',
      body: {
        query: 'mutation($id:ID!){markPullRequestReadyForReview(input:{pullRequestId:$id}){pullRequest{isDraft}}}',
        variables: { id: pull.node_id },
      },
    });
    if (ready?.errors?.length || ready?.data?.markPullRequestReadyForReview?.pullRequest?.isDraft !== false) {
      fail('Could not mark the gated pull request ready for review.');
    }
  }
  const readyPull = await github(`/repos/${OWNER}/${REPO}/pulls/${number}`);
  assertPullBinding(readyPull, { head, base, headSha: sha, baseSha });
  await verifyPageDelta({
    head,
    base,
    headSha: sha,
    baseSha,
    sourceCycle,
    targetCycle,
  });
  const result = await github(`/repos/${OWNER}/${REPO}/pulls/${number}/merge`, {
    method: 'PUT',
    body: {
      sha,
      merge_method: 'squash',
      commit_title: `Progress Prize rollover ${targetCycle}`,
    },
  });
  if (result?.merged !== true || !/^[a-f0-9]{40}$/.test(result.sha ?? '')) {
    fail('GitHub did not merge the gated pull request.');
  }
  if (process.env.GITHUB_OUTPUT) await appendFile(process.env.GITHUB_OUTPUT, `merge-sha=${result.sha}\n`);
  return { mergeSha: result.sha };
}

async function main(argv) {
  const { command, options } = parseArgs(argv);
  if (command === 'gate') return gate(options);
  if (command === 'activation-state') return activationState(options);
  if (command === 'wait-preview') return waitPreview(options);
  if (command === 'ensure-pr') return ensurePull(options);
  if (command === 'merge') return merge(options);
  fail('Unknown trusted GitHub helper command.');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await main(process.argv.slice(2));
  } catch {
    process.stderr.write('Progress Prize GitHub coordination failed safely.\n');
    process.exitCode = 1;
  }
}
