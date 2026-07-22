#!/usr/bin/env node

import { appendFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const OWNER = 'ScrollPrize';
const REPOSITORY = 'villa';
const REPOSITORY_ID = 890972577;
const API_ORIGIN = 'https://api.github.com';
const WEB_ORIGIN = 'https://github.com';
const MAIN_REF = 'main';
const PRODUCTION_WORKFLOW = 'progress-prizes-production.yml';
const PRODUCTION_WORKFLOW_PATH = `.github/workflows/${PRODUCTION_WORKFLOW}`;
const PAGE_PATH = 'scrollprize.org/docs/34_prizes.md';
const API_VERSION = '2022-11-28';
const SAFE_FAILURE = 'Progress Prize scheduler GitHub coordination failed safely.';

export const MAX_GITHUB_RESPONSE_BYTES = 256 * 1024;
export const NONTERMINAL_RUN_STATUSES = Object.freeze([
  'requested',
  'queued',
  'pending',
  'waiting',
  'in_progress',
]);

const NONTERMINAL_STATUS_SET = new Set(NONTERMINAL_RUN_STATUSES);
const RUN_STATUS_SET = new Set([...NONTERMINAL_RUN_STATUSES, 'completed']);
const OPERATIONS = new Set(['dry-run', 'prepare', 'activate']);
const DEDUPE_REASONS = new Set([
  'production-idle',
  'production-run-nonterminal',
  'prepare-required',
  'prepare-exact',
  'prepare-refresh',
]);
const DISPATCH_REASONS = new Set([
  'production-dispatched',
  'production-run-nonterminal',
]);

function failClosed() {
  throw new Error(SAFE_FAILURE);
}

function isPlainObject(value) {
  return value !== null
    && typeof value === 'object'
    && !Array.isArray(value)
    && (Object.getPrototypeOf(value) === Object.prototype
      || Object.getPrototypeOf(value) === null);
}

function assertPlainObject(value) {
  if (!isPlainObject(value)) failClosed();
  return value;
}

function assertPositiveInteger(value) {
  if (!Number.isSafeInteger(value) || value < 1) failClosed();
  return value;
}

function assertSha(value) {
  if (typeof value !== 'string' || !/^[a-f0-9]{40}$/.test(value)) failClosed();
  return value;
}

function assertOperation(value) {
  if (typeof value !== 'string' || !OPERATIONS.has(value)) failClosed();
  return value;
}

function parseCycle(value) {
  if (typeof value !== 'string') failClosed();
  const match = /^([1-9]\d{3})-(0[1-9]|1[0-2])$/.exec(value);
  if (!match) failClosed();
  return { value, year: Number(match[1]), month: Number(match[2]) };
}

function assertConsecutiveCycles(sourceCycle, targetCycle) {
  const source = parseCycle(sourceCycle);
  const target = parseCycle(targetCycle);
  const nextYear = source.month === 12 ? source.year + 1 : source.year;
  const nextMonth = source.month === 12 ? 1 : source.month + 1;
  if (
    nextYear > 9999
    || target.year !== nextYear
    || target.month !== nextMonth
  ) {
    failClosed();
  }
  return Object.freeze({ sourceCycle: source.value, targetCycle: target.value });
}

function assertRequestId(value) {
  if (typeof value === 'number' && (!Number.isSafeInteger(value) || value < 1)) failClosed();
  const text = typeof value === 'number' ? String(value) : value;
  if (
    typeof text !== 'string'
    || !/^[1-9]\d*$/.test(text)
    || text.length > 20
    || BigInt(text) > 18_446_744_073_709_551_615n
  ) {
    failClosed();
  }
  return text;
}

function assertToken(token) {
  if (
    typeof token !== 'string'
    || token.length < 1
    || token.length > 16_384
    || /[\r\n]/.test(token)
  ) {
    failClosed();
  }
  return token;
}

function assertRepository(value) {
  const repository = assertPlainObject(value);
  if (
    repository.id !== REPOSITORY_ID
    || repository.full_name !== `${OWNER}/${REPOSITORY}`
  ) {
    failClosed();
  }
  return repository;
}

function workflowRunApiUrl(id) {
  return `${API_ORIGIN}/repos/${OWNER}/${REPOSITORY}/actions/runs/${id}`;
}

function workflowRunHtmlUrl(id) {
  return `${WEB_ORIGIN}/${OWNER}/${REPOSITORY}/actions/runs/${id}`;
}

function pullApiUrl(number) {
  return `${API_ORIGIN}/repos/${OWNER}/${REPOSITORY}/pulls/${number}`;
}

function pullHtmlUrl(number) {
  return `${WEB_ORIGIN}/${OWNER}/${REPOSITORY}/pull/${number}`;
}

function assertWorkflowRun(run, expectedStatus) {
  const value = assertPlainObject(run);
  const id = assertPositiveInteger(value.id);
  if (
    !RUN_STATUS_SET.has(value.status)
    || (expectedStatus !== undefined && value.status !== expectedStatus)
    || value.event !== 'workflow_dispatch'
    || value.head_branch !== MAIN_REF
    || value.path !== PRODUCTION_WORKFLOW_PATH
    || value.url !== workflowRunApiUrl(id)
    || value.html_url !== workflowRunHtmlUrl(id)
  ) {
    failClosed();
  }
  assertSha(value.head_sha);
  assertRepository(value.repository);
  assertRepository(value.head_repository);
  return Object.freeze({
    runId: id,
    runUrl: value.html_url,
    status: value.status,
  });
}

async function readBoundedText(response) {
  let reader;
  try {
    if (
      response?.redirected !== false
      || typeof response.headers?.get !== 'function'
      || typeof response.body?.getReader !== 'function'
    ) {
      failClosed();
    }
    const contentLength = response.headers.get('content-length');
    if (contentLength !== null) {
      if (!/^\d+$/.test(contentLength)) failClosed();
      const length = Number(contentLength);
      if (!Number.isSafeInteger(length) || length > MAX_GITHUB_RESPONSE_BYTES) failClosed();
    }
    reader = response.body.getReader();
  } catch {
    failClosed();
  }

  const decoder = new TextDecoder();
  let byteLength = 0;
  let result = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!(value instanceof Uint8Array)) failClosed();
      byteLength += value.byteLength;
      if (byteLength > MAX_GITHUB_RESPONSE_BYTES) {
        await reader.cancel();
        failClosed();
      }
      result += decoder.decode(value, { stream: true });
    }
    result += decoder.decode();
  } catch {
    failClosed();
  } finally {
    try {
      reader.releaseLock();
    } catch {
      failClosed();
    }
  }
  return result;
}

async function readBoundedJson(response) {
  const text = await readBoundedText(response);
  try {
    return JSON.parse(text);
  } catch {
    failClosed();
  }
}

async function githubJson(path, {
  token,
  fetchImpl = globalThis.fetch,
  method = 'GET',
  body,
  expectedStatus = 200,
} = {}) {
  const credential = assertToken(token);
  if (typeof fetchImpl !== 'function') failClosed();
  if (typeof path !== 'string' || !path.startsWith('/repos/ScrollPrize/villa/')) failClosed();

  let response;
  try {
    response = await fetchImpl(`${API_ORIGIN}${path}`, {
      method,
      redirect: 'error',
      headers: {
        accept: 'application/vnd.github+json',
        authorization: `Bearer ${credential}`,
        'content-type': 'application/json',
        'user-agent': 'progress-prize-trusted-scheduler',
        'x-github-api-version': API_VERSION,
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    failClosed();
  }

  try {
    if (response?.status !== expectedStatus || response.redirected !== false) failClosed();
    return await readBoundedJson(response);
  } catch {
    failClosed();
  }
}

function workflowRunsPath(status) {
  const query = new URLSearchParams({
    branch: MAIN_REF,
    event: 'workflow_dispatch',
    status,
    per_page: '100',
  });
  return `/repos/${OWNER}/${REPOSITORY}/actions/workflows/${PRODUCTION_WORKFLOW}/runs?${query}`;
}

function recentWorkflowRunsPath() {
  const query = new URLSearchParams({
    branch: MAIN_REF,
    event: 'workflow_dispatch',
    per_page: '10',
  });
  return `/repos/${OWNER}/${REPOSITORY}/actions/workflows/${PRODUCTION_WORKFLOW}/runs?${query}`;
}

function assertWorkflowRunsPage(body, expectedStatus) {
  const value = assertPlainObject(body);
  if (
    !Number.isSafeInteger(value.total_count)
    || value.total_count < 0
    || value.total_count > 100
    || !Array.isArray(value.workflow_runs)
    || value.workflow_runs.length !== value.total_count
  ) {
    failClosed();
  }
  const ids = new Set();
  return value.workflow_runs.map((run) => {
    const inspected = assertWorkflowRun(run, expectedStatus);
    if (ids.has(inspected.runId)) failClosed();
    ids.add(inspected.runId);
    return inspected;
  });
}

function assertRecentWorkflowRunsPage(body) {
  const value = assertPlainObject(body);
  if (
    !Number.isSafeInteger(value.total_count)
    || value.total_count < 0
    || !Array.isArray(value.workflow_runs)
    || value.workflow_runs.length !== Math.min(value.total_count, 10)
  ) {
    failClosed();
  }
  const ids = new Set();
  return value.workflow_runs.map((run) => {
    const inspected = assertWorkflowRun(run);
    if (ids.has(inspected.runId)) failClosed();
    ids.add(inspected.runId);
    return inspected;
  });
}

export async function inspectProductionWorkflowRuns({
  token,
  fetchImpl = globalThis.fetch,
} = {}) {
  const pages = await Promise.all(NONTERMINAL_RUN_STATUSES.map(async (status) => {
    const body = await githubJson(workflowRunsPath(status), { token, fetchImpl });
    return assertWorkflowRunsPage(body, status);
  }));

  // A single final snapshot closes the crossing-state gap between the filtered
  // requests (for example, requested -> queued while those pages are read).
  const recentBody = await githubJson(recentWorkflowRunsPath(), { token, fetchImpl });
  const recentRuns = assertRecentWorkflowRunsPage(recentBody);

  for (let index = 0; index < pages.length; index += 1) {
    const run = pages[index][0];
    if (run) {
      return Object.freeze({
        dispatch: false,
        reason: 'production-run-nonterminal',
        blockingStatus: NONTERMINAL_RUN_STATUSES[index],
        blockingRunId: run.runId,
        blockingRunUrl: run.runUrl,
      });
    }
  }
  const recentBlocker = recentRuns.find((run) => NONTERMINAL_STATUS_SET.has(run.status));
  if (recentBlocker) {
    return Object.freeze({
      dispatch: false,
      reason: 'production-run-nonterminal',
      blockingStatus: recentBlocker.status,
      blockingRunId: recentBlocker.runId,
      blockingRunUrl: recentBlocker.runUrl,
    });
  }
  return Object.freeze({ dispatch: true, reason: 'production-idle' });
}

function assertMainRef(body) {
  const value = assertPlainObject(body);
  const object = assertPlainObject(value.object);
  const sha = assertSha(object.sha);
  if (
    value.ref !== 'refs/heads/main'
    || object.type !== 'commit'
    || object.url !== `${API_ORIGIN}/repos/${OWNER}/${REPOSITORY}/git/commits/${sha}`
  ) {
    failClosed();
  }
  return sha;
}

function automationBranch(targetCycle) {
  const target = parseCycle(targetCycle);
  return `codex/progress-prize-${target.value}`;
}

function assertPull(pull, { branch }) {
  const value = assertPlainObject(pull);
  const number = assertPositiveInteger(value.number);
  assertPositiveInteger(value.id);
  const head = assertPlainObject(value.head);
  const base = assertPlainObject(value.base);
  const headSha = assertSha(head.sha);
  const baseSha = assertSha(base.sha);
  if (
    value.state !== 'open'
    || value.url !== pullApiUrl(number)
    || value.html_url !== pullHtmlUrl(number)
    || head.ref !== branch
    || base.ref !== MAIN_REF
  ) {
    failClosed();
  }
  assertRepository(head.repo);
  assertRepository(base.repo);
  return Object.freeze({
    number,
    url: value.html_url,
    headSha,
    baseSha,
  });
}

function assertPageOnlyCommit(commit, { headSha }) {
  const value = assertPlainObject(commit);
  if (
    value.sha !== headSha
    || value.url !== `${API_ORIGIN}/repos/${OWNER}/${REPOSITORY}/commits/${headSha}`
    || !Array.isArray(value.parents)
    || value.parents.length !== 1
    || !Array.isArray(value.files)
    || value.files.length !== 1
    || value.files[0]?.filename !== PAGE_PATH
    || value.files[0]?.status !== 'modified'
  ) {
    failClosed();
  }
  return assertSha(value.parents[0]?.sha);
}

export async function inspectProductionPrepare({
  targetCycle,
  token,
  fetchImpl = globalThis.fetch,
} = {}) {
  const branch = automationBranch(targetCycle);
  const query = new URLSearchParams({
    state: 'open',
    head: `${OWNER}:${branch}`,
    base: MAIN_REF,
    per_page: '2',
  });
  const pulls = await githubJson(
    `/repos/${OWNER}/${REPOSITORY}/pulls?${query}`,
    { token, fetchImpl },
  );
  if (!Array.isArray(pulls) || pulls.length > 1) failClosed();
  if (pulls.length === 0) {
    return Object.freeze({ dispatch: true, reason: 'prepare-required' });
  }

  const pull = assertPull(pulls[0], { branch });
  const commitBody = await githubJson(
    `/repos/${OWNER}/${REPOSITORY}/commits/${pull.headSha}?per_page=2`,
    { token, fetchImpl },
  );
  const parentSha = assertPageOnlyCommit(commitBody, { headSha: pull.headSha });
  // Read main last so the skip decision is based on the freshest ref observed by
  // this helper, rather than on a ref read before the pull and commit requests.
  const mainBody = await githubJson(
    `/repos/${OWNER}/${REPOSITORY}/git/ref/heads/${MAIN_REF}`,
    { token, fetchImpl },
  );
  const mainSha = assertMainRef(mainBody);
  if (parentSha !== mainSha || pull.baseSha !== mainSha) {
    return Object.freeze({
      dispatch: true,
      reason: 'prepare-refresh',
      pullNumber: pull.number,
      pullUrl: pull.url,
    });
  }
  return Object.freeze({
    dispatch: false,
    reason: 'prepare-exact',
    pullNumber: pull.number,
    pullUrl: pull.url,
  });
}

export async function dedupeProductionDispatch({
  operation,
  targetCycle,
  token,
  fetchImpl = globalThis.fetch,
} = {}) {
  const selectedOperation = assertOperation(operation);
  parseCycle(targetCycle);
  const runState = await inspectProductionWorkflowRuns({ token, fetchImpl });
  if (!runState.dispatch || selectedOperation !== 'prepare') return runState;
  return inspectProductionPrepare({ targetCycle, token, fetchImpl });
}

export function buildProductionDispatch({
  operation,
  sourceCycle,
  targetCycle,
  requestId,
} = {}) {
  const selectedOperation = assertOperation(operation);
  const cycles = assertConsecutiveCycles(sourceCycle, targetCycle);
  const numericRequestId = assertRequestId(requestId);
  return Object.freeze({
    ref: MAIN_REF,
    inputs: Object.freeze({
      operation: selectedOperation,
      'source-cycle': cycles.sourceCycle,
      'target-cycle': cycles.targetCycle,
      'verify-mode': 'prepared',
      'request-id': numericRequestId,
    }),
    return_run_details: true,
  });
}

export async function dispatchProductionWorkflow({
  operation,
  sourceCycle,
  targetCycle,
  requestId,
  token,
  fetchImpl = globalThis.fetch,
} = {}) {
  const dispatch = buildProductionDispatch({
    operation,
    sourceCycle,
    targetCycle,
    requestId,
  });
  const runState = await inspectProductionWorkflowRuns({ token, fetchImpl });
  if (!runState.dispatch) {
    return Object.freeze({
      dispatched: false,
      reason: runState.reason,
      blockingStatus: runState.blockingStatus,
      blockingRunId: runState.blockingRunId,
      blockingRunUrl: runState.blockingRunUrl,
    });
  }
  const body = await githubJson(
    `/repos/${OWNER}/${REPOSITORY}/actions/workflows/${PRODUCTION_WORKFLOW}/dispatches`,
    {
      token,
      fetchImpl,
      method: 'POST',
      body: dispatch,
      expectedStatus: 200,
    },
  );
  const value = assertPlainObject(body);
  const runId = assertPositiveInteger(value.workflow_run_id);
  const apiUrl = workflowRunApiUrl(runId);
  const htmlUrl = workflowRunHtmlUrl(runId);
  if (value.run_url !== apiUrl || value.html_url !== htmlUrl) failClosed();
  return Object.freeze({
    dispatched: true,
    reason: 'production-dispatched',
    runId,
    apiUrl,
    htmlUrl,
  });
}

function parseArgs(argv) {
  if (!Array.isArray(argv) || argv.length < 1) failClosed();
  const [command, ...tokens] = argv;
  if (command !== 'dedupe' && command !== 'dispatch') failClosed();
  if (tokens.length % 2 !== 0) failClosed();
  const options = Object.create(null);
  for (let index = 0; index < tokens.length; index += 2) {
    const name = tokens[index];
    const value = tokens[index + 1];
    if (
      typeof name !== 'string'
      || !/^--[a-z][a-z0-9-]*$/.test(name)
      || typeof value !== 'string'
      || value.length === 0
      || value.startsWith('--')
    ) {
      failClosed();
    }
    const key = name.slice(2);
    if (Object.hasOwn(options, key)) failClosed();
    options[key] = value;
  }
  const allowed = command === 'dedupe'
    ? new Set(['operation', 'target-cycle'])
    : new Set(['operation', 'source-cycle', 'target-cycle', 'request-id']);
  if (Object.keys(options).some((key) => !allowed.has(key))) failClosed();
  if (Object.keys(options).length !== allowed.size) failClosed();
  for (const name of allowed) {
    if (!Object.hasOwn(options, name)) failClosed();
  }
  return { command, options };
}

function assertPublicOutputValue(value) {
  const text = String(value);
  if (!/^[A-Za-z0-9:/._-]+$/.test(text) || /[\r\n]/.test(text)) failClosed();
  return text;
}

function outputEntries(command, result) {
  if (command === 'dedupe') {
    if (
      typeof result?.dispatch !== 'boolean'
      || !DEDUPE_REASONS.has(result.reason)
    ) {
      failClosed();
    }
    const entries = [
      ['dispatch', result.dispatch ? 'true' : 'false'],
      ['dedupe-reason', result.reason],
    ];
    if (result.blockingStatus !== undefined) {
      if (!NONTERMINAL_STATUS_SET.has(result.blockingStatus)) failClosed();
      entries.push(['blocking-status', result.blockingStatus]);
    }
    if (result.blockingRunId !== undefined) entries.push(['blocking-run-id', result.blockingRunId]);
    if (result.blockingRunUrl !== undefined) entries.push(['blocking-run-url', result.blockingRunUrl]);
    if (result.pullNumber !== undefined) entries.push(['pull-number', result.pullNumber]);
    if (result.pullUrl !== undefined) entries.push(['pull-url', result.pullUrl]);
    return entries;
  }
  if (
    typeof result?.dispatched !== 'boolean'
    || !DISPATCH_REASONS.has(result.reason)
    || result.dispatched !== (result.reason === 'production-dispatched')
  ) {
    failClosed();
  }
  const entries = [
    ['dispatched', result.dispatched ? 'true' : 'false'],
    ['dispatch-reason', result.reason],
  ];
  if (result.dispatched) {
    const runId = assertPositiveInteger(result.runId);
    if (
      result.apiUrl !== workflowRunApiUrl(runId)
      || result.htmlUrl !== workflowRunHtmlUrl(runId)
    ) {
      failClosed();
    }
    entries.push(
      ['child-run-id', runId],
      ['child-run-api-url', result.apiUrl],
      ['child-run-html-url', result.htmlUrl],
    );
  } else {
    if (!NONTERMINAL_STATUS_SET.has(result.blockingStatus)) failClosed();
    const runId = assertPositiveInteger(result.blockingRunId);
    if (result.blockingRunUrl !== workflowRunHtmlUrl(runId)) failClosed();
    entries.push(
      ['blocking-status', result.blockingStatus],
      ['blocking-run-id', runId],
      ['blocking-run-url', result.blockingRunUrl],
    );
  }
  return entries;
}

async function appendPublicOutputs(command, result, {
  outputPath,
  appendFileImpl = appendFile,
} = {}) {
  if (
    typeof outputPath !== 'string'
    || outputPath.length < 1
    || outputPath.length > 4096
    || /[\0\r\n]/.test(outputPath)
    || typeof appendFileImpl !== 'function'
  ) {
    failClosed();
  }
  const content = `${outputEntries(command, result)
    .map(([key, value]) => `${key}=${assertPublicOutputValue(value)}`)
    .join('\n')}\n`;
  try {
    await appendFileImpl(outputPath, content, 'utf8');
  } catch {
    failClosed();
  }
}

export async function runSchedulerCli(argv, {
  env = process.env,
  fetchImpl = globalThis.fetch,
  appendFileImpl = appendFile,
} = {}) {
  const { command, options } = parseArgs(argv);
  const token = env?.GITHUB_TOKEN;
  let result;
  if (command === 'dedupe') {
    result = await dedupeProductionDispatch({
      operation: options.operation,
      targetCycle: options['target-cycle'],
      token,
      fetchImpl,
    });
  } else {
    result = await dispatchProductionWorkflow({
      operation: options.operation,
      sourceCycle: options['source-cycle'],
      targetCycle: options['target-cycle'],
      requestId: options['request-id'],
      token,
      fetchImpl,
    });
  }
  await appendPublicOutputs(command, result, {
    outputPath: env?.GITHUB_OUTPUT,
    appendFileImpl,
  });
  return result;
}

async function main() {
  try {
    await runSchedulerCli(process.argv.slice(2));
  } catch {
    process.stderr.write(`${SAFE_FAILURE}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
