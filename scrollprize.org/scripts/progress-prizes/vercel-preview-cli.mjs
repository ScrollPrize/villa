#!/usr/bin/env node

import { constants as fsConstants } from 'node:fs';
import { open } from 'node:fs/promises';
import { isAbsolute, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import { verifyVercelPreview } from './vercel-preview.mjs';

const EXPECTED_REPOSITORY = 'ScrollPrize/villa';
const EXPECTED_REPOSITORY_ID = '890972577';
const EXPECTED_REPOSITORY_OWNER_ID = '121906140';
const MAX_EVENT_BYTES = 1024 * 1024;

const REQUIRED_ENVIRONMENT = Object.freeze([
  'GITHUB_EVENT_NAME',
  'GITHUB_EVENT_PATH',
  'GITHUB_REPOSITORY',
  'GITHUB_TOKEN',
  'PROGRESS_PRIZE_EXPECTED_SHA',
  'PROGRESS_PRIZE_EXPECTED_REF',
  'VERCEL_PROJECT_ID',
  'VERCEL_AUTOMATION_BYPASS_SECRET',
]);

const SAFE_MESSAGES = Object.freeze({
  CONFIG: 'The preview verifier configuration is incomplete or invalid.',
  CONTEXT: 'The preview verifier is not running in the expected repository context.',
  EVENT: 'The GitHub repository-dispatch event could not be read safely.',
  VERIFY: 'The Vercel preview verification failed.',
  SUMMARY: 'The preview was verified, but its safe runner summary could not be written.',
});

export class PreviewCliError extends Error {
  constructor(code) {
    super(SAFE_MESSAGES[code] ?? 'The Vercel preview verification command failed.');
    this.name = 'PreviewCliError';
    this.code = code;
  }
}

function fail(code) {
  throw new PreviewCliError(code);
}

function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function requireEnvironment(env) {
  if (!isRecord(env)) fail('CONFIG');
  for (const name of REQUIRED_ENVIRONMENT) {
    const value = env[name];
    if (typeof value !== 'string' || !value) fail('CONFIG');
  }
  if (
    env.PROGRESS_PRIZE_EXPECTED_CYCLE !== undefined
    && (
      typeof env.PROGRESS_PRIZE_EXPECTED_CYCLE !== 'string'
      || !/^\d{4}-(?:0[1-9]|1[0-2])$/.test(env.PROGRESS_PRIZE_EXPECTED_CYCLE)
    )
  ) {
    fail('CONFIG');
  }
  return env;
}

function assertRepositoryContext(env, event) {
  if (
    env.GITHUB_EVENT_NAME !== 'repository_dispatch'
    || env.GITHUB_REPOSITORY !== EXPECTED_REPOSITORY
  ) {
    fail('CONTEXT');
  }

  const environmentIds = [
    [env.GITHUB_REPOSITORY_ID, EXPECTED_REPOSITORY_ID],
    [env.GITHUB_REPOSITORY_OWNER_ID, EXPECTED_REPOSITORY_OWNER_ID],
  ];
  for (const [actual, expected] of environmentIds) {
    if (actual !== undefined && (typeof actual !== 'string' || actual !== expected)) {
      fail('CONTEXT');
    }
  }

  if (isRecord(event.repository)) {
    const eventValues = [
      [event.repository.full_name, EXPECTED_REPOSITORY],
      [event.repository.id, EXPECTED_REPOSITORY_ID],
      [isRecord(event.repository.owner) ? event.repository.owner.id : undefined, EXPECTED_REPOSITORY_OWNER_ID],
    ];
    for (const [actual, expected] of eventValues) {
      if (actual !== undefined && String(actual) !== expected) fail('CONTEXT');
    }
  }
}

function assertSafeAbsolutePath(value, errorCode) {
  if (
    typeof value !== 'string'
    || !value
    || value.includes('\u0000')
    || !isAbsolute(value)
    || resolve(value) !== value
  ) {
    fail(errorCode);
  }
  return value;
}

function noFollowFlag() {
  return typeof fsConstants.O_NOFOLLOW === 'number' ? fsConstants.O_NOFOLLOW : 0;
}

/** Read a bounded, regular, non-symlink GitHub event file. */
export async function readGitHubEvent(path) {
  const safePath = assertSafeAbsolutePath(path, 'EVENT');
  let handle;
  try {
    handle = await open(safePath, fsConstants.O_RDONLY | noFollowFlag());
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size < 2 || stat.size > MAX_EVENT_BYTES) fail('EVENT');
    const contents = await handle.readFile('utf8');
    if (Buffer.byteLength(contents, 'utf8') > MAX_EVENT_BYTES) fail('EVENT');
    const event = JSON.parse(contents);
    if (!isRecord(event)) fail('EVENT');
    return event;
  } catch (error) {
    if (error instanceof PreviewCliError) throw error;
    fail('EVENT');
  } finally {
    if (handle) {
      try {
        await handle.close();
      } catch {
        // The command has already consumed the file; never expose a private path.
      }
    }
  }
}

async function appendRunnerFile(path, contents, appendFileImpl) {
  const safePath = assertSafeAbsolutePath(path, 'SUMMARY');
  if (appendFileImpl !== undefined) {
    if (typeof appendFileImpl !== 'function') fail('SUMMARY');
    try {
      await appendFileImpl(safePath, contents);
      return;
    } catch {
      fail('SUMMARY');
    }
  }

  let handle;
  try {
    handle = await open(
      safePath,
      fsConstants.O_WRONLY | fsConstants.O_APPEND | fsConstants.O_CREAT | noFollowFlag(),
      0o600,
    );
    const stat = await handle.stat();
    if (!stat.isFile()) fail('SUMMARY');
    await handle.writeFile(contents, 'utf8');
  } catch (error) {
    if (error instanceof PreviewCliError) throw error;
    fail('SUMMARY');
  } finally {
    if (handle) {
      try {
        await handle.close();
      } catch {
        // Never expose the runner-file path in a close error.
      }
    }
  }
}

function safeSummary(cycle) {
  return [
    '### Progress Prize Vercel preview',
    '',
    '- Status: verified',
    `- Cycle: \`${cycle}\``,
    '- Page: `/prizes`',
    '',
  ].join('\n');
}

/**
 * Bind trusted runner context to the verifier. All errors crossing this boundary
 * have fixed messages and do not retain a cause or configuration values.
 */
export async function runVercelPreviewCli({
  env = process.env,
  readEventImpl = readGitHubEvent,
  verifyImpl = verifyVercelPreview,
  appendFileImpl,
} = {}) {
  requireEnvironment(env);
  if (typeof readEventImpl !== 'function' || typeof verifyImpl !== 'function') fail('CONFIG');

  let event;
  try {
    event = await readEventImpl(env.GITHUB_EVENT_PATH);
  } catch {
    fail('EVENT');
  }
  assertRepositoryContext(env, event);

  let verified;
  try {
    verified = await verifyImpl({
      event,
      expectedProjectId: env.VERCEL_PROJECT_ID,
      expectedSha: env.PROGRESS_PRIZE_EXPECTED_SHA,
      expectedRef: env.PROGRESS_PRIZE_EXPECTED_REF,
      expectedCycle: env.PROGRESS_PRIZE_EXPECTED_CYCLE,
      github: {
        owner: 'ScrollPrize',
        repo: 'villa',
        githubToken: env.GITHUB_TOKEN,
      },
      protectionBypassSecret: env.VERCEL_AUTOMATION_BYPASS_SECRET,
    });
  } catch {
    fail('VERIFY');
  }
  if (
    !isRecord(verified)
    || verified.ok !== true
    || typeof verified.cycle !== 'string'
    || !/^\d{4}-(?:0[1-9]|1[0-2])$/.test(verified.cycle)
    || (
      env.PROGRESS_PRIZE_EXPECTED_CYCLE !== undefined
      && verified.cycle !== env.PROGRESS_PRIZE_EXPECTED_CYCLE
    )
  ) {
    fail('VERIFY');
  }

  const summary = safeSummary(verified.cycle);
  if (env.GITHUB_STEP_SUMMARY !== undefined) {
    await appendRunnerFile(env.GITHUB_STEP_SUMMARY, summary, appendFileImpl);
  }
  if (env.GITHUB_OUTPUT !== undefined) {
    await appendRunnerFile(
      env.GITHUB_OUTPUT,
      `status=verified\ncycle=${verified.cycle}\n`,
      appendFileImpl,
    );
  }

  return Object.freeze({ ok: true, status: 'verified', cycle: verified.cycle });
}

function isDirectExecution() {
  if (!process.argv[1]) return false;
  return import.meta.url === pathToFileURL(resolve(process.argv[1])).href;
}

if (isDirectExecution()) {
  runVercelPreviewCli().catch((error) => {
    const message = error instanceof PreviewCliError
      ? error.message
      : SAFE_MESSAGES.VERIFY;
    process.stderr.write(`${message}\n`);
    process.exitCode = 1;
  });
}
