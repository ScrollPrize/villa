#!/usr/bin/env node

import { appendFile, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { createClock } from './clock.mjs';
import { createGoogleApiClient, redactForLog } from './google-api.mjs';
import { assertPublicResponderUri } from './markdown.mjs';
import {
  assertRolloverRuntimeSafety,
  createRolloverService,
} from './rollover.mjs';
import { parseCycle, previousCycle } from './dates.mjs';
import {
  GOOGLE_CLI_USAGE,
  GOOGLE_COMMANDS,
  parseCliArgs,
  parsePositiveIntegerOption,
  requireOption,
} from './cli-args.mjs';

const DEFAULT_PAGE_PATH = resolve(
  fileURLToPath(new URL('../../docs/34_prizes.md', import.meta.url)),
);
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);
const GIT_SHA_PATTERN = /^[a-f0-9]{40}$/i;
const OUTPUT_STATUSES = new Set([
  'active',
  'archived',
  'planned',
  'prepared',
  'valid',
  'waiting',
]);

export const AUTOMATION_ENV = Object.freeze({
  ACCESS_TOKEN: 'GOOGLE_ACCESS_TOKEN',
  ARCHIVE_FOLDER_ID: 'PROGRESS_PRIZE_ARCHIVE_FOLDER_ID',
  BRANCH: 'PROGRESS_PRIZE_BRANCH',
  DEFAULT_TARGET_BRANCH: 'PROGRESS_PRIZE_DEFAULT_TARGET_BRANCH',
  DRIVE_ID: 'PROGRESS_PRIZE_DRIVE_ID',
  EDITOR_GROUP_EMAIL: 'PROGRESS_PRIZE_EDITOR_GROUP_EMAIL',
  ENVIRONMENT: 'PROGRESS_PRIZE_ENVIRONMENT',
  FOLDER_ID: 'PROGRESS_PRIZE_FOLDER_ID',
  HEAD_SHA: 'PROGRESS_PRIZE_HEAD_SHA',
  SERVICE_ACCOUNT_EMAIL: 'PROGRESS_PRIZE_SERVICE_ACCOUNT_EMAIL',
  SMOKE_BRANCH_PREFIX: 'PROGRESS_PRIZE_SMOKE_BRANCH_PREFIX',
  SMOKE_DATE: 'PROGRESS_PRIZE_SMOKE_DATE',
  SOURCE_FORM_ID: 'PROGRESS_PRIZE_SOURCE_FORM_ID',
  STAGING_FOLDER_ID: 'PROGRESS_PRIZE_STAGING_FOLDER_ID',
  STAGING_SERVICE_ACCOUNT_EMAIL: 'PROGRESS_PRIZE_STAGING_SERVICE_ACCOUNT_EMAIL',
  TARGET_BRANCH: 'PROGRESS_PRIZE_TARGET_BRANCH',
  VERIFIED_SHA: 'PROGRESS_PRIZE_VERIFIED_SHA',
});

const PRIVATE_ENV_NAMES = Object.freeze([
  AUTOMATION_ENV.ACCESS_TOKEN,
  AUTOMATION_ENV.ARCHIVE_FOLDER_ID,
  AUTOMATION_ENV.DRIVE_ID,
  AUTOMATION_ENV.EDITOR_GROUP_EMAIL,
  AUTOMATION_ENV.FOLDER_ID,
  AUTOMATION_ENV.SERVICE_ACCOUNT_EMAIL,
  AUTOMATION_ENV.SOURCE_FORM_ID,
  AUTOMATION_ENV.STAGING_FOLDER_ID,
  AUTOMATION_ENV.STAGING_SERVICE_ACCOUNT_EMAIL,
]);

export class ResponderUrlResolutionError extends Error {
  constructor() {
    super('The public Google Forms short URL could not be resolved safely.');
    this.name = 'ResponderUrlResolutionError';
  }
}

function requiredEnv(env, name) {
  const value = env[name];
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`Required environment variable ${name} is missing`);
  }
  return value;
}

function optionalEnv(env, name) {
  const value = env[name];
  return typeof value === 'string' && value.trim() !== '' ? value : undefined;
}

function normalizeCanonicalResponderUri(value) {
  try {
    const normalized = assertPublicResponderUri(value);
    const url = new URL(normalized);
    if (url.hostname !== 'docs.google.com') {
      throw new ResponderUrlResolutionError();
    }
    url.search = '';
    if (url.pathname.endsWith('/')) url.pathname = url.pathname.slice(0, -1);
    return assertPublicResponderUri(url.toString());
  } catch {
    throw new ResponderUrlResolutionError();
  }
}

/**
 * Resolve only the public forms.gle redirect. Redirects are never followed by
 * fetch, and the destination must already be a canonical public Forms URL.
 */
export async function resolveGoogleResponderUri(value, {
  fetchImpl = globalThis.fetch,
} = {}) {
  const normalized = assertPublicResponderUri(value);
  const input = new URL(normalized);
  if (input.hostname === 'docs.google.com') {
    return normalizeCanonicalResponderUri(input.toString());
  }
  if (typeof fetchImpl !== 'function') throw new ResponderUrlResolutionError();

  let response;
  try {
    response = await fetchImpl(input, {
      method: 'GET',
      redirect: 'manual',
      headers: {
        accept: 'text/html',
        'user-agent': 'scrollprize-progress-prize-responder-resolver',
      },
    });
  } catch {
    throw new ResponderUrlResolutionError();
  }
  if (!response || !REDIRECT_STATUSES.has(response.status)) {
    throw new ResponderUrlResolutionError();
  }

  const location = response.headers?.get?.('location');
  if (typeof location !== 'string' || location === '') {
    throw new ResponderUrlResolutionError();
  }
  let destination;
  try {
    destination = new URL(location, input);
  } catch {
    throw new ResponderUrlResolutionError();
  }
  return normalizeCanonicalResponderUri(destination.toString());
}

/** File boundary consumed by createRolloverService. */
export function createFilePageFacade({
  pagePath = DEFAULT_PAGE_PATH,
  read = readFile,
  write = writeFile,
  fetchImpl = globalThis.fetch,
} = {}) {
  if (typeof pagePath !== 'string' || pagePath.trim() === '') {
    throw new TypeError('pagePath must be a non-empty string');
  }
  if (typeof read !== 'function' || typeof write !== 'function') {
    throw new TypeError('read and write must be functions');
  }
  const file = resolve(pagePath);
  return Object.freeze({
    file,
    read: () => read(file, 'utf8'),
    write: (content) => write(file, content, 'utf8'),
    resolveResponderUri: (uri) => resolveGoogleResponderUri(uri, { fetchImpl }),
  });
}

function resolvePagePath(options) {
  if (options.file !== undefined && options['page-path'] !== undefined) {
    throw new Error('Use either --file or --page-path, not both');
  }
  return options['page-path'] ?? options.file ?? DEFAULT_PAGE_PATH;
}

function buildRuntime(options, env) {
  const environment = options.environment
    ?? optionalEnv(env, AUTOMATION_ENV.ENVIRONMENT)
    ?? 'production';
  const eventName = options['event-name'] ?? optionalEnv(env, 'GITHUB_EVENT_NAME');
  const branch = optionalEnv(env, AUTOMATION_ENV.BRANCH)
    ?? optionalEnv(env, 'GITHUB_HEAD_REF')
    ?? optionalEnv(env, 'GITHUB_REF_NAME');

  return {
    environment,
    eventName,
    folderId: requiredEnv(env, AUTOMATION_ENV.FOLDER_ID),
    driveId: requiredEnv(env, AUTOMATION_ENV.DRIVE_ID),
    serviceAccountEmail: requiredEnv(env, AUTOMATION_ENV.SERVICE_ACCOUNT_EMAIL),
    stagingServiceAccountEmail: optionalEnv(env, AUTOMATION_ENV.STAGING_SERVICE_ACCOUNT_EMAIL),
    stagingFolderId: optionalEnv(env, AUTOMATION_ENV.STAGING_FOLDER_ID),
    archiveFolderId: optionalEnv(env, AUTOMATION_ENV.ARCHIVE_FOLDER_ID),
    branch,
    targetBranch: optionalEnv(env, AUTOMATION_ENV.TARGET_BRANCH) ?? 'main',
    defaultTargetBranch: optionalEnv(env, AUTOMATION_ENV.DEFAULT_TARGET_BRANCH) ?? 'main',
    smokeBranchPrefix: optionalEnv(env, AUTOMATION_ENV.SMOKE_BRANCH_PREFIX)
      ?? 'codex/progress-prize-smoke-',
    smokeDate: optionalEnv(env, AUTOMATION_ENV.SMOKE_DATE),
    simulatedNow: options['simulated-now'],
  };
}

function assertExactGitSha(value, variableName) {
  if (typeof value !== 'string' || !GIT_SHA_PATTERN.test(value)) {
    throw new Error(`${variableName} must be an exact 40-character Git commit SHA`);
  }
  return value.toLowerCase();
}

/**
 * The preview/check job writes VERIFIED_SHA. Activation separately receives the
 * intended PR head as HEAD_SHA. Equality is checked both at construction and
 * when the rollover service invokes the gate.
 */
export function createExactShaActivationGate(env, {
  headSha: headShaOption,
  verifiedSha: verifiedShaOption,
} = {}) {
  const headSha = assertExactGitSha(
    headShaOption ?? requiredEnv(env, AUTOMATION_ENV.HEAD_SHA),
    '--head-sha',
  );
  const verifiedSha = assertExactGitSha(
    verifiedShaOption ?? requiredEnv(env, AUTOMATION_ENV.VERIFIED_SHA),
    '--verified-sha',
  );
  if (headSha !== verifiedSha) {
    throw new Error('Activation is not bound to the exact verified commit SHA');
  }
  return Object.freeze({
    headSha,
    gate: async ({ headSha: requestedSha }) => (
      typeof requestedSha === 'string'
      && requestedSha.toLowerCase() === verifiedSha
    ),
  });
}

function privateValues(env) {
  return PRIVATE_ENV_NAMES.map((name) => optionalEnv(env, name)).filter(Boolean);
}

function assertSourceCycleMatchesTarget(options, targetCycle) {
  const expected = previousCycle(targetCycle);
  if (options['source-cycle'] !== undefined) {
    parseCycle(options['source-cycle']);
    if (options['source-cycle'] !== expected) {
      throw new Error(`Source cycle must be ${expected} for target cycle ${targetCycle}`);
    }
  }
  return expected;
}

function assertAutomationOptions(command, options) {
  if (command !== 'validate' && options.google === true) {
    throw new Error('--google is accepted only by validate');
  }
  if (options.mode !== undefined && command !== 'verify') {
    throw new Error('--mode is accepted only by verify');
  }
  if (options['dry-run'] === true && !['bootstrap', 'prepare'].includes(command)) {
    throw new Error('--dry-run is accepted only by bootstrap and prepare');
  }
  if (options.fault !== undefined && !['bootstrap', 'prepare', 'activate'].includes(command)) {
    throw new Error('--fault is accepted only by bootstrap, prepare, and activate');
  }
  for (const shaOption of ['head-sha', 'verified-sha']) {
    if (options[shaOption] !== undefined && command !== 'activate') {
      throw new Error(`--${shaOption} is accepted only by activate`);
    }
  }
  if (
    command === 'activate'
    && ((options['head-sha'] === undefined) !== (options['verified-sha'] === undefined))
  ) {
    throw new Error('--head-sha and --verified-sha must be supplied together');
  }
  for (const coreOnly of [
    'expected-current-cycle',
    'expected-current-responder-uri',
    'responder-uri',
  ]) {
    if (options[coreOnly] !== undefined) {
      throw new Error(`--${coreOnly} is not accepted by Google automation commands`);
    }
  }
}

function collaboratorPermissions(command, env) {
  if (!['validate', 'bootstrap', 'prepare', 'activate', 'verify'].includes(command)) return [];
  return [{
    type: 'group',
    role: 'writer',
    emailAddress: requiredEnv(env, AUTOMATION_ENV.EDITOR_GROUP_EMAIL),
  }];
}

async function emitAutomationResult(result, {
  env,
  output,
  append = appendFile,
} = {}) {
  const summary = redactForLog(result, { secrets: privateValues(env) });
  output(JSON.stringify(summary));

  const outputLines = [];
  if (OUTPUT_STATUSES.has(result?.status)) {
    outputLines.push(`status=${result.status}`);
  }
  for (const [key, value] of [
    ['cycle', result?.cycle],
    ['source_cycle', result?.sourceCycle],
    ['target_cycle', result?.targetCycle],
  ]) {
    if (value !== undefined) {
      parseCycle(value);
      outputLines.push(`${key}=${value}`);
    }
  }
  if (result?.responderUri !== undefined) {
    const responderUri = assertPublicResponderUri(result.responderUri);
    outputLines.push(`responder_uri=${responderUri}`);
  }
  const githubOutput = optionalEnv(env, 'GITHUB_OUTPUT');
  if (githubOutput !== undefined && outputLines.length > 0) {
    await append(githubOutput, `${outputLines.join('\n')}\n`, 'utf8');
  }
  return summary;
}

/** Execute only the Google-backed command family. */
export async function runAutomationCli(argv, {
  env = process.env,
  read = readFile,
  write = writeFile,
  append = appendFile,
  fetchImpl = globalThis.fetch,
  googleFactory = createGoogleApiClient,
  rolloverFactory = createRolloverService,
  output = (value) => process.stdout.write(`${value}\n`),
} = {}) {
  const { command, options } = parseCliArgs(argv);
  if (!GOOGLE_COMMANDS.includes(command)) {
    throw new Error(`${command} is not a Google automation command`);
  }
  if (options.help) {
    output(GOOGLE_CLI_USAGE);
    return { help: true };
  }
  assertAutomationOptions(command, options);

  // This validation deliberately precedes token access, Google client creation,
  // responder URL resolution, and every filesystem/network call.
  const runtime = buildRuntime(options, env);
  assertRolloverRuntimeSafety(runtime, { faultInjection: options.fault });
  const clock = createClock({
    environment: runtime.environment,
    eventName: runtime.eventName,
    simulatedNow: runtime.simulatedNow,
  });

  let activation;
  if (command === 'activate') {
    activation = createExactShaActivationGate(env, {
      headSha: options['head-sha'],
      verifiedSha: options['verified-sha'],
    });
  }

  const page = createFilePageFacade({
    pagePath: resolvePagePath(options),
    read,
    write,
    fetchImpl,
  });
  const google = googleFactory({
    accessToken: requiredEnv(env, AUTOMATION_ENV.ACCESS_TOKEN),
    fetchImpl,
  });
  const rollover = rolloverFactory({
    google,
    page,
    clock,
    runtime,
    activationGate: activation?.gate,
  });
  const preparationDays = parsePositiveIntegerOption(options, 'preparation-days', 7);
  const collaborators = collaboratorPermissions(command, env);
  let result;
  if (command === 'validate') {
    const sourceCycle = requireOption(options, 'source-cycle');
    parseCycle(sourceCycle);
    result = await rollover.validate({
      sourceFormId: requiredEnv(env, AUTOMATION_ENV.SOURCE_FORM_ID),
      sourceCycle,
      collaboratorPermissions: collaborators,
    });
  } else if (command === 'bootstrap') {
    const sourceCycle = requireOption(options, 'source-cycle');
    parseCycle(sourceCycle);
    result = await rollover.bootstrapStagingSource({
      sourceFormId: requiredEnv(env, AUTOMATION_ENV.SOURCE_FORM_ID),
      sourceCycle,
      collaboratorPermissions: collaborators,
      dryRun: options['dry-run'] === true,
      faultInjection: options.fault,
    });
  } else {
    const targetCycle = requireOption(options, 'target-cycle');
    parseCycle(targetCycle);
    assertSourceCycleMatchesTarget(options, targetCycle);
    const productionSourceFormId = runtime.environment === 'production'
      ? requiredEnv(env, AUTOMATION_ENV.SOURCE_FORM_ID)
      : undefined;

    if (command === 'prepare') {
      result = await rollover.prepare({
        targetCycle,
        sourceFormId: productionSourceFormId,
        collaboratorPermissions: collaborators,
        preparationDays,
        dryRun: options['dry-run'] === true,
        faultInjection: options.fault,
      });
    } else if (command === 'activate') {
      result = await rollover.activate({
        targetCycle,
        sourceFormId: productionSourceFormId,
        collaboratorPermissions: collaborators,
        preparationDays,
        faultInjection: options.fault,
        headSha: activation.headSha,
      });
    } else if (command === 'verify') {
      result = await rollover.verify({
        targetCycle,
        sourceFormId: productionSourceFormId,
        mode: options.mode ?? 'prepared',
        collaboratorPermissions: collaborators,
      });
    } else {
      result = await rollover.cleanup({ targetCycle });
    }
  }

  return emitAutomationResult(result, { env, output, append });
}

async function main() {
  try {
    await runAutomationCli(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`progress-prizes: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}
