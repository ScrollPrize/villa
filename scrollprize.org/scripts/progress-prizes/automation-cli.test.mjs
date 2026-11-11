import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { Buffer } from 'node:buffer';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
  ResponderUrlResolutionError,
  AUTOMATION_ERROR_FALLBACK,
  createFilePageFacade,
  formatAutomationError,
  resolveGoogleResponderUri,
  runAutomationCli,
  safeAutomationDiagnostic,
} from './automation-cli.mjs';
import { runCli } from './cli.mjs';
import { PROGRESS_PRIZE_MARKERS } from './markdown.mjs';

const PRIVATE = Object.freeze({
  token: 'private-wif-access-token',
  folder: 'private-production-folder-id',
  drive: 'private-production-drive-id',
  form: 'private-production-editor-form-id',
  account: 'production-bot@private.example',
  reader: 'staging-bot@private.example',
  admin: 'production-break-glass@private.example',
  group: 'production-editors@private.example',
});
const RESPONDER = 'https://docs.google.com/forms/d/e/publicResponder123/viewform';
const SHORT_RESPONDER = 'https://forms.gle/PublicShort123';

function productionEnv(overrides = {}) {
  return {
    GOOGLE_ACCESS_TOKEN: PRIVATE.token,
    GITHUB_EVENT_NAME: 'workflow_dispatch',
    PROGRESS_PRIZE_ENVIRONMENT: 'production',
    PROGRESS_PRIZE_FOLDER_ID: PRIVATE.folder,
    PROGRESS_PRIZE_DRIVE_ID: PRIVATE.drive,
    PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL: PRIVATE.admin,
    PROGRESS_PRIZE_SERVICE_ACCOUNT_EMAIL: PRIVATE.account,
    PROGRESS_PRIZE_STAGING_SERVICE_ACCOUNT_EMAIL: PRIVATE.reader,
    PROGRESS_PRIZE_SOURCE_FORM_ID: PRIVATE.form,
    PROGRESS_PRIZE_EDITOR_GROUP_EMAIL: PRIVATE.group,
    PROGRESS_PRIZE_BRANCH: 'codex/progress-prize-2026-08',
    PROGRESS_PRIZE_TARGET_BRANCH: 'main',
    PROGRESS_PRIZE_DEFAULT_TARGET_BRANCH: 'main',
    ...overrides,
  };
}

function fakeRolloverFactory({ result, capture }) {
  return (dependencies) => {
    capture.dependencies = dependencies;
    return Object.fromEntries(['validate', 'prepare', 'verify'].map((method) => [method, async (input) => {
      capture.method = method;
      capture.input = input;
      return typeof result === 'function' ? result(method, input, dependencies) : result;
    }]));
  };
}

function pageMarkdown() {
  return [
    '## Progress Prizes',
    PROGRESS_PRIZE_MARKERS.deadlineStart,
    'Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is 11:59pm Pacific, July 31st, 2026!',
    PROGRESS_PRIZE_MARKERS.deadlineEnd,
    PROGRESS_PRIZE_MARKERS.formStart,
    `[Submission Form](${SHORT_RESPONDER})`,
    PROGRESS_PRIZE_MARKERS.formEnd,
    '## Terms and Conditions',
  ].join('\n');
}

test('resolves a forms.gle URL without following redirects and canonicalizes the public destination', async () => {
  const calls = [];
  const resolved = await resolveGoogleResponderUri(`${SHORT_RESPONDER}/`, {
    fetchImpl: async (url, options) => {
      calls.push({ url: new URL(url), options });
      return new Response(null, {
        status: 302,
        headers: { location: `${RESPONDER}/?usp=sf_link` },
      });
    },
  });

  assert.equal(resolved, RESPONDER);
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url.hostname, 'forms.gle');
  assert.equal(calls[0].options.redirect, 'manual');
  assert.equal(calls[0].options.headers.authorization, undefined);
});

test('short URL resolution rejects unsafe redirect targets with a fixed log-safe error', async () => {
  const privateDestination = 'https://accounts.google.com/private-editor-token';
  await assert.rejects(
    resolveGoogleResponderUri(SHORT_RESPONDER, {
      fetchImpl: async () => new Response(null, {
        status: 302,
        headers: { location: privateDestination },
      }),
    }),
    (error) => {
      assert.ok(error instanceof ResponderUrlResolutionError);
      assert.doesNotMatch(`${error}\n${JSON.stringify(error)}`, /accounts|private-editor-token/);
      return true;
    },
  );
});

test('automation failure diagnostics are single-line, bounded, and fully redacted', () => {
  const unknownOpaqueId = '1UnknownGoogleOpaqueIdentifier9876543210';
  const unconfiguredEmail = 'private-user@localhost';
  const diagnostic = formatAutomationError(new Error([
    'Google API request failed',
    PRIVATE.token,
    PRIVATE.form,
    PRIVATE.folder,
    PRIVATE.account,
    `https://docs.google.com/forms/d/${unknownOpaqueId}/edit`,
    unknownOpaqueId,
    unconfiguredEmail,
    '\n\u0085\u2028::warning title=unsafe::injected',
    'multibyte-\u00e9\u00e9\u00e9\u00e9',
    'x'.repeat(10_000),
  ].join(' ')), { env: productionEnv() });

  assert.match(diagnostic, /^progress-prizes: /);
  assert.ok(Buffer.byteLength(diagnostic, 'utf8') <= 600);
  assert.doesNotMatch(diagnostic, /[\p{Cc}\p{Cf}\p{Zl}\p{Zp}]/u);
  assert.doesNotMatch(diagnostic, /https?:\/\//);
  assert.doesNotMatch(diagnostic, /::/);
  assert.equal(diagnostic.includes(unconfiguredEmail), false);
  assert.equal(diagnostic.includes(unknownOpaqueId), false);
  for (const privateValue of Object.values(PRIVATE)) {
    assert.equal(diagnostic.includes(privateValue), false);
  }
  assert.equal(safeAutomationDiagnostic(Buffer.from(`${diagnostic}\n`)), diagnostic);
  assert.equal(
    safeAutomationDiagnostic(Buffer.from(`${diagnostic}\nsecond line\n`)),
    AUTOMATION_ERROR_FALLBACK,
  );
  assert.equal(
    safeAutomationDiagnostic(Buffer.alloc(602, 0x61)),
    AUTOMATION_ERROR_FALLBACK,
  );
  assert.equal(
    safeAutomationDiagnostic(Buffer.from([0xff, 0x0a])),
    AUTOMATION_ERROR_FALLBACK,
  );

  const forgedOpaqueId = '-UnknownGoogleOpaqueIdentifier9876543210-';
  const forged = safeAutomationDiagnostic(Buffer.from([
    'progress-prizes:',
    PRIVATE.token,
    PRIVATE.account,
    `https://docs.google.com/forms/d/${forgedOpaqueId}/edit`,
    forgedOpaqueId,
    'private-user@localhost',
    '::warning title=unsafe::injected',
  ].join(' ') + '\n'), { env: productionEnv() });
  assert.match(forged, /^progress-prizes: /);
  assert.doesNotMatch(forged, /https?:\/\/|::|@/);
  assert.equal(forged.includes(PRIVATE.token), false);
  assert.equal(forged.includes(forgedOpaqueId), false);
});

test('automation failure formatter fails closed for hostile errors and environments', () => {
  const hostileError = new Proxy(new Error('private-message'), {
    get(target, property, receiver) {
      if (property === 'message') throw new Error('hostile getter');
      return Reflect.get(target, property, receiver);
    },
  });
  assert.equal(
    formatAutomationError(hostileError, { env: productionEnv() }),
    'progress-prizes: Automation failed without a safe diagnostic',
  );
  assert.equal(
    formatAutomationError(new Error('private-message'), {
      env: new Proxy({}, { get() { throw new Error('hostile environment'); } }),
    }),
    'progress-prizes: Automation failed without a safe diagnostic',
  );
  const multibyte = formatAutomationError(new Error('\u20ac'.repeat(1_000)), { env: productionEnv() });
  assert.ok(Buffer.byteLength(multibyte, 'utf8') <= 600);
});

test('automation CLI subprocess emits only one bounded redacted diagnostic line', () => {
  const unknownOpaqueId = '1UnknownGoogleOpaqueIdentifier9876543210';
  const cli = fileURLToPath(new URL('./automation-cli.mjs', import.meta.url));
  const child = spawnSync(process.execPath, [cli, 'prepare', `--${unknownOpaqueId}`], {
    encoding: 'utf8',
    env: { ...process.env, ...productionEnv() },
  });
  assert.equal(child.status, 1);
  assert.equal(child.stdout, '');
  assert.match(child.stderr, /^progress-prizes: [^\r\n]+\n$/);
  assert.ok(Buffer.byteLength(child.stderr.trimEnd(), 'utf8') <= 600);
  assert.equal(child.stderr.includes(unknownOpaqueId), false);
});

test('file page facade keeps file IO local and resolves canonical URLs without fetch', async () => {
  const calls = [];
  const page = createFilePageFacade({
    pagePath: 'relative-prize-page.md',
    read: async (...args) => {
      calls.push(['read', ...args]);
      return pageMarkdown();
    },
    write: async (...args) => calls.push(['write', ...args]),
    fetchImpl: async () => {
      throw new Error('canonical URLs must not fetch');
    },
  });
  assert.equal(await page.read(), pageMarkdown());
  await page.write('updated');
  assert.equal(await page.resolveResponderUri(`${RESPONDER}?usp=sf_link`), RESPONDER);
  assert.deepEqual(calls.map(([operation]) => operation), ['read', 'write']);
  assert.equal(calls[0][2], 'utf8');
  assert.equal(calls[1][2], 'updated');
  assert.equal(calls[1][3], 'utf8');
});

test('production-only guards reject retired controls before token access, client creation, file IO, or network', async () => {
  for (const [argv, env, expected] of [
    [
      ['validate', '--source-cycle', '2026-07', '--simulated-now', '2026-07-20T12:00:00Z'],
      productionEnv({ GOOGLE_ACCESS_TOKEN: undefined }),
      /simulated time is forbidden in production/,
    ],
    [
      ['prepare', '--target-cycle', '2026-08', '--environment', 'staging'],
      productionEnv({ GOOGLE_ACCESS_TOKEN: undefined }),
      /run only in production/,
    ],
    [
      ['prepare', '--target-cycle', '2026-08'],
      productionEnv({
        GOOGLE_ACCESS_TOKEN: undefined,
        PROGRESS_PRIZE_TARGET_BRANCH: 'private-test-base',
      }),
      /alternate target branch/,
    ],
    [
      ['prepare', '--target-cycle', '2026-08', '--fault', 'after-copy'],
      productionEnv({ GOOGLE_ACCESS_TOKEN: undefined }),
      /Unknown option --fault/,
    ],
    [
      ['verify', '--target-cycle', '2026-08', '--mode', 'active'],
      productionEnv({ GOOGLE_ACCESS_TOKEN: undefined }),
      /Unknown option --mode/,
    ],
    [
      ['verify', '--target-cycle', '2026-08', '--dry-run'],
      productionEnv({ GOOGLE_ACCESS_TOKEN: undefined }),
      /--dry-run is accepted only by prepare/,
    ],
    [
      ['prepare', '--target-cycle', '2026-08', '--source-cycle', '2026-06'],
      productionEnv({ GOOGLE_ACCESS_TOKEN: undefined }),
      /Source cycle must be 2026-07/,
    ],
    [
      ['activate', '--target-cycle', '2026-08'],
      productionEnv({ GOOGLE_ACCESS_TOKEN: undefined }),
      /Command must be one of/,
    ],
    [
      ['reconcile-active', '--target-cycle', '2026-08'],
      productionEnv({ GOOGLE_ACCESS_TOKEN: undefined }),
      /Command must be one of/,
    ],
  ]) {
    let clientCalls = 0;
    let fileCalls = 0;
    let fetchCalls = 0;
    await assert.rejects(
      runAutomationCli(argv, {
        env,
        googleFactory: () => { clientCalls += 1; return {}; },
        read: async () => { fileCalls += 1; return ''; },
        write: async () => { fileCalls += 1; },
        fetchImpl: async () => { fetchCalls += 1; return new Response(); },
        output: () => {},
      }),
      expected,
    );
    assert.equal(clientCalls, 0);
    assert.equal(fileCalls, 0);
    assert.equal(fetchCalls, 0);
  }
});

test('prepare passes only the configured source and editor group ACL and emits no private value', async () => {
  const capture = {};
  const outputs = [];
  const appends = [];
  const env = productionEnv({ GITHUB_OUTPUT: '/private/github-output-file' });
  const summary = await runAutomationCli([
    'prepare',
    '--target-cycle', '2026-08',
    '--source-cycle', '2026-07',
    '--dry-run',
    '--file', 'prize-page.md',
  ], {
    env,
    googleFactory: ({ accessToken }) => {
      assert.equal(accessToken, PRIVATE.token);
      return { fake: true };
    },
    rolloverFactory: fakeRolloverFactory({
      capture,
      result: {
        action: 'prepare',
        status: 'prepared',
        sourceCycle: '2026-07',
        targetCycle: '2026-08',
        responderUri: RESPONDER,
        privateEcho: `${PRIVATE.group} ${PRIVATE.folder} ${PRIVATE.admin} ${PRIVATE.token}`,
      },
    }),
    output: (value) => outputs.push(value),
    append: async (...args) => appends.push(args),
  });

  assert.equal(capture.method, 'prepare');
  assert.deepEqual(capture.input.collaboratorPermissions, [{
    type: 'group',
    role: 'writer',
    emailAddress: PRIVATE.group,
  }]);
  assert.equal(capture.input.sourceFormId, PRIVATE.form);
  assert.equal(capture.input.dryRun, true);
  assert.equal(capture.input.preparationDays, 7);
  assert.equal(capture.dependencies.runtime.driveAdminEmail, PRIVATE.admin);
  assert.equal(capture.dependencies.page.file.endsWith('prize-page.md'), true);
  assert.equal(summary.privateEcho, '[REDACTED] [REDACTED] [REDACTED] [REDACTED]');
  assert.equal(outputs.length, 1);
  for (const privateValue of Object.values(PRIVATE)) {
    assert.equal(outputs[0].includes(privateValue), false);
  }
  assert.deepEqual(appends, [[
    '/private/github-output-file',
    `status=prepared\nsource_cycle=2026-07\ntarget_cycle=2026-08\nresponder_uri=${RESPONDER}\n`,
    'utf8',
  ]]);
});

test('CLI requires the private Drive administrator before token or I/O access', async () => {
  let clientCalls = 0;
  let fileCalls = 0;
  await assert.rejects(
    runAutomationCli(['prepare', '--target-cycle', '2026-08', '--dry-run'], {
      env: productionEnv({
        GOOGLE_ACCESS_TOKEN: undefined,
        PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL: undefined,
      }),
      googleFactory: () => { clientCalls += 1; return {}; },
      read: async () => { fileCalls += 1; return ''; },
      write: async () => { fileCalls += 1; },
      output: () => {},
    }),
    /Required environment variable PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL is missing/,
  );
  assert.equal(clientCalls, 0);
  assert.equal(fileCalls, 0);
});

test('GitHub output fields use fixed keys and reject newline injection before writing', async () => {
  const appends = [];
  const capture = {};
  await assert.rejects(
    runAutomationCli(['prepare', '--target-cycle', '2026-08', '--dry-run'], {
      env: productionEnv({ GITHUB_OUTPUT: '/tmp/github-output' }),
      googleFactory: () => ({}),
      rolloverFactory: fakeRolloverFactory({
        capture,
        result: {
          status: 'prepared',
          targetCycle: '2026-08\nleaked=true',
          responderUri: RESPONDER,
        },
      }),
      output: () => {},
      append: async (...args) => appends.push(args),
    }),
    /expected YYYY-MM/,
  );
  assert.deepEqual(appends, []);
});

test('top-level CLI keeps page validate compatibility and routes source-cycle validate to Google', async () => {
  const coreOutputs = [];
  const core = await runCli(['validate', '--file', 'page.md'], {
    read: async () => pageMarkdown(),
    output: (value) => coreOutputs.push(value),
  });
  assert.equal(core.page.cycle, '2026-07');

  const capture = {};
  await runCli(['validate', '--source-cycle', '2026-07'], {
    env: productionEnv(),
    googleFactory: () => ({}),
    rolloverFactory: fakeRolloverFactory({
      capture,
      result: { status: 'valid', cycle: '2026-07', responderUri: RESPONDER },
    }),
    output: () => {},
  });
  assert.equal(capture.method, 'validate');
  assert.equal(capture.input.sourceFormId, PRIVATE.form);
  assert.equal(capture.dependencies.runtime.stagingServiceAccountEmail, PRIVATE.reader);
});

test('verify maps to the read-only service operation with the protected source', async () => {
  const capture = {};
  await runAutomationCli(['verify', '--target-cycle', '2026-08', '--source-cycle', '2026-07'], {
    env: productionEnv({ PROGRESS_PRIZE_BRANCH: 'main' }),
    googleFactory: () => ({}),
    rolloverFactory: fakeRolloverFactory({
      capture,
      result: { status: 'valid', sourceCycle: '2026-07', targetCycle: '2026-08' },
    }),
    output: () => {},
  });
  assert.equal(capture.method, 'verify');
  assert.equal(capture.input.sourceFormId, PRIVATE.form);
  assert.deepEqual(capture.input.collaboratorPermissions, [{
    type: 'group',
    role: 'writer',
    emailAddress: PRIVATE.group,
  }]);
});
