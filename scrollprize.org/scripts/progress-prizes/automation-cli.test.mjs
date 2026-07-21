import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { Buffer } from 'node:buffer';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

import {
  ResponderUrlResolutionError,
  AUTOMATION_ERROR_FALLBACK,
  createExactShaActivationGate,
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
  folder: 'private-staging-folder-id',
  drive: 'private-staging-drive-id',
  form: 'private-production-editor-form-id',
  account: 'staging-bot@private.example',
  admin: 'staging-break-glass@private.example',
  group: 'staging-editors@private.example',
  archive: 'private-staging-archive-id',
});
const RESPONDER = 'https://docs.google.com/forms/d/e/publicResponder123/viewform';
const SHORT_RESPONDER = 'https://forms.gle/PublicShort123';
const SHA = '0123456789abcdef0123456789abcdef01234567';

function productionEnv(overrides = {}) {
  return {
    GOOGLE_ACCESS_TOKEN: PRIVATE.token,
    GITHUB_EVENT_NAME: 'workflow_dispatch',
    PROGRESS_PRIZE_ENVIRONMENT: 'production',
    PROGRESS_PRIZE_FOLDER_ID: 'private-production-folder-id',
    PROGRESS_PRIZE_DRIVE_ID: 'private-production-drive-id',
    PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL: 'production-break-glass@private.example',
    PROGRESS_PRIZE_SERVICE_ACCOUNT_EMAIL: 'production-bot@private.example',
    PROGRESS_PRIZE_SOURCE_FORM_ID: PRIVATE.form,
    PROGRESS_PRIZE_EDITOR_GROUP_EMAIL: 'production-editors@private.example',
    PROGRESS_PRIZE_BRANCH: 'codex/progress-prize-2026-08',
    PROGRESS_PRIZE_TARGET_BRANCH: 'main',
    PROGRESS_PRIZE_DEFAULT_TARGET_BRANCH: 'main',
    ...overrides,
  };
}

function stagingEnv(overrides = {}) {
  return {
    GOOGLE_ACCESS_TOKEN: PRIVATE.token,
    GITHUB_EVENT_NAME: 'workflow_dispatch',
    PROGRESS_PRIZE_ENVIRONMENT: 'staging',
    PROGRESS_PRIZE_FOLDER_ID: PRIVATE.folder,
    PROGRESS_PRIZE_STAGING_FOLDER_ID: PRIVATE.folder,
    PROGRESS_PRIZE_ARCHIVE_FOLDER_ID: PRIVATE.archive,
    PROGRESS_PRIZE_DRIVE_ID: PRIVATE.drive,
    PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL: PRIVATE.admin,
    PROGRESS_PRIZE_SERVICE_ACCOUNT_EMAIL: PRIVATE.account,
    PROGRESS_PRIZE_STAGING_SERVICE_ACCOUNT_EMAIL: PRIVATE.account,
    PROGRESS_PRIZE_SOURCE_FORM_ID: PRIVATE.form,
    PROGRESS_PRIZE_EDITOR_GROUP_EMAIL: PRIVATE.group,
    PROGRESS_PRIZE_BRANCH: 'codex/progress-prize-smoke-20260720',
    PROGRESS_PRIZE_TARGET_BRANCH: 'codex/progress-prize-smoke-base-20260720',
    PROGRESS_PRIZE_DEFAULT_TARGET_BRANCH: 'main',
    PROGRESS_PRIZE_SMOKE_BRANCH_PREFIX: 'codex/progress-prize-smoke-',
    PROGRESS_PRIZE_SMOKE_DATE: '2026-07-20',
    ...overrides,
  };
}

function fakeRolloverFactory({ result, capture }) {
  return (dependencies) => {
    capture.dependencies = dependencies;
    return Object.fromEntries([
      'validate',
      'bootstrapStagingSource',
      'prepare',
      'activate',
      'verify',
      'cleanup',
    ].map((method) => [method, async (input) => {
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
  ].join(' ')), { env: stagingEnv() });

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
  ].join(' ') + '\n'), { env: stagingEnv() });
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
    formatAutomationError(hostileError, { env: stagingEnv() }),
    'progress-prizes: Automation failed without a safe diagnostic',
  );
  assert.equal(
    formatAutomationError(new Error('private-message'), {
      env: new Proxy({}, { get() { throw new Error('hostile environment'); } }),
    }),
    'progress-prizes: Automation failed without a safe diagnostic',
  );
  const multibyte = formatAutomationError(new Error('\u20ac'.repeat(1_000)), { env: stagingEnv() });
  assert.ok(Buffer.byteLength(multibyte, 'utf8') <= 600);
});

test('automation CLI subprocess emits only one bounded redacted diagnostic line', () => {
  const unknownOpaqueId = '1UnknownGoogleOpaqueIdentifier9876543210';
  const cli = fileURLToPath(new URL('./automation-cli.mjs', import.meta.url));
  const child = spawnSync(process.execPath, [cli, 'prepare', `--${unknownOpaqueId}`], {
    encoding: 'utf8',
    env: { ...process.env, ...stagingEnv() },
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

test('production-only guards reject staging controls before token access, client creation, file IO, or network', async () => {
  for (const [argv, env, expected] of [
    [
      ['validate', '--source-cycle', '2026-07', '--simulated-now', '2026-07-20T12:00:00Z'],
      productionEnv({ GOOGLE_ACCESS_TOKEN: undefined }),
      /simulated time is forbidden in production/,
    ],
    [
      ['prepare', '--target-cycle', '2026-08', '--fault', 'after-copy'],
      productionEnv({ GOOGLE_ACCESS_TOKEN: undefined }),
      /fault injection is forbidden in production/,
    ],
    [
      ['prepare', '--target-cycle', '2026-08'],
      productionEnv({
        GOOGLE_ACCESS_TOKEN: undefined,
        PROGRESS_PRIZE_TARGET_BRANCH: 'private-test-base',
      }),
      /alternate target branch/,
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

test('staging prepare passes only the configured editor group ACL and emits no private value', async () => {
  const capture = {};
  const outputs = [];
  const appends = [];
  const env = stagingEnv({ GITHUB_OUTPUT: '/private/github-output-file' });
  const summary = await runAutomationCli([
    'prepare',
    '--target-cycle', '2026-08',
    '--source-cycle', '2026-07',
    '--simulated-now', '2026-07-26T12:00:00Z',
    '--fault', 'after-copy',
    '--dry-run',
    '--file', 'smoke-prize-page.md',
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
  assert.equal(capture.input.sourceFormId, undefined);
  assert.equal(capture.input.dryRun, true);
  assert.equal(capture.input.faultInjection, 'after-copy');
  assert.equal(capture.dependencies.runtime.stagingFolderId, PRIVATE.folder);
  assert.equal(capture.dependencies.runtime.driveAdminEmail, PRIVATE.admin);
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
      env: stagingEnv({
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
      env: stagingEnv({ GITHUB_OUTPUT: '/tmp/github-output' }),
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

test('activation requires the preview-verified exact SHA and binds the gate invocation to it', async () => {
  const mismatched = stagingEnv({
    PROGRESS_PRIZE_HEAD_SHA: SHA,
    PROGRESS_PRIZE_VERIFIED_SHA: 'f'.repeat(40),
  });
  let clientCalls = 0;
  await assert.rejects(
    runAutomationCli([
      'activate',
      '--target-cycle', '2026-08',
      '--simulated-now', '2026-08-01T07:00:01Z',
    ], {
      env: mismatched,
      googleFactory: () => { clientCalls += 1; return {}; },
      output: () => {},
    }),
    /exact verified commit SHA/,
  );
  assert.equal(clientCalls, 0);

  const capture = {};
  await runAutomationCli([
    'activate',
    '--target-cycle', '2026-08',
    '--simulated-now', '2026-08-01T07:00:01Z',
    '--head-sha', SHA.toUpperCase(),
    '--verified-sha', SHA,
  ], {
    env: stagingEnv(),
    googleFactory: () => ({}),
    rolloverFactory: fakeRolloverFactory({
      capture,
      result: async (_method, input, dependencies) => {
        assert.equal(await dependencies.activationGate({ headSha: input.headSha }), true);
        assert.equal(await dependencies.activationGate({ headSha: 'f'.repeat(40) }), false);
        return {
          status: 'active',
          sourceCycle: '2026-07',
          targetCycle: '2026-08',
          responderUri: RESPONDER,
        };
      },
    }),
    output: () => {},
  });
  assert.equal(capture.method, 'activate');
  assert.equal(capture.input.headSha, SHA);
});

test('exact SHA gate accepts env-only workflow wiring', async () => {
  const activation = createExactShaActivationGate({
    PROGRESS_PRIZE_HEAD_SHA: SHA.toUpperCase(),
    PROGRESS_PRIZE_VERIFIED_SHA: SHA,
  });
  assert.equal(activation.headSha, SHA);
  assert.equal(await activation.gate({ headSha: SHA }), true);
  assert.equal(await activation.gate({ headSha: '0'.repeat(40) }), false);
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
});

test('bootstrap, verify, and cleanup commands map to the corresponding service operations', async () => {
  for (const [argv, method, expectedMode] of [
    [
      ['bootstrap', '--source-cycle', '2026-07', '--dry-run'],
      'bootstrapStagingSource',
      undefined,
    ],
    [
      ['verify', '--target-cycle', '2026-08', '--mode', 'active'],
      'verify',
      'active',
    ],
    [
      ['cleanup', '--target-cycle', '2026-08'],
      'cleanup',
      undefined,
    ],
  ]) {
    const capture = {};
    await runAutomationCli(argv, {
      env: stagingEnv(),
      googleFactory: () => ({}),
      rolloverFactory: fakeRolloverFactory({
        capture,
        result: { status: 'valid', targetCycle: '2026-08' },
      }),
      output: () => {},
    });
    assert.equal(capture.method, method);
    if (expectedMode !== undefined) assert.equal(capture.input.mode, expectedMode);
  }
});
