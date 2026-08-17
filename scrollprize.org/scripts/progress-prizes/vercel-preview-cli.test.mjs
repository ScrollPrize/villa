import assert from 'node:assert/strict';
import { mkdtemp, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import {
  PreviewCliError,
  readGitHubEvent,
  runVercelPreviewCli,
} from './vercel-preview-cli.mjs';

const sha = '0123456789abcdef0123456789abcdef01234567';
const ref = 'refs/heads/codex/progress-prize-smoke-20260720';

function environment(overrides = {}) {
  return {
    GITHUB_EVENT_NAME: 'repository_dispatch',
    GITHUB_EVENT_PATH: '/runner/event.json',
    GITHUB_REPOSITORY: 'ScrollPrize/villa',
    GITHUB_REPOSITORY_ID: '890972577',
    GITHUB_REPOSITORY_OWNER_ID: '121906140',
    GITHUB_TOKEN: 'private-github-token',
    GITHUB_STEP_SUMMARY: '/runner/summary.md',
    GITHUB_OUTPUT: '/runner/output.txt',
    PROGRESS_PRIZE_EXPECTED_SHA: sha,
    PROGRESS_PRIZE_EXPECTED_REF: ref,
    PROGRESS_PRIZE_EXPECTED_CYCLE: '2026-08',
    VERCEL_PROJECT_ID: 'private-project-identifier',
    VERCEL_AUTOMATION_BYPASS_SECRET: 'private-vercel-bypass',
    ...overrides,
  };
}

function event(overrides = {}) {
  return {
    action: 'vercel.deployment.ready',
    client_payload: { deployment: 'payload' },
    repository: {
      full_name: 'ScrollPrize/villa',
      id: 890972577,
      owner: { id: 121906140 },
    },
    ...overrides,
  };
}

test('binds trusted environment values to the verifier and writes only safe status data', async () => {
  const env = environment();
  const dispatch = event();
  const reads = [];
  const verificationCalls = [];
  const appends = [];

  const result = await runVercelPreviewCli({
    env,
    readEventImpl: async (path) => {
      reads.push(path);
      return dispatch;
    },
    verifyImpl: async (options) => {
      verificationCalls.push(options);
      return {
        ok: true,
        cycle: '2026-08',
        responderUri: 'https://forms.gle/PublicSmokeForm',
        sha,
        ref,
      };
    },
    appendFileImpl: async (path, contents) => appends.push({ path, contents }),
  });

  assert.deepEqual(result, { ok: true, status: 'verified', cycle: '2026-08' });
  assert.deepEqual(reads, ['/runner/event.json']);
  assert.equal(verificationCalls.length, 1);
  assert.equal(verificationCalls[0].event, dispatch);
  assert.equal(verificationCalls[0].expectedProjectId, env.VERCEL_PROJECT_ID);
  assert.equal(verificationCalls[0].expectedSha, sha);
  assert.equal(verificationCalls[0].expectedRef, ref);
  assert.equal(verificationCalls[0].expectedCycle, '2026-08');
  assert.deepEqual(verificationCalls[0].github, {
    owner: 'ScrollPrize',
    repo: 'villa',
    githubToken: env.GITHUB_TOKEN,
  });
  assert.equal(
    verificationCalls[0].protectionBypassSecret,
    env.VERCEL_AUTOMATION_BYPASS_SECRET,
  );

  assert.deepEqual(appends.map(({ path }) => path), [
    '/runner/summary.md',
    '/runner/output.txt',
  ]);
  assert.match(appends[0].contents, /Status: verified/);
  assert.match(appends[0].contents, /Cycle: `2026-08`/);
  assert.equal(appends[1].contents, 'status=verified\ncycle=2026-08\n');
  const written = appends.map(({ contents }) => contents).join('\n');
  assert.doesNotMatch(
    written,
    /private|0123456789abcdef|progress-prize-smoke|forms\.gle|890972577|121906140/,
  );
});

test('requires every sensitive binding and rejects output-injecting optional cycles before reading the event', async () => {
  for (const missing of [
    'GITHUB_EVENT_PATH',
    'GITHUB_TOKEN',
    'PROGRESS_PRIZE_EXPECTED_SHA',
    'PROGRESS_PRIZE_EXPECTED_REF',
    'VERCEL_PROJECT_ID',
    'VERCEL_AUTOMATION_BYPASS_SECRET',
  ]) {
    const env = environment();
    delete env[missing];
    let read = false;
    await assert.rejects(
      runVercelPreviewCli({
        env,
        readEventImpl: async () => {
          read = true;
          return event();
        },
        verifyImpl: async () => ({ ok: true, cycle: '2026-08' }),
      }),
      (error) => error instanceof PreviewCliError && error.code === 'CONFIG',
    );
    assert.equal(read, false);
  }

  let read = false;
  await assert.rejects(
    runVercelPreviewCli({
      env: environment({ PROGRESS_PRIZE_EXPECTED_CYCLE: '2026-08\nleak=value' }),
      readEventImpl: async () => {
        read = true;
        return event();
      },
      verifyImpl: async () => ({ ok: true, cycle: '2026-08\nleak=value' }),
    }),
    (error) => error instanceof PreviewCliError && error.code === 'CONFIG',
  );
  assert.equal(read, false);
});

test('requires the exact repository, event type, and immutable IDs when supplied', async () => {
  const cases = [
    { env: environment({ GITHUB_EVENT_NAME: 'pull_request' }), event: event() },
    { env: environment({ GITHUB_REPOSITORY: 'attacker/villa' }), event: event() },
    { env: environment({ GITHUB_REPOSITORY_ID: '999999999' }), event: event() },
    { env: environment({ GITHUB_REPOSITORY_OWNER_ID: '999999999' }), event: event() },
    { env: environment(), event: event({ repository: { full_name: 'attacker/villa' } }) },
    { env: environment(), event: event({ repository: { id: 999999999 } }) },
    { env: environment(), event: event({ repository: { owner: { id: 999999999 } } }) },
  ];

  for (const testCase of cases) {
    let verified = false;
    await assert.rejects(
      runVercelPreviewCli({
        env: testCase.env,
        readEventImpl: async () => testCase.event,
        verifyImpl: async () => {
          verified = true;
          return { ok: true, cycle: '2026-08' };
        },
      }),
      (error) => error instanceof PreviewCliError && error.code === 'CONTEXT',
    );
    assert.equal(verified, false);
  }

  const withoutOptionalIds = environment({
    GITHUB_REPOSITORY_ID: undefined,
    GITHUB_REPOSITORY_OWNER_ID: undefined,
    GITHUB_STEP_SUMMARY: undefined,
    GITHUB_OUTPUT: undefined,
    PROGRESS_PRIZE_EXPECTED_CYCLE: undefined,
  });
  let verifierOptions;
  const result = await runVercelPreviewCli({
    env: withoutOptionalIds,
    readEventImpl: async () => event({ repository: undefined }),
    verifyImpl: async (options) => {
      verifierOptions = options;
      return { ok: true, cycle: '2026-08' };
    },
  });
  assert.equal(result.ok, true);
  assert.equal(verifierOptions.expectedCycle, undefined);
  assert.equal(result.cycle, '2026-08');
});

test('redacts verifier, event reader, and runner-writer failures', async () => {
  const privateValues = [
    'private-github-token',
    'private-project-identifier',
    'private-vercel-bypass',
    sha,
    ref,
  ];
  const privateMessage = privateValues.join(' / ');

  await assert.rejects(
    runVercelPreviewCli({
      env: environment(),
      readEventImpl: async () => event(),
      verifyImpl: async () => { throw new Error(privateMessage); },
    }),
    (error) => {
      assert.ok(error instanceof PreviewCliError);
      assert.equal(error.code, 'VERIFY');
      const serialized = `${error}\n${JSON.stringify(error)}`;
      for (const value of privateValues) assert.equal(serialized.includes(value), false);
      return true;
    },
  );

  await assert.rejects(
    runVercelPreviewCli({
      env: environment(),
      readEventImpl: async () => { throw new Error(privateMessage); },
      verifyImpl: async () => ({ ok: true, cycle: '2026-08' }),
    }),
    (error) => {
      assert.ok(error instanceof PreviewCliError);
      assert.equal(error.code, 'EVENT');
      assert.equal(`${error}`.includes(privateMessage), false);
      return true;
    },
  );

  await assert.rejects(
    runVercelPreviewCli({
      env: environment(),
      readEventImpl: async () => event(),
      verifyImpl: async () => ({ ok: true, cycle: '2026-08' }),
      appendFileImpl: async () => { throw new Error(privateMessage); },
    }),
    (error) => {
      assert.ok(error instanceof PreviewCliError);
      assert.equal(error.code, 'SUMMARY');
      assert.equal(`${error}`.includes(privateMessage), false);
      return true;
    },
  );
});

test('reads only a bounded regular absolute event file and rejects symlinks', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'progress-prize-event-'));
  try {
    const eventPath = join(directory, 'event.json');
    await writeFile(eventPath, JSON.stringify(event()), 'utf8');
    const parsed = await readGitHubEvent(eventPath);
    assert.equal(parsed.action, 'vercel.deployment.ready');

    await assert.rejects(
      readGitHubEvent('relative-event.json'),
      (error) => error instanceof PreviewCliError && error.code === 'EVENT',
    );

    const invalidPath = join(directory, 'invalid.json');
    await writeFile(invalidPath, '[]', 'utf8');
    await assert.rejects(
      readGitHubEvent(invalidPath),
      (error) => error instanceof PreviewCliError && error.code === 'EVENT',
    );

    const symlinkPath = join(directory, 'event-link.json');
    await symlink(eventPath, symlinkPath);
    await assert.rejects(
      readGitHubEvent(symlinkPath),
      (error) => error instanceof PreviewCliError && error.code === 'EVENT',
    );
  } finally {
    await rm(directory, { recursive: true });
  }
});
