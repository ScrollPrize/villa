import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const googleJobs = [
  'validate',
  'dry-run',
  'prepare-google',
  'activate-production',
  'verify-completed-activation',
  'verify-prepared',
  'verify-active',
];
const githubOnlyJobs = [
  'preflight',
  'prepare-page',
  'activation-state',
  'activation-page-state',
  'refresh-page',
  'activation-binding',
  'activation-gate',
  'activation-approval',
  'merge',
  'verify-prepared-state',
  'report-scheduled-failure',
];
const protectedMappings = new Map([
  ['workload-identity-provider', 'GOOGLE_WORKLOAD_IDENTITY_PROVIDER'],
  ['service-account-email', 'GOOGLE_SERVICE_ACCOUNT_EMAIL'],
  ['drive-admin-email', 'PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL'],
  ['drive-id', 'PROGRESS_PRIZE_DRIVE_ID'],
  ['folder-id', 'PROGRESS_PRIZE_FOLDER_ID'],
  ['source-form-id', 'PROGRESS_PRIZE_SOURCE_FORM_ID'],
  ['editor-group-email', 'PROGRESS_PRIZE_EDITOR_GROUP_EMAIL'],
]);

async function productionWorkflow() {
  return readFile(
    resolve(repositoryRoot, '.github/workflows/progress-prizes-production.yml'),
    'utf8',
  );
}

async function googleAction() {
  return readFile(
    resolve(repositoryRoot, '.github/actions/progress-prizes-google/action.yml'),
    'utf8',
  );
}

function jobBlock(source, name) {
  const startMarker = `  ${name}:\n`;
  const start = source.indexOf(startMarker);
  assert.notEqual(start, -1, `missing ${name} job`);
  const remainder = source.slice(start + startMarker.length);
  const next = remainder.search(/^  [a-z0-9-]+:\n/m);
  return next === -1
    ? source.slice(start)
    : source.slice(start, start + startMarker.length + next);
}

function jobNames(source) {
  const jobs = source.slice(source.indexOf('\njobs:\n') + '\njobs:\n'.length);
  return [...jobs.matchAll(/^  ([a-z][a-z0-9-]*):\n/gm)].map((match) => match[1]);
}

function literalRunScripts(source) {
  const lines = source.split('\n');
  const scripts = [];
  for (let index = 0; index < lines.length; index += 1) {
    const run = lines[index].match(/^(\s*)run:\s*\|[-+]?\s*$/);
    if (!run) continue;
    let firstContent = index + 1;
    while (firstContent < lines.length && lines[firstContent].trim() === '') firstContent += 1;
    const contentIndent = lines[firstContent]?.match(/^ */)?.[0].length ?? 0;
    assert.ok(contentIndent > run[1].length, `run block at line ${index + 1} has no body`);
    const body = [];
    for (let cursor = index + 1; cursor < lines.length; cursor += 1) {
      const line = lines[cursor];
      const indentation = line.match(/^ */)?.[0].length ?? 0;
      if (line.trim() !== '' && indentation < contentIndent) break;
      body.push(line.trim() === '' ? '' : line.slice(contentIndent));
    }
    scripts.push(`${body.join('\n')}\n`);
  }
  return scripts;
}

test('production accepts only a secret-free manual dispatch contract', async () => {
  const source = await productionWorkflow();
  const inputSection = source.slice(0, source.indexOf('\npermissions:'));
  const preflight = jobBlock(source, 'preflight');

  assert.match(source, /^on:\n  workflow_dispatch:/m);
  assert.doesNotMatch(source, /workflow_call:|pull_request(?:_target)?:/);
  assert.doesNotMatch(inputSection, /^  (?:schedule|repository_dispatch):/m);
  assert.doesNotMatch(inputSection, /^\s+(?:simulated-now|fault|base-branch|head-sha):/m);
  assert.match(source, /^concurrency:\n  group: progress-prizes-production\n  cancel-in-progress: false$/m);
  assert.match(preflight, /permissions: \{\}/);
  assert.doesNotMatch(preflight, /environment:|id-token|secrets\./);
  assert.match(preflight, /assert\.equal\(process\.env\.REPOSITORY, 'ScrollPrize\/villa'\)/);
  assert.match(preflight, /assert\.equal\(process\.env\.REPOSITORY_ID, '890972577'\)/);
  assert.match(preflight, /assert\.equal\(process\.env\.REPOSITORY_OWNER_ID, '121906140'\)/);
  assert.match(preflight, /assert\.equal\(process\.env\.REF, 'refs\/heads\/main'\)/);
  assert.match(preflight, /assert\.equal\(process\.env\.EVENT_NAME, 'workflow_dispatch'\)/);
  assert.match(preflight, /process\.env\.TARGET_CYCLE, target/);
  assert.match(preflight, /\^\[1-9\]\\d\*\$/);
  assert.deepEqual(
    new Set(jobNames(source)),
    new Set([...googleJobs, ...githubOnlyJobs]),
    'every production job must be assigned to the Google or secret-free control plane',
  );
});

test('production preflight executes a strict valid and invalid control matrix', async () => {
  const source = await productionWorkflow();
  const [guard] = literalRunScripts(jobBlock(source, 'preflight'));
  assert.ok(guard);
  const valid = {
    REPOSITORY: 'ScrollPrize/villa',
    REPOSITORY_ID: '890972577',
    REPOSITORY_OWNER_ID: '121906140',
    REF: 'refs/heads/main',
    HEAD_SHA: 'a'.repeat(40),
    EVENT_NAME: 'workflow_dispatch',
    OPERATION: 'activate',
    SOURCE_CYCLE: '2026-07',
    TARGET_CYCLE: '2026-08',
    VERIFY_MODE: 'prepared',
    REQUEST_ID: '',
  };
  const execute = (overrides = {}) => spawnSync('bash', ['--noprofile', '--norc', '-c', guard], {
    env: { ...process.env, ...valid, ...overrides },
    encoding: 'utf8',
  });
  assert.equal(execute().status, 0);
  for (const overrides of [
    { REPOSITORY_ID: '1' },
    { REF: 'refs/heads/feature' },
    { EVENT_NAME: 'schedule' },
    { OPERATION: 'cleanup' },
    { TARGET_CYCLE: '2026-09' },
    { REQUEST_ID: 'untrusted' },
  ]) {
    assert.notEqual(execute(overrides).status, 0, JSON.stringify(overrides));
  }
});

test('every production Google job is literal, exact-SHA, and explicitly secret mapped', async () => {
  const source = await productionWorkflow();

  assert.doesNotMatch(source, /progress-prizes-google\.yml|secrets:\s*inherit/);
  assert.doesNotMatch(source, /environment:\s*\$\{\{/);
  for (const name of googleJobs) {
    const job = jobBlock(source, name);
    assert.match(job, /runs-on: ubuntu-24\.04/);
    assert.match(job, /environment: progress-prizes-production/);
    assert.match(job, /contents: read/);
    assert.match(job, /id-token: write/);
    assert.match(job, /ref: \$\{\{ github\.sha \}\}/);
    assert.match(job, /persist-credentials: false/);
    assert.match(job, /uses: \.\/\.github\/actions\/progress-prizes-google/);
    assert.match(job, /^          environment: production$/m);
    for (const [input, secret] of protectedMappings) {
      assert.match(
        job,
        new RegExp(`^          ${input}: \\$\\{\\{ secrets\\.${secret} \\}\\}$`, 'm'),
      );
    }
    assert.doesNotMatch(job, /staging-folder-id|archive-folder-id/);
  }

  const validation = jobBlock(source, 'validate');
  assert.match(
    validation,
    /^          staging-service-account-email: \$\{\{ inputs\['source-cycle'\] == '2026-07' && secrets\.PROGRESS_PRIZE_STAGING_SERVICE_ACCOUNT_EMAIL \|\| '' \}\}$/m,
  );
  for (const name of googleJobs.filter((name) => name !== 'validate')) {
    assert.doesNotMatch(jobBlock(source, name), /staging-service-account-email/);
  }
});

test('production dry-run immediately exercises a read-only public proposal', async () => {
  const production = await productionWorkflow();
  const action = await googleAction();
  const dryRun = jobBlock(production, 'dry-run');

  assert.match(dryRun, /^          operation: prepare$/m);
  assert.match(dryRun, /^          dry-run: "true"$/m);
  assert.match(dryRun, /Summarize the public no-mutation proposal/);
  assert.match(dryRun, /getCycleDeadline\(process\.env\.TARGET_CYCLE\)/);
  assert.match(dryRun, /Google mutation: `disabled`/);
  assert.match(dryRun, /Website mutation: `disabled`/);
  assert.match(
    action,
    /PROGRESS_PRIZE_ENVIRONMENT" = production \\\n\s+-a "\$OPERATION" = prepare \\\n\s+-a "\$DRY_RUN" = true/,
  );
  assert.match(action, /arguments\+=\(--preparation-days 31\)/);
  assert.match(
    action,
    /if: inputs\.operation == 'validate' \|\| inputs\.operation == 'verify' \|\| inputs\['dry-run'\] == 'true'/,
  );
});

test('normal preparation no-ops safely outside the seven-day window', async () => {
  const production = await productionWorkflow();
  const google = jobBlock(production, 'prepare-google');
  const page = jobBlock(production, 'prepare-page');

  assert.doesNotMatch(google, /preparation-days/);
  assert.match(
    page,
    /inputs\.operation == 'prepare' &&\n\s+needs\.prepare-google\.outputs\['responder-uri'\] != ''/,
  );
  assert.match(page, /responder-uri: \$\{\{ needs\.prepare-google\.outputs\['responder-uri'\] \}\}/);
  assert.match(page, /head-branch: codex\/progress-prize-\$\{\{ inputs\['target-cycle'\] \}\}/);
  assert.match(page, /base-branch: main/);
});

test('GitHub control-plane jobs have no Google identity or OIDC', async () => {
  const source = await productionWorkflow();
  for (const name of githubOnlyJobs) {
    const job = jobBlock(source, name);
    assert.doesNotMatch(job, /id-token:\s*write/);
    assert.doesNotMatch(job, /secrets\.(?:GOOGLE_|PROGRESS_PRIZE_)/);
    assert.doesNotMatch(job, /google-github-actions\/auth|GOOGLE_ACCESS_TOKEN/);
  }

  const approval = jobBlock(source, 'activation-approval');
  assert.match(approval, /environment: progress-prizes-production-activation/);
  assert.match(approval, /permissions: \{\}/);
  const reporter = jobBlock(source, 'report-scheduled-failure');
  assert.match(reporter, /permissions:\n      issues: write/);
  assert.match(reporter, /\/search\/issues\?\$\{query\}/);
  assert.match(reporter, /search\.total_count > 100/);
  assert.match(reporter, /Google identifiers, ACL identities, and operation output are intentionally omitted/);
});

test('every privileged checkout executes the immutable trigger code', async () => {
  const source = await productionWorkflow();
  for (const name of [
    ...googleJobs,
    'activation-state',
    'activation-page-state',
    'activation-gate',
    'merge',
    'verify-prepared-state',
  ]) {
    const job = jobBlock(source, name);
    assert.match(job, /uses: actions\/checkout@[a-f0-9]{40}/);
    assert.match(job, /ref: \$\{\{ github\.sha \}\}/);
    assert.doesNotMatch(job, /ref: (?:refs\/heads\/main|main)/);
  }
});

test('activation gates, approves, waits, authenticates, leases, verifies, then merges', async () => {
  const production = await productionWorkflow();
  const action = await googleAction();
  const gate = jobBlock(production, 'activation-gate');
  const approval = jobBlock(production, 'activation-approval');
  const activation = jobBlock(production, 'activate-production');
  const merge = jobBlock(production, 'merge');

  assert.match(gate, /needs:\n      - activation-state\n      - activation-binding/);
  assert.match(gate, /progress-prizes-github\.mjs gate/);
  assert.match(gate, /--timeout-seconds 1800/);
  assert.match(gate, /Open approval only inside the bounded cutoff window/);
  assert.match(gate, /waitMilliseconds > 60 \* 60 \* 1000/);
  assert.match(approval, /needs:\n      - activation-state\n      - activation-gate/);
  assert.match(activation, /needs:\n      - activation-state\n      - activation-gate\n      - activation-approval/);
  assert.match(activation, /base-sha: \$\{\{ needs\.activation-gate\.outputs\['base-sha'\] \}\}/);
  assert.match(activation, /github-token: \$\{\{ github\.token \}\}/);

  const wait = action.indexOf('Wait for the real production cutoff without a Google token');
  const auth = action.indexOf('Authenticate mutating operations without a credential file');
  const zeroLease = action.indexOf('--timeout-seconds 0');
  const mutation = action.indexOf('automation-cli.mjs "${arguments[@]}"');
  assert.ok(wait !== -1 && wait < auth && auth < zeroLease && zeroLease < mutation);
  assert.match(action, /waitMilliseconds > 60 \* 60 \* 1000/);
  assert.match(action, /GATED_BASE_SHA: \$\{\{ inputs\['base-sha'\] \}\}/);
  assert.match(action, /result\.sourceAcceptingResponses !== false/);
  assert.match(action, /result\.targetAcceptingResponses !== true/);

  assert.match(merge, /needs:\n      - activation-state\n      - activate-production/);
  assert.match(merge, /HEAD_SHA: \$\{\{ needs\.activate-production\.outputs\['head-sha'\] \}\}/);
  assert.match(merge, /BASE_SHA: \$\{\{ needs\.activate-production\.outputs\['base-sha'\] \}\}/);
  assert.match(merge, /PR_NUMBER: \$\{\{ needs\.activate-production\.outputs\['pr-number'\] \}\}/);
  assert.match(merge, /--sha "\$HEAD_SHA"/);
  assert.match(merge, /--base-sha "\$BASE_SHA"/);
});

test('activation reuses an exact prepared commit and refreshes only stale parents', async () => {
  const production = await productionWorkflow();
  const state = jobBlock(production, 'activation-state');
  const page = jobBlock(production, 'activation-page-state');
  const refresh = jobBlock(production, 'refresh-page');
  const binding = jobBlock(production, 'activation-binding');
  const completed = jobBlock(production, 'verify-completed-activation');

  assert.match(state, /refresh-required: \$\{\{ steps\.state\.outputs\['refresh-required'\] \}\}/);
  assert.match(state, /--expected-base-sha "\$TRIGGER_SHA"/);
  assert.match(page, /outputs\['refresh-required'\] == 'true'/);
  assert.match(page, /parseProgressPrizeMarkdown/);
  assert.match(refresh, /outputs\['refresh-required'\] == 'true'/);
  assert.match(refresh, /progress-prizes-page-pr\.yml/);
  assert.match(binding, /if test "\$REFRESH_REQUIRED" = true/);
  assert.match(binding, /test "\$base_sha" = "\$TRIGGER_SHA"/);
  assert.match(completed, /outputs\.state == 'completed'/);
  assert.match(completed, /operation: verify/);
  assert.match(completed, /verify-mode: active/);
});

test('prepared verification binds the pending page while active verification binds main', async () => {
  const production = await productionWorkflow();
  const resolver = jobBlock(production, 'verify-prepared-state');
  const prepared = jobBlock(production, 'verify-prepared');
  const active = jobBlock(production, 'verify-active');

  assert.match(resolver, /progress-prizes-github\.mjs activation-state/);
  assert.match(resolver, /--expected-base-sha "\$TRIGGER_SHA"/);
  assert.match(resolver, /steps\.state\.outputs\.state != 'pending'/);
  assert.match(resolver, /steps\.state\.outputs\['refresh-required'\] != 'false'/);
  assert.match(prepared, /needs:\n      - preflight\n      - verify-prepared-state/);
  assert.match(prepared, /branch: \$\{\{ needs\.verify-prepared-state\.outputs\['head-branch'\] \}\}/);
  assert.match(prepared, /head-sha: \$\{\{ needs\.verify-prepared-state\.outputs\['head-sha'\] \}\}/);
  assert.match(active, /branch: main/);
  assert.match(active, /head-sha: \$\{\{ github\.sha \}\}/);
});

test('documentation pins production WIF to only the exact top-level dispatch workflow', async () => {
  const readme = await readFile(
    resolve(repositoryRoot, 'scrollprize.org/scripts/progress-prizes/README.md'),
    'utf8',
  );
  assert.match(
    readme,
    /attribute\.workflow_ref == 'ScrollPrize\/villa\/\.github\/workflows\/progress-prizes-production\.yml@refs\/heads\/main'/,
  );
  assert.match(readme, /attribute\.event_name == 'workflow_dispatch'/);
  assert.match(readme, /attribute\.environment == 'progress-prizes-production'/);
  assert.match(readme, /assertion\.workflow_sha == assertion\.sha/);
  assert.match(readme, /schedule milestone does not receive Google configuration or OIDC/);
});
