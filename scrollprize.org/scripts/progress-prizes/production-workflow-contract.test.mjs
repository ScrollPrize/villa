import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const googleJobs = ['validate', 'dry-run', 'prepare-google', 'verify'];
const githubOnlyJobs = [
  'preflight',
  'prepare-page',
  'ready',
  'verify-state',
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
  assert.match(
    source,
    /^concurrency:\n  group: progress-prizes-production\n  cancel-in-progress: false\n  queue: max$/m,
  );
  assert.match(preflight, /permissions: \{\}/);
  assert.doesNotMatch(preflight, /environment:|id-token|secrets\./);
  assert.match(preflight, /assert\.equal\(process\.env\.REPOSITORY, 'ScrollPrize\/villa'\)/);
  assert.match(preflight, /assert\.equal\(process\.env\.REPOSITORY_ID, '890972577'\)/);
  assert.match(preflight, /assert\.equal\(process\.env\.REPOSITORY_OWNER_ID, '121906140'\)/);
  assert.match(preflight, /assert\.equal\(process\.env\.REF, 'refs\/heads\/main'\)/);
  assert.match(preflight, /assert\.equal\(process\.env\.EVENT_NAME, 'workflow_dispatch'\)/);
  assert.match(preflight, /process\.env\.TARGET_CYCLE, target/);
  assert.match(preflight, /\^\[1-9\]\\d\*\$/);
  assert.doesNotMatch(source, /verify-mode|reconcile|activate|merge_method|progress-prizes-production-activation/);
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
    OPERATION: 'ready',
    SOURCE_CYCLE: '2026-07',
    TARGET_CYCLE: '2026-08',
    REQUEST_ID: '',
  };
  const execute = (overrides = {}) => spawnSync('bash', ['--noprofile', '--norc', '-c', guard], {
    env: { ...process.env, ...valid, ...overrides },
    encoding: 'utf8',
  });
  assert.equal(execute().status, 0);
  assert.equal(execute({ OPERATION: 'prepare', REQUEST_ID: '123' }).status, 0);
  for (const overrides of [
    { REPOSITORY_ID: '1' },
    { REF: 'refs/heads/feature' },
    { EVENT_NAME: 'schedule' },
    { OPERATION: 'cleanup' },
    { OPERATION: 'activate' },
    { OPERATION: 'reconcile-active' },
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
  assert.match(action, /if test "\$DRY_RUN" = true; then/);
  assert.match(action, /arguments\+=\(--dry-run --preparation-days 31\)/);
  assert.match(action, /if test "\$DRY_RUN" = true; then test "\$OPERATION" = prepare; fi/);
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

  for (const name of ['preflight', 'ready', 'verify-state', 'report-scheduled-failure']) {
    assert.doesNotMatch(jobBlock(source, name), /environment:/);
  }
  const reporter = jobBlock(source, 'report-scheduled-failure');
  assert.match(reporter, /permissions:\n      issues: write/);
  assert.match(reporter, /\/search\/issues\?\$\{query\}/);
  assert.match(reporter, /search\.total_count > 100/);
  assert.match(reporter, /Google identifiers, ACL identities, and operation output are intentionally omitted/);
});

test('every privileged checkout executes the immutable trigger code', async () => {
  const source = await productionWorkflow();
  for (const name of [...googleJobs, 'ready', 'verify-state']) {
    const job = jobBlock(source, name);
    assert.match(job, /uses: actions\/checkout@[a-f0-9]{40}/);
    assert.match(job, /ref: \$\{\{ github\.sha \}\}/);
    assert.doesNotMatch(job, /ref: (?:refs\/heads\/main|main)/);
  }
});

test('ready waits for the public tests and preview, flips the draft, and never merges', async () => {
  const production = await productionWorkflow();
  const ready = jobBlock(production, 'ready');

  assert.match(ready, /^    if: inputs\.operation == 'ready'$/m);
  assert.match(ready, /^    needs: preflight$/m);
  assert.match(ready, /permissions:\n      actions: read\n      checks: read\n      contents: read\n      pull-requests: write\n      statuses: read/);
  assert.match(ready, /progress-prizes-github\.mjs rollover-state/);
  assert.match(ready, /if: steps\.state\.outputs\.state == 'pending' && steps\.state\.outputs\.draft == 'true'/);
  assert.match(ready, /waitMilliseconds > 60 \* 60 \* 1000/);
  assert.match(ready, /if: steps\.state\.outputs\.state == 'pending'/);
  assert.match(ready, /progress-prizes-github\.mjs ready/);
  assert.match(ready, /--timeout-seconds 1800/);
  assert.match(ready, /if: steps\.state\.outputs\.state == 'completed'/);
  assert.doesNotMatch(ready, /pulls\/\d+\/merge|merge_method|environment:|id-token|secrets\./);
  assert.doesNotMatch(production, /progress-prizes-google[\s\S]*operation: (?:activate|reconcile-active)/);
});

test('verification binds the Google check to the pending rollover page commit', async () => {
  const production = await productionWorkflow();
  const resolver = jobBlock(production, 'verify-state');
  const verify = jobBlock(production, 'verify');

  assert.match(resolver, /progress-prizes-github\.mjs rollover-state/);
  assert.match(resolver, /steps\.state\.outputs\.state != 'pending'/);
  assert.match(verify, /needs:\n      - preflight\n      - verify-state/);
  assert.match(verify, /^          operation: verify$/m);
  assert.match(verify, /branch: \$\{\{ needs\.verify-state\.outputs\['head-branch'\] \}\}/);
  assert.match(verify, /head-sha: \$\{\{ needs\.verify-state\.outputs\['head-sha'\] \}\}/);
  assert.doesNotMatch(verify, /verify-mode/);
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
