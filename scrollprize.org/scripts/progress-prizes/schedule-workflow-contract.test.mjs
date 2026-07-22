import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const schedulePath = resolve(
  repositoryRoot,
  '.github/workflows/progress-prizes-schedule.yml',
);
const productionPath = resolve(
  repositoryRoot,
  '.github/workflows/progress-prizes-production.yml',
);
const helperPath = resolve(repositoryRoot, '.github/progress-prizes-schedule.mjs');

async function source(path) {
  return readFile(path, 'utf8');
}

function jobBlock(workflow, name) {
  const marker = `  ${name}:\n`;
  const start = workflow.indexOf(marker);
  assert.notEqual(start, -1, `missing ${name} job`);
  const remainder = workflow.slice(start + marker.length);
  const next = remainder.search(/^  [a-z][a-z0-9-]*:\n/m);
  return next === -1
    ? workflow.slice(start)
    : workflow.slice(start, start + marker.length + next);
}

function jobNames(workflow) {
  const jobs = workflow.slice(workflow.indexOf('\njobs:\n') + '\njobs:\n'.length);
  return [...jobs.matchAll(/^  ([a-z][a-z0-9-]*):\n/gm)].map((match) => match[1]);
}

function literalRunScripts(workflow) {
  const lines = workflow.split('\n');
  const scripts = [];
  for (let index = 0; index < lines.length; index += 1) {
    const run = lines[index].match(/^(\s*)run:\s*\|[-+]?\s*$/);
    if (!run) continue;
    let firstContent = index + 1;
    while (firstContent < lines.length && lines[firstContent].trim() === '') firstContent += 1;
    const contentIndent = lines[firstContent]?.match(/^ */)?.[0].length ?? 0;
    assert.ok(contentIndent > run[1].length);
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

test('scheduler exposes only a real-clock manual smoke and four Pacific schedules', async () => {
  const workflow = await source(schedulePath);
  const triggers = workflow.slice(0, workflow.indexOf('\npermissions:'));
  const expectedCrons = [
    '17 6 * * *',
    '40 23 28-31 * *',
    '17 0 1 * *',
    '47 6 1 * *',
  ];

  assert.match(triggers, /^on:\n  workflow_dispatch:\n  schedule:/m);
  assert.doesNotMatch(
    triggers,
    /^  (?:pull_request|pull_request_target|push|workflow_call|workflow_run|repository_dispatch):/m,
  );
  assert.doesNotMatch(triggers, /^    inputs:/m);
  assert.deepEqual(
    [...triggers.matchAll(/^    - cron: "([^"]+)"$/gm)].map((match) => match[1]),
    expectedCrons,
  );
  assert.equal(
    [...triggers.matchAll(/^      timezone: America\/Los_Angeles$/gm)].length,
    expectedCrons.length,
  );
  for (const cron of expectedCrons) {
    assert.notEqual(Number(cron.split(' ')[0]), 0, `${cron} must avoid minute zero`);
  }
});

test('scheduler jobs keep GitHub write authority isolated from planning and Google', async () => {
  const workflow = await source(schedulePath);
  assert.match(workflow, /^permissions: \{\}$/m);
  assert.match(
    workflow,
    /^concurrency:\n  group: progress-prizes-schedule\n  cancel-in-progress: false\n  queue: max$/m,
  );
  assert.deepEqual(
    new Set(jobNames(workflow)),
    new Set(['preflight', 'plan', 'dedupe', 'dispatch', 'report-failure']),
  );

  const preflight = jobBlock(workflow, 'preflight');
  const plan = jobBlock(workflow, 'plan');
  const dedupe = jobBlock(workflow, 'dedupe');
  const dispatch = jobBlock(workflow, 'dispatch');
  const reporter = jobBlock(workflow, 'report-failure');
  assert.match(preflight, /permissions: \{\}/);
  assert.doesNotMatch(preflight, /uses:/);
  assert.match(plan, /permissions:\n      contents: read/);
  assert.match(dedupe, /permissions:\n      actions: read\n      contents: read\n      pull-requests: read/);
  assert.match(dispatch, /permissions:\n      actions: write\n      contents: read/);
  assert.match(reporter, /permissions:\n      issues: write/);
  assert.equal([...workflow.matchAll(/actions: write/g)].length, 1);
  assert.equal([...workflow.matchAll(/issues: write/g)].length, 1);

  assert.doesNotMatch(workflow, /id-token:|environment:|secrets\.|vars\./);
  assert.doesNotMatch(
    workflow,
    /GOOGLE_(?:ACCESS_TOKEN|WORKLOAD_IDENTITY_PROVIDER|SERVICE_ACCOUNT)|PROGRESS_PRIZE_(?:DRIVE|FOLDER|SOURCE_FORM|EDITOR_GROUP)|VERCEL_/,
  );
  assert.doesNotMatch(
    workflow,
    /google-github-actions\/auth|actions\/(?:cache|upload-artifact|download-artifact)|secrets:\s*inherit/,
  );
  assert.doesNotMatch(workflow, /uses: \.\/\.github\/workflows\/progress-prizes-production\.yml/);
});

test('every scheduler checkout and third-party action is immutable', async () => {
  const workflow = await source(schedulePath);
  const approved = new Set([
    'actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0',
    'actions/setup-node@820762786026740c76f36085b0efc47a31fe5020',
  ]);
  for (const match of workflow.matchAll(/^\s*uses:\s*([^\s#]+).*$/gm)) {
    assert.ok(approved.has(match[1]), match[1]);
    assert.match(match[1], /@[a-f0-9]{40}$/);
  }
  const checkoutCount = [...workflow.matchAll(/uses: actions\/checkout@/g)].length;
  assert.equal(checkoutCount, 3);
  assert.equal([...workflow.matchAll(/ref: \$\{\{ github\.sha \}\}/g)].length, checkoutCount);
  assert.equal([...workflow.matchAll(/persist-credentials: false/g)].length, checkoutCount);
  assert.doesNotMatch(workflow, /ref: (?:main|refs\/heads\/main)$/m);
});

test('secret-free preflight executes an exact valid and invalid context matrix', async () => {
  const workflow = await source(schedulePath);
  const [guard] = literalRunScripts(jobBlock(workflow, 'preflight'));
  assert.ok(guard);
  const valid = {
    REPOSITORY: 'ScrollPrize/villa',
    REPOSITORY_ID: '890972577',
    REPOSITORY_OWNER_ID: '121906140',
    REF: 'refs/heads/main',
    HEAD_SHA: 'a'.repeat(40),
    WORKFLOW_REF: 'ScrollPrize/villa/.github/workflows/progress-prizes-schedule.yml@refs/heads/main',
    WORKFLOW_SHA: 'a'.repeat(40),
    EVENT_NAME: 'workflow_dispatch',
    SCHEDULE_EXPRESSION: '',
    RUN_ID: '12345',
  };
  const execute = (overrides = {}) => spawnSync(
    'bash',
    ['--noprofile', '--norc', '-c', guard],
    { cwd: repositoryRoot, env: { ...process.env, ...valid, ...overrides }, encoding: 'utf8' },
  );
  assert.equal(execute().status, 0);
  assert.equal(execute({
    EVENT_NAME: 'schedule',
    SCHEDULE_EXPRESSION: '17 6 * * *',
  }).status, 0);
  for (const overrides of [
    { REPOSITORY_ID: '1' },
    { REPOSITORY_OWNER_ID: '1' },
    { REF: 'refs/heads/feature' },
    { WORKFLOW_REF: 'ScrollPrize/villa/.github/workflows/other.yml@refs/heads/main' },
    { WORKFLOW_SHA: 'b'.repeat(40) },
    { EVENT_NAME: 'pull_request' },
    { EVENT_NAME: 'schedule', SCHEDULE_EXPRESSION: '0 0 * * *' },
    { EVENT_NAME: 'workflow_dispatch', SCHEDULE_EXPRESSION: '17 6 * * *' },
    { RUN_ID: 'not-numeric' },
  ]) {
    assert.notEqual(execute(overrides).status, 0, JSON.stringify(overrides));
  }
  assert.doesNotMatch(guard, /ACTOR|actor/);
});

test('workflow wiring makes manual scheduler dispatch read-only dry-run only', async () => {
  const workflow = await source(schedulePath);
  const planJob = jobBlock(workflow, 'plan');
  const [planScript] = literalRunScripts(planJob);
  const temporary = await mkdtemp(join(tmpdir(), 'progress-prize-schedule-contract-'));
  try {
    const outputPath = join(temporary, 'output');
    const summaryPath = join(temporary, 'summary');
    const result = spawnSync('bash', ['--noprofile', '--norc', '-c', planScript], {
      cwd: repositoryRoot,
      env: {
        ...process.env,
        REPOSITORY: 'ScrollPrize/villa',
        REPOSITORY_ID: '890972577',
        REPOSITORY_OWNER_ID: '121906140',
        REF: 'refs/heads/main',
        HEAD_SHA: 'a'.repeat(40),
        WORKFLOW_REF: 'ScrollPrize/villa/.github/workflows/progress-prizes-schedule.yml@refs/heads/main',
        WORKFLOW_SHA: 'a'.repeat(40),
        EVENT_NAME: 'workflow_dispatch',
        SCHEDULE_EXPRESSION: '',
        GITHUB_OUTPUT: outputPath,
        GITHUB_STEP_SUMMARY: summaryPath,
      },
      encoding: 'utf8',
    });
    assert.equal(result.status, 0, result.stderr);
    const output = await readFile(outputPath, 'utf8');
    assert.match(output, /^operation=dry-run$/m);
    assert.doesNotMatch(output, /^operation=(?:prepare|activate)$/m);
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
});

test('dedupe and dispatch use only the tested fixed helper contract', async () => {
  const workflow = await source(schedulePath);
  const helper = await source(helperPath);
  const dedupe = jobBlock(workflow, 'dedupe');
  const dispatch = jobBlock(workflow, 'dispatch');

  assert.match(dedupe, /progress-prizes-schedule\.mjs dedupe/);
  assert.match(dedupe, /--operation "\$OPERATION"/);
  assert.match(dedupe, /--target-cycle "\$TARGET_CYCLE"/);
  assert.match(dispatch, /needs\.dedupe\.outputs\['should-dispatch'\] == 'true'/);
  assert.match(dispatch, /progress-prizes-schedule\.mjs dispatch/);
  assert.match(dispatch, /--request-id "\$REQUEST_ID"/);
  assert.match(dispatch, /REQUEST_ID: \$\{\{ github\.run_id \}\}/);
  assert.match(dispatch, /if: steps\.child\.outputs\.dispatched == 'true'/);
  assert.match(dispatch, /if: steps\.child\.outputs\.dispatched == 'false'/);
  assert.match(dispatch, /production-run-nonterminal/);

  assert.match(
    helper,
    /\/actions\/workflows\/\$\{PRODUCTION_WORKFLOW\}\/dispatches/,
  );
  assert.match(helper, /const PRODUCTION_WORKFLOW = 'progress-prizes-production\.yml'/);
  assert.match(helper, /ref: MAIN_REF/);
  assert.match(helper, /return_run_details: true/);
  assert.match(helper, /expectedStatus: 200/);
  assert.match(helper, /redirect: 'error'/);
  assert.match(helper, /MAX_GITHUB_RESPONSE_BYTES = 256 \* 1024/);
  assert.match(helper, /new Set\(\['dry-run', 'prepare', 'activate'\]\)/);
  assert.match(helper, /'requested'[\s\S]*'queued'[\s\S]*'pending'[\s\S]*'waiting'[\s\S]*'in_progress'/);
  assert.match(
    helper,
    /export async function dispatchProductionWorkflow[\s\S]*const runState = await inspectProductionWorkflowRuns[\s\S]*if \(!runState\.dispatch\)[\s\S]*githubJson\(/,
  );
  assert.doesNotMatch(
    helper,
    /actions\/(?:runs\/[^'"`]*\/(?:cancel|rerun|approve)|workflows\/[^'"`]*\/(?:enable|disable))/,
  );
});

test('scheduler never waits or exposes simulated and staging controls', async () => {
  const workflow = await source(schedulePath);
  assert.doesNotMatch(workflow, /\bsleep\b|setTimeout|simulated-now|SIMULATED_NOW|fault-injection|after-copy|after-close-source/);
  assert.doesNotMatch(workflow, /staging-folder|archive-folder|smoke-branch/);
});

test('redacted reporting is main-bound and cannot expose operation responses', async () => {
  const workflow = await source(schedulePath);
  const reporter = jobBlock(workflow, 'report-failure');
  assert.match(reporter, /github\.ref == 'refs\/heads\/main'/);
  assert.match(reporter, /github\.workflow_sha == github\.sha/);
  assert.match(reporter, /issues: write/);
  assert.match(reporter, /Google identifiers, ACL identities, and operation output are intentionally omitted/);
  assert.doesNotMatch(reporter, /needs\.plan\.outputs|steps\.child\.outputs|response\.text/);
});

test('production remains dispatch-only, queued, and WIF-isolated from scheduler', async () => {
  const production = await source(productionPath);
  const readme = await source(resolve(
    repositoryRoot,
    'scrollprize.org/scripts/progress-prizes/README.md',
  ));
  const triggers = production.slice(0, production.indexOf('\npermissions:'));
  assert.match(triggers, /^on:\n  workflow_dispatch:/m);
  assert.doesNotMatch(triggers, /^  (?:schedule|workflow_call|repository_dispatch|pull_request):/m);
  assert.match(
    production,
    /^concurrency:\n  group: progress-prizes-production\n  cancel-in-progress: false\n  queue: max$/m,
  );
  assert.match(
    readme,
    /attribute\.workflow_ref == 'ScrollPrize\/villa\/\.github\/workflows\/progress-prizes-production\.yml@refs\/heads\/main'/,
  );
  assert.match(readme, /attribute\.event_name == 'workflow_dispatch'/);
  assert.doesNotMatch(
    readme,
    /attribute\.workflow_ref == 'ScrollPrize\/villa\/\.github\/workflows\/progress-prizes-schedule\.yml/,
  );
});
