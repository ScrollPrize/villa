import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import {
  assertAutomationBranch,
  assertDeterministicPageDelta,
  assertPullBinding,
  assertSinglePageCommit,
  gateSnapshot,
  isTrustedPreviewRun,
  pageCommitParent,
  resolveRolloverState,
  safeDiagnosticCode,
  waitForPullBinding,
} from '../../../.github/progress-prizes-github.mjs';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const workflowNames = [
  'progress-prizes-page-pr.yml',
  'progress-prizes-pr-safety.yml',
  'progress-prizes-production.yml',
  'progress-prizes-schedule.yml',
  'progress-prizes-vercel-preview.yml',
];

async function workflow(name) {
  return readFile(resolve(repositoryRoot, '.github/workflows', name), 'utf8');
}

async function googleAction() {
  return readFile(
    resolve(repositoryRoot, '.github/actions/progress-prizes-google/action.yml'),
    'utf8',
  );
}

function jobBlock(source, name) {
  const marker = `  ${name}:\n`;
  const start = source.indexOf(marker);
  assert.notEqual(start, -1, `missing ${name} job`);
  const following = source.slice(start + marker.length);
  const next = following.search(/^  [a-z][a-z0-9-]*:\n/m);
  return source.slice(start, next === -1 ? source.length : start + marker.length + next);
}

function jobNames(source) {
  return [...source.matchAll(/^  ([a-z][a-z0-9-]*):\n/gm)].map((match) => match[1]);
}

function literalRunScripts(source) {
  const lines = source.split('\n');
  const scripts = [];
  for (let index = 0; index < lines.length; index += 1) {
    const run = lines[index].match(/^(\s*)run:\s*\|[-+]?\s*$/);
    if (!run) continue;

    let firstContent = index + 1;
    while (firstContent < lines.length && lines[firstContent].trim() === '') {
      firstContent += 1;
    }
    const contentIndent = lines[firstContent]?.match(/^ */)?.[0].length ?? 0;
    assert.ok(contentIndent > run[1].length, `run block at line ${index + 1} has no body`);

    const body = [];
    for (let cursor = index + 1; cursor < lines.length; cursor += 1) {
      const line = lines[cursor];
      const indentation = line.match(/^ */)?.[0].length ?? 0;
      if (line.trim() !== '' && indentation < contentIndent) break;
      body.push(line.trim() === '' ? '' : line.slice(contentIndent));
    }
    scripts.push({ line: index + 1, source: `${body.join('\n')}\n` });
  }
  return scripts;
}

test('every third-party action is pinned to an approved immutable commit', async () => {
  const approved = new Set([
    'actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0',
    'actions/setup-node@820762786026740c76f36085b0efc47a31fe5020',
    'google-github-actions/auth@7c6bc770dae815cd3e89ee6cdf493a5fab2cc093',
  ]);
  const sources = await Promise.all([
    ...workflowNames.map(async (name) => [name, await workflow(name)]),
    ['progress-prizes-google/action.yml', await googleAction()],
  ]);
  for (const [name, source] of sources) {
    for (const match of source.matchAll(/^\s*uses:\s*([^\s#]+).*$/gm)) {
      const action = match[1];
      if (action.startsWith('./')) continue;
      assert.ok(approved.has(action), `${name} uses an unapproved action: ${action}`);
      assert.match(action, /@[a-f0-9]{40}$/);
    }
  }
});

test('every literal Progress Prize run block has valid Bash syntax after YAML indentation', async () => {
  const sources = await Promise.all([
    ...workflowNames.map(async (name) => [name, await workflow(name)]),
    ['progress-prizes-google/action.yml', await googleAction()],
  ]);
  for (const [name, source] of sources) {
    for (const script of literalRunScripts(source)) {
      const result = spawnSync('bash', ['--noprofile', '--norc', '-n'], {
        input: script.source,
        encoding: 'utf8',
      });
      assert.equal(
        result.status,
        0,
        `${name}:${script.line} is not valid Bash:\n${result.stderr}`,
      );
    }
  }
});

function guardEnvironment(overrides = {}) {
  return {
    REPOSITORY: 'ScrollPrize/villa',
    REPOSITORY_ID: '890972577',
    REPOSITORY_OWNER_ID: '121906140',
    REF: 'refs/heads/main',
    AUTOMATION_ENVIRONMENT: 'production',
    EVENT_NAME: 'workflow_dispatch',
    OPERATION: 'prepare',
    DRY_RUN: 'false',
    BRANCH: 'main',
    TARGET_BRANCH: 'main',
    SOURCE_CYCLE: '2026-07',
    TARGET_CYCLE: '2026-08',
    HEAD_SHA: '',
    ...overrides,
  };
}

test('composite action guard accepts only production validate, prepare, and verify', async () => {
  const action = await googleAction();
  const guard = literalRunScripts(action)[0]?.source;
  const run = (override = {}) => spawnSync('bash', ['--noprofile', '--norc'], {
    input: guard,
    encoding: 'utf8',
    env: guardEnvironment(override),
  });

  assert.equal(run().status, 0);
  assert.equal(run({ OPERATION: 'validate' }).status, 0);
  assert.equal(run({ OPERATION: 'prepare', DRY_RUN: 'true' }).status, 0);
  assert.equal(run({ OPERATION: 'verify', BRANCH: 'codex/progress-prize-2026-08' }).status, 0);
  assert.equal(run({ OPERATION: 'verify', HEAD_SHA: 'a'.repeat(40) }).status, 0);
  for (const override of [
    { REPOSITORY_ID: '1' },
    { REF: 'refs/heads/feature' },
    { EVENT_NAME: 'schedule' },
    { AUTOMATION_ENVIRONMENT: 'staging' },
    { OPERATION: 'activate' },
    { OPERATION: 'reconcile-active' },
    { OPERATION: 'bootstrap' },
    { OPERATION: 'cleanup' },
    { OPERATION: 'verify', DRY_RUN: 'true' },
    { DRY_RUN: 'maybe' },
    { HEAD_SHA: 'a'.repeat(40) },
    { OPERATION: 'verify', HEAD_SHA: 'not-a-sha' },
    { TARGET_BRANCH: 'release' },
    { BRANCH: 'feature/other' },
    { TARGET_CYCLE: '' },
    { OPERATION: 'validate', SOURCE_CYCLE: '' },
    { SOURCE_CYCLE: '2026-13' },
  ]) {
    assert.notEqual(run(override).status, 0, JSON.stringify(override));
  }
});

test('composite action carries no retired staging, fault, or activation controls', async () => {
  const action = await googleAction();
  assert.doesNotMatch(action, /simulated-now|expect-fault|allow-activation-rewind|verify-mode/);
  assert.doesNotMatch(action, /staging-folder-id|archive-folder-id|github-token|base-sha/);
  assert.doesNotMatch(action, /progress-prizes-github\.mjs|Wait for the real production cutoff/);
  assert.doesNotMatch(action, /reconcile|activate|bootstrap|cleanup/);
});

test('Vercel verification runs trusted default-branch code and requires GitHub association', async () => {
  const vercel = await workflow('progress-prizes-vercel-preview.yml');
  assert.match(vercel, /environment: progress-prizes-preview/);
  assert.match(vercel, /github\.actor == 'vercel\[bot\]'/);
  assert.match(vercel, /- vercel\.deployment\.ready/);
  assert.doesNotMatch(vercel, /- vercel\.deployment\.success/);
  assert.match(
    vercel,
    /startsWith\(github\.event\.client_payload\.git\.ref, 'refs\/heads\/codex\/progress-prize-'\)/,
  );
  assert.match(vercel, /!startsWith\(github\.event\.client_payload\.git\.ref, 'refs\/heads\/codex\/progress-prize-smoke-'\)/);
  assert.match(vercel, /run-name: Progress Prize Vercel preview \$\{\{ github\.event\.client_payload\.git\.sha \}\}/);
  assert.match(vercel, /ref: \$\{\{ github\.sha \}\}/);
  assert.doesNotMatch(vercel, /ref: refs\/heads\/main/);
  assert.match(vercel, /pulls\?state=open&head=/);
  assert.doesNotMatch(vercel, /git\/ref\/heads/);
  assert.match(vercel, /payload\.git\?\.sha/);
  assert.match(vercel, /payload\.git\?\.ref/);
  assert.match(vercel, /progress-prizes\/vercel-preview/);
  assert.match(vercel, /VERCEL_AUTOMATION_BYPASS_SECRET/);
  assert.doesNotMatch(vercel, /\$\{\{\s*vars\.VERCEL_PROJECT_ID\s*\}\}/);
  assert.equal(
    [...vercel.matchAll(/\$\{\{\s*secrets\.VERCEL_PROJECT_ID\s*\}\}/g)].length,
    2,
    'the Vercel project identifier must be auto-masked at both uses',
  );
  assert.match(vercel, /printf '::add-mask::%s\\n' \"\$VERCEL_PROJECT_ID\"/);
  assert.match(vercel, /printf '::add-mask::%s\\n' \"\$VERCEL_AUTOMATION_BYPASS_SECRET\"/);
  assert.match(vercel, /!process\.env\.VERCEL_AUTOMATION_BYPASS_SECRET/);
  assert.match(vercel, /redirect: 'error'/);
  assert.doesNotMatch(vercel, /id-token|google-github-actions|GOOGLE_/);
  assert.doesNotMatch(vercel, /checkout.*(?:HEAD_SHA|deployment|pull_request)/i);
});

test('preview gates use creator-bearing newest-first commit status history', async () => {
  const helper = await readFile(
    resolve(repositoryRoot, '.github/progress-prizes-github.mjs'),
    'utf8',
  );
  const historyEndpoint = 'github(`/repos/${OWNER}/${REPO}/commits/${sha}/statuses?per_page=100`)';
  const combinedEndpoint = 'github(`/repos/${OWNER}/${REPO}/commits/${sha}/status`)';
  assert.equal(helper.split(historyEndpoint).length - 1, 1);
  assert.equal(helper.includes(combinedEndpoint), false);
});

test('PR preparation discovers by immutable head and waits for exact SHA convergence', async () => {
  const helper = await readFile(
    resolve(repositoryRoot, '.github/progress-prizes-github.mjs'),
    'utf8',
  );
  const start = helper.indexOf('async function ensurePull(options)');
  const end = helper.indexOf('async function ready(options)', start);
  assert.ok(start >= 0 && end > start);
  const ensurePull = helper.slice(start, end);
  assert.match(ensurePull, /pulls\?state=open&head=/);
  assert.doesNotMatch(ensurePull, /&base=/);
  assert.match(ensurePull, /waitForPullBinding/);
});

test('trusted branch and exact-check gate helpers reject ambiguous automation state', () => {
  assert.equal(assertAutomationBranch('codex/progress-prize-2026-08', 'main').kind, 'production');
  assert.throws(() => assertAutomationBranch('feature/untrusted', 'main'));

  const expectedSha = 'a'.repeat(40);
  const baseSha = 'b'.repeat(40);
  const statuses = [{
    context: 'progress-prizes/vercel-preview',
    state: 'success',
    description: 'Exact Progress Prize preview verified',
    creator: { login: 'github-actions[bot]', id: 41898282, type: 'Bot' },
    target_url: 'https://github.com/ScrollPrize/villa/actions/runs/12345',
  }];
  const previewRun = {
    id: 12345,
    html_url: statuses[0].target_url,
    name: `Progress Prize Vercel preview ${expectedSha}`,
    path: '.github/workflows/progress-prizes-vercel-preview.yml',
    event: 'repository_dispatch',
    actor: { login: 'vercel[bot]', type: 'Bot' },
    triggering_actor: { login: 'vercel[bot]', type: 'Bot' },
    repository: {
      id: 890972577,
      owner: { id: 121906140 },
      full_name: 'ScrollPrize/villa',
    },
    head_repository: { id: 890972577 },
    head_branch: 'main',
    status: 'completed',
    conclusion: 'success',
    display_title: `Progress Prize Vercel preview ${expectedSha}`,
  };
  const successfulChecks = {
    total_count: 2,
    check_runs: [
      {
        name: 'Public no-secret tests',
        status: 'completed',
        conclusion: 'success',
        app: { slug: 'github-actions' },
      },
      {
        name: 'Another required check',
        status: 'completed',
        conclusion: 'success',
        app: { slug: 'github-actions' },
      },
    ],
  };
  assert.equal(isTrustedPreviewRun({ status: statuses[0], run: previewRun, expectedSha }), true);
  assert.equal(gateSnapshot({ statuses, checks: successfulChecks, previewRun, expectedSha }).ready, true);
  assert.equal(gateSnapshot({
    statuses: { statuses },
    checks: successfulChecks,
    previewRun,
    expectedSha,
  }).ready, false, 'the obsolete combined-status response shape must fail closed');
  assert.equal(gateSnapshot({
    statuses: [{ context: 'unrelated', state: 'failure' }, statuses[0]],
    checks: successfulChecks,
    previewRun,
    expectedSha,
  }).ready, true, 'the first matching context is the newest preview status');
  assert.equal(gateSnapshot({
    statuses,
    checks: { total_count: 1, check_runs: successfulChecks.check_runs.slice(1) },
    previewRun,
    expectedSha,
  }).ready, false);
  assert.equal(gateSnapshot({
    statuses,
    checks: {
      total_count: 2,
      check_runs: [
        successfulChecks.check_runs[0],
        { name: 'Unrelated cancelled workflow', status: 'completed', conclusion: 'cancelled' },
      ],
    },
    previewRun,
    expectedSha,
  }).ready, true, 'unrelated repository checks never block the rollover');
  assert.equal(gateSnapshot({
    statuses,
    checks: {
      total_count: 2,
      check_runs: [
        { ...successfulChecks.check_runs[0], conclusion: 'failure' },
        successfulChecks.check_runs[1],
      ],
    },
    previewRun,
    expectedSha,
  }).ready, false);
  assert.equal(gateSnapshot({
    statuses,
    checks: {
      total_count: 2,
      check_runs: [
        { ...successfulChecks.check_runs[0], app: { slug: 'third-party-app' } },
        successfulChecks.check_runs[1],
      ],
    },
    previewRun,
    expectedSha,
  }).ready, false);
  assert.equal(gateSnapshot({
    statuses,
    checks: { total_count: 101, check_runs: successfulChecks.check_runs },
    previewRun,
    expectedSha,
  }).ready, false);
  assert.equal(gateSnapshot({
    statuses: [{
      ...statuses[0],
      creator: { login: 'spoofed-user' },
    }, statuses[0]],
    checks: successfulChecks,
    previewRun,
    expectedSha,
  }).ready, false, 'an older trusted success must not override the newest matching status');
  assert.equal(gateSnapshot({
    statuses: [{ ...statuses[0], state: 'pending' }, statuses[0]],
    checks: successfulChecks,
    previewRun,
    expectedSha,
  }).ready, false, 'an older success must not override a newer pending status');
  assert.equal(gateSnapshot({
    statuses: [statuses[0], { ...statuses[0], state: 'failure' }],
    checks: successfulChecks,
    previewRun,
    expectedSha,
  }).ready, true, 'a newer trusted success supersedes older statuses');

  for (const status of [
    { ...statuses[0], context: 'untrusted/context' },
    { ...statuses[0], state: 'failure' },
    { ...statuses[0], description: 'Looks plausible but is not exact' },
    { ...statuses[0], creator: undefined },
    { ...statuses[0], creator: null },
    { ...statuses[0], creator: { ...statuses[0].creator, login: 'spoofed-user' } },
    { ...statuses[0], creator: { ...statuses[0].creator, id: 1 } },
    { ...statuses[0], creator: { ...statuses[0].creator, type: 'User' } },
  ]) {
    assert.equal(isTrustedPreviewRun({ status, run: previewRun, expectedSha }), false);
  }

  for (const [field, value] of [
    ['name', `Progress Prize Vercel preview ${baseSha}`],
    ['path', '.github/workflows/untrusted.yml'],
    ['event', 'workflow_dispatch'],
    ['head_branch', 'feature/untrusted'],
    ['status', 'in_progress'],
    ['conclusion', 'failure'],
    ['display_title', `Progress Prize Vercel preview ${baseSha}`],
  ]) {
    assert.equal(isTrustedPreviewRun({
      status: statuses[0],
      run: { ...previewRun, [field]: value },
      expectedSha,
    }), false);
  }
  assert.equal(isTrustedPreviewRun({
    status: statuses[0],
    run: { ...previewRun, actor: { login: 'attacker', type: 'User' } },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses[0],
    run: { ...previewRun, actor: { login: 'vercel[bot]', type: 'User' } },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses[0],
    run: { ...previewRun, triggering_actor: { login: 'human', type: 'User' } },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses[0],
    run: { ...previewRun, triggering_actor: { login: 'vercel[bot]', type: 'User' } },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses[0],
    run: {
      ...previewRun,
      repository: { ...previewRun.repository, id: 1 },
    },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses[0],
    run: {
      ...previewRun,
      repository: {
        ...previewRun.repository,
        owner: { id: 1 },
      },
    },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses[0],
    run: { ...previewRun, head_repository: { id: 1 } },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses[0],
    run: { ...previewRun, id: 999 },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses[0],
    run: { ...previewRun, html_url: 'https://github.com/ScrollPrize/villa/actions/runs/999' },
    expectedSha,
  }), false);
  for (const target_url of [
    'https://github.com/ScrollPrize/villa/actions/runs/12345/',
    'https://github.com/ScrollPrize/villa/actions/runs/12345?trusted=false',
    'https://github.com/ScrollPrize/villa/actions/runs/12345#spoofed',
    'https://github.com/attacker/villa/actions/runs/12345',
  ]) {
    assert.equal(isTrustedPreviewRun({
      status: { ...statuses[0], target_url },
      run: previewRun,
      expectedSha,
    }), false);
  }
  assert.equal(isTrustedPreviewRun({
    status: { ...statuses[0], target_url: 'https://github.com/ScrollPrize/villa/actions/runs/999' },
    run: previewRun,
    expectedSha,
  }), false);
});

test('the GitHub helper binds the PR and exact deterministic page-only commit', () => {
  const headSha = 'a'.repeat(40);
  const baseSha = 'b'.repeat(40);
  const head = 'codex/progress-prize-2026-08';
  const base = 'main';
  const baseMarkdown = [
    '# Prizes',
    '',
    '## Progress Prizes',
    '',
    '{/* progress-prizes:deadline:start */}',
    'Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is 11:59pm Pacific, July 31st, 2026!',
    '{/* progress-prizes:deadline:end */}',
    '',
    '{/* progress-prizes:form:start */}',
    '[Submission Form](https://forms.gle/JulyForm)',
    '{/* progress-prizes:form:end */}',
    '',
    '## Terms and Conditions',
    '',
  ].join('\n');
  const headMarkdown = baseMarkdown
    .replace('July 31st, 2026', 'August 31st, 2026')
    .replace('https://forms.gle/JulyForm', 'https://forms.gle/AugustForm');

  assert.equal(assertDeterministicPageDelta({
    baseMarkdown,
    headMarkdown,
    sourceCycle: '2026-07',
    targetCycle: '2026-08',
    responderUri: 'https://forms.gle/AugustForm',
  }).target.cycle, '2026-08');
  assert.throws(() => assertDeterministicPageDelta({
    baseMarkdown,
    headMarkdown: `${headMarkdown}unmanaged change\n`,
    sourceCycle: '2026-07',
    targetCycle: '2026-08',
  }), /deterministic marker-only/);

  const pull = {
    state: 'open',
    head: { ref: head, sha: headSha, repo: { id: 890972577, full_name: 'ScrollPrize/villa' } },
    base: { ref: base, sha: baseSha, repo: { id: 890972577, full_name: 'ScrollPrize/villa' } },
  };
  assert.equal(assertPullBinding(pull, { head, base, headSha, baseSha }), pull);
  assert.throws(
    () => assertPullBinding(pull, { head, base, headSha, baseSha: 'c'.repeat(40) }),
    /immutable refs/,
  );

  const commit = {
    sha: headSha,
    parents: [{ sha: baseSha }],
    files: [{ filename: 'scrollprize.org/docs/34_prizes.md', status: 'modified' }],
  };
  assert.equal(assertSinglePageCommit(commit, { headSha, baseSha }), commit);
  assert.equal(pageCommitParent(commit, { headSha }), baseSha);
  assert.throws(() => assertSinglePageCommit({
    ...commit,
    files: [...commit.files, { filename: '.github/workflows/untrusted.yml', status: 'added' }],
  }, { headSha, baseSha }), /one page-only commit/);
  assert.equal(pageCommitParent({
    ...commit,
    parents: [{ sha: 'c'.repeat(40) }],
  }, { headSha }), 'c'.repeat(40), 'an older parent is fine; merging is manual');
  assert.throws(() => pageCommitParent({
    ...commit,
    files: [...commit.files, { filename: '.github/workflows/untrusted.yml', status: 'added' }],
  }, { headSha }), /one page-only commit/);
  assert.throws(() => pageCommitParent({
    ...commit,
    parents: [{ sha: baseSha }, { sha: 'c'.repeat(40) }],
  }, { headSha }), /one page-only commit/);
});

test('rollover-state distinguishes a pending draft, a ready PR, and a merged page', async () => {
  const head = 'codex/progress-prize-2026-08';
  const base = 'main';
  const headSha = 'a'.repeat(40);
  const baseSha = 'b'.repeat(40);
  const pull = {
    number: 42,
    draft: true,
    state: 'open',
    head: {
      ref: head,
      sha: headSha,
      repo: { id: 890972577, full_name: 'ScrollPrize/villa' },
    },
    base: {
      ref: base,
      sha: baseSha,
      repo: { id: 890972577, full_name: 'ScrollPrize/villa' },
    },
  };
  const mainRef = async () => ({ ref: 'refs/heads/main', object: { type: 'commit', sha: baseSha } });
  const options = { head, base, cycle: '2026-08' };

  assert.deepEqual(await resolveRolloverState(options, { listPulls: async () => [pull] }), {
    state: 'pending',
    headSha,
    pullNumber: '42',
    draft: true,
  });
  assert.deepEqual(
    await resolveRolloverState(options, { listPulls: async () => [{ ...pull, draft: false }] }),
    { state: 'pending', headSha, pullNumber: '42', draft: false },
  );
  assert.deepEqual(await resolveRolloverState(options, {
    listPulls: async () => [],
    readMainRef: mainRef,
    readPage: async () => ({ cycle: '2026-08' }),
  }), {
    state: 'completed',
    headSha: baseSha,
    pullNumber: '',
    draft: false,
  });

  await assert.rejects(
    resolveRolloverState(options, { listPulls: async () => [pull, pull] }),
    (error) => safeDiagnosticCode(error) === 'ambiguous-pr',
  );
  await assert.rejects(
    resolveRolloverState(options, {
      listPulls: async () => [],
      readMainRef: async () => ({ ref: 'refs/heads/main', object: { type: 'tag', sha: baseSha } }),
    }),
    (error) => safeDiagnosticCode(error) === 'invalid-main-ref',
  );
  await assert.rejects(
    resolveRolloverState(options, {
      listPulls: async () => [],
      readMainRef: mainRef,
      readPage: async () => ({ cycle: '2026-07' }),
    }),
    (error) => safeDiagnosticCode(error) === 'completed-state-missing',
  );
  await assert.rejects(
    resolveRolloverState(options, {
      listPulls: async () => [{
        ...pull,
        head: { ...pull.head, repo: { id: 123, full_name: 'untrusted/fork' } },
      }],
    }),
    (error) => safeDiagnosticCode(error) === 'invalid-pr-association',
  );
});

test('GitHub coordination diagnostics are bounded to an allowlist', () => {
  assert.equal(safeDiagnosticCode({ diagnosticCode: 'ambiguous-pr' }), 'ambiguous-pr');
  assert.equal(safeDiagnosticCode({
    diagnosticCode: 'private-id-from-api-body',
    message: 'sensitive response body',
  }), 'unexpected-error');

  const result = spawnSync(process.execPath, [
    resolve(repositoryRoot, '.github/progress-prizes-github.mjs'),
    'unknown-command',
  ], { encoding: 'utf8' });
  assert.equal(result.status, 1);
  assert.equal(result.stdout, '');
  assert.equal(
    result.stderr,
    'Progress Prize GitHub coordination failed safely [unexpected-error].\n',
  );
});

test('PR binding retry tolerates only bounded stale SHAs', async () => {
  const head = 'codex/progress-prize-2026-08';
  const base = 'main';
  const headSha = 'a'.repeat(40);
  const baseSha = 'b'.repeat(40);
  const exactPull = {
    number: 1194,
    state: 'open',
    head: {
      ref: head,
      sha: headSha,
      repo: { id: 890972577, full_name: 'ScrollPrize/villa' },
    },
    base: {
      ref: base,
      sha: baseSha,
      repo: { id: 890972577, full_name: 'ScrollPrize/villa' },
    },
  };
  const stalePull = {
    ...exactPull,
    head: { ...exactPull.head, sha: 'c'.repeat(40) },
    base: { ...exactPull.base, sha: 'd'.repeat(40) },
  };
  const responses = [stalePull, stalePull, exactPull];
  const sleeps = [];
  const result = await waitForPullBinding({
    number: exactPull.number,
    head,
    base,
    headSha,
    baseSha,
  }, {
    attempts: 3,
    delayMs: 7,
    readPull: async () => responses.shift(),
    sleep: async (milliseconds) => sleeps.push(milliseconds),
  });
  assert.equal(result, exactPull);
  assert.deepEqual(sleeps, [7, 7]);

  let unsafeReads = 0;
  let unsafeSleeps = 0;
  await assert.rejects(
    waitForPullBinding({
      number: exactPull.number,
      head,
      base,
      headSha,
      baseSha,
    }, {
      attempts: 3,
      delayMs: 0,
      readPull: async () => {
        unsafeReads += 1;
        return { ...stalePull, base: { ...stalePull.base, ref: 'release' } };
      },
      sleep: async () => { unsafeSleeps += 1; },
    }),
    /association failed/,
  );
  assert.equal(unsafeReads, 1);
  assert.equal(unsafeSleeps, 0);

  let staleReads = 0;
  await assert.rejects(
    waitForPullBinding({
      number: exactPull.number,
      head,
      base,
      headSha,
      baseSha,
    }, {
      attempts: 2,
      delayMs: 0,
      readPull: async () => {
        staleReads += 1;
        return stalePull;
      },
      sleep: async () => {},
    }),
    /immutable refs did not converge/,
  );
  assert.equal(staleReads, 2);
});
