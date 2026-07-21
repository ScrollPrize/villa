import assert from 'node:assert/strict';
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
} from '../../../.github/progress-prizes-github.mjs';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const workflowNames = [
  'progress-prizes-google.yml',
  'progress-prizes-page-pr.yml',
  'progress-prizes-pr-safety.yml',
  'progress-prizes-rehearsal.yml',
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

test('Google OIDC uses two literal protected Environment jobs and one secret-free composite', async () => {
  const google = await workflow('progress-prizes-google.yml');
  const action = await googleAction();
  assert.match(google, /^on:\n  workflow_call:/m);
  assert.doesNotMatch(google, /pull_request(?:_target)?:/);
  assert.doesNotMatch(google, /^    secrets:/m, 'Environment secrets must not be caller inputs');
  assert.doesNotMatch(google, /environment:\s*\$\{\{/);

  const production = jobBlock(google, 'production');
  const staging = jobBlock(google, 'staging');
  for (const [name, protectedJob] of [['production', production], ['staging', staging]]) {
    assert.match(protectedJob, new RegExp(`if: inputs\\.environment == '${name}'`));
    assert.match(protectedJob, new RegExp(`environment: progress-prizes-${name}`));
    assert.match(protectedJob, /contents: read\n      id-token: write/);
    assert.match(protectedJob, /ref: \$\{\{ github\.sha \}\}/);
    assert.match(protectedJob, /uses: \.\/\.github\/actions\/progress-prizes-google/);
    assert.match(protectedJob, new RegExp(`^          environment: ${name}$`, 'm'));
    assert.match(
      protectedJob,
      /responder-uri: \$\{\{ steps\.google\.outputs\['responder-uri'\] \}\}/,
    );
  }
  assert.equal(
    [...google.matchAll(/ref: \$\{\{ github\.sha \}\}/g)].length,
    2,
    'both protected jobs must check out the exact approved workflow commit',
  );

  const secretMappings = new Map([
    ['workload-identity-provider', 'GOOGLE_WORKLOAD_IDENTITY_PROVIDER'],
    ['service-account-email', 'GOOGLE_SERVICE_ACCOUNT_EMAIL'],
    ['drive-admin-email', 'PROGRESS_PRIZE_DRIVE_ADMIN_EMAIL'],
    ['drive-id', 'PROGRESS_PRIZE_DRIVE_ID'],
    ['folder-id', 'PROGRESS_PRIZE_FOLDER_ID'],
    ['archive-folder-id', 'PROGRESS_PRIZE_ARCHIVE_FOLDER_ID'],
    ['source-form-id', 'PROGRESS_PRIZE_SOURCE_FORM_ID'],
    ['editor-group-email', 'PROGRESS_PRIZE_EDITOR_GROUP_EMAIL'],
  ]);
  for (const [input, secret] of secretMappings) {
    const mapping = `          ${input}: \${{ secrets.${secret} }}`;
    assert.equal(
      google.split(mapping).length - 1,
      2,
      `${secret} must be mapped once in each literal Environment job`,
    );
    if (input === 'archive-folder-id') {
      assert.match(
        action,
        /^  archive-folder-id:\n(?:    .*\n)*?    required: false\n    default: ""$/m,
      );
    } else {
      assert.match(
        action,
        new RegExp(`^  ${input}:\\n(?:    .*\\n)*?    required: true$`, 'm'),
      );
    }
  }
  assert.doesNotMatch(
    action,
    /\$\{\{\s*(?:secrets|vars)(?:\.|\[)/,
    'the composite must receive protected values only through explicit inputs',
  );
  assert.doesNotMatch(google, /google-github-actions\/auth/);
  assert.equal([...action.matchAll(/google-github-actions\/auth@/g)].length, 2);
  assert.match(action, /create_credentials_file: false/);
  assert.match(action, /export_environment_variables: false/);
  assert.match(action, /printf '::add-mask::%s\\n' "\$value"/);
  const maskStep = action.indexOf('- name: Register protected Google configuration masks');
  const validationStep = action.indexOf('- name: Validate protected environment configuration');
  assert.ok(maskStep >= 0 && maskStep < validationStep, 'mask registration must precede validation');
  assert.equal([...action.matchAll(/access_token_lifetime: 1200s/g)].length, 2);
  assert.doesNotMatch(action, /access_token_scopes: >-/);
  assert.match(
    action,
    /access_token_scopes: \|-\n\s+https:\/\/www\.googleapis\.com\/auth\/forms\.body\.readonly\n\s+https:\/\/www\.googleapis\.com\/auth\/drive\.readonly/,
  );
  assert.match(
    action,
    /access_token_scopes: \|-\n\s+https:\/\/www\.googleapis\.com\/auth\/forms\.body\n\s+https:\/\/www\.googleapis\.com\/auth\/drive\n/,
  );
  assert.doesNotMatch(action, /drive\.file/);
  assert.match(action, /automation-cli\.mjs/);
  assert.doesNotMatch(`${google}\n${action}`, /credentials_json|service_account_key|private_key/i);

  const publicSafety = await workflow('progress-prizes-pr-safety.yml');
  assert.match(publicSafety, /\.github\/actions\/progress-prizes-google\/\*\*/);
  assert.doesNotMatch(publicSafety, /id-token|google-github-actions|GOOGLE_/);
  assert.doesNotMatch(publicSafety, /secrets\./);
  assert.doesNotMatch(publicSafety, /pull_request_target/);
});

test('Google preflight and result selector fail closed around one protected job', async () => {
  const google = await workflow('progress-prizes-google.yml');
  const action = await googleAction();
  const preflight = jobBlock(google, 'preflight');
  const selector = jobBlock(google, 'select-result');

  assert.match(preflight, /name: Reject unsafe Google routing/);
  assert.match(preflight, /permissions: \{\}/);
  assert.match(preflight, /WORKFLOW_SHA: \$\{\{ github\.sha \}\}/);
  assert.match(preflight, /test "\$REPOSITORY" = ScrollPrize\/villa/);
  assert.match(preflight, /test "\$REPOSITORY_ID" = 890972577/);
  assert.match(preflight, /test "\$REPOSITORY_OWNER_ID" = 121906140/);
  assert.match(preflight, /test "\$REF" = refs\/heads\/main/);
  assert.match(preflight, /\[\[ "\$WORKFLOW_SHA" =~ \^\[a-f0-9\]\{40\}\$ \]\]/);
  assert.match(preflight, /case "\$AUTOMATION_ENVIRONMENT" in staging\|production/);
  assert.match(preflight, /test "\$OPERATION" != bootstrap/);
  assert.match(action, /test "\$OPERATION" != bootstrap/);

  assert.match(selector, /needs:\n      - preflight\n      - production\n      - staging/);
  assert.match(selector, /if: always\(\)/);
  assert.match(selector, /permissions: \{\}/);
  assert.match(selector, /test "\$PREFLIGHT_RESULT" = success/);
  assert.match(selector, /test "\$PRODUCTION_RESULT" = success/);
  assert.match(selector, /test "\$STAGING_RESULT" = skipped/);
  assert.match(selector, /test "\$STAGING_RESULT" = success/);
  assert.match(selector, /test "\$PRODUCTION_RESULT" = skipped/);
  assert.match(selector, /https:\/\/docs\.google\.com\/forms\/d\/e\/\*\/viewform/);

  const workflowOutputs = google.slice(
    google.indexOf('    outputs:\n'),
    google.indexOf('\n\npermissions:'),
  );
  const workflowOutputNames = [
    ...workflowOutputs.matchAll(/^      ([a-z][a-z-]+):\n        description:/gm),
  ].map((match) => match[1]);
  assert.deepEqual(workflowOutputNames, ['responder-uri']);
  assert.match(
    google,
    /value: \$\{\{ jobs\.select-result\.outputs\['responder-uri'\] \}\}/,
  );

  const actionOutputs = action.slice(
    action.indexOf('outputs:\n'),
    action.indexOf('\nruns:'),
  );
  const actionOutputNames = [...actionOutputs.matchAll(/^  ([a-z][a-z-]+):\n/gm)]
    .map((match) => match[1]);
  assert.deepEqual(actionOutputNames, ['responder-uri']);
  assert.match(selector, /printf 'responder-uri=%s\\n' "\$selected_uri" >>"\$GITHUB_OUTPUT"/);
  assert.doesNotMatch(
    `${google}\n${action}`,
    /GITHUB_OUTPUT.*(?:form-id|folder-id|drive-id|editor)/i,
  );
});

test('composite boolean inputs stay strings across authentication and fault handling', async () => {
  const google = await workflow('progress-prizes-google.yml');
  const action = await googleAction();
  for (const input of ['expect-fault', 'dry-run']) {
    assert.match(
      action,
      new RegExp(`^  ${input}:\\n(?:    .*\\n)*?    default: "false"$`, 'm'),
    );
    const forwarding = `          ${input}: \${{ inputs['${input}'] }}`;
    assert.equal(
      google.split(forwarding).length - 1,
      2,
      `${input} must be passed to both protected jobs`,
    );
  }
  assert.match(action, /inputs\['dry-run'\] == 'true'/);
  assert.match(action, /inputs\['dry-run'\] != 'true'/);
  assert.match(action, /case "\$EXPECT_FAULT" in true\|false/);
  assert.match(action, /case "\$DRY_RUN" in true\|false/);
  assert.match(action, /test "\$DRY_RUN" = false \|\| arguments\+=\(--dry-run\)/);
});

test('rehearsal controls are fixed to staging and its ephemeral branches', async () => {
  const rehearsal = await workflow('progress-prizes-rehearsal.yml');
  assert.match(rehearsal, /^on:\n  workflow_dispatch:/m);
  assert.doesNotMatch(rehearsal, /schedule:|pull_request/);
  assert.doesNotMatch(
    rehearsal,
    /^ {4}secrets:/m,
    'callers must not pass or override the Google job Environment secrets',
  );
  assert.match(rehearsal, /codex\/progress-prize-smoke-20260720/);
  assert.match(rehearsal, /codex\/progress-prize-smoke-base-20260720/);
  assert.match(rehearsal, /fault: after-copy/);
  assert.match(rehearsal, /fault: after-close-source/);
  assert.match(rehearsal, /expect-fault: true/g);
  assert.match(rehearsal, /verify-post-merge-preview/);
  assert.match(rehearsal, /prove-activation-idempotency/);
  assert.match(rehearsal, /verify-cleaned-full-rehearsal/);
  assert.match(rehearsal, /actions: read\n      checks: read/);
  assert.match(rehearsal, /actions: read\n      contents: read\n      statuses: read/);
  assert.match(rehearsal, /--base-sha \"\$EXPECTED_BASE_SHA\"/);
  assert.match(rehearsal, /--source-cycle \"\$SOURCE_CYCLE\"/);

  const pagePr = await workflow('progress-prizes-page-pr.yml');
  assert.match(pagePr, /--force-with-lease=\"refs\/heads\/\$HEAD_BRANCH:\$REMOTE_HEAD_SHA\"/);
  assert.match(pagePr, /--force-with-lease=\"refs\/heads\/\$HEAD_BRANCH:\"/);
  assert.match(pagePr, /--head-sha \"\$HEAD_SHA\"/);
  assert.match(pagePr, /--base-sha \"\$BASE_SHA\"/);
  assert.match(pagePr, /--source-cycle \"\$SOURCE_CYCLE\"/);
  assert.match(pagePr, /--responder-uri \"\$RESPONDER_URI\"/);
  assert.match(pagePr, /test \"\$\{remote_after%%\[\[:space:\]\]\*\}\" = \"\$HEAD_SHA\"/);
  assert.match(pagePr, /test \"\$base_sha\" = \"\$\(git rev-parse refs\/remotes\/origin\/main\^\{commit\}\)\"/);

  const google = await workflow('progress-prizes-google.yml');
  const action = await googleAction();
  assert.doesNotMatch(google, /passedCopyFault|passedCloseFault/);
  assert.match(action, /passedCopyFault/);
  assert.match(action, /passedCloseFault/);
  assert.match(action, /result\.created === false/);
  assert.match(action, /result\.resumed === true/);
});

test('Vercel verification runs trusted default-branch code and requires GitHub association', async () => {
  const vercel = await workflow('progress-prizes-vercel-preview.yml');
  assert.match(vercel, /environment: progress-prizes-preview/);
  assert.match(vercel, /github\.actor == 'vercel\[bot\]'/);
  assert.match(vercel, /run-name: Progress Prize Vercel preview \$\{\{ github\.event\.client_payload\.git\.sha \}\}/);
  assert.match(vercel, /ref: refs\/heads\/main/);
  assert.match(vercel, /pulls\?state=open&head=/);
  assert.match(vercel, /git\/ref\/heads/);
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
  assert.match(vercel, /!process\.env\.VERCEL_AUTOMATION_BYPASS_SECRET/);
  assert.match(vercel, /redirect: 'error'/);
  assert.doesNotMatch(vercel, /id-token|google-github-actions|GOOGLE_/);
  assert.doesNotMatch(vercel, /checkout.*(?:HEAD_SHA|deployment|pull_request)/i);
});

test('trusted branch and exact-check gate helpers reject ambiguous automation state', () => {
  assert.deepEqual(
    assertAutomationBranch(
      'codex/progress-prize-smoke-20260720',
      'codex/progress-prize-smoke-base-20260720',
    ).kind,
    'smoke',
  );
  assert.equal(assertAutomationBranch('codex/progress-prize-2026-08', 'main').kind, 'production');
  assert.throws(() => assertAutomationBranch('feature/untrusted', 'main'));

  const expectedSha = 'a'.repeat(40);
  const baseSha = 'b'.repeat(40);
  const statuses = {
    statuses: [{
      context: 'progress-prizes/vercel-preview',
      state: 'success',
      creator: { login: 'github-actions[bot]' },
      target_url: 'https://github.com/ScrollPrize/villa/actions/runs/12345',
    }],
  };
  const previewRun = {
    id: 12345,
    html_url: statuses.statuses[0].target_url,
    name: 'Progress Prize Vercel preview gate',
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
  assert.equal(isTrustedPreviewRun({ status: statuses.statuses[0], run: previewRun, expectedSha }), true);
  assert.equal(gateSnapshot({ statuses, checks: successfulChecks, previewRun, expectedSha }).ready, true);
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
        { name: 'Failing check', status: 'completed', conclusion: 'failure' },
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
    statuses: {
      statuses: [{
        ...statuses.statuses[0],
        creator: { login: 'spoofed-user' },
      }],
    },
    checks: successfulChecks,
    previewRun,
    expectedSha,
  }).ready, false);

  for (const [field, value] of [
    ['name', 'Untrusted workflow'],
    ['path', '.github/workflows/untrusted.yml'],
    ['event', 'workflow_dispatch'],
    ['head_branch', 'feature/untrusted'],
    ['status', 'in_progress'],
    ['conclusion', 'failure'],
    ['display_title', `Progress Prize Vercel preview ${baseSha}`],
  ]) {
    assert.equal(isTrustedPreviewRun({
      status: statuses.statuses[0],
      run: { ...previewRun, [field]: value },
      expectedSha,
    }), false);
  }
  assert.equal(isTrustedPreviewRun({
    status: statuses.statuses[0],
    run: { ...previewRun, actor: { login: 'attacker', type: 'User' } },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses.statuses[0],
    run: { ...previewRun, triggering_actor: { login: 'human', type: 'User' } },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses.statuses[0],
    run: {
      ...previewRun,
      repository: { ...previewRun.repository, id: 1 },
    },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: statuses.statuses[0],
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
    status: statuses.statuses[0],
    run: { ...previewRun, head_repository: { id: 1 } },
    expectedSha,
  }), false);
  assert.equal(isTrustedPreviewRun({
    status: { ...statuses.statuses[0], target_url: 'https://github.com/ScrollPrize/villa/actions/runs/999' },
    run: previewRun,
    expectedSha,
  }), false);
});

test('the GitHub helper binds the PR and exact deterministic page-only commit', () => {
  const headSha = 'a'.repeat(40);
  const baseSha = 'b'.repeat(40);
  const head = 'codex/progress-prize-smoke-20260720';
  const base = 'codex/progress-prize-smoke-base-20260720';
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
  assert.throws(() => assertSinglePageCommit({
    ...commit,
    files: [...commit.files, { filename: '.github/workflows/untrusted.yml', status: 'added' }],
  }, { headSha, baseSha }), /one page-only commit/);
});
