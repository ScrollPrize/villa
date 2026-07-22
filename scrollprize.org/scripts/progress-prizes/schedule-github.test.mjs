import assert from 'node:assert/strict';
import test from 'node:test';

import {
  MAX_GITHUB_RESPONSE_BYTES,
  NONTERMINAL_RUN_STATUSES,
  buildProductionDispatch,
  dedupeProductionDispatch,
  dispatchProductionWorkflow,
  inspectProductionPrepare,
  inspectProductionWorkflowRuns,
  runSchedulerCli,
} from '../../../.github/progress-prizes-schedule.mjs';

const API = 'https://api.github.com';
const WEB = 'https://github.com';
const OWNER = 'ScrollPrize';
const REPO = 'villa';
const REPOSITORY_ID = 890972577;
const TOKEN = 'github_pat_private_test_value';
const SAFE_FAILURE = 'Progress Prize scheduler GitHub coordination failed safely.';
const MAIN_SHA = 'a'.repeat(40);
const STALE_SHA = 'b'.repeat(40);
const HEAD_SHA = 'c'.repeat(40);
const TARGET_CYCLE = '2026-08';
const BRANCH = `codex/progress-prize-${TARGET_CYCLE}`;
const WORKFLOW_RUNS_PATH = '/repos/ScrollPrize/villa/actions/workflows/progress-prizes-production.yml/runs';

function repository() {
  return { id: REPOSITORY_ID, full_name: `${OWNER}/${REPO}` };
}

function jsonResponse(body, { status = 200, headers = {} } = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json', ...headers },
  });
}

function workflowRun(status, id = 1234) {
  return {
    id,
    status,
    event: 'workflow_dispatch',
    head_branch: 'main',
    head_sha: 'd'.repeat(40),
    path: '.github/workflows/progress-prizes-production.yml',
    url: `${API}/repos/${OWNER}/${REPO}/actions/runs/${id}`,
    html_url: `${WEB}/${OWNER}/${REPO}/actions/runs/${id}`,
    repository: repository(),
    head_repository: repository(),
  };
}

function assertTrustedRequest(url, options, { method = 'GET' } = {}) {
  assert.equal(options.method, method);
  assert.equal(options.redirect, 'error');
  assert.equal(options.headers.accept, 'application/vnd.github+json');
  assert.equal(options.headers.authorization, `Bearer ${TOKEN}`);
  assert.equal(options.headers['content-type'], 'application/json');
  assert.equal(options.headers['user-agent'], 'progress-prize-trusted-scheduler');
  assert.equal(options.headers['x-github-api-version'], '2022-11-28');
  return new URL(url);
}

function runsResponse(parsed, { runsByStatus = {}, recentRuns = [] } = {}) {
  assert.equal(parsed.pathname, WORKFLOW_RUNS_PATH);
  assert.equal(parsed.searchParams.get('branch'), 'main');
  assert.equal(parsed.searchParams.get('event'), 'workflow_dispatch');
  const status = parsed.searchParams.get('status');
  if (status === null) {
    assert.equal(parsed.searchParams.get('per_page'), '10');
    return jsonResponse({ total_count: recentRuns.length, workflow_runs: recentRuns });
  }
  assert.equal(parsed.searchParams.get('per_page'), '100');
  assert.equal(NONTERMINAL_RUN_STATUSES.includes(status), true);
  const runs = runsByStatus[status] ?? [];
  return jsonResponse({ total_count: runs.length, workflow_runs: runs });
}

function runListFetch(runsByStatus = {}, { recentRuns = [] } = {}) {
  const calls = [];
  const fetchImpl = async (url, options) => {
    const parsed = assertTrustedRequest(url, options);
    const status = parsed.searchParams.get('status');
    calls.push(status ?? 'recent');
    return runsResponse(parsed, { runsByStatus, recentRuns });
  };
  return { fetchImpl, calls };
}

function mainRefBody() {
  return {
    ref: 'refs/heads/main',
    object: {
      type: 'commit',
      sha: MAIN_SHA,
      url: `${API}/repos/${OWNER}/${REPO}/git/commits/${MAIN_SHA}`,
    },
  };
}

function pull(number = 42) {
  return {
    id: 9876 + number,
    number,
    state: 'open',
    url: `${API}/repos/${OWNER}/${REPO}/pulls/${number}`,
    html_url: `${WEB}/${OWNER}/${REPO}/pull/${number}`,
    head: {
      ref: BRANCH,
      sha: HEAD_SHA,
      repo: repository(),
    },
    base: {
      ref: 'main',
      sha: MAIN_SHA,
      repo: repository(),
    },
  };
}

function pageOnlyCommit(parentSha = MAIN_SHA) {
  return {
    sha: HEAD_SHA,
    url: `${API}/repos/${OWNER}/${REPO}/commits/${HEAD_SHA}`,
    parents: [{ sha: parentSha }],
    files: [{
      filename: 'scrollprize.org/docs/34_prizes.md',
      status: 'modified',
    }],
  };
}

function productionStateFetch({
  pulls = [pull()],
  commit = pageOnlyCommit(),
} = {}) {
  const calls = [];
  const fetchImpl = async (url, options) => {
    const parsed = assertTrustedRequest(url, options);
    calls.push(parsed.href);
    if (parsed.pathname === WORKFLOW_RUNS_PATH) {
      return runsResponse(parsed);
    }
    if (parsed.pathname === `/repos/${OWNER}/${REPO}/git/ref/heads/main`) {
      return jsonResponse(mainRefBody());
    }
    if (parsed.pathname === `/repos/${OWNER}/${REPO}/pulls`) {
      assert.equal(parsed.searchParams.get('state'), 'open');
      assert.equal(parsed.searchParams.get('head'), `${OWNER}:${BRANCH}`);
      assert.equal(parsed.searchParams.get('base'), 'main');
      assert.equal(parsed.searchParams.get('per_page'), '2');
      return jsonResponse(pulls);
    }
    if (parsed.pathname === `/repos/${OWNER}/${REPO}/commits/${HEAD_SHA}`) {
      assert.equal(parsed.searchParams.get('per_page'), '2');
      return jsonResponse(commit);
    }
    assert.fail(`Unexpected mocked request: ${parsed.href}`);
  };
  return { fetchImpl, calls };
}

async function thrownError(promise) {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  assert.fail('Expected operation to fail closed.');
}

test('every nonterminal production state suppresses dispatch', async (t) => {
  for (const [index, status] of NONTERMINAL_RUN_STATUSES.entries()) {
    await t.test(status, async () => {
      const run = workflowRun(status, 1200 + index);
      const { fetchImpl, calls } = runListFetch({ [status]: [run] });
      const result = await inspectProductionWorkflowRuns({ token: TOKEN, fetchImpl });
      assert.deepEqual(result, {
        dispatch: false,
        reason: 'production-run-nonterminal',
        blockingStatus: status,
        blockingRunId: run.id,
        blockingRunUrl: run.html_url,
      });
      assert.deepEqual(new Set(calls), new Set([...NONTERMINAL_RUN_STATUSES, 'recent']));
    });
  }
});

test('production run fixture binds the API path without an @ref suffix to main', async () => {
  const run = workflowRun('in_progress', 29857030793);
  assert.equal(run.path, '.github/workflows/progress-prizes-production.yml');
  assert.equal(run.head_branch, 'main');
  const { fetchImpl } = runListFetch({ in_progress: [run] });
  const result = await inspectProductionWorkflowRuns({ token: TOKEN, fetchImpl });
  assert.equal(result.dispatch, false);
  assert.equal(result.blockingRunId, 29857030793);
});

test('an exact empty scan of every nonterminal state allows dispatch', async () => {
  const { fetchImpl, calls } = runListFetch();
  assert.deepEqual(
    await inspectProductionWorkflowRuns({ token: TOKEN, fetchImpl }),
    { dispatch: true, reason: 'production-idle' },
  );
  assert.deepEqual(new Set(calls), new Set([...NONTERMINAL_RUN_STATUSES, 'recent']));
});

test('a final recent-run snapshot catches requested to queued scan crossings', async () => {
  const blocker = workflowRun('queued', 2026072101);
  let phase = 'requested';
  const fetchImpl = async (url, options) => {
    const parsed = assertTrustedRequest(url, options);
    const status = parsed.searchParams.get('status');
    if (status === 'requested') {
      await new Promise((resolve) => setImmediate(resolve));
      phase = 'queued';
      return runsResponse(parsed);
    }
    if (status === 'queued') {
      assert.equal(phase, 'requested');
      return runsResponse(parsed);
    }
    if (status === null) {
      assert.equal(phase, 'queued');
      return runsResponse(parsed, { recentRuns: [blocker] });
    }
    return runsResponse(parsed);
  };

  assert.deepEqual(
    await inspectProductionWorkflowRuns({ token: TOKEN, fetchImpl }),
    {
      dispatch: false,
      reason: 'production-run-nonterminal',
      blockingStatus: 'queued',
      blockingRunId: blocker.id,
      blockingRunUrl: blocker.html_url,
    },
  );
});

test('recent snapshot validates terminal run identity while allowing an idle result', async () => {
  const completed = workflowRun('completed', 2026072102);
  const { fetchImpl } = runListFetch({}, { recentRuns: [completed] });
  assert.deepEqual(
    await inspectProductionWorkflowRuns({ token: TOKEN, fetchImpl }),
    { dispatch: true, reason: 'production-idle' },
  );

  completed.path = '.github/workflows/a-different-workflow.yml';
  const malformed = runListFetch({}, { recentRuns: [completed] });
  await assert.rejects(
    inspectProductionWorkflowRuns({ token: TOKEN, fetchImpl: malformed.fetchImpl }),
    { message: SAFE_FAILURE },
  );
});

test('run inspection fails closed for malformed, mismatched, and oversized pages', async (t) => {
  const cases = [
    {
      name: 'malformed JSON',
      response: () => new Response(`{"private":"${TOKEN}"`, { status: 200 }),
    },
    {
      name: 'oversized JSON',
      response: () => new Response('x'.repeat(MAX_GITHUB_RESPONSE_BYTES + 1), { status: 200 }),
    },
    {
      name: 'oversized count',
      response: () => jsonResponse({ total_count: 101, workflow_runs: [] }),
    },
    {
      name: 'status mismatch',
      response: () => jsonResponse({
        total_count: 1,
        workflow_runs: [workflowRun('queued')],
      }),
    },
    {
      name: 'repository mismatch',
      response: () => {
        const run = workflowRun('requested');
        run.repository.id += 1;
        return jsonResponse({ total_count: 1, workflow_runs: [run] });
      },
    },
  ];

  for (const item of cases) {
    await t.test(item.name, async () => {
      const fetchImpl = async (url, options) => {
        const parsed = assertTrustedRequest(url, options);
        const status = parsed.searchParams.get('status');
        if (status === 'requested') return item.response();
        return jsonResponse({ total_count: 0, workflow_runs: [] });
      };
      const error = await thrownError(inspectProductionWorkflowRuns({ token: TOKEN, fetchImpl }));
      assert.equal(error.message, SAFE_FAILURE);
      assert.equal(error.message.includes(TOKEN), false);
    });
  }
});

test('prepare skips only one exact page-only PR directly above current main', async () => {
  const { fetchImpl } = productionStateFetch();
  assert.deepEqual(
    await dedupeProductionDispatch({
      operation: 'prepare',
      targetCycle: TARGET_CYCLE,
      token: TOKEN,
      fetchImpl,
    }),
    {
      dispatch: false,
      reason: 'prepare-exact',
      pullNumber: 42,
      pullUrl: `${WEB}/${OWNER}/${REPO}/pull/42`,
    },
  );
});

test('prepare dispatches when its fixed PR does not exist', async () => {
  const { fetchImpl } = productionStateFetch({ pulls: [] });
  assert.deepEqual(
    await inspectProductionPrepare({
      targetCycle: TARGET_CYCLE,
      token: TOKEN,
      fetchImpl,
    }),
    { dispatch: true, reason: 'prepare-required' },
  );
});

test('a stale page-only PR is refreshed rather than skipped', async () => {
  const { fetchImpl } = productionStateFetch({ commit: pageOnlyCommit(STALE_SHA) });
  assert.deepEqual(
    await inspectProductionPrepare({
      targetCycle: TARGET_CYCLE,
      token: TOKEN,
      fetchImpl,
    }),
    {
      dispatch: true,
      reason: 'prepare-refresh',
      pullNumber: 42,
      pullUrl: `${WEB}/${OWNER}/${REPO}/pull/42`,
    },
  );
});

test('a PR whose reported base is not current main is refreshed rather than skipped', async () => {
  const mismatchedPull = pull();
  mismatchedPull.base.sha = STALE_SHA;
  const { fetchImpl } = productionStateFetch({ pulls: [mismatchedPull] });
  assert.deepEqual(
    await inspectProductionPrepare({
      targetCycle: TARGET_CYCLE,
      token: TOKEN,
      fetchImpl,
    }),
    {
      dispatch: true,
      reason: 'prepare-refresh',
      pullNumber: 42,
      pullUrl: `${WEB}/${OWNER}/${REPO}/pull/42`,
    },
  );
});

test('ambiguous and malformed automation PR state fails closed', async (t) => {
  await t.test('two matching PRs', async () => {
    const { fetchImpl } = productionStateFetch({ pulls: [pull(42), pull(43)] });
    await assert.rejects(
      inspectProductionPrepare({ targetCycle: TARGET_CYCLE, token: TOKEN, fetchImpl }),
      { message: SAFE_FAILURE },
    );
  });

  await t.test('non-page-only commit', async () => {
    const commit = pageOnlyCommit();
    commit.files.push({ filename: 'README.md', status: 'modified' });
    const { fetchImpl } = productionStateFetch({ commit });
    await assert.rejects(
      inspectProductionPrepare({ targetCycle: TARGET_CYCLE, token: TOKEN, fetchImpl }),
      { message: SAFE_FAILURE },
    );
  });

  await t.test('wrong fixed branch association', async () => {
    const wrongPull = pull();
    wrongPull.head.ref = 'codex/not-the-fixed-branch';
    const { fetchImpl } = productionStateFetch({ pulls: [wrongPull] });
    await assert.rejects(
      inspectProductionPrepare({ targetCycle: TARGET_CYCLE, token: TOKEN, fetchImpl }),
      { message: SAFE_FAILURE },
    );
  });
});

test('dispatch body is fixed, consecutive, prepared, and numeric', () => {
  assert.deepEqual(
    buildProductionDispatch({
      operation: 'activate',
      sourceCycle: '2026-12',
      targetCycle: '2027-01',
      requestId: 123456,
    }),
    {
      ref: 'main',
      inputs: {
        operation: 'activate',
        'source-cycle': '2026-12',
        'target-cycle': '2027-01',
        'verify-mode': 'prepared',
        'request-id': '123456',
      },
      return_run_details: true,
    },
  );

  assert.equal(
    buildProductionDispatch({
      operation: 'prepare',
      sourceCycle: '2026-07',
      targetCycle: '2026-08',
      requestId: '9007199254740993',
    }).inputs['request-id'],
    '9007199254740993',
  );

  for (const input of [
    { operation: 'none', sourceCycle: '2026-07', targetCycle: '2026-08', requestId: '1' },
    { operation: 'prepare', sourceCycle: '2026-07', targetCycle: '2026-09', requestId: '1' },
    { operation: 'prepare', sourceCycle: '2026-07', targetCycle: '2026-08', requestId: '01' },
    { operation: 'prepare', sourceCycle: '2026-07', targetCycle: '2026-08', requestId: 'not-numeric' },
  ]) {
    assert.throws(() => buildProductionDispatch(input), { message: SAFE_FAILURE });
  }
});

test('dispatch posts only the fixed production request and validates exact child URLs', async () => {
  const childId = 7654321;
  let calls = 0;
  const fetchImpl = async (url, options) => {
    calls += 1;
    const parsed = new URL(url);
    if (parsed.pathname === WORKFLOW_RUNS_PATH) {
      assertTrustedRequest(url, options);
      return runsResponse(parsed);
    }
    assertTrustedRequest(url, options, { method: 'POST' });
    assert.equal(
      parsed.pathname,
      '/repos/ScrollPrize/villa/actions/workflows/progress-prizes-production.yml/dispatches',
    );
    assert.deepEqual(JSON.parse(options.body), {
      ref: 'main',
      inputs: {
        operation: 'dry-run',
        'source-cycle': '2026-07',
        'target-cycle': '2026-08',
        'verify-mode': 'prepared',
        'request-id': '9988',
      },
      return_run_details: true,
    });
    return jsonResponse({
      workflow_run_id: childId,
      run_url: `${API}/repos/${OWNER}/${REPO}/actions/runs/${childId}`,
      html_url: `${WEB}/${OWNER}/${REPO}/actions/runs/${childId}`,
    });
  };

  assert.deepEqual(
    await dispatchProductionWorkflow({
      operation: 'dry-run',
      sourceCycle: '2026-07',
      targetCycle: '2026-08',
      requestId: '9988',
      token: TOKEN,
      fetchImpl,
    }),
    {
      dispatched: true,
      reason: 'production-dispatched',
      runId: childId,
      apiUrl: `${API}/repos/${OWNER}/${REPO}/actions/runs/${childId}`,
      htmlUrl: `${WEB}/${OWNER}/${REPO}/actions/runs/${childId}`,
    },
  );
  assert.equal(calls, NONTERMINAL_RUN_STATUSES.length + 2);
});

test('dispatch rejects non-200, redirect, malformed URLs, and oversized bodies generically', async (t) => {
  const childId = 77;
  const bodies = [
    {
      name: 'non-200',
      response: () => jsonResponse({ private: TOKEN }, { status: 403 }),
    },
    {
      name: 'redirect',
      response: () => new Response(null, { status: 302, headers: { location: 'https://example.test/' } }),
    },
    {
      name: 'wrong child URL',
      response: () => jsonResponse({
        workflow_run_id: childId,
        run_url: `${API}/repos/attacker/repository/actions/runs/${childId}`,
        html_url: `${WEB}/${OWNER}/${REPO}/actions/runs/${childId}`,
      }),
    },
    {
      name: 'non-positive child ID',
      response: () => jsonResponse({ workflow_run_id: 0, run_url: '', html_url: '' }),
    },
    {
      name: 'oversized response',
      response: () => new Response('x'.repeat(MAX_GITHUB_RESPONSE_BYTES + 1), { status: 200 }),
    },
  ];

  for (const item of bodies) {
    await t.test(item.name, async () => {
      const fetchImpl = async (url, options) => {
        const parsed = new URL(url);
        if (parsed.pathname === WORKFLOW_RUNS_PATH) {
          assertTrustedRequest(url, options);
          return runsResponse(parsed);
        }
        assertTrustedRequest(url, options, { method: 'POST' });
        return item.response();
      };
      const error = await thrownError(dispatchProductionWorkflow({
        operation: 'prepare',
        sourceCycle: '2026-07',
        targetCycle: '2026-08',
        requestId: '123',
        token: TOKEN,
        fetchImpl,
      }));
      assert.equal(error.message, SAFE_FAILURE);
      assert.equal(error.message.includes(TOKEN), false);
    });
  }
});

test('dedupe CLI writes only public enums and run identifiers to GITHUB_OUTPUT', async () => {
  const blocker = workflowRun('waiting', 20260720);
  const { fetchImpl } = runListFetch({ waiting: [blocker] });
  let appended;
  const result = await runSchedulerCli([
    'dedupe',
    '--operation', 'activate',
    '--target-cycle', TARGET_CYCLE,
  ], {
    env: { GITHUB_TOKEN: TOKEN, GITHUB_OUTPUT: '/tmp/github-output' },
    fetchImpl,
    appendFileImpl: async (path, content, encoding) => {
      appended = { path, content, encoding };
    },
  });
  assert.equal(result.dispatch, false);
  assert.deepEqual(appended, {
    path: '/tmp/github-output',
    content: [
      'dispatch=false',
      'dedupe-reason=production-run-nonterminal',
      'blocking-status=waiting',
      'blocking-run-id=20260720',
      `blocking-run-url=${WEB}/${OWNER}/${REPO}/actions/runs/20260720`,
      '',
    ].join('\n'),
    encoding: 'utf8',
  });
  assert.equal(appended.content.includes(TOKEN), false);
});

test('dispatch CLI suppresses a run that appears after an idle dedupe without POSTing', async () => {
  const blocker = workflowRun('queued', 2026072199);
  let phase = 'dedupe';
  let posted = false;
  const fetchImpl = async (url, options) => {
    const parsed = new URL(url);
    if (parsed.pathname !== WORKFLOW_RUNS_PATH) {
      posted = true;
      assert.fail('A production dispatch must not be posted while another run is nonterminal.');
    }
    assertTrustedRequest(url, options);
    if (phase === 'dedupe') return runsResponse(parsed);
    const status = parsed.searchParams.get('status');
    return runsResponse(parsed, {
      runsByStatus: status === 'queued' ? { queued: [blocker] } : {},
      recentRuns: status === null ? [blocker] : [],
    });
  };

  let dedupeOutput = '';
  const dedupe = await runSchedulerCli([
    'dedupe',
    '--operation', 'activate',
    '--target-cycle', TARGET_CYCLE,
  ], {
    env: { GITHUB_TOKEN: TOKEN, GITHUB_OUTPUT: '/tmp/dedupe-output' },
    fetchImpl,
    appendFileImpl: async (_path, content) => {
      dedupeOutput = content;
    },
  });
  assert.deepEqual(dedupe, { dispatch: true, reason: 'production-idle' });
  assert.equal(dedupeOutput, 'dispatch=true\ndedupe-reason=production-idle\n');

  phase = 'dispatch';
  let dispatchOutput = '';
  const dispatch = await runSchedulerCli([
    'dispatch',
    '--operation', 'activate',
    '--source-cycle', '2026-07',
    '--target-cycle', TARGET_CYCLE,
    '--request-id', '20260721',
  ], {
    env: { GITHUB_TOKEN: TOKEN, GITHUB_OUTPUT: '/tmp/dispatch-output' },
    fetchImpl,
    appendFileImpl: async (_path, content) => {
      dispatchOutput = content;
    },
  });
  assert.deepEqual(dispatch, {
    dispatched: false,
    reason: 'production-run-nonterminal',
    blockingStatus: 'queued',
    blockingRunId: blocker.id,
    blockingRunUrl: blocker.html_url,
  });
  assert.equal(dispatchOutput, [
    'dispatched=false',
    'dispatch-reason=production-run-nonterminal',
    'blocking-status=queued',
    `blocking-run-id=${blocker.id}`,
    `blocking-run-url=${blocker.html_url}`,
    '',
  ].join('\n'));
  assert.equal(posted, false);
  assert.equal(dispatchOutput.includes(TOKEN), false);
});

test('dispatch CLI exposes only the positive child ID and exact public URLs', async () => {
  const childId = 20260801;
  const fetchImpl = async (url, options) => {
    const parsed = new URL(url);
    if (parsed.pathname === WORKFLOW_RUNS_PATH) {
      assertTrustedRequest(url, options);
      return runsResponse(parsed);
    }
    assertTrustedRequest(url, options, { method: 'POST' });
    return jsonResponse({
      workflow_run_id: childId,
      run_url: `${API}/repos/${OWNER}/${REPO}/actions/runs/${childId}`,
      html_url: `${WEB}/${OWNER}/${REPO}/actions/runs/${childId}`,
    });
  };
  let output = '';
  await runSchedulerCli([
    'dispatch',
    '--operation', 'activate',
    '--source-cycle', '2026-07',
    '--target-cycle', TARGET_CYCLE,
    '--request-id', '445566',
  ], {
    env: { GITHUB_TOKEN: TOKEN, GITHUB_OUTPUT: '/tmp/github-output' },
    fetchImpl,
    appendFileImpl: async (_path, content) => {
      output = content;
    },
  });
  assert.equal(output, [
    'dispatched=true',
    'dispatch-reason=production-dispatched',
    'child-run-id=20260801',
    `child-run-api-url=${API}/repos/${OWNER}/${REPO}/actions/runs/20260801`,
    `child-run-html-url=${WEB}/${OWNER}/${REPO}/actions/runs/20260801`,
    '',
  ].join('\n'));
  assert.equal(output.includes(TOKEN), false);
});

test('CLI failure neither writes outputs nor exposes token or response body', async () => {
  const privateBody = `private-response-${TOKEN}`;
  let appendCalled = false;
  const error = await thrownError(runSchedulerCli([
    'dispatch',
    '--operation', 'prepare',
    '--source-cycle', '2026-07',
    '--target-cycle', TARGET_CYCLE,
    '--request-id', '42',
  ], {
    env: { GITHUB_TOKEN: TOKEN, GITHUB_OUTPUT: '/tmp/github-output' },
    fetchImpl: async (url, options) => {
      const parsed = new URL(url);
      if (parsed.pathname === WORKFLOW_RUNS_PATH) {
        assertTrustedRequest(url, options);
        return runsResponse(parsed);
      }
      assertTrustedRequest(url, options, { method: 'POST' });
      return new Response(privateBody, { status: 500 });
    },
    appendFileImpl: async () => {
      appendCalled = true;
    },
  }));
  assert.equal(error.message, SAFE_FAILURE);
  assert.equal(error.message.includes(TOKEN), false);
  assert.equal(error.message.includes(privateBody), false);
  assert.equal(appendCalled, false);
});
