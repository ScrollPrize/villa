import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PreviewVerificationError,
  assertVercelPreviewUrl,
  fetchExpectedProgressPrizeState,
  validateVercelDispatch,
  verifyVercelPreview,
} from './vercel-preview.mjs';

const projectId = 'project_internal_identifier';
const sha = '0123456789abcdef0123456789abcdef01234567';
const ref = 'refs/heads/codex/progress-prize-smoke-20260720';
const responderUri = 'https://forms.gle/SmokeForm_20260720';
const bypassSecret = 'vercel-bypass-private-value';

function dispatch(overrides = {}) {
  const clientPayload = {
    environment: 'preview',
    project: { id: projectId },
    url: 'https://villa-smoke-abc123.vercel.app',
    git: {
      sha,
      ref,
    },
    ...(overrides.client_payload ?? {}),
  };
  return {
    action: 'vercel.deployment.ready',
    ...overrides,
    client_payload: clientPayload,
  };
}

function expectedState(overrides = {}) {
  return {
    cycle: '2026-08',
    deadlineLabel: '11:59pm Pacific, August 31st, 2026',
    responderUri,
    ...overrides,
  };
}

function prizePageMarkdown() {
  return [
    '# Prizes',
    '',
    '## Progress Prizes',
    '',
    '{/* progress-prizes:deadline:start */}',
    'Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is 11:59pm Pacific, August 31st, 2026!',
    '{/* progress-prizes:deadline:end */}',
    '',
    '{/* progress-prizes:form:start */}',
    `[Submission Form](${responderUri})`,
    '{/* progress-prizes:form:end */}',
    '',
    '## Terms and Conditions',
    '',
  ].join('\n');
}

function htmlResponse({
  deadline = '11:59pm Pacific, August 31st, 2026',
  href = responderUri,
  status = 200,
  headers = {},
  body,
} = {}) {
  return new Response(
    body ?? `<html><body><p>The next deadline is ${deadline}!</p><a href="${href}">Submission Form</a></body></html>`,
    {
      status,
      headers: { 'content-type': 'text/html; charset=utf-8', ...headers },
    },
  );
}

test('validates Vercel\'s documented repository dispatch payload', () => {
  const result = validateVercelDispatch({
    event: dispatch(),
    expectedProjectId: projectId,
    expectedSha: sha,
    expectedRef: ref,
  });

  assert.deepEqual(result, {
    action: 'vercel.deployment.ready',
    environment: 'preview',
    deploymentOrigin: 'https://villa-smoke-abc123.vercel.app',
    sha,
    ref,
  });
});

test('requires every supplied deployment alias to be consistent', () => {
  const event = dispatch({
    event_type: 'vercel.deployment.ready',
    client_payload: {
      target: 'preview',
      project: { id: projectId },
      deployment: {
        url: 'https://villa-smoke-abc123.vercel.app',
        meta: { githubCommitSha: sha, githubCommitRef: ref },
      },
    },
  });
  const result = validateVercelDispatch({
    event,
    expectedProjectId: projectId,
    expectedSha: sha.toUpperCase(),
    expectedRef: ref,
  });

  assert.deepEqual(result, {
    action: 'vercel.deployment.ready',
    environment: 'preview',
    deploymentOrigin: 'https://villa-smoke-abc123.vercel.app',
    sha,
    ref,
  });

  assert.throws(
    () => validateVercelDispatch({
      event: dispatch({ client_payload: { target: 'production' } }),
      expectedProjectId: projectId,
      expectedSha: sha,
      expectedRef: ref,
    }),
    (error) => error instanceof PreviewVerificationError && error.code === 'EVENT',
  );

  const bareRef = 'codex/progress-prize-smoke-20260720';
  assert.equal(
    validateVercelDispatch({
      event: dispatch({
        client_payload: {
          git: { sha, ref: bareRef },
        },
      }),
      expectedProjectId: projectId,
      expectedSha: sha,
      expectedRef: bareRef,
    }).ref,
    bareRef,
  );
});

test('rejects non-preview dispatches and project, SHA, or ref mismatches without disclosing values', () => {
  const cases = [
    dispatch({ action: 'deployment.ready' }),
    dispatch({ client_payload: { environment: 'production' } }),
    dispatch({ client_payload: { project: { id: 'wrong-private-project' } } }),
    dispatch({ client_payload: { git: { sha: 'f'.repeat(40), ref } } }),
    dispatch({ client_payload: { git: { sha, ref: 'refs/heads/wrong-private-ref' } } }),
  ];

  for (const event of cases) {
    assert.throws(
      () => validateVercelDispatch({
        event,
        expectedProjectId: projectId,
        expectedSha: sha,
        expectedRef: ref,
      }),
      (error) => {
        assert.ok(error instanceof PreviewVerificationError);
        const serialized = `${error}\n${JSON.stringify(error)}`;
        assert.doesNotMatch(serialized, /project_internal_identifier|wrong-private|0123456789abcdef/);
        return true;
      },
    );
  }
});

test('allows only root HTTPS subdomains of vercel.app', () => {
  assert.equal(
    assertVercelPreviewUrl('https://project-team.vercel.app'),
    'https://project-team.vercel.app',
  );
  assert.equal(
    assertVercelPreviewUrl('https://nested.project.vercel.app/'),
    'https://nested.project.vercel.app',
  );

  for (const url of [
    'http://project.vercel.app',
    'https://vercel.app',
    'https://project.vercel.app.evil.example',
    'https://project-vercel.app',
    'https://project.vercel.app@evil.example',
    'https://user:password@project.vercel.app',
    'https://project.vercel.app:443',
    'https://project.vercel.app.evil.example/',
    'https://project.vercel.app/prizes',
    'https://project.vercel.app/?redirect=https://evil.example',
    'https://project.vercel.app./',
  ]) {
    assert.throws(
      () => assertVercelPreviewUrl(url),
      (error) => error instanceof PreviewVerificationError && error.code === 'HOST',
    );
  }
});

test('reads the exact managed state from the trusted GitHub contents API at the commit SHA', async () => {
  const calls = [];
  const githubToken = 'github-private-token';
  const result = await fetchExpectedProgressPrizeState({
    owner: 'ScrollPrize',
    repo: 'villa',
    commitSha: sha,
    githubToken,
    fetchImpl: async (url, options) => {
      calls.push({ url: new URL(url), options });
      return new Response(prizePageMarkdown(), {
        status: 200,
        headers: { 'content-type': 'text/plain; charset=utf-8' },
      });
    },
  });

  assert.deepEqual(result, expectedState());
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url.origin, 'https://api.github.com');
  assert.equal(
    calls[0].url.pathname,
    '/repos/ScrollPrize/villa/contents/scrollprize.org/docs/34_prizes.md',
  );
  assert.equal(calls[0].url.searchParams.get('ref'), sha);
  assert.equal(calls[0].options.redirect, 'error');
  assert.equal(calls[0].options.headers.authorization, `Bearer ${githubToken}`);
  assert.doesNotMatch(calls[0].url.href, /github-private-token/);
});

test('verifies the exact deployed page and sends bypass only to an already validated host', async () => {
  const calls = [];
  const result = await verifyVercelPreview({
    event: dispatch(),
    expectedProjectId: projectId,
    expectedSha: sha,
    expectedRef: ref,
    expectedCycle: '2026-08',
    expectedState: expectedState(),
    protectionBypassSecret: bypassSecret,
    fetchImpl: async (url, options) => {
      calls.push({ url: new URL(url), options });
      return htmlResponse({
        body: `<html><body><p>11:59pm Pacific, <strong>August 31st, 2026</strong></p><a href="${responderUri}">Submission Form</a></body></html>`,
      });
    },
  });

  assert.equal(result.ok, true);
  assert.equal(result.cycle, '2026-08');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url.href, 'https://villa-smoke-abc123.vercel.app/prizes');
  assert.equal(calls[0].options.redirect, 'manual');
  assert.equal(calls[0].options.headers['x-vercel-protection-bypass'], bypassSecret);
});

test('infers the cycle only from managed state fetched at the exact GitHub SHA', async () => {
  const calls = [];
  const githubToken = 'github-private-token';
  const result = await verifyVercelPreview({
    event: dispatch(),
    expectedProjectId: projectId,
    expectedSha: sha,
    expectedRef: ref,
    github: { owner: 'ScrollPrize', repo: 'villa', githubToken },
    protectionBypassSecret: bypassSecret,
    fetchImpl: async (url, options) => {
      const parsedUrl = new URL(url);
      calls.push({ url: parsedUrl, options });
      if (parsedUrl.origin === 'https://api.github.com') {
        return new Response(prizePageMarkdown(), { status: 200 });
      }
      return htmlResponse();
    },
  });

  assert.equal(result.ok, true);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url.origin, 'https://api.github.com');
  assert.equal(calls[0].options.headers.authorization, `Bearer ${githubToken}`);
  assert.equal(calls[0].options.headers['x-vercel-protection-bypass'], undefined);
  assert.equal(calls[1].url.origin, 'https://villa-smoke-abc123.vercel.app');
  assert.equal(calls[1].options.headers.authorization, undefined);
  assert.equal(calls[1].options.headers['x-vercel-protection-bypass'], bypassSecret);
});

test('does not infer a cycle from injected expected state', async () => {
  await assert.rejects(
    verifyVercelPreview({
      event: dispatch(),
      expectedProjectId: projectId,
      expectedSha: sha,
      expectedRef: ref,
      expectedState: expectedState(),
      fetchImpl: async () => htmlResponse(),
    }),
    (error) => error instanceof PreviewVerificationError && error.code === 'EXPECTED',
  );
});

test('rejects redirects without following them or leaking the bypass secret', async () => {
  const calls = [];
  await assert.rejects(
    verifyVercelPreview({
      event: dispatch(),
      expectedProjectId: projectId,
      expectedSha: sha,
      expectedRef: ref,
      expectedCycle: '2026-08',
      expectedState: expectedState(),
      protectionBypassSecret: bypassSecret,
      fetchImpl: async (url, options) => {
        calls.push({ url: new URL(url), options });
        return new Response(null, {
          status: 307,
          headers: { location: `https://evil.example/${bypassSecret}` },
        });
      },
    }),
    (error) => {
      assert.ok(error instanceof PreviewVerificationError);
      assert.equal(error.code, 'REDIRECT');
      assert.doesNotMatch(`${error}\n${JSON.stringify(error)}`, new RegExp(bypassSecret));
      return true;
    },
  );
  assert.equal(calls.length, 1);
  assert.equal(calls[0].options.redirect, 'manual');
});

test('rejects an unsafe host before fetch and never exposes the bypass header', async () => {
  let called = false;
  await assert.rejects(
    verifyVercelPreview({
      event: dispatch({ client_payload: { url: 'https://preview.vercel.app.evil.example' } }),
      expectedProjectId: projectId,
      expectedSha: sha,
      expectedRef: ref,
      expectedCycle: '2026-08',
      expectedState: expectedState(),
      protectionBypassSecret: bypassSecret,
      fetchImpl: async () => {
        called = true;
        return htmlResponse();
      },
    }),
    (error) => error instanceof PreviewVerificationError && error.code === 'HOST',
  );
  assert.equal(called, false);
});

test('requires the rendered deadline and an anchor with the exact responder URI', async () => {
  async function expectFailure(body, code) {
    await assert.rejects(
      verifyVercelPreview({
        event: dispatch(),
        expectedProjectId: projectId,
        expectedSha: sha,
        expectedRef: ref,
        expectedCycle: '2026-08',
        expectedState: expectedState(),
        fetchImpl: async () => htmlResponse({ body }),
      }),
      (error) => error instanceof PreviewVerificationError && error.code === code,
    );
  }

  await expectFailure(
    `<html><body><script>"11:59pm Pacific, August 31st, 2026"</script><a href="${responderUri}">Form</a></body></html>`,
    'DEADLINE',
  );
  await expectFailure(
    `<html><body><p>11:59pm Pacific, August 31st, 2026</p><p>${responderUri}</p><script><a href="${responderUri}">not rendered</a></script><a href="${responderUri}-lookalike">Form</a></body></html>`,
    'RESPONDER',
  );
});

test('rejects redirect-like preview paths before issuing a request', async () => {
  let called = false;
  for (const previewPath of ['//evil.example/prizes', '/\\evil.example/prizes']) {
    await assert.rejects(
      verifyVercelPreview({
        event: dispatch(),
        expectedProjectId: projectId,
        expectedSha: sha,
        expectedRef: ref,
        expectedCycle: '2026-08',
        expectedState: expectedState(),
        previewPath,
        fetchImpl: async () => {
          called = true;
          return htmlResponse();
        },
      }),
      (error) => error instanceof PreviewVerificationError && error.code === 'HOST',
    );
  }
  assert.equal(called, false);
});

test('uses fixed errors that do not include response bodies or fetch failures', async () => {
  const privateResponse = `private editor id and ${bypassSecret}`;
  await assert.rejects(
    verifyVercelPreview({
      event: dispatch(),
      expectedProjectId: projectId,
      expectedSha: sha,
      expectedRef: ref,
      expectedCycle: '2026-08',
      expectedState: expectedState(),
      protectionBypassSecret: bypassSecret,
      fetchImpl: async () => new Response(privateResponse, { status: 500 }),
    }),
    (error) => {
      assert.equal(error.code, 'STATUS');
      assert.doesNotMatch(`${error}\n${JSON.stringify(error)}`, /private editor|vercel-bypass/);
      return true;
    },
  );

  await assert.rejects(
    verifyVercelPreview({
      event: dispatch(),
      expectedProjectId: projectId,
      expectedSha: sha,
      expectedRef: ref,
      expectedCycle: '2026-08',
      expectedState: expectedState(),
      protectionBypassSecret: bypassSecret,
      fetchImpl: async () => { throw new Error(privateResponse); },
    }),
    (error) => {
      assert.equal(error.code, 'REQUEST');
      assert.doesNotMatch(`${error}\n${JSON.stringify(error)}`, /private editor|vercel-bypass/);
      return true;
    },
  );
});
