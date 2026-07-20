import assert from "node:assert/strict";
import test from "node:test";

import {
  GOOGLE_API_SCOPES,
  GOOGLE_API_READ_SCOPES,
  GoogleApiError,
  buildAppPropertiesQuery,
  createGoogleApiClient,
  filterDirectCollaboratorPermissions,
  filterPublishedResponderPermissions,
  isInheritedPermission,
  normalizePermissionForCreate,
  permissionIdentityKey,
  redactForLog,
} from "./google-api.mjs";

function jsonResponse(body, init = {}) {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: {
      "content-type": "application/json",
      ...(init.headers ?? {}),
    },
  });
}

function sequenceFetch(sequence, calls = []) {
  let index = 0;
  return {
    calls,
    fetch: async (url, options) => {
      calls.push({ url: new URL(url), options });
      const result = sequence[index];
      index += 1;
      if (result instanceof Error) throw result;
      if (typeof result === "function") return result(url, options);
      if (result === undefined) throw new Error(`Unexpected fetch call ${index}`);
      return result;
    },
  };
}

function clientFor(sequence, options = {}) {
  const mock = sequenceFetch(sequence);
  return {
    mock,
    client: createGoogleApiClient({
      accessToken: "short-lived-token",
      fetchImpl: mock.fetch,
      sleep: options.sleep ?? (async () => {}),
      random: options.random ?? (() => 0.5),
      now: options.now ?? (() => 0),
      retry: options.retry ?? { maxAttempts: 3, baseDelayMs: 100, maxDelayMs: 5_000, jitterRatio: 0 },
    }),
  };
}

test("requires injected authentication and never puts the token in the URL", async () => {
  assert.throws(() => createGoogleApiClient(), /accessToken or getAccessToken is required/);

  const { client, mock } = clientFor([jsonResponse({ formId: "private-form-id" })]);
  await client.getForm({ formId: "private-form-id" });

  assert.equal(mock.calls[0].options.headers.authorization, "Bearer short-lived-token");
  assert.doesNotMatch(mock.calls[0].url.href, /short-lived-token/);
  assert.equal(mock.calls[0].url.pathname, "/v1/forms/private-form-id");
});

test("retries transient responses with Retry-After and injected sleep", async () => {
  const sleeps = [];
  const { client, mock } = clientFor(
    [
      jsonResponse({ error: { code: 429, status: "RESOURCE_EXHAUSTED" } }, { status: 429, headers: { "retry-after": "2" } }),
      jsonResponse({ formId: "form" }),
    ],
    { sleep: async (milliseconds) => sleeps.push(milliseconds) },
  );

  assert.deepEqual(await client.getForm({ formId: "form" }), { formId: "form" });
  assert.deepEqual(sleeps, [2_000]);
  assert.equal(mock.calls.length, 2);
});

test("retries fetch network failures with deterministic exponential backoff", async () => {
  const sleeps = [];
  const { client } = clientFor(
    [new TypeError("socket contained private host details"), new TypeError("again"), jsonResponse({ id: "file" })],
    { sleep: async (milliseconds) => sleeps.push(milliseconds) },
  );

  await client.getFile({ fileId: "file" });
  assert.deepEqual(sleeps, [100, 200]);
});

test("throws structured errors without retaining Google's private response body", async () => {
  const privateValue = "private-editor-form-id";
  const { client } = clientFor([
    jsonResponse(
      {
        error: {
          code: 403,
          status: "PERMISSION_DENIED",
          message: `No access to ${privateValue}`,
          errors: [{ reason: "insufficientPermissions" }],
        },
      },
      { status: 403 },
    ),
  ]);

  await assert.rejects(
    client.getFile({ fileId: privateValue }),
    (error) => {
      assert.ok(error instanceof GoogleApiError);
      assert.equal(error.status, 403);
      assert.equal(error.googleCode, 403);
      assert.equal(error.googleStatus, "PERMISSION_DENIED");
      assert.deepEqual(error.reasons, ["insufficientPermissions"]);
      assert.doesNotMatch(error.message, new RegExp(privateValue));
      assert.doesNotMatch(JSON.stringify(error), new RegExp(privateValue));
      return true;
    },
  );
});

test("builds an escaped appProperties query and paginates shared-drive file results", async () => {
  const { client, mock } = clientFor([
    jsonResponse({ files: [{ id: "one" }], nextPageToken: "page-2" }),
    jsonResponse({ files: [{ id: "two" }] }),
  ]);

  const files = await client.listFilesByAppProperties({
    appProperties: { cycle: "2026-08", environment: "stage's" },
    parentId: "staging-folder",
    driveId: "shared-drive",
  });

  assert.deepEqual(files, [{ id: "one" }, { id: "two" }]);
  assert.equal(mock.calls.length, 2);
  const first = mock.calls[0].url.searchParams;
  assert.match(first.get("q"), /appProperties has \{ key='cycle' and value='2026-08' \}/);
  assert.match(first.get("q"), /value='stage\\'s'/);
  assert.match(first.get("q"), /'staging-folder' in parents/);
  assert.match(first.get("q"), /trashed = false/);
  assert.equal(first.get("corpora"), "drive");
  assert.equal(first.get("driveId"), "shared-drive");
  assert.equal(first.get("includeItemsFromAllDrives"), "true");
  assert.equal(mock.calls[1].url.searchParams.get("pageToken"), "page-2");
});

test("rejects empty appProperties searches and repeated pagination tokens", async () => {
  assert.throws(() => buildAppPropertiesQuery({}), /at least one key/);

  const { client } = clientFor([
    jsonResponse({ files: [], nextPageToken: "same" }),
    jsonResponse({ files: [], nextPageToken: "same" }),
  ]);
  await assert.rejects(
    client.listFilesByAppProperties({ appProperties: { cycle: "2026-08" } }),
    /repeated pagination token/,
  );
});

test("copies and updates Drive files with appProperties and Shared Drive flags", async () => {
  const { client, mock } = clientFor([
    jsonResponse({ id: "copy" }),
    jsonResponse({ id: "copy", name: "renamed" }),
  ]);

  await client.copyFile({
    fileId: "source/id",
    name: "August 2026 Progress Prizes",
    parentId: "target-folder",
    appProperties: { environment: "staging", cycle: "2026-08" },
  });
  await client.updateFile({
    fileId: "copy",
    name: "renamed",
    appProperties: { state: "prepared" },
    addParentIds: ["archive"],
    removeParentIds: ["active"],
  });

  assert.equal(mock.calls[0].options.method, "POST");
  assert.equal(mock.calls[0].url.pathname, "/drive/v3/files/source%2Fid/copy");
  assert.equal(mock.calls[0].url.searchParams.get("supportsAllDrives"), "true");
  assert.equal(mock.calls[0].url.searchParams.get("ignoreDefaultVisibility"), "true");
  assert.deepEqual(JSON.parse(mock.calls[0].options.body), {
    name: "August 2026 Progress Prizes",
    parents: ["target-folder"],
    appProperties: { environment: "staging", cycle: "2026-08" },
  });

  assert.equal(mock.calls[1].options.method, "PATCH");
  assert.equal(mock.calls[1].url.searchParams.get("addParents"), "archive");
  assert.equal(mock.calls[1].url.searchParams.get("removeParents"), "active");
  assert.deepEqual(JSON.parse(mock.calls[1].options.body), {
    name: "renamed",
    appProperties: { state: "prepared" },
  });
});

test("does not retry non-idempotent Drive copies after an ambiguous network failure", async () => {
  const { client, mock } = clientFor([
    new TypeError("connection closed after the server may have copied the file"),
    jsonResponse({ id: "duplicate" }),
  ]);

  await assert.rejects(
    client.copyFile({
      fileId: "source",
      name: "August 2026 Progress Prizes",
      parentId: "target-folder",
      appProperties: { environment: "staging", cycle: "2026-08" },
    }),
    /network error after 1 attempts/,
  );
  assert.equal(mock.calls.length, 1);
});

test("gets, updates the title of, and changes publishing for a form", async () => {
  const { client, mock } = clientFor([
    jsonResponse({ formId: "form", revisionId: "rev-1" }),
    jsonResponse({ form: { info: { title: "August 2026 Progress Prizes" } } }),
    jsonResponse({ publishSettings: { publishState: { isPublished: true, isAcceptingResponses: false } } }),
  ]);

  await client.getForm({ formId: "form" });
  await client.updateFormTitle({
    formId: "form",
    title: "August 2026 Progress Prizes",
    requiredRevisionId: "rev-1",
  });
  await client.setPublishState({
    formId: "form",
    isPublished: true,
    isAcceptingResponses: false,
  });

  assert.equal(mock.calls[1].url.pathname, "/v1/forms/form:batchUpdate");
  assert.deepEqual(JSON.parse(mock.calls[1].options.body), {
    includeFormInResponse: true,
    requests: [
      {
        updateFormInfo: {
          info: { title: "August 2026 Progress Prizes" },
          updateMask: "title",
        },
      },
    ],
    writeControl: { requiredRevisionId: "rev-1" },
  });
  assert.equal(mock.calls[2].url.pathname, "/v1/forms/form:setPublishSettings");
  assert.deepEqual(JSON.parse(mock.calls[2].options.body), {
    publishSettings: {
      publishState: { isPublished: true, isAcceptingResponses: false },
    },
    updateMask: "publishState",
  });
});

test("validates Form write controls and impossible publish states before fetching", async () => {
  const { client, mock } = clientFor([]);

  await assert.rejects(
    client.forms.batchUpdate({
      formId: "form",
      requests: [{}],
      requiredRevisionId: "one",
      targetRevisionId: "two",
    }),
    /either requiredRevisionId or targetRevisionId/,
  );
  await assert.rejects(
    client.setPublishState({ formId: "form", isPublished: false, isAcceptingResponses: true }),
    /cannot accept responses/,
  );
  assert.equal(mock.calls.length, 0);
});

test("paginates, creates, and deletes Drive permissions", async () => {
  const { client, mock } = clientFor([
    jsonResponse({ permissions: [{ id: "p1" }], nextPageToken: "next" }),
    jsonResponse({ permissions: [{ id: "p2" }] }),
    jsonResponse({ id: "p3", type: "group", role: "writer" }),
    new Response(null, { status: 204 }),
  ]);

  assert.deepEqual(await client.getAllPermissions({ fileId: "form" }), [{ id: "p1" }, { id: "p2" }]);
  await client.createPermission({
    fileId: "form",
    permission: { type: "group", role: "writer", emailAddress: "editors@example.org" },
  });
  assert.equal(await client.deletePermission({ fileId: "form", permissionId: "p3" }), undefined);

  assert.equal(mock.calls[1].url.searchParams.get("pageToken"), "next");
  assert.equal(mock.calls[0].url.searchParams.get("includePermissionsForView"), "published");
  assert.equal(mock.calls[2].options.method, "POST");
  assert.equal(mock.calls[2].url.searchParams.get("sendNotificationEmail"), "false");
  assert.deepEqual(JSON.parse(mock.calls[2].options.body), {
    type: "group",
    role: "writer",
    emailAddress: "editors@example.org",
  });
  assert.equal(mock.calls[3].options.method, "DELETE");
  assert.equal(mock.calls[3].url.pathname, "/drive/v3/files/form/permissions/p3");
});

test("permission helpers conservatively select and normalize direct collaborators", () => {
  const permissions = [
    { type: "group", role: "writer", emailAddress: "Editors@Example.org" },
    { type: "user", role: "commenter", emailAddress: "person@example.org", permissionDetails: [{ inherited: false }] },
    { type: "user", role: "writer", emailAddress: "inherited@example.org", permissionDetails: [{ inherited: true }] },
    { type: "anyone", role: "reader", allowFileDiscovery: false, view: "published" },
    { type: "user", role: "owner", emailAddress: "owner@example.org" },
    { type: "group", role: "writer", emailAddress: "deleted@example.org", deleted: true },
  ];

  assert.equal(isInheritedPermission(permissions[2]), true);
  assert.deepEqual(filterDirectCollaboratorPermissions(permissions), permissions.slice(0, 2));
  assert.deepEqual(filterPublishedResponderPermissions(permissions), [permissions[3]]);
  assert.deepEqual(normalizePermissionForCreate(permissions[0]), {
    type: "group",
    role: "writer",
    emailAddress: "editors@example.org",
  });
  assert.equal(
    permissionIdentityKey(permissions[0]),
    "group:writer:editors@example.org:::",
  );
  assert.deepEqual(normalizePermissionForCreate(permissions[3]), {
    type: "anyone",
    role: "reader",
    allowFileDiscovery: false,
    view: "published",
  });
});

test("redacts tokens, internal IDs, ACL identities, and editor links but keeps responderUri public", () => {
  const responderUri = "https://docs.google.com/forms/d/e/public-id/viewform";
  const redacted = redactForLog(
    {
      formId: "private-form-id",
      permission: { emailAddress: "editor@example.org" },
      responderUri,
      note: "Bearer token-value and secret-value at https://docs.google.com/forms/d/editor-id/viewform",
    },
    { secrets: ["secret-value"] },
  );

  assert.deepEqual(redacted, {
    formId: "[REDACTED]",
    permission: { emailAddress: "[REDACTED]" },
    responderUri,
    note: "Bearer [REDACTED] and [REDACTED] at [REDACTED]",
  });
});

test("exports read-only validation and ACL-bounded headless write scopes", () => {
  assert.deepEqual(GOOGLE_API_READ_SCOPES, [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/forms.body.readonly",
  ]);
  assert.deepEqual(GOOGLE_API_SCOPES, [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/forms.body",
  ]);
});
