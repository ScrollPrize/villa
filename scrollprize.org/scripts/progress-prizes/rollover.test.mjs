import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PROGRESS_PRIZE_MARKERS,
  getCycleDeadline,
} from './index.mjs';
import {
  ROLLOVER_FAULTS,
  ROLLOVER_FILE_ROLES,
  ROLLOVER_FILE_STATES,
  ROLLOVER_MANAGED_BY,
  RolloverFaultError,
  assertRolloverRuntimeSafety,
  createRolloverService,
} from './rollover.mjs';

const LIVE_FORM_ID = 'private-live-form';
const LIVE_RESPONDER = 'https://docs.google.com/forms/d/e/live-public-id/viewform';
const LIVE_SHORT_URL = 'https://forms.gle/livePublic';
const STAGING_EDITOR = 'progress-prizes-staging@example.org';

const STAGING_EDITOR_PERMISSION = Object.freeze({
  type: 'group',
  role: 'writer',
  emailAddress: STAGING_EDITOR,
});

const PRODUCTION_EDITOR_PERMISSION = Object.freeze({
  type: 'group',
  role: 'writer',
  emailAddress: 'private-team@example.org',
});

const PUBLIC_PERMISSION = Object.freeze({
  id: 'published-reader',
  type: 'anyone',
  role: 'reader',
  allowFileDiscovery: false,
  view: 'published',
});

function clone(value) {
  return value === undefined ? undefined : structuredClone(value);
}

function pageMarkdown(cycle = '2026-07', responderUri = LIVE_SHORT_URL) {
  const deadline = getCycleDeadline(cycle);
  return [
    '# Prize page',
    '',
    '## Progress Prizes',
    '',
    PROGRESS_PRIZE_MARKERS.deadlineStart,
    `Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is ${deadline.label}!`,
    PROGRESS_PRIZE_MARKERS.deadlineEnd,
    '',
    '<details>',
    '<summary>Submission criteria and requirements</summary>',
    'Requirements stay untouched.',
    '</details>',
    '',
    PROGRESS_PRIZE_MARKERS.formStart,
    `[Submission Form](${responderUri})`,
    PROGRESS_PRIZE_MARKERS.formEnd,
    '',
    '***',
    '',
    '## Terms and Conditions',
    '',
  ].join('\n');
}

class MemoryPage {
  constructor(content = pageMarkdown()) {
    this.content = content;
    this.writes = [];
  }

  async read() {
    return this.content;
  }

  async write(content) {
    this.writes.push(content);
    this.content = content;
  }

  async resolveResponderUri(uri) {
    return uri === new URL(LIVE_SHORT_URL).toString() ? LIVE_RESPONDER : uri;
  }
}

class FakeGoogle {
  constructor() {
    this.calls = [];
    this.nextFile = 1;
    this.nextPermission = 1;
    this.files = new Map([
      [LIVE_FORM_ID, {
        id: LIVE_FORM_ID,
        name: 'July 2026 Progress Prizes',
        parents: ['private-production-folder'],
        driveId: 'private-production-drive',
        appProperties: {},
        capabilities: { canCopy: true, canEdit: true, canShare: true },
      }],
    ]);
    this.forms = new Map([
      [LIVE_FORM_ID, {
        formId: LIVE_FORM_ID,
        revisionId: 'revision-1',
        info: {
          title: 'July 2026 Progress Prizes',
          documentTitle: 'July 2026 Progress Prizes',
          description: 'Monthly open-source progress prizes',
        },
        settings: { quizSettings: { isQuiz: false } },
        items: [
          {
            itemId: 'source-name-item',
            title: 'Your full name',
            questionItem: { question: { questionId: 'source-name-question', required: true } },
          },
        ],
        responderUri: LIVE_RESPONDER,
        publishSettings: {
          publishState: { isPublished: true, isAcceptingResponses: true },
        },
      }],
    ]);
    this.permissions = new Map([
      [LIVE_FORM_ID, [
        { id: 'production-editor', type: 'group', role: 'writer', emailAddress: 'private-team@example.org' },
        PUBLIC_PERMISSION,
      ]],
    ]);
  }

  record(method, details = {}) {
    this.calls.push({ method, ...clone(details) });
  }

  requireFile(fileId) {
    const file = this.files.get(fileId);
    if (!file) throw new Error(`Missing fake file ${fileId}`);
    return file;
  }

  requireForm(formId) {
    const form = this.forms.get(formId);
    if (!form) throw new Error(`Missing fake form ${formId}`);
    return form;
  }

  async getForm({ formId }) {
    this.record('getForm', { formId });
    return clone(this.requireForm(formId));
  }

  async getFile({ fileId }) {
    this.record('getFile', { fileId });
    return clone(this.requireFile(fileId));
  }

  async listFilesByAppProperties({ appProperties, parentId, driveId }) {
    this.record('listFilesByAppProperties', { appProperties, parentId, driveId });
    return [...this.files.values()]
      .filter((file) => Object.entries(appProperties).every(
        ([key, value]) => file.appProperties?.[key] === value,
      ))
      .filter((file) => parentId === undefined || file.parents?.includes(parentId))
      .map(clone);
  }

  async copyFile({ fileId, name, parentId, appProperties }) {
    this.record('copyFile', { fileId, name, parentId, appProperties });
    const sourceForm = this.requireForm(fileId);
    const id = `managed-form-${this.nextFile++}`;
    const file = {
      id,
      name,
      parents: [parentId],
      driveId: parentId === 'private-staging-folder'
        ? 'private-staging-drive'
        : 'private-production-drive',
      appProperties: clone(appProperties),
      capabilities: { canCopy: true, canEdit: true, canShare: true },
    };
    const form = clone(sourceForm);
    form.formId = id;
    form.revisionId = `revision-${this.nextFile}`;
    form.info.documentTitle = name;
    form.items[0].itemId = `copied-item-${this.nextFile}`;
    form.items[0].questionItem.question.questionId = `copied-question-${this.nextFile}`;
    form.responderUri = `https://docs.google.com/forms/d/e/public-${id}/viewform`;
    this.files.set(id, file);
    this.forms.set(id, form);
    this.permissions.set(id, []);
    return clone(file);
  }

  async updateFile({ fileId, name, appProperties, addParentIds, removeParentIds }) {
    this.record('updateFile', { fileId, name, appProperties, addParentIds, removeParentIds });
    const file = this.requireFile(fileId);
    if (name !== undefined) {
      file.name = name;
      this.requireForm(fileId).info.documentTitle = name;
    }
    if (appProperties !== undefined) file.appProperties = clone(appProperties);
    const parents = new Set(file.parents ?? []);
    for (const parent of addParentIds ?? []) parents.add(parent);
    for (const parent of removeParentIds ?? []) parents.delete(parent);
    file.parents = [...parents];
    return clone(file);
  }

  async getAllPermissions({ fileId }) {
    this.record('getAllPermissions', { fileId });
    return clone(this.permissions.get(fileId) ?? []);
  }

  async createPermission({ fileId, permission, sendNotificationEmail }) {
    this.record('createPermission', { fileId, permission, sendNotificationEmail });
    const created = {
      id: `permission-${this.nextPermission++}`,
      ...clone(permission),
    };
    this.permissions.get(fileId).push(created);
    return clone(created);
  }

  async deletePermission({ fileId, permissionId }) {
    this.record('deletePermission', { fileId, permissionId });
    const permissions = this.permissions.get(fileId) ?? [];
    this.permissions.set(fileId, permissions.filter(({ id }) => id !== permissionId));
  }

  async updateFormTitle({ formId, title, requiredRevisionId }) {
    this.record('updateFormTitle', { formId, title, requiredRevisionId });
    const form = this.requireForm(formId);
    form.info.title = title;
    form.revisionId = `${form.revisionId}-updated`;
    return { form: clone(form) };
  }

  async setPublishState({ formId, isPublished, isAcceptingResponses }) {
    this.record('setPublishState', { formId, isPublished, isAcceptingResponses });
    const form = this.requireForm(formId);
    form.publishSettings.publishState = { isPublished, isAcceptingResponses };
    return clone(form.publishSettings);
  }

  managed(role, cycle) {
    return [...this.files.values()].filter((file) =>
      file.appProperties?.managedBy === ROLLOVER_MANAGED_BY
      && file.appProperties?.role === role
      && file.appProperties?.cycle === cycle);
  }
}

function fixedClock(instant) {
  return { now: () => new Date(instant) };
}

function stagingRuntime(overrides = {}) {
  return {
    environment: 'staging',
    eventName: 'workflow_dispatch',
    folderId: 'private-staging-folder',
    stagingFolderId: 'private-staging-folder',
    archiveFolderId: 'private-staging-archive',
    driveId: 'private-staging-drive',
    serviceAccountEmail: 'staging-automation@example.org',
    stagingServiceAccountEmail: 'staging-automation@example.org',
    branch: 'codex/progress-prize-smoke-20260720',
    targetBranch: 'codex/progress-prize-smoke-base-20260720',
    defaultTargetBranch: 'main',
    smokeBranchPrefix: 'codex/progress-prize-smoke-',
    smokeDate: '2026-07-20',
    simulatedNow: '2026-07-26T12:00:00Z',
    ...overrides,
  };
}

function productionRuntime(overrides = {}) {
  return {
    environment: 'production',
    eventName: 'schedule',
    folderId: 'private-production-folder',
    driveId: 'private-production-drive',
    serviceAccountEmail: 'production-automation@example.org',
    stagingServiceAccountEmail: 'staging-automation@example.org',
    branch: 'codex/progress-prize-2026-08',
    targetBranch: 'main',
    defaultTargetBranch: 'main',
    ...overrides,
  };
}

function service({
  google = new FakeGoogle(),
  page = new MemoryPage(),
  clock = fixedClock('2026-07-26T12:00:00Z'),
  runtime = stagingRuntime(),
  activationGate = async () => true,
} = {}) {
  return {
    google,
    page,
    rollover: createRolloverService({ google, page, clock, runtime, activationGate }),
  };
}

async function bootstrapAndPrepare(context) {
  await context.rollover.bootstrapStagingSource({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
  });
  return context.rollover.prepare({
    targetCycle: '2026-08',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    expectedCurrentResponderUri: LIVE_RESPONDER,
  });
}

test('validate performs a read-only production preflight and resolves the public short URL', async () => {
  const google = new FakeGoogle();
  const context = service({
    google,
    runtime: productionRuntime(),
    clock: fixedClock('2026-07-20T12:00:00Z'),
  });

  const result = await context.rollover.validate({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
  });

  assert.equal(result.status, 'valid');
  assert.equal(result.responderUri, LIVE_RESPONDER);
  assert.equal(result.isAcceptingResponses, true);
  assert.equal(result.hasLinkedSheet, false);
  assert.equal(
    google.calls.some(({ method }) => [
      'copyFile',
      'updateFile',
      'createPermission',
      'deletePermission',
      'updateFormTitle',
      'setPublishState',
    ].includes(method)),
    false,
  );
});

test('validate safely stops for linked Sheets and legacy publish settings', async () => {
  const linkedGoogle = new FakeGoogle();
  linkedGoogle.forms.get(LIVE_FORM_ID).linkedSheetId = 'private-sheet';
  const linked = service({
    google: linkedGoogle,
    runtime: productionRuntime(),
  });
  await assert.rejects(
    linked.rollover.validate({ sourceFormId: LIVE_FORM_ID, sourceCycle: '2026-07' }),
    /linked response Sheet/,
  );

  const legacyGoogle = new FakeGoogle();
  delete legacyGoogle.forms.get(LIVE_FORM_ID).publishSettings;
  const legacy = service({
    google: legacyGoogle,
    runtime: productionRuntime(),
  });
  await assert.rejects(
    legacy.rollover.validate({ sourceFormId: LIVE_FORM_ID, sourceCycle: '2026-07' }),
    /modern publishSettings/,
  );
});

test('production preflight requires the configured Shared Drive and explicit capabilities', async () => {
  const wrongDriveGoogle = new FakeGoogle();
  wrongDriveGoogle.files.get(LIVE_FORM_ID).driveId = 'unexpected-drive';
  await assert.rejects(
    service({ google: wrongDriveGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
    }),
    /configured Shared Drive/,
  );
  assert.equal(wrongDriveGoogle.calls.some(({ method }) => method === 'getForm'), false);

  const unknownCapabilityGoogle = new FakeGoogle();
  delete unknownCapabilityGoogle.files.get(LIVE_FORM_ID).capabilities.canShare;
  await assert.rejects(
    service({ google: unknownCapabilityGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
    }),
    /required canShare capability/,
  );
});

test('production preflight verifies the published reader and configured editor group', async () => {
  const missingPublishedGoogle = new FakeGoogle();
  missingPublishedGoogle.permissions.set(LIVE_FORM_ID, [
    { id: 'production-editor', ...PRODUCTION_EDITOR_PERMISSION },
  ]);
  await assert.rejects(
    service({ google: missingPublishedGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /published responder permission/,
  );

  const missingEditorGoogle = new FakeGoogle();
  await assert.rejects(
    service({ google: missingEditorGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    }),
    /configured internal collaborator/,
  );
});

test('bootstrap copies the live form into staging, limits ACLs, and never mutates the live source', async () => {
  const context = service();
  context.google.files.get(LIVE_FORM_ID).capabilities.canEdit = false;
  context.google.files.get(LIVE_FORM_ID).capabilities.canShare = false;
  const before = clone(context.google.forms.get(LIVE_FORM_ID));

  const first = await context.rollover.bootstrapStagingSource({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
  });
  const second = await context.rollover.bootstrapStagingSource({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
  });

  assert.equal(first.created, true);
  assert.equal(second.resumed, true);
  assert.deepEqual(context.google.forms.get(LIVE_FORM_ID), before);
  const sources = context.google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07');
  assert.equal(sources.length, 1);
  assert.equal(sources[0].name, '[SMOKE SOURCE 2026-07-20] July 2026 Progress Prizes');
  assert.equal(sources[0].appProperties.state, ROLLOVER_FILE_STATES.ACTIVE);
  const stagedPermissions = context.google.permissions.get(sources[0].id);
  assert.deepEqual(
    stagedPermissions.map(({ type, role, emailAddress }) => ({ type, role, emailAddress })),
    [
      { type: 'group', role: 'writer', emailAddress: STAGING_EDITOR },
      { type: 'anyone', role: 'reader', emailAddress: undefined },
    ],
  );
  const liveMutations = context.google.calls.filter(({ method, fileId, formId }) =>
    (fileId === LIVE_FORM_ID || formId === LIVE_FORM_ID)
    && ['updateFile', 'createPermission', 'deletePermission', 'updateFormTitle', 'setPublishState'].includes(method));
  assert.deepEqual(liveMutations, []);
});

test('prepare resumes the one appProperties copy after an injected failure and writes only the managed page lines', async () => {
  const context = service();
  await context.rollover.bootstrapStagingSource({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
  });

  await assert.rejects(
    context.rollover.prepare({
      targetCycle: '2026-08',
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
      expectedCurrentResponderUri: LIVE_RESPONDER,
      faultInjection: ROLLOVER_FAULTS.AFTER_COPY,
    }),
    (error) => error instanceof RolloverFaultError && error.step === ROLLOVER_FAULTS.AFTER_COPY,
  );
  const faultedTarget = context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  assert.ok(faultedTarget);
  assert.deepEqual(context.google.forms.get(faultedTarget.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });
  assert.equal(context.page.writes.length, 0);

  const result = await context.rollover.prepare({
    targetCycle: '2026-08',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    expectedCurrentResponderUri: LIVE_RESPONDER,
  });

  assert.equal(result.status, 'prepared');
  assert.equal(result.created, false);
  assert.equal(result.resumed, true);
  assert.equal(result.isPublished, true);
  assert.equal(result.isAcceptingResponses, false);
  assert.equal(context.page.writes.length, 1);
  assert.match(context.page.content, /August 31st, 2026/);
  assert.match(context.page.content, new RegExp(result.responderUri.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  assert.match(context.page.content, /Requirements stay untouched\./);

  const target = context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  assert.equal(target.name, '[SMOKE TARGET 2026-07-20] August 2026 Progress Prizes');
  assert.equal(target.appProperties.state, ROLLOVER_FILE_STATES.PREPARED);
  assert.equal(context.google.forms.get(target.id).info.title, target.name);
  assert.equal(context.google.forms.get(target.id).items[0].title, 'Your full name');
  assert.equal(
    context.google.calls.filter(({ method, appProperties }) =>
      method === 'copyFile' && appProperties?.role === ROLLOVER_FILE_ROLES.TARGET).length,
    1,
  );

  const idempotent = await context.rollover.prepare({
    targetCycle: '2026-08',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    expectedCurrentResponderUri: LIVE_RESPONDER,
  });
  assert.equal(idempotent.pageChanged, false);
  assert.equal(context.page.writes.length, 1);

  assert.deepEqual(await context.rollover.verify({ targetCycle: '2026-08' }), {
    action: 'verify',
    status: 'valid',
    mode: 'prepared',
    sourceCycle: '2026-07',
    targetCycle: '2026-08',
    responderUri: result.responderUri,
  });
});

test('prepare refuses to copy a source that is no longer live', async () => {
  const context = service();
  await context.rollover.bootstrapStagingSource({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
  });
  const source = context.google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
  context.google.forms.get(source.id).publishSettings.publishState.isAcceptingResponses = false;

  await assert.rejects(
    context.rollover.prepare({
      targetCycle: '2026-08',
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    }),
    /publishing state/,
  );
  assert.equal(context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08').length, 0);
});

test('later production cycles discover the previously activated managed target', async () => {
  const google = new FakeGoogle();
  const page = new MemoryPage();
  const july = service({
    google,
    page,
    runtime: productionRuntime(),
    clock: fixedClock('2026-07-26T12:00:00Z'),
  });
  await july.rollover.prepare({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  });
  const augustTarget = google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];

  const cutoff = service({
    google,
    page,
    runtime: productionRuntime(),
    clock: fixedClock('2026-08-01T07:00:01Z'),
  });
  await cutoff.rollover.activate({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    headSha: 'a'.repeat(40),
  });

  const august = service({
    google,
    page,
    runtime: productionRuntime(),
    clock: fixedClock('2026-08-25T12:00:00Z'),
  });
  await august.rollover.prepare({
    targetCycle: '2026-09',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  });

  assert.equal(google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-09').length, 1);
  const septemberCopy = google.calls.findLast(({ method, appProperties }) => (
    method === 'copyFile' && appProperties?.cycle === '2026-09'
  ));
  assert.equal(septemberCopy.fileId, augustTarget.id);
});

test('activate requires the injected preview gate before closing the source', async () => {
  const google = new FakeGoogle();
  const page = new MemoryPage();
  const preparation = service({ google, page });
  await bootstrapAndPrepare(preparation);
  const source = google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];

  const noGate = createRolloverService({
    google,
    page,
    clock: fixedClock('2026-08-01T07:00:01Z'),
    runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
  });
  await assert.rejects(noGate.activate({ targetCycle: '2026-08' }), /requires an injected.*preview gate/);
  assert.equal(
    google.forms.get(source.id).publishSettings.publishState.isAcceptingResponses,
    true,
  );

  const failedGate = createRolloverService({
    google,
    page,
    clock: fixedClock('2026-08-01T07:00:01Z'),
    runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
    activationGate: async () => ({ ok: false }),
  });
  await assert.rejects(failedGate.activate({ targetCycle: '2026-08' }), /gate did not pass/);
  assert.equal(
    google.forms.get(source.id).publishSettings.publishState.isAcceptingResponses,
    true,
  );
});

test('activate recovers after closing the source, opens the target once, and is idempotent', async () => {
  const google = new FakeGoogle();
  const page = new MemoryPage();
  await bootstrapAndPrepare(service({ google, page }));
  const gateCalls = [];
  const active = service({
    google,
    page,
    clock: fixedClock('2026-08-01T07:00:01Z'),
    runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
    activationGate: async (input) => {
      gateCalls.push(input);
      return { ok: true };
    },
  });

  await assert.rejects(
    active.rollover.activate({
      targetCycle: '2026-08',
      faultInjection: ROLLOVER_FAULTS.AFTER_CLOSE_SOURCE,
    }),
    (error) => error instanceof RolloverFaultError
      && error.step === ROLLOVER_FAULTS.AFTER_CLOSE_SOURCE,
  );
  const source = google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
  const target = google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  assert.deepEqual(google.forms.get(source.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });
  assert.deepEqual(google.forms.get(target.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });

  const first = await active.rollover.activate({ targetCycle: '2026-08' });
  const publishCallsAfterFirst = google.calls.filter(({ method }) => method === 'setPublishState').length;
  const second = await active.rollover.activate({ targetCycle: '2026-08' });
  assert.equal(first.status, 'active');
  assert.equal(second.status, 'active');
  assert.equal(
    google.calls.filter(({ method }) => method === 'setPublishState').length,
    publishCallsAfterFirst,
  );
  assert.equal(source.appProperties.state, ROLLOVER_FILE_STATES.CLOSED);
  assert.equal(target.appProperties.state, ROLLOVER_FILE_STATES.ACTIVE);
  assert.equal(gateCalls.length, 3);
  assert.equal(gateCalls[0].responderUri, google.forms.get(target.id).responderUri);
  assert.equal((await active.rollover.verify({ targetCycle: '2026-08', mode: 'active' })).status, 'valid');
});

test('activate rejects publishing drift and resumes an explicitly marked target activation', async () => {
  const driftGoogle = new FakeGoogle();
  const driftPage = new MemoryPage();
  await bootstrapAndPrepare(service({ google: driftGoogle, page: driftPage }));
  const driftTarget = driftGoogle.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  driftGoogle.forms.get(driftTarget.id).publishSettings.publishState.isAcceptingResponses = true;
  const driftService = service({
    google: driftGoogle,
    page: driftPage,
    clock: fixedClock('2026-08-01T07:00:01Z'),
    runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
  });
  await assert.rejects(
    driftService.rollover.activate({ targetCycle: '2026-08' }),
    /allowed activation or recovery state/,
  );
  assert.equal(
    driftGoogle.forms.get(
      driftGoogle.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0].id,
    ).publishSettings.publishState.isAcceptingResponses,
    true,
  );

  const recoveryGoogle = new FakeGoogle();
  const recoveryPage = new MemoryPage();
  await bootstrapAndPrepare(service({ google: recoveryGoogle, page: recoveryPage }));
  const recoverySource = recoveryGoogle.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
  const recoveryTarget = recoveryGoogle.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  recoveryGoogle.forms.get(recoverySource.id).publishSettings.publishState.isAcceptingResponses = false;
  recoverySource.appProperties.state = ROLLOVER_FILE_STATES.CLOSED;
  recoveryGoogle.forms.get(recoveryTarget.id).publishSettings.publishState.isAcceptingResponses = true;
  recoveryTarget.appProperties.state = ROLLOVER_FILE_STATES.ACTIVATING;
  const recovered = service({
    google: recoveryGoogle,
    page: recoveryPage,
    clock: fixedClock('2026-08-01T07:00:01Z'),
    runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
  });
  assert.equal((await recovered.rollover.activate({ targetCycle: '2026-08' })).status, 'active');
  assert.equal(recoveryTarget.appProperties.state, ROLLOVER_FILE_STATES.ACTIVE);
});

test('activate resumes when the source closed but its CLOSED marker write failed', async () => {
  const google = new FakeGoogle();
  const page = new MemoryPage();
  await bootstrapAndPrepare(service({ google, page }));
  const source = google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
  const target = google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  const updateFile = google.updateFile.bind(google);
  let failClosedMarkerOnce = true;
  google.updateFile = async (input) => {
    if (
      failClosedMarkerOnce
      && input.fileId === source.id
      && input.appProperties?.state === ROLLOVER_FILE_STATES.CLOSED
    ) {
      failClosedMarkerOnce = false;
      throw new Error('simulated marker write failure');
    }
    return updateFile(input);
  };
  const active = service({
    google,
    page,
    clock: fixedClock('2026-08-01T07:00:01Z'),
    runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
  });

  await assert.rejects(
    active.rollover.activate({ targetCycle: '2026-08' }),
    /simulated marker write failure/,
  );
  assert.deepEqual(google.forms.get(source.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });
  assert.equal(source.appProperties.state, ROLLOVER_FILE_STATES.ACTIVE);
  assert.deepEqual(google.forms.get(target.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });

  assert.equal((await active.rollover.activate({ targetCycle: '2026-08' })).status, 'active');
  assert.equal(source.appProperties.state, ROLLOVER_FILE_STATES.CLOSED);
  assert.equal(target.appProperties.state, ROLLOVER_FILE_STATES.ACTIVE);
});

test('cleanup is staging-only, unpublishes both forms, removes public access, archives, and reruns safely', async () => {
  const google = new FakeGoogle();
  const page = new MemoryPage();
  await bootstrapAndPrepare(service({ google, page }));
  const active = service({
    google,
    page,
    clock: fixedClock('2026-08-01T07:00:01Z'),
    runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
  });
  await active.rollover.activate({ targetCycle: '2026-08' });

  for (const file of [
    google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0],
    google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0],
  ]) {
    google.permissions.get(file.id).push({
      id: `ordinary-anyone-${file.id}`,
      type: 'anyone',
      role: 'reader',
      allowFileDiscovery: false,
    });
  }

  const first = await active.rollover.cleanup({ targetCycle: '2026-08' });
  const second = await active.rollover.cleanup({ targetCycle: '2026-08' });
  assert.equal(first.archivedCount, 2);
  assert.equal(second.archivedCount, 2);
  for (const file of [
    google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0],
    google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0],
  ]) {
    assert.deepEqual(google.forms.get(file.id).publishSettings.publishState, {
      isPublished: false,
      isAcceptingResponses: false,
    });
    assert.equal(
      google.permissions.get(file.id).some(({ view }) => view === 'published'),
      false,
    );
    assert.equal(
      google.permissions.get(file.id).some(({ id }) => id === `ordinary-anyone-${file.id}`),
      true,
    );
    assert.deepEqual(file.parents, ['private-staging-archive']);
    assert.equal(file.appProperties.state, ROLLOVER_FILE_STATES.ARCHIVED);
  }
  assert.equal((await active.rollover.verify({ targetCycle: '2026-08', mode: 'cleaned' })).status, 'valid');

  page.content = pageMarkdown();
  await assert.rejects(
    active.rollover.bootstrapStagingSource({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    }),
    /configured active folder/,
  );
  assert.equal(google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07').length, 1);
  assert.equal(google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08').length, 1);

  const target = google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  target.parents = ['private-staging-folder'];
  target.appProperties.state = ROLLOVER_FILE_STATES.ACTIVE;
  await assert.rejects(
    active.rollover.verify({ targetCycle: '2026-08', mode: 'cleaned' }),
    /configured archive state/,
  );
});

test('production safety rejects simulated time, faults, alternate bases, and staging identities before I/O', () => {
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({
      eventName: 'workflow_dispatch',
      simulatedNow: '2026-07-26T12:00:00Z',
    })),
    /simulated time is forbidden in production/,
  );
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({ eventName: 'workflow_dispatch' }), {
      faultInjection: ROLLOVER_FAULTS.AFTER_COPY,
    }),
    /fault injection is forbidden in production/,
  );
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({ targetBranch: 'test-base' })),
    /alternate target branch/,
  );
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({
      stagingFolderId: 'private-staging-folder',
    })),
    /supplied a staging folder/,
  );
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({
      serviceAccountEmail: 'staging-automation@example.org',
    })),
    /staging service account/,
  );
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({ eventName: 'pull_request' })),
    /only for schedule or workflow_dispatch/,
  );
});

test('fault injection also requires the staging account, folder, dispatch event, and smoke branch', () => {
  for (const [override, expected] of [
    [{ eventName: 'schedule', simulatedNow: undefined }, /manually dispatched staging run/],
    [{ serviceAccountEmail: 'wrong@example.org' }, /staging service account/],
    [{ folderId: 'wrong-folder' }, /staging folder/],
    [{ branch: 'feature/not-a-smoke-run' }, /smoke branch prefix/],
  ]) {
    assert.throws(
      () => assertRolloverRuntimeSafety(stagingRuntime(override), {
        faultInjection: ROLLOVER_FAULTS.AFTER_COPY,
      }),
      expected,
    );
  }
});

test('daily preparation is a read-only no-op before the seven-day window', async () => {
  const context = service({
    runtime: stagingRuntime({ simulatedNow: '2026-07-20T12:00:00Z' }),
    clock: fixedClock('2026-07-20T12:00:00Z'),
  });
  const result = await context.rollover.prepare({ targetCycle: '2026-08' });
  assert.equal(result.status, 'waiting');
  assert.equal(context.google.calls.length, 0);
  assert.equal(context.page.writes.length, 0);
});
