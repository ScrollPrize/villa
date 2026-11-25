import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PROGRESS_PRIZE_MARKERS,
  getCycleDeadline,
} from './index.mjs';
import {
  ROLLOVER_FILE_ROLES,
  ROLLOVER_FILE_STATES,
  ROLLOVER_MANAGED_BY,
  assertRolloverRuntimeSafety,
  createRolloverService,
} from './rollover.mjs';

const LIVE_FORM_ID = 'private-live-form';
const LIVE_RESPONDER = 'https://docs.google.com/forms/d/e/live-public-id/viewform';
const LIVE_SHORT_URL = 'https://forms.gle/livePublic';
const EXTERNAL_PRODUCTION_EDITOR = 'external-progress-prize-editor@example.net';
const STAGING_SERVICE_ACCOUNT = 'staging-automation@private-project.iam.gserviceaccount.com';
const PRODUCTION_SERVICE_ACCOUNT = 'production-automation@private-project.iam.gserviceaccount.com';
const PRODUCTION_DRIVE_ADMIN = 'production-break-glass@example.org';

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

function sharedDriveManagerDetail(driveId, overrides = {}) {
  return {
    permissionType: 'member',
    role: 'organizer',
    inherited: true,
    inheritedFrom: driveId,
    ...overrides,
  };
}

function sharedDriveManagerPermission({ id, emailAddress, driveId }) {
  return {
    id,
    type: 'user',
    role: 'organizer',
    emailAddress,
    permissionDetails: [sharedDriveManagerDetail(driveId)],
  };
}

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
        mimeType: 'application/vnd.google-apps.form',
        parents: ['private-owner-my-drive-folder'],
        appProperties: {},
        trashed: false,
        capabilities: { canCopy: true, canEdit: false, canShare: false },
      }],
      ['private-production-folder', {
        id: 'private-production-folder',
        name: 'Production active forms',
        mimeType: 'application/vnd.google-apps.folder',
        parents: [],
        driveId: 'private-production-drive',
        appProperties: {},
        trashed: false,
        capabilities: { canAddChildren: true, canShare: true },
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
        { id: 'owner', type: 'user', role: 'owner', emailAddress: 'private-owner@example.org' },
        {
          id: 'production-automation',
          type: 'user',
          role: 'writer',
          emailAddress: PRODUCTION_SERVICE_ACCOUNT,
        },
        {
          id: 'staging-automation',
          type: 'user',
          role: 'reader',
          emailAddress: STAGING_SERVICE_ACCOUNT,
        },
        { id: 'production-editor', type: 'group', role: 'writer', emailAddress: 'private-team@example.org' },
        PUBLIC_PERMISSION,
      ]],
      ['private-production-folder', [
        sharedDriveManagerPermission({
          id: 'production-manager',
          emailAddress: PRODUCTION_SERVICE_ACCOUNT,
          driveId: 'private-production-drive',
        }),
        sharedDriveManagerPermission({
          id: 'production-break-glass-manager',
          emailAddress: PRODUCTION_DRIVE_ADMIN,
          driveId: 'private-production-drive',
        }),
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
      .filter((file) => driveId === undefined || file.driveId === driveId)
      .filter((file) => parentId === undefined || file.parents?.includes(parentId))
      .map(clone);
  }

  async copyFile({ fileId, name, parentId, appProperties }) {
    this.record('copyFile', { fileId, name, parentId, appProperties });
    const sourceForm = this.requireForm(fileId);
    const destination = this.requireFile(parentId);
    const id = `managed-form-${this.nextFile++}`;
    const file = {
      id,
      name,
      mimeType: 'application/vnd.google-apps.form',
      parents: [parentId],
      driveId: destination.driveId,
      appProperties: clone(appProperties),
      trashed: false,
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
    this.permissions.set(id, clone(this.permissions.get(parentId) ?? []));
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

function grantProductionSourceAccess(google) {
  google.files.get(LIVE_FORM_ID).capabilities = {
    canCopy: true,
    canEdit: true,
    canShare: false,
  };
  return google;
}

function productionRuntime(overrides = {}) {
  return {
    environment: 'production',
    eventName: 'schedule',
    folderId: 'private-production-folder',
    driveId: 'private-production-drive',
    driveAdminEmail: PRODUCTION_DRIVE_ADMIN,
    serviceAccountEmail: PRODUCTION_SERVICE_ACCOUNT,
    stagingServiceAccountEmail: STAGING_SERVICE_ACCOUNT,
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
  runtime = productionRuntime(),
} = {}) {
  return {
    google,
    page,
    rollover: createRolloverService({ google, page, clock, runtime }),
  };
}

// A production context whose July source has already been copied into the
// prepared August form; the live form keeps its one-time copy access.
async function preparedProductionContext() {
  const context = service({ google: grantProductionSourceAccess(new FakeGoogle()) });
  const result = await context.rollover.prepare({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  });
  const target = context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  return { ...context, result, target };
}

function productionVerify(context) {
  return context.rollover.verify({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  });
}

test('validate performs a read-only production preflight and resolves the public short URL', async () => {
  const google = grantProductionSourceAccess(new FakeGoogle());
  const context = service({
    google,
    runtime: productionRuntime(),
    clock: fixedClock('2026-07-20T12:00:00Z'),
  });

  const result = await context.rollover.validate({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  });

  assert.equal(result.status, 'valid');
  assert.equal(result.responderUri, LIVE_RESPONDER);
  assert.equal(result.isAcceptingResponses, true);
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

test('validate tolerates a natively linked response Sheet without writing', async () => {
  const linkedGoogle = grantProductionSourceAccess(new FakeGoogle());
  linkedGoogle.forms.get(LIVE_FORM_ID).linkedSheetId = 'private-sheet';
  const linked = service({
    google: linkedGoogle,
    runtime: productionRuntime(),
  });
  const result = await linked.rollover.validate({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  });
  assert.equal(result.status, 'valid');
  assert.equal(
    linkedGoogle.calls.some(({ method }) => [
      'updateFile',
      'deletePermission',
    ].includes(method)),
    false,
  );
});

test('validate still stops safely for legacy publish settings', async () => {

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

  const unpublishedModernGoogle = grantProductionSourceAccess(new FakeGoogle());
  unpublishedModernGoogle.forms.get(LIVE_FORM_ID).publishSettings = {};
  await assert.rejects(
    service({
      google: unpublishedModernGoogle,
      runtime: productionRuntime(),
    }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /not in the expected live publishing state/,
  );
  assert.equal(
    unpublishedModernGoogle.calls.some(({ method }) => method === 'setPublishState'),
    false,
  );
});

test('production accepts only its explicit My Drive bootstrap source with copy and edit access', async () => {
  const wrongDriveGoogle = new FakeGoogle();
  wrongDriveGoogle.files.get(LIVE_FORM_ID).driveId = 'unexpected-drive';
  wrongDriveGoogle.files.get(LIVE_FORM_ID).parents = ['unexpected-folder'];
  await assert.rejects(
    service({ google: wrongDriveGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
    }),
    /configured Shared Drive/,
  );
  assert.equal(wrongDriveGoogle.calls.some(({ method }) => method === 'getForm'), false);

  const disguisedManagedGoogle = grantProductionSourceAccess(new FakeGoogle());
  const disguisedFile = disguisedManagedGoogle.files.get(LIVE_FORM_ID);
  disguisedFile.driveId = 'private-production-drive';
  disguisedFile.parents = ['private-production-folder'];
  disguisedFile.appProperties = {
    managedBy: ROLLOVER_MANAGED_BY,
    schemaVersion: '1',
    environment: 'production',
    role: ROLLOVER_FILE_ROLES.TARGET,
    cycle: '2026-06',
  };
  await assert.rejects(
    service({ google: disguisedManagedGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
    }),
    /explicit fallback cannot claim managed rollover metadata/,
  );
  assert.equal(disguisedManagedGoogle.calls.some(({ method }) => method === 'getForm'), false);

  const missingEditGoogle = new FakeGoogle();
  await assert.rejects(
    service({ google: missingEditGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
    }),
    /required canEdit capability/,
  );

  const leastPrivilegeGoogle = grantProductionSourceAccess(new FakeGoogle());
  assert.equal(leastPrivilegeGoogle.files.get(LIVE_FORM_ID).capabilities.canShare, false);
  assert.equal((await service({
    google: leastPrivilegeGoogle,
    runtime: productionRuntime(),
  }).rollover.validate({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
  })).status, 'valid');

  for (const [mutate, expected] of [
    [(file) => { file.mimeType = 'application/vnd.google-apps.spreadsheet'; }, /not a Google Form/],
    [(file) => { file.trashed = true; }, /source form is trashed/],
  ]) {
    const invalidGoogle = grantProductionSourceAccess(new FakeGoogle());
    mutate(invalidGoogle.files.get(LIVE_FORM_ID));
    await assert.rejects(
      service({ google: invalidGoogle, runtime: productionRuntime() }).rollover.validate({
        sourceFormId: LIVE_FORM_ID,
        sourceCycle: '2026-07',
      }),
      expected,
    );
    assert.equal(invalidGoogle.calls.some(({ method }) => method === 'getForm'), false);
  }

  const driftingGoogle = grantProductionSourceAccess(new FakeGoogle());
  const originalGetFile = driftingGoogle.getFile.bind(driftingGoogle);
  let liveReads = 0;
  driftingGoogle.getFile = async (input) => {
    const file = await originalGetFile(input);
    if (input.fileId === LIVE_FORM_ID && ++liveReads === 2) {
      file.driveId = 'unexpected-drive';
      file.parents = ['unexpected-folder'];
    }
    return file;
  };
  await assert.rejects(
    service({ google: driftingGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
    }),
    /My Drive fallback source changed location during preflight/,
  );
});

test('production preflight verifies the published reader and configured editor group', async () => {
  const missingPublishedGoogle = grantProductionSourceAccess(new FakeGoogle());
  missingPublishedGoogle.permissions.set(LIVE_FORM_ID, [
    {
      id: 'production-automation',
      type: 'user',
      role: 'writer',
      emailAddress: PRODUCTION_SERVICE_ACCOUNT,
    },
    { id: 'production-editor', ...PRODUCTION_EDITOR_PERMISSION },
    { id: 'domain-published', type: 'domain', role: 'reader', domain: 'example.org', view: 'published' },
  ]);
  await assert.rejects(
    service({ google: missingPublishedGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /published responder permission/,
  );

  const missingEditorGoogle = grantProductionSourceAccess(new FakeGoogle());
  await assert.rejects(
    service({ google: missingEditorGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [{
        type: 'group',
        role: 'writer',
        emailAddress: 'unconfigured-editors@example.org',
      }],
    }),
    /configured internal collaborator/,
  );

  const missingAutomationGoogle = grantProductionSourceAccess(new FakeGoogle());
  missingAutomationGoogle.permissions.set(
    LIVE_FORM_ID,
    missingAutomationGoogle.permissions.get(LIVE_FORM_ID).filter(
      ({ emailAddress }) => emailAddress !== PRODUCTION_SERVICE_ACCOUNT,
    ),
  );
  await assert.rejects(
    service({ google: missingAutomationGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /required direct automation access/,
  );

  const extraPublishedGoogle = grantProductionSourceAccess(new FakeGoogle());
  extraPublishedGoogle.permissions.get(LIVE_FORM_ID).push({
    id: 'unexpected-domain-published',
    type: 'domain',
    role: 'reader',
    domain: 'example.org',
    view: 'published',
  });
  await assert.rejects(
    service({ google: extraPublishedGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /unexpected non-public published responder permission/,
  );

  for (const mutate of [
    (permissions) => permissions.filter(
      ({ emailAddress }) => emailAddress !== STAGING_SERVICE_ACCOUNT,
    ),
    (permissions) => [...permissions, {
      id: 'unexpected-human-reader',
      type: 'user',
      role: 'reader',
      emailAddress: 'unexpected-reader@example.org',
    }],
    (permissions) => permissions.map((permission) => (
      permission.emailAddress === STAGING_SERVICE_ACCOUNT
        ? { ...permission, expirationTime: '2027-01-01T00:00:00Z' }
        : permission
    )),
    (permissions) => permissions.map((permission) => (
      permission.emailAddress === STAGING_SERVICE_ACCOUNT
        ? { ...permission, role: 'commenter' }
        : permission
    )),
    (permissions) => permissions.map((permission) => (
      permission.emailAddress === STAGING_SERVICE_ACCOUNT
        ? { ...permission, permissionDetails: [{ inherited: true }] }
        : permission
    )),
    (permissions) => [...permissions, {
      id: 'duplicate-automation-reader',
      type: 'user',
      role: 'reader',
      emailAddress: 'duplicate-reader@private-project.iam.gserviceaccount.com',
    }],
    (permissions) => permissions.map((permission) => (
      permission.emailAddress === STAGING_SERVICE_ACCOUNT
        ? {
          ...permission,
          emailAddress: 'replacement-reader@private-project.iam.gserviceaccount.com',
        }
        : permission
    )),
  ]) {
    const invalidBootstrapReaderGoogle = grantProductionSourceAccess(new FakeGoogle());
    invalidBootstrapReaderGoogle.permissions.set(
      LIVE_FORM_ID,
      mutate(invalidBootstrapReaderGoogle.permissions.get(LIVE_FORM_ID)),
    );
    await assert.rejects(
      service({
        google: invalidBootstrapReaderGoogle,
        runtime: productionRuntime(),
      }).rollover.validate({
        sourceFormId: LIVE_FORM_ID,
        sourceCycle: '2026-07',
        collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
      }),
      /exactly one direct copy-only automation reader/,
    );
  }

  await assert.rejects(
    service({
      google: grantProductionSourceAccess(new FakeGoogle()),
      runtime: productionRuntime({ stagingServiceAccountEmail: undefined }),
    }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /requires the configured staging automation identity/,
  );
});

test('destination folder preflight blocks both dry-run and real copies', async () => {
  for (const mutate of [
    (google, folder) => { folder.driveId = 'unexpected-drive'; },
    (google, folder) => { folder.capabilities.canShare = false; },
    (google, folder) => {
      google.permissions.set(folder.id, [{
        id: 'unexpected-folder-manager',
        type: 'user',
        role: 'organizer',
        emailAddress: 'unexpected-manager@example.org',
        permissionDetails: [{ inherited: true }],
      }]);
    },
  ]) {
    for (const dryRun of [true, false]) {
      const google = grantProductionSourceAccess(new FakeGoogle());
      mutate(google, google.files.get('private-production-folder'));
      await assert.rejects(
        service({ google, runtime: productionRuntime() }).rollover.prepare({
          targetCycle: '2026-08',
          sourceFormId: LIVE_FORM_ID,
          collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
          dryRun,
        }),
        /writable Shared Drive folder|unexpected inherited edit access|inherited Manager permission/,
      );
      assert.equal(google.calls.some(({ method }) => method === 'copyFile'), false);
    }
  }
});

test('destination permits only the exact human break-glass Manager', async () => {
  const google = grantProductionSourceAccess(new FakeGoogle());
  const result = await service({
    google,
    runtime: productionRuntime(),
  }).rollover.prepare({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    dryRun: true,
  });

  assert.equal(result.status, 'planned');
  assert.equal(google.calls.some(({ method }) => method === 'copyFile'), false);
});

test('break-glass access must be one inherited, non-expiring Manager permission', async (t) => {
  for (const scenario of [
    { name: 'writer', role: 'writer', detailRole: 'writer' },
    { name: 'commenter', role: 'commenter', detailRole: 'commenter' },
    { name: 'Content manager', role: 'fileOrganizer', detailRole: 'fileOrganizer' },
    {
      name: 'direct Manager',
      role: 'organizer',
      permissionType: 'file',
      inherited: false,
      inheritedFrom: undefined,
    },
    { name: 'expiring Manager', role: 'organizer', inherited: true, expirationTime: '2026-08-01T00:00:00Z' },
    { name: 'group Manager', role: 'organizer', type: 'group' },
    { name: 'wrong permission type', role: 'organizer', permissionType: 'file' },
    { name: 'wrong detail role', role: 'organizer', detailRole: 'fileOrganizer' },
    { name: 'wrong inheritance source', role: 'organizer', inheritedFrom: 'different-drive' },
  ]) {
    await t.test(scenario.name, async () => {
      const google = grantProductionSourceAccess(new FakeGoogle());
      const admin = google.permissions.get('private-production-folder').find(
        ({ emailAddress }) => emailAddress === PRODUCTION_DRIVE_ADMIN,
      );
      admin.type = scenario.type ?? 'user';
      admin.role = scenario.role;
      admin.permissionDetails = [sharedDriveManagerDetail('private-production-drive', {
        permissionType: scenario.permissionType ?? 'member',
        role: scenario.detailRole ?? 'organizer',
        inherited: scenario.inherited ?? true,
        inheritedFrom: Object.hasOwn(scenario, 'inheritedFrom')
          ? scenario.inheritedFrom
          : 'private-production-drive',
      })];
      if (scenario.expirationTime !== undefined) admin.expirationTime = scenario.expirationTime;

      await assert.rejects(
        service({ google, runtime: productionRuntime() }).rollover.prepare({
          targetCycle: '2026-08',
          sourceFormId: LIVE_FORM_ID,
          collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
          dryRun: true,
        }),
        /break-glass administrator access is not exactly one inherited Manager permission/,
      );
      assert.equal(google.calls.some(({ method }) => method === 'copyFile'), false);
    });
  }

  await t.test('multiple role-source details', async () => {
    const google = grantProductionSourceAccess(new FakeGoogle());
    const admin = google.permissions.get('private-production-folder').find(
      ({ emailAddress }) => emailAddress === PRODUCTION_DRIVE_ADMIN,
    );
    admin.permissionDetails.push({
      permissionType: 'file',
      role: 'writer',
      inherited: false,
    });
    await assert.rejects(
      service({ google, runtime: productionRuntime() }).rollover.prepare({
        targetCycle: '2026-08',
        sourceFormId: LIVE_FORM_ID,
        collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
        dryRun: true,
      }),
      /break-glass administrator access is not exactly one inherited Manager permission/,
    );
  });
});

test('automation access must also be exactly one inherited, non-expiring Manager', async (t) => {
  for (const scenario of [
    { name: 'reader', role: 'reader', detailRole: 'reader' },
    { name: 'writer', role: 'writer', detailRole: 'writer' },
    { name: 'Content manager', role: 'fileOrganizer', detailRole: 'fileOrganizer' },
    {
      name: 'direct Manager',
      role: 'organizer',
      permissionType: 'file',
      inherited: false,
      inheritedFrom: undefined,
    },
    { name: 'expiring Manager', role: 'organizer', inherited: true, expirationTime: '2026-08-01T00:00:00Z' },
    { name: 'wrong permission type', role: 'organizer', permissionType: 'file' },
    { name: 'wrong detail role', role: 'organizer', detailRole: 'fileOrganizer' },
    { name: 'wrong inheritance source', role: 'organizer', inheritedFrom: 'different-drive' },
  ]) {
    await t.test(scenario.name, async () => {
      const google = grantProductionSourceAccess(new FakeGoogle());
      const automation = google.permissions.get('private-production-folder').find(
        ({ emailAddress }) => emailAddress === PRODUCTION_SERVICE_ACCOUNT,
      );
      automation.role = scenario.role;
      automation.permissionDetails = [sharedDriveManagerDetail('private-production-drive', {
        permissionType: scenario.permissionType ?? 'member',
        role: scenario.detailRole ?? 'organizer',
        inherited: scenario.inherited ?? true,
        inheritedFrom: Object.hasOwn(scenario, 'inheritedFrom')
          ? scenario.inheritedFrom
          : 'private-production-drive',
      })];
      if (scenario.expirationTime !== undefined) {
        automation.expirationTime = scenario.expirationTime;
      }

      await assert.rejects(
        service({ google, runtime: productionRuntime() }).rollover.prepare({
          targetCycle: '2026-08',
          sourceFormId: LIVE_FORM_ID,
          collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
          dryRun: true,
        }),
        /automation access is not exactly one inherited Manager permission/,
      );
      assert.equal(google.calls.some(({ method }) => method === 'copyFile'), false);
    });
  }
});

test('destination rejects a merged direct role source for either permitted Manager', async (t) => {
  for (const [name, emailAddress, expected] of [
    ['automation', PRODUCTION_SERVICE_ACCOUNT, /automation access is not exactly one inherited Manager permission/],
    ['break-glass administrator', PRODUCTION_DRIVE_ADMIN, /break-glass administrator access is not exactly one inherited Manager permission/],
  ]) {
    await t.test(name, async () => {
      const google = grantProductionSourceAccess(new FakeGoogle());
      const permission = google.permissions.get('private-production-folder').find(
        ({ emailAddress: current }) => current === emailAddress,
      );
      permission.permissionDetails.push({
        permissionType: 'file',
        role: 'writer',
        inherited: false,
      });

      await assert.rejects(
        service({ google, runtime: productionRuntime() }).rollover.prepare({
          targetCycle: '2026-08',
          sourceFormId: LIVE_FORM_ID,
          collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
          dryRun: true,
        }),
        expected,
      );
      assert.equal(google.calls.some(({ method }) => method === 'copyFile'), false);
    });
  }
});

test('destination rejects every other editable human, group, or domain identity', async (t) => {
  for (const permission of [
    {
      id: 'unexpected-human-writer',
      type: 'user',
      role: 'writer',
      emailAddress: 'unexpected-human@example.org',
      permissionDetails: [{ inherited: true }],
    },
    {
      id: 'configured-editor-on-drive',
      ...PRODUCTION_EDITOR_PERMISSION,
      permissionDetails: [{ inherited: true }],
    },
    {
      id: 'unexpected-domain-writer',
      type: 'domain',
      role: 'writer',
      domain: 'example.org',
      permissionDetails: [{ inherited: true }],
    },
    {
      id: 'unexpected-human-manager',
      type: 'user',
      role: 'organizer',
      emailAddress: 'unexpected-manager@example.org',
      permissionDetails: [{ inherited: true }],
    },
    {
      id: 'unexpected-human-owner',
      type: 'user',
      role: 'owner',
      emailAddress: 'unexpected-owner@example.org',
    },
  ]) {
    await t.test(permission.id, async () => {
      const google = grantProductionSourceAccess(new FakeGoogle());
      google.permissions.get('private-production-folder').push(permission);
      await assert.rejects(
        service({ google, runtime: productionRuntime() }).rollover.prepare({
          targetCycle: '2026-08',
          sourceFormId: LIVE_FORM_ID,
          collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
          dryRun: true,
        }),
        /destination folder has unexpected inherited edit access/,
      );
      assert.equal(google.calls.some(({ method }) => method === 'copyFile'), false);
    });
  }
});

test('destination folders reject an unexpected service account at every Drive role', async (t) => {
  for (const role of ['reader', 'commenter', 'writer', 'fileOrganizer', 'organizer', 'owner']) {
    await t.test(role, async () => {
      const google = grantProductionSourceAccess(new FakeGoogle());
      google.permissions.set('private-production-folder', [{
        id: `unexpected-service-account-${role}`,
        type: 'user',
        role,
        emailAddress: role === 'reader'
          ? 'unexpected-legacy@developer.gserviceaccount.com'
          : STAGING_SERVICE_ACCOUNT,
        permissionDetails: [{ inherited: true }],
      }]);

      await assert.rejects(
        service({ google, runtime: productionRuntime() }).rollover.prepare({
          targetCycle: '2026-08',
          sourceFormId: LIVE_FORM_ID,
          collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
          dryRun: true,
        }),
        /unexpected service-account permission/,
      );
      assert.equal(google.calls.some(({ method }) => method === 'copyFile'), false);
    });
  }
});

test('service accounts cannot be configured as copied collaborators', async () => {
  const google = grantProductionSourceAccess(new FakeGoogle());
  await assert.rejects(
    service({ google, runtime: productionRuntime() }).rollover.prepare({
      targetCycle: '2026-08',
      sourceFormId: LIVE_FORM_ID,
      collaboratorPermissions: [{
        type: 'user',
        role: 'writer',
        emailAddress: PRODUCTION_SERVICE_ACCOUNT,
      }],
    }),
    /service account cannot be configured as a form collaborator/,
  );
  assert.equal(google.calls.length, 0);

  const privilegedStagingGoogle = grantProductionSourceAccess(new FakeGoogle());
  privilegedStagingGoogle.permissions.get(LIVE_FORM_ID).find(
    ({ emailAddress }) => emailAddress === STAGING_SERVICE_ACCOUNT,
  ).role = 'writer';
  await assert.rejects(
    service({
      google: privilegedStagingGoogle,
      runtime: productionRuntime({ stagingServiceAccountEmail: undefined }),
    }).rollover.prepare({
      targetCycle: '2026-08',
      sourceFormId: LIVE_FORM_ID,
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /unexpected service-account collaborator/,
  );
  assert.equal(
    privilegedStagingGoogle.calls.some(({ method }) => method === 'copyFile'),
    false,
  );

  const domainGoogle = grantProductionSourceAccess(new FakeGoogle());
  domainGoogle.permissions.get(LIVE_FORM_ID).push({
    id: 'domain-writer',
    type: 'domain',
    role: 'writer',
    domain: 'example.org',
  });
  await assert.rejects(
    service({ google: domainGoogle, runtime: productionRuntime() }).rollover.validate({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /unsupported domain collaborator/,
  );
});

test('a prepared target rejects unexpected service accounts at every role', async (t) => {
  for (const role of ['reader', 'commenter', 'writer', 'fileOrganizer', 'organizer', 'owner']) {
    await t.test(role, async () => {
      const context = await preparedProductionContext();
      const unexpectedPermission = {
        id: `unexpected-service-account-${role}`,
        type: 'user',
        role,
        emailAddress: STAGING_SERVICE_ACCOUNT,
        permissionDetails: [{ inherited: true }],
      };
      context.google.permissions.get(context.target.id).push(unexpectedPermission);
      await assert.rejects(productionVerify(context), /unexpected service-account permission/);

      context.google.permissions.set(
        context.target.id,
        context.google.permissions.get(context.target.id).filter(
          ({ id }) => id !== unexpectedPermission.id,
        ),
      );
      context.google.permissions.get(context.target.id).push({
        ...unexpectedPermission,
        id: `duplicate-automation-${role}`,
        emailAddress: PRODUCTION_SERVICE_ACCOUNT,
      });
      await assert.rejects(
        productionVerify(context),
        /Shared Drive automation access is not exactly one inherited Manager permission/,
      );
    });
  }
});

test('managed forms ignore only the exact inherited break-glass Manager permission', async (t) => {
  for (const scenario of [
    { name: 'writer', role: 'writer', detailRole: 'writer' },
    { name: 'commenter', role: 'commenter', detailRole: 'commenter' },
    { name: 'Content manager', role: 'fileOrganizer', detailRole: 'fileOrganizer' },
    {
      name: 'direct Manager',
      role: 'organizer',
      permissionType: 'file',
      inherited: false,
      inheritedFrom: undefined,
    },
    { name: 'wrong inheritance source', role: 'organizer', inheritedFrom: 'different-drive' },
  ]) {
    await t.test(scenario.name, async () => {
      const context = await preparedProductionContext();
      const admin = context.google.permissions.get(context.target.id).find(
        ({ emailAddress }) => emailAddress === PRODUCTION_DRIVE_ADMIN,
      );
      admin.role = scenario.role;
      admin.permissionDetails = [sharedDriveManagerDetail('private-production-drive', {
        permissionType: scenario.permissionType ?? 'member',
        role: scenario.detailRole ?? 'organizer',
        inherited: scenario.inherited ?? true,
        inheritedFrom: Object.hasOwn(scenario, 'inheritedFrom')
          ? scenario.inheritedFrom
          : 'private-production-drive',
      })];

      await assert.rejects(
        productionVerify(context),
        /break-glass administrator access is not exactly one inherited Manager permission/,
      );
    });
  }

  for (const [name, emailAddress, expected] of [
    ['automation with a merged direct form grant', PRODUCTION_SERVICE_ACCOUNT, /automation access is not exactly one inherited Manager permission/],
    ['break-glass administrator with a merged direct form grant', PRODUCTION_DRIVE_ADMIN, /break-glass administrator access is not exactly one inherited Manager permission/],
  ]) {
    await t.test(name, async () => {
      const context = await preparedProductionContext();
      const permission = context.google.permissions.get(context.target.id).find(
        ({ emailAddress: current }) => current === emailAddress,
      );
      permission.permissionDetails.push({
        permissionType: 'file',
        role: 'writer',
        inherited: false,
      });

      await assert.rejects(productionVerify(context), expected);
    });
  }

  await t.test('editor group inherited from the Shared Drive', async () => {
    const context = await preparedProductionContext();
    const editor = context.google.permissions.get(context.target.id).find(
      ({ emailAddress }) => emailAddress === PRODUCTION_EDITOR_PERMISSION.emailAddress,
    );
    editor.permissionDetails = [{ inherited: true }];
    await assert.rejects(
      productionVerify(context),
      /unexpected inherited or administrative edit access/,
    );
  });
});

test('prepare resumes an interrupted copy and writes only the managed page lines', async () => {
  const context = await preparedProductionContext();
  assert.equal(context.result.status, 'prepared');
  assert.equal(context.result.created, true);
  assert.equal(context.page.writes.length, 1);
  assert.match(context.page.content, /August 31st, 2026/);
  assert.match(
    context.page.content,
    new RegExp(context.result.responderUri.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
  );
  assert.match(context.page.content, /Requirements stay untouched\./);
  assert.equal(context.target.name, 'August 2026 Progress Prizes');
  assert.equal(context.target.appProperties.state, ROLLOVER_FILE_STATES.PREPARED);
  assert.equal(context.google.forms.get(context.target.id).info.title, context.target.name);
  assert.deepEqual(context.google.forms.get(context.target.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });

  // Rewind to the state an interrupted first run leaves behind: a closed copy
  // marked COPIED and an untouched page.
  context.target.appProperties.state = ROLLOVER_FILE_STATES.COPIED;
  context.page.content = pageMarkdown();
  context.page.writes.length = 0;
  const resumed = await context.rollover.prepare({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  });
  assert.equal(resumed.status, 'prepared');
  assert.equal(resumed.created, false);
  assert.equal(resumed.resumed, true);
  assert.equal(resumed.responderUri, context.result.responderUri);
  assert.equal(context.page.writes.length, 1);
  assert.equal(
    context.google.calls.filter(({ method }) => method === 'copyFile').length,
    1,
  );
  assert.equal(context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08').length, 1);

  const idempotent = await context.rollover.prepare({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  });
  assert.equal(idempotent.pageChanged, false);
  assert.equal(context.page.writes.length, 1);

  assert.deepEqual(await productionVerify(context), {
    action: 'verify',
    status: 'valid',
    sourceCycle: '2026-07',
    targetCycle: '2026-08',
    responderUri: context.result.responderUri,
  });
});

test('prepare proceeds from a manually closed source without reopening it', async () => {
  const google = grantProductionSourceAccess(new FakeGoogle());
  google.forms.get(LIVE_FORM_ID).publishSettings.publishState.isAcceptingResponses = false;
  const context = service({ google });

  const result = await context.rollover.prepare({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  });
  assert.equal(result.status, 'prepared');
  const target = google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  assert.equal(target.appProperties.state, ROLLOVER_FILE_STATES.PREPARED);
  assert.deepEqual(google.forms.get(LIVE_FORM_ID).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });
  assert.deepEqual(google.forms.get(target.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });
  assert.equal((await productionVerify(context)).status, 'valid');
});

test('prepare refuses to copy an unpublished source', async () => {
  const google = grantProductionSourceAccess(new FakeGoogle());
  google.forms.get(LIVE_FORM_ID).publishSettings.publishState = {
    isPublished: false,
    isAcceptingResponses: false,
  };
  const context = service({ google });

  await assert.rejects(
    context.rollover.prepare({
      targetCycle: '2026-08',
      sourceFormId: LIVE_FORM_ID,
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /publishing state/,
  );
  assert.equal(google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08').length, 0);
});

test('a newly copied target is closed before later capability validation can fail', async () => {
  const google = grantProductionSourceAccess(new FakeGoogle());
  const context = service({ google });
  const originalCopyFile = google.copyFile.bind(google);
  google.copyFile = async (input) => {
    const copied = await originalCopyFile(input);
    copied.capabilities.canShare = false;
    google.files.get(copied.id).capabilities.canShare = false;
    return copied;
  };

  await assert.rejects(
    context.rollover.prepare({
      targetCycle: '2026-08',
      sourceFormId: LIVE_FORM_ID,
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /required canShare capability/,
  );
  const target = google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  assert.deepEqual(google.forms.get(target.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });
});

test('production preserves its group and direct external editor without copying either service account', async () => {
  const google = grantProductionSourceAccess(new FakeGoogle());
  google.permissions.get(LIVE_FORM_ID).push({
    id: 'direct-external-human-editor',
    type: 'user',
    role: 'writer',
    emailAddress: EXTERNAL_PRODUCTION_EDITOR,
  });
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
  assert.equal(augustTarget.driveId, 'private-production-drive');
  assert.deepEqual(augustTarget.parents, ['private-production-folder']);
  assert.equal(augustTarget.appProperties.managedBy, ROLLOVER_MANAGED_BY);
  assert.equal(augustTarget.appProperties.environment, 'production');
  assert.equal(augustTarget.appProperties.role, ROLLOVER_FILE_ROLES.TARGET);
  assert.equal(augustTarget.appProperties.cycle, '2026-08');
  assert.equal(augustTarget.appProperties.state, ROLLOVER_FILE_STATES.PREPARED);
  assert.match(augustTarget.appProperties.sourceFingerprint, /^[a-f0-9]{64}$/);
  assert.deepEqual(google.files.get(LIVE_FORM_ID).appProperties, {});
  const firstCopy = google.calls.find(({ method }) => method === 'copyFile');
  assert.equal(firstCopy.fileId, LIVE_FORM_ID);
  assert.equal(firstCopy.parentId, 'private-production-folder');
  const copiedCollaborators = google.permissions.get(augustTarget.id)
    .filter(({ type, role }) => (
      (type === 'user' || type === 'group')
      && (role === 'writer' || role === 'commenter')
    ))
    .map(({ emailAddress }) => emailAddress)
    .sort();
  assert.deepEqual(copiedCollaborators, [
    EXTERNAL_PRODUCTION_EDITOR,
    'private-team@example.org',
  ]);
  assert.equal(google.permissions.get(augustTarget.id).some((permission) => (
    (permission.emailAddress === PRODUCTION_SERVICE_ACCOUNT
      || permission.emailAddress === STAGING_SERVICE_ACCOUNT)
    && !permission.permissionDetails?.some(({ inherited }) => inherited === true)
  )), false);
  assert.equal((await july.rollover.verify({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  })).status, 'valid');
  google.files.get(LIVE_FORM_ID).capabilities.canCopy = false;

  // The cutoff is manual: a human closes July and opens August in the Forms UI.
  google.forms.get(LIVE_FORM_ID).publishSettings.publishState.isAcceptingResponses = false;
  google.forms.get(augustTarget.id).publishSettings.publishState.isAcceptingResponses = true;

  google.files.delete(LIVE_FORM_ID);
  google.forms.delete(LIVE_FORM_ID);
  google.permissions.delete(LIVE_FORM_ID);
  google.calls.length = 0;

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

  const septemberTarget = google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-09')[0];
  assert.ok(septemberTarget);
  const septemberCopy = google.calls.findLast(({ method, appProperties }) => (
    method === 'copyFile' && appProperties?.cycle === '2026-09'
  ));
  assert.equal(septemberCopy.fileId, augustTarget.id);
  assert.deepEqual(
    google.permissions.get(septemberTarget.id)
      .filter(({ type, role }) => (
        (type === 'user' || type === 'group')
        && (role === 'writer' || role === 'commenter')
      ))
      .map(({ emailAddress }) => emailAddress)
      .sort(),
    [EXTERNAL_PRODUCTION_EDITOR, 'private-team@example.org'],
  );
  assert.equal(google.permissions.get(septemberTarget.id).some((permission) => (
    (permission.emailAddress === PRODUCTION_SERVICE_ACCOUNT
      || permission.emailAddress === STAGING_SERVICE_ACCOUNT)
    && !permission.permissionDetails?.some(({ inherited }) => inherited === true)
  )), false);
  assert.equal(
    google.calls.some(({ fileId, formId }) => fileId === LIVE_FORM_ID || formId === LIVE_FORM_ID),
    false,
  );
});

test('a missing later managed source cannot fall back to the stale initial form ID', async () => {
  const google = grantProductionSourceAccess(new FakeGoogle());
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
  google.files.delete(augustTarget.id);
  google.forms.delete(augustTarget.id);
  google.permissions.delete(augustTarget.id);
  google.calls.length = 0;

  const august = service({
    google,
    page,
    runtime: productionRuntime(),
    clock: fixedClock('2026-08-25T12:00:00Z'),
  });
  await assert.rejects(
    august.rollover.prepare({
      targetCycle: '2026-09',
      sourceFormId: LIVE_FORM_ID,
      collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    }),
    /explicit fallback is restricted to the initial bootstrap cycle/,
  );
  assert.equal(
    google.calls.some(({ fileId, formId }) => fileId === LIVE_FORM_ID || formId === LIVE_FORM_ID),
    false,
  );
});

test('production safety rejects simulated time, staging, alternate bases, and staging identities before I/O', () => {
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({
      eventName: 'workflow_dispatch',
      simulatedNow: '2026-07-26T12:00:00Z',
    })),
    /simulated time is forbidden in production/,
  );
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({ environment: 'staging' })),
    /run only in production/,
  );
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({ targetBranch: 'test-base' })),
    /alternate target branch/,
  );
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({
      serviceAccountEmail: STAGING_SERVICE_ACCOUNT,
    })),
    /staging service account/,
  );
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({ eventName: 'pull_request' })),
    /only for schedule or workflow_dispatch/,
  );
  assert.throws(
    () => createRolloverService({
      google: new FakeGoogle(),
      page: new MemoryPage(),
      clock: fixedClock('2026-07-26T12:00:00Z'),
      runtime: productionRuntime({ environment: 'staging' }),
    }),
    /run only in production/,
  );
});

test('daily preparation is a read-only no-op before the seven-day window', async () => {
  const context = service({ clock: fixedClock('2026-07-20T12:00:00Z') });
  const result = await context.rollover.prepare({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
  });
  assert.equal(result.status, 'waiting');
  assert.equal(context.google.calls.length, 0);
  assert.equal(context.page.writes.length, 0);
});

test('runtime requires a separate non-service-account break-glass email', () => {
  for (const [driveAdminEmail, expected] of [
    [undefined, /driveAdminEmail must be a non-empty string/],
    ['not-an-email', /driveAdminEmail must be an email address/],
    [PRODUCTION_SERVICE_ACCOUNT, /separate human break-glass administrator/],
    ['other@private-project.iam.gserviceaccount.com', /separate human break-glass administrator/],
  ]) {
    assert.throws(
      () => assertRolloverRuntimeSafety(productionRuntime({ driveAdminEmail })),
      expected,
    );
  }
});

test('break-glass administrator cannot also be the explicit editor collaborator', async () => {
  const google = grantProductionSourceAccess(new FakeGoogle());
  await assert.rejects(
    service({ google, runtime: productionRuntime() }).rollover.prepare({
      targetCycle: '2026-08',
      sourceFormId: LIVE_FORM_ID,
      collaboratorPermissions: [{
        type: 'user',
        role: 'writer',
        emailAddress: PRODUCTION_DRIVE_ADMIN,
      }],
    }),
    /break-glass administrator cannot be configured as a form collaborator/,
  );
  assert.equal(google.calls.length, 0);
});
