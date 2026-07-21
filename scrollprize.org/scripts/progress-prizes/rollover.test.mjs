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
const EXTERNAL_PRODUCTION_EDITOR = 'external-progress-prize-editor@example.net';
const STAGING_SERVICE_ACCOUNT = 'staging-automation@private-project.iam.gserviceaccount.com';
const PRODUCTION_SERVICE_ACCOUNT = 'production-automation@private-project.iam.gserviceaccount.com';
const STAGING_DRIVE_ADMIN = 'staging-break-glass@example.org';
const PRODUCTION_DRIVE_ADMIN = 'production-break-glass@example.org';

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
      ['private-staging-folder', {
        id: 'private-staging-folder',
        name: 'Staging active forms',
        mimeType: 'application/vnd.google-apps.folder',
        parents: [],
        driveId: 'private-staging-drive',
        appProperties: {},
        trashed: false,
        capabilities: { canAddChildren: true, canShare: true },
      }],
      ['private-staging-archive', {
        id: 'private-staging-archive',
        name: 'Staging archive',
        mimeType: 'application/vnd.google-apps.folder',
        parents: [],
        driveId: 'private-staging-drive',
        appProperties: {},
        trashed: false,
        capabilities: { canAddChildren: true, canShare: true },
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
      ['private-staging-folder', [
        sharedDriveManagerPermission({
          id: 'staging-manager',
          emailAddress: STAGING_SERVICE_ACCOUNT,
          driveId: 'private-staging-drive',
        }),
        sharedDriveManagerPermission({
          id: 'staging-break-glass-manager',
          emailAddress: STAGING_DRIVE_ADMIN,
          driveId: 'private-staging-drive',
        }),
      ]],
      ['private-staging-archive', [
        sharedDriveManagerPermission({
          id: 'staging-archive-manager',
          emailAddress: STAGING_SERVICE_ACCOUNT,
          driveId: 'private-staging-drive',
        }),
        sharedDriveManagerPermission({
          id: 'staging-archive-break-glass-manager',
          emailAddress: STAGING_DRIVE_ADMIN,
          driveId: 'private-staging-drive',
        }),
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

function stagingRuntime(overrides = {}) {
  return {
    environment: 'staging',
    eventName: 'workflow_dispatch',
    folderId: 'private-staging-folder',
    stagingFolderId: 'private-staging-folder',
    archiveFolderId: 'private-staging-archive',
    driveId: 'private-staging-drive',
    driveAdminEmail: STAGING_DRIVE_ADMIN,
    serviceAccountEmail: STAGING_SERVICE_ACCOUNT,
    stagingServiceAccountEmail: STAGING_SERVICE_ACCOUNT,
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

  await assert.rejects(
    service().rollover.prepare({
      targetCycle: '2026-08',
      sourceFormId: LIVE_FORM_ID,
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    }),
    /My Drive fallback source is restricted to production/,
  );

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
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
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
    stagedPermissions
      .filter(({ role }) => role === 'writer' || role === 'commenter' || role === 'reader')
      .map(({ type, role, emailAddress }) => ({ type, role, emailAddress })),
    [
      { type: 'group', role: 'writer', emailAddress: STAGING_EDITOR },
      { type: 'anyone', role: 'reader', emailAddress: undefined },
    ],
  );
  const liveMutations = context.google.calls.filter(({ method, fileId, formId }) =>
    (fileId === LIVE_FORM_ID || formId === LIVE_FORM_ID)
    && ['updateFile', 'createPermission', 'deletePermission', 'updateFormTitle', 'setPublishState'].includes(method));
  assert.deepEqual(liveMutations, []);
  assert.equal(
    context.google.calls.some(
      ({ method, fileId }) => method === 'getAllPermissions' && fileId === LIVE_FORM_ID,
    ),
    false,
  );
});

test('bootstrap does not require a staging Viewer to enumerate the production ACL', async () => {
  const context = service();
  context.google.files.get(LIVE_FORM_ID).capabilities.canEdit = false;
  context.google.files.get(LIVE_FORM_ID).capabilities.canShare = false;
  const originalGetAllPermissions = context.google.getAllPermissions.bind(context.google);
  let livePermissionReads = 0;
  context.google.getAllPermissions = async (input) => {
    if (input.fileId === LIVE_FORM_ID) {
      livePermissionReads += 1;
      throw new Error('A Viewer cannot enumerate the production ACL');
    }
    return originalGetAllPermissions(input);
  };

  const result = await context.rollover.bootstrapStagingSource({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
  });

  assert.equal(result.status, 'active');
  assert.equal(livePermissionReads, 0);
});

test('bootstrap rejects an overprivileged staging identity before copying production', async () => {
  for (const capability of ['canEdit', 'canShare']) {
    for (const invalidValue of [true, null, 'false', undefined]) {
      const context = service();
      if (invalidValue === undefined) {
        delete context.google.files.get(LIVE_FORM_ID).capabilities[capability];
      } else {
        context.google.files.get(LIVE_FORM_ID).capabilities[capability] = invalidValue;
      }
      await assert.rejects(
        context.rollover.bootstrapStagingSource({
          sourceFormId: LIVE_FORM_ID,
          sourceCycle: '2026-07',
          collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
        }),
        /forbidden edit or share access/,
      );
      assert.equal(context.google.calls.some(({ method }) => method === 'copyFile'), false);
    }
  }
});

test('staging reader bootstrap is limited to the explicit July My Drive source', async () => {
  const wrongCycle = service();
  await assert.rejects(
    wrongCycle.rollover.bootstrapStagingSource({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-08',
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    }),
    /initial explicit source cycle/,
  );
  assert.equal(wrongCycle.google.calls.length, 0);

  for (const mutate of [
    (file) => { file.driveId = 'private-production-drive'; },
    (file) => { file.appProperties.managedBy = ROLLOVER_MANAGED_BY; },
  ]) {
    const context = service();
    mutate(context.google.files.get(LIVE_FORM_ID));
    await assert.rejects(
      context.rollover.bootstrapStagingSource({
        sourceFormId: LIVE_FORM_ID,
        sourceCycle: '2026-07',
        collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
      }),
      /initial explicit My Drive source/,
    );
    assert.equal(context.google.calls.some(({ method }) => method === 'getForm'), false);
  }
});

test('managed sources and targets reject other service accounts at every role', async (t) => {
  for (const role of ['reader', 'commenter', 'writer', 'fileOrganizer', 'organizer', 'owner']) {
    await t.test(role, async () => {
      const context = service();
      await context.rollover.bootstrapStagingSource({
        sourceFormId: LIVE_FORM_ID,
        sourceCycle: '2026-07',
        collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
      });
      const source = context.google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
      const unexpectedPermission = {
        id: `unexpected-production-service-account-${role}`,
        type: 'user',
        role,
        emailAddress: PRODUCTION_SERVICE_ACCOUNT,
        permissionDetails: [{ inherited: true }],
      };
      context.google.permissions.get(source.id).push(unexpectedPermission);

      await assert.rejects(
        context.rollover.prepare({
          targetCycle: '2026-08',
          collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
        }),
        /unexpected service-account permission/,
      );
      assert.equal(
        context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08').length,
        0,
      );

      context.google.permissions.set(
        source.id,
        context.google.permissions.get(source.id).filter(
          ({ id }) => id !== unexpectedPermission.id,
        ),
      );
      await context.rollover.prepare({
        targetCycle: '2026-08',
        collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
      });
      const target = context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
      context.google.permissions.get(target.id).push(unexpectedPermission);

      await assert.rejects(
        context.rollover.verify({
          targetCycle: '2026-08',
          collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
        }),
        /unexpected service-account permission/,
      );

      context.google.permissions.set(
        target.id,
        context.google.permissions.get(target.id).filter(
          ({ id }) => id !== unexpectedPermission.id,
        ),
      );
      context.google.permissions.get(target.id).push({
        ...unexpectedPermission,
        id: `current-staging-service-account-${role}`,
        emailAddress: STAGING_SERVICE_ACCOUNT,
      });
      await assert.rejects(
        context.rollover.verify({
          targetCycle: '2026-08',
          collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
        }),
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
      const context = service();
      await bootstrapAndPrepare(context);
      const target = context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
      const admin = context.google.permissions.get(target.id).find(
        ({ emailAddress }) => emailAddress === STAGING_DRIVE_ADMIN,
      );
      admin.role = scenario.role;
      admin.permissionDetails = [sharedDriveManagerDetail('private-staging-drive', {
        permissionType: scenario.permissionType ?? 'member',
        role: scenario.detailRole ?? 'organizer',
        inherited: scenario.inherited ?? true,
        inheritedFrom: Object.hasOwn(scenario, 'inheritedFrom')
          ? scenario.inheritedFrom
          : 'private-staging-drive',
      })];

      await assert.rejects(
        context.rollover.verify({
          targetCycle: '2026-08',
          collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
        }),
        /break-glass administrator access is not exactly one inherited Manager permission/,
      );
    });
  }

  for (const [name, emailAddress, expected] of [
    ['automation with a merged direct form grant', STAGING_SERVICE_ACCOUNT, /automation access is not exactly one inherited Manager permission/],
    ['break-glass administrator with a merged direct form grant', STAGING_DRIVE_ADMIN, /break-glass administrator access is not exactly one inherited Manager permission/],
  ]) {
    await t.test(name, async () => {
      const context = service();
      await bootstrapAndPrepare(context);
      const target = context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
      const permission = context.google.permissions.get(target.id).find(
        ({ emailAddress: current }) => current === emailAddress,
      );
      permission.permissionDetails.push({
        permissionType: 'file',
        role: 'writer',
        inherited: false,
      });

      await assert.rejects(
        context.rollover.verify({
          targetCycle: '2026-08',
          collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
        }),
        expected,
      );
    });
  }

  await t.test('editor group inherited from the Shared Drive', async () => {
    const context = service();
    await bootstrapAndPrepare(context);
    const target = context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
    const editor = context.google.permissions.get(target.id).find(
      ({ emailAddress }) => emailAddress === STAGING_EDITOR,
    );
    editor.permissionDetails = [{ inherited: true }];
    await assert.rejects(
      context.rollover.verify({
        targetCycle: '2026-08',
        collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
      }),
      /unexpected inherited or administrative edit access/,
    );
  });
});

test('bootstrap closes a fresh staging source before the copy fault and reopens only after reconciliation', async () => {
  const context = service();

  await assert.rejects(
    context.rollover.bootstrapStagingSource({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
      faultInjection: ROLLOVER_FAULTS.AFTER_COPY,
    }),
    (error) => error instanceof RolloverFaultError && error.step === ROLLOVER_FAULTS.AFTER_COPY,
  );

  const stagedSource = context.google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
  assert.ok(stagedSource);
  assert.equal(stagedSource.appProperties.state, ROLLOVER_FILE_STATES.COPIED);
  assert.deepEqual(context.google.forms.get(stagedSource.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });
  assert.deepEqual(
    context.google.permissions.get(stagedSource.id).filter(
      ({ permissionDetails }) => !permissionDetails?.some(({ inherited }) => inherited === true),
    ),
    [],
  );
  assert.equal(
    context.google.calls.some(({ method, fileId, formId }) =>
      (fileId === stagedSource.id || formId === stagedSource.id)
      && ['updateFile', 'createPermission', 'updateFormTitle'].includes(method)),
    false,
  );

  const originalCreatePermission = context.google.createPermission.bind(context.google);
  let failAclReconciliation = true;
  context.google.createPermission = async (input) => {
    if (failAclReconciliation) throw new Error('injected ACL reconciliation failure');
    return originalCreatePermission(input);
  };
  await assert.rejects(
    context.rollover.bootstrapStagingSource({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    }),
    /injected ACL reconciliation failure/,
  );
  assert.deepEqual(context.google.forms.get(stagedSource.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: false,
  });

  failAclReconciliation = false;
  const recoveryCallStart = context.google.calls.length;
  const recovered = await context.rollover.bootstrapStagingSource({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
  });
  const recoveryCalls = context.google.calls.slice(recoveryCallStart);
  const lastAclWrite = recoveryCalls.findLastIndex(
    ({ method, fileId }) => method === 'createPermission' && fileId === stagedSource.id,
  );
  const reopen = recoveryCalls.findIndex(
    ({ method, formId, isAcceptingResponses }) =>
      method === 'setPublishState'
      && formId === stagedSource.id
      && isAcceptingResponses === true,
  );

  assert.equal(recovered.status, 'active');
  assert.equal(recovered.created, false);
  assert.equal(recovered.resumed, true);
  assert.ok(lastAclWrite >= 0);
  assert.ok(reopen > lastAclWrite);
  assert.equal(
    context.google.calls.filter(({ method, appProperties }) =>
      method === 'copyFile' && appProperties?.role === ROLLOVER_FILE_ROLES.SOURCE).length,
    1,
  );
  assert.deepEqual(context.google.forms.get(stagedSource.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: true,
  });
});

test('bootstrap closes a copied staging source before validating its Drive metadata and capabilities', async (t) => {
  const cases = [
    {
      name: 'malformed managed metadata',
      expected: /Managed form metadata/,
      mutate(copied) {
        copied.appProperties.state = 'invalid';
      },
    },
    {
      name: 'missing canShare capability',
      expected: /required canShare capability/,
      mutate(copied, google) {
        copied.capabilities.canShare = false;
        google.files.get(copied.id).capabilities.canShare = false;
      },
    },
  ];

  for (const scenario of cases) {
    await t.test(scenario.name, async () => {
      const context = service();
      const originalCopyFile = context.google.copyFile.bind(context.google);
      let copiedId;
      context.google.copyFile = async (input) => {
        const copied = await originalCopyFile(input);
        copiedId = copied.id;
        scenario.mutate(copied, context.google);
        return copied;
      };

      await assert.rejects(
        context.rollover.bootstrapStagingSource({
          sourceFormId: LIVE_FORM_ID,
          sourceCycle: '2026-07',
          collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
        }),
        scenario.expected,
      );

      assert.ok(copiedId);
      assert.deepEqual(context.google.forms.get(copiedId).publishSettings.publishState, {
        isPublished: true,
        isAcceptingResponses: false,
      });
      const copyCall = context.google.calls.findIndex(
        ({ method, appProperties }) =>
          method === 'copyFile' && appProperties?.role === ROLLOVER_FILE_ROLES.SOURCE,
      );
      const closeCall = context.google.calls.findIndex(
        ({ method, formId, isAcceptingResponses }) =>
          method === 'setPublishState'
          && formId === copiedId
          && isAcceptingResponses === false,
      );
      const copiedDriveReads = context.google.calls
        .map(({ method, fileId }, index) => ({ method, fileId, index }))
        .filter(({ method, fileId }) =>
          fileId === copiedId && ['getFile', 'getAllPermissions'].includes(method));
      assert.ok(closeCall > copyCall);
      assert.equal(copiedDriveReads.every(({ index }) => closeCall < index), true);
      assert.equal(
        context.google.calls.some(({ method, fileId, formId }) =>
          (fileId === copiedId || formId === copiedId)
          && ['updateFile', 'createPermission', 'updateFormTitle'].includes(method)),
        false,
      );
    });
  }
});

test('bootstrap re-closes a reconciled copy after its ACTIVE marker write fails', async () => {
  const context = service();
  const originalUpdateFile = context.google.updateFile.bind(context.google);
  let failActiveMarker = true;
  context.google.updateFile = async (input) => {
    if (
      failActiveMarker
      && input.appProperties?.state === ROLLOVER_FILE_STATES.ACTIVE
    ) {
      throw new Error('injected ACTIVE marker failure');
    }
    return originalUpdateFile(input);
  };

  await assert.rejects(
    context.rollover.bootstrapStagingSource({
      sourceFormId: LIVE_FORM_ID,
      sourceCycle: '2026-07',
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    }),
    /injected ACTIVE marker failure/,
  );
  const stagedSource = context.google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
  assert.equal(stagedSource.appProperties.state, ROLLOVER_FILE_STATES.COPIED);
  assert.deepEqual(context.google.forms.get(stagedSource.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: true,
  });

  failActiveMarker = false;
  const recoveryCallStart = context.google.calls.length;
  const recovered = await context.rollover.bootstrapStagingSource({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
  });
  const publishWrites = context.google.calls
    .slice(recoveryCallStart)
    .filter(({ method, formId }) => method === 'setPublishState' && formId === stagedSource.id);

  assert.deepEqual(
    publishWrites.map(({ isAcceptingResponses }) => isAcceptingResponses),
    [false, true],
  );
  assert.equal(recovered.status, 'active');
  assert.equal(recovered.resumed, true);
  assert.equal(
    context.google.calls.filter(({ method, appProperties }) =>
      method === 'copyFile' && appProperties?.role === ROLLOVER_FILE_ROLES.SOURCE).length,
    1,
  );
  assert.equal(
    context.google.files.get(stagedSource.id).appProperties.state,
    ROLLOVER_FILE_STATES.ACTIVE,
  );
  assert.deepEqual(context.google.forms.get(stagedSource.id).publishSettings.publishState, {
    isPublished: true,
    isAcceptingResponses: true,
  });
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

test('a newly copied target is closed before later capability validation can fail', async () => {
  const context = service();
  await context.rollover.bootstrapStagingSource({
    sourceFormId: LIVE_FORM_ID,
    sourceCycle: '2026-07',
    collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
  });
  const originalCopyFile = context.google.copyFile.bind(context.google);
  context.google.copyFile = async (input) => {
    const copied = await originalCopyFile(input);
    copied.capabilities.canShare = false;
    context.google.files.get(copied.id).capabilities.canShare = false;
    return copied;
  };

  await assert.rejects(
    context.rollover.prepare({
      targetCycle: '2026-08',
      collaboratorPermissions: [STAGING_EDITOR_PERMISSION],
    }),
    /required canShare capability/,
  );
  const target = context.google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  assert.deepEqual(context.google.forms.get(target.id).publishSettings.publishState, {
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
  assert.equal(
    google.forms.get(LIVE_FORM_ID).publishSettings.publishState.isAcceptingResponses,
    false,
  );
  assert.equal(
    google.forms.get(augustTarget.id).publishSettings.publishState.isAcceptingResponses,
    true,
  );
  const sourceCloseIndex = google.calls.findIndex((call) => (
    call.method === 'setPublishState'
    && call.formId === LIVE_FORM_ID
    && call.isAcceptingResponses === false
  ));
  const targetOpenIndex = google.calls.findIndex((call) => (
    call.method === 'setPublishState'
    && call.formId === augustTarget.id
    && call.isAcceptingResponses === true
  ));
  assert.ok(sourceCloseIndex >= 0 && targetOpenIndex > sourceCloseIndex);
  assert.equal(google.files.get(LIVE_FORM_ID).appProperties.managedBy, undefined);
  assert.equal(google.files.get(LIVE_FORM_ID).appProperties.state, ROLLOVER_FILE_STATES.CLOSED);
  assert.equal((await cutoff.rollover.verify({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    mode: 'active',
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
  })).status, 'valid');
  assert.equal((await cutoff.rollover.activate({
    targetCycle: '2026-08',
    sourceFormId: LIVE_FORM_ID,
    collaboratorPermissions: [PRODUCTION_EDITOR_PERMISSION],
    headSha: 'a'.repeat(40),
  })).status, 'active');

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

test('activate rejects target capability or source-binding drift before closing the source', async () => {
  for (const mutate of [
    (target) => { target.capabilities.canShare = false; },
    (target) => { target.appProperties.sourceFingerprint = '0'.repeat(64); },
  ]) {
    const google = new FakeGoogle();
    const page = new MemoryPage();
    await bootstrapAndPrepare(service({ google, page }));
    const source = google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
    const target = google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
    mutate(target);
    let gateCalled = false;
    const active = service({
      google,
      page,
      clock: fixedClock('2026-08-01T07:00:01Z'),
      runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
      activationGate: async () => {
        gateCalled = true;
        return true;
      },
    });

    await assert.rejects(
      active.rollover.activate({ targetCycle: '2026-08' }),
      /required canShare capability|not bound to the resolved source form/,
    );
    assert.equal(gateCalled, false);
    assert.equal(
      google.forms.get(source.id).publishSettings.publishState.isAcceptingResponses,
      true,
    );
    assert.equal(
      google.forms.get(target.id).publishSettings.publishState.isAcceptingResponses,
      false,
    );
  }
});

test('activate revalidates target binding and ACLs after the preview gate', async () => {
  for (const mutate of [
    (google, target) => { target.appProperties.sourceFingerprint = '0'.repeat(64); },
    (google, target) => {
      google.permissions.get(target.id).push({
        id: 'unexpected-inherited-editor',
        type: 'user',
        role: 'writer',
        emailAddress: 'unexpected-editor@example.org',
        permissionDetails: [{ inherited: true }],
      });
    },
    (google, target) => {
      google.permissions.get(target.id).push({
        id: 'unexpected-rogue-service-account',
        type: 'user',
        role: 'writer',
        emailAddress: 'rogue@private-project.iam.gserviceaccount.com',
        permissionDetails: [{ inherited: true }],
      });
    },
    (google, target) => {
      google.permissions.get(target.id).push({
        id: 'unexpected-drive-manager',
        type: 'user',
        role: 'organizer',
        emailAddress: 'unexpected-manager@example.org',
        permissionDetails: [{ inherited: true }],
      });
    },
  ]) {
    const google = new FakeGoogle();
    const page = new MemoryPage();
    await bootstrapAndPrepare(service({ google, page }));
    const source = google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
    const target = google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
    const active = service({
      google,
      page,
      clock: fixedClock('2026-08-01T07:00:01Z'),
      runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
      activationGate: async () => {
        mutate(google, target);
        return true;
      },
    });

    await assert.rejects(
      active.rollover.activate({ targetCycle: '2026-08' }),
      /not bound to the resolved source form|permissions do not match|unexpected service-account permission|unexpected inherited or administrative edit access/,
    );
    assert.equal(
      google.forms.get(source.id).publishSettings.publishState.isAcceptingResponses,
      true,
    );
    assert.equal(
      google.forms.get(target.id).publishSettings.publishState.isAcceptingResponses,
      false,
    );
    assert.equal(target.appProperties.state, ROLLOVER_FILE_STATES.PREPARED);
  }
});

for (const scenario of [
  {
    name: 'ACL drift',
    error: /permissions do not match/,
    mutate(google, target) {
      const permission = {
        id: 'post-open-unexpected-editor',
        type: 'user',
        role: 'writer',
        emailAddress: 'post-open-unexpected-editor@example.org',
      };
      google.permissions.get(target.id).push(permission);
      return () => {
        google.permissions.set(
          target.id,
          google.permissions.get(target.id).filter(({ id }) => id !== permission.id),
        );
      };
    },
  },
  {
    name: 'visible-title drift',
    error: /visible form title do not match/,
    mutate(google, target) {
      const form = google.forms.get(target.id);
      const original = form.info.title;
      form.info.title = 'Tampered after opening';
      return () => { form.info.title = original; };
    },
  },
  {
    name: 'structure drift',
    error: /structure or settings do not match/,
    mutate(google, target) {
      const form = google.forms.get(target.id);
      const original = form.items[0].title;
      form.items[0].title = 'Tampered question';
      return () => { form.items[0].title = original; };
    },
  },
  {
    name: 'source-fingerprint drift',
    error: /not bound to the resolved source form/,
    mutate(google, target) {
      const original = target.appProperties.sourceFingerprint;
      target.appProperties.sourceFingerprint = '0'.repeat(64);
      return () => { target.appProperties.sourceFingerprint = original; };
    },
  },
  {
    name: 'active-marker drift',
    error: /fully verified active transition/,
    mutate(google, target) {
      target.appProperties.state = ROLLOVER_FILE_STATES.ACTIVATING;
      // ACTIVATING plus an already-open target is the supported recovery
      // state, so the retry itself repairs this injected marker failure.
      return () => {};
    },
  },
]) {
  test(`activate catches post-open ${scenario.name} and recovers idempotently`, async () => {
    const google = new FakeGoogle();
    const page = new MemoryPage();
    await bootstrapAndPrepare(service({ google, page }));
    const source = google.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
    const target = google.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
    const updateFile = google.updateFile.bind(google);
    let injectOnce = true;
    let repair = () => {};
    google.updateFile = async (input) => {
      const updated = await updateFile(input);
      if (
        injectOnce
        && input.fileId === target.id
        && input.appProperties?.state === ROLLOVER_FILE_STATES.ACTIVE
      ) {
        injectOnce = false;
        repair = scenario.mutate(google, target);
      }
      return updated;
    };
    const active = service({
      google,
      page,
      clock: fixedClock('2026-08-01T07:00:01Z'),
      runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
    });

    await assert.rejects(
      active.rollover.activate({ targetCycle: '2026-08' }),
      scenario.error,
    );
    assert.equal(
      google.forms.get(source.id).publishSettings.publishState.isAcceptingResponses,
      false,
    );
    assert.equal(
      google.forms.get(target.id).publishSettings.publishState.isAcceptingResponses,
      true,
    );

    repair();
    const publishCallsBeforeRecovery = google.calls.filter(
      ({ method }) => method === 'setPublishState',
    ).length;
    assert.equal(
      (await active.rollover.activate({ targetCycle: '2026-08' })).status,
      'active',
    );
    assert.equal(
      google.calls.filter(({ method }) => method === 'setPublishState').length,
      publishCallsBeforeRecovery,
    );
    assert.equal(source.appProperties.state, ROLLOVER_FILE_STATES.CLOSED);
    assert.equal(target.appProperties.state, ROLLOVER_FILE_STATES.ACTIVE);
    assert.equal(
      (await active.rollover.verify({ targetCycle: '2026-08', mode: 'active' })).status,
      'valid',
    );
  });
}

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
  source.capabilities.canCopy = false;
  source.capabilities.canShare = false;

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
  const manualGoogle = new FakeGoogle();
  const manualPage = new MemoryPage();
  await bootstrapAndPrepare(service({ google: manualGoogle, page: manualPage }));
  const manualSource = manualGoogle.managed(ROLLOVER_FILE_ROLES.SOURCE, '2026-07')[0];
  const manualTarget = manualGoogle.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  manualGoogle.forms.get(manualSource.id).publishSettings.publishState.isAcceptingResponses = false;
  const manualService = service({
    google: manualGoogle,
    page: manualPage,
    clock: fixedClock('2026-08-01T07:00:01Z'),
    runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
  });
  await assert.rejects(
    manualService.rollover.activate({ targetCycle: '2026-08' }),
    /not in an allowed activation or recovery state/,
  );
  assert.equal(
    manualGoogle.forms.get(manualTarget.id).publishSettings.publishState.isAcceptingResponses,
    false,
  );

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

  const intentGoogle = new FakeGoogle();
  const intentPage = new MemoryPage();
  await bootstrapAndPrepare(service({ google: intentGoogle, page: intentPage }));
  const intentTarget = intentGoogle.managed(ROLLOVER_FILE_ROLES.TARGET, '2026-08')[0];
  intentTarget.appProperties.state = ROLLOVER_FILE_STATES.ACTIVATING;
  const intentRecovery = service({
    google: intentGoogle,
    page: intentPage,
    clock: fixedClock('2026-08-01T07:00:01Z'),
    runtime: stagingRuntime({ simulatedNow: '2026-08-01T07:00:01Z' }),
  });
  assert.equal((await intentRecovery.rollover.activate({ targetCycle: '2026-08' })).status, 'active');
  assert.equal(intentTarget.appProperties.state, ROLLOVER_FILE_STATES.ACTIVE);

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
  assert.equal(target.appProperties.state, ROLLOVER_FILE_STATES.ACTIVATING);
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
      serviceAccountEmail: STAGING_SERVICE_ACCOUNT,
    })),
    /staging service account/,
  );
  assert.throws(
    () => assertRolloverRuntimeSafety(productionRuntime({ eventName: 'pull_request' })),
    /only for schedule or workflow_dispatch/,
  );
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
