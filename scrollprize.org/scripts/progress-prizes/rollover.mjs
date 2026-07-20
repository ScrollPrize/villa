import {
  AUTOMATION_ENVIRONMENTS,
  ROLLOVER_PHASES,
  assertAutomationEnvironment,
  assertPublicResponderUri,
  assertStagingOnlyControl,
  getCycleDeadline,
  parseCycle,
  parseProgressPrizeMarkdown,
  planRollover,
  previousCycle,
  updateProgressPrizeMarkdown,
} from './index.mjs';
import {
  filterDirectCollaboratorPermissions,
  filterPublishedResponderPermissions,
  normalizePermissionForCreate,
  permissionIdentityKey,
} from './google-api.mjs';

export const ROLLOVER_MANAGED_BY = 'scrollprize-progress-prizes';

export const ROLLOVER_FILE_ROLES = Object.freeze({
  SOURCE: 'source',
  TARGET: 'target',
});

export const ROLLOVER_FILE_STATES = Object.freeze({
  COPIED: 'copied',
  PREPARED: 'prepared',
  ACTIVATING: 'activating',
  ACTIVE: 'active',
  CLOSED: 'closed',
  ARCHIVED: 'archived',
});

export const ROLLOVER_FAULTS = Object.freeze({
  AFTER_COPY: 'after-copy',
  AFTER_CLOSE_SOURCE: 'after-close-source',
});

const REQUIRED_GOOGLE_METHODS = Object.freeze([
  'getForm',
  'getFile',
  'listFilesByAppProperties',
  'copyFile',
  'updateFile',
  'getAllPermissions',
  'createPermission',
  'deletePermission',
  'updateFormTitle',
  'setPublishState',
]);

const ALLOWED_EVENTS = new Set(['schedule', 'workflow_dispatch']);
const ALLOWED_VERIFY_MODES = new Set(['prepared', 'active', 'cleaned']);

export class RolloverFaultError extends Error {
  constructor(step) {
    super(`Injected staging rollover failure at ${step}`);
    this.name = 'RolloverFaultError';
    this.step = step;
  }
}

function assertNonEmptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new TypeError(`${label} must be a non-empty string`);
  }
  return value;
}

function normalizeEmail(value) {
  return assertNonEmptyString(value, 'service account identity').trim().toLowerCase();
}

function hasValue(value) {
  return value !== undefined && value !== null && value !== '';
}

function normalizeRuntime(runtime = {}) {
  const environment = assertAutomationEnvironment(runtime.environment);
  const eventName = assertNonEmptyString(runtime.eventName, 'eventName');
  if (!ALLOWED_EVENTS.has(eventName)) {
    throw new Error('Google rollover jobs may run only for schedule or workflow_dispatch events');
  }

  const normalized = {
    ...runtime,
    environment,
    eventName,
    folderId: assertNonEmptyString(runtime.folderId, 'folderId'),
    driveId: assertNonEmptyString(runtime.driveId, 'driveId'),
    serviceAccountEmail: normalizeEmail(runtime.serviceAccountEmail),
    stagingServiceAccountEmail: hasValue(runtime.stagingServiceAccountEmail)
      ? normalizeEmail(runtime.stagingServiceAccountEmail)
      : undefined,
    stagingFolderId: runtime.stagingFolderId,
    branch: assertNonEmptyString(runtime.branch, 'branch'),
    targetBranch: runtime.targetBranch ?? 'main',
    defaultTargetBranch: runtime.defaultTargetBranch ?? 'main',
    smokeBranchPrefix: runtime.smokeBranchPrefix ?? 'codex/progress-prize-smoke-',
    smokeDate: runtime.smokeDate,
  };

  assertNonEmptyString(normalized.targetBranch, 'targetBranch');
  assertNonEmptyString(normalized.defaultTargetBranch, 'defaultTargetBranch');
  assertNonEmptyString(normalized.smokeBranchPrefix, 'smokeBranchPrefix');
  if (normalized.stagingFolderId !== undefined) {
    assertNonEmptyString(normalized.stagingFolderId, 'stagingFolderId');
  }
  if (normalized.archiveFolderId !== undefined) {
    assertNonEmptyString(normalized.archiveFolderId, 'archiveFolderId');
  }
  if (normalized.smokeDate !== undefined && !/^\d{4}-\d{2}-\d{2}$/.test(normalized.smokeDate)) {
    throw new TypeError('smokeDate must use YYYY-MM-DD format');
  }

  return Object.freeze(normalized);
}

function assertStagingIdentity(runtime) {
  if (runtime.environment !== AUTOMATION_ENVIRONMENTS.STAGING) {
    throw new Error('This operation is restricted to the staging environment');
  }
  if (runtime.eventName !== 'workflow_dispatch') {
    throw new Error('Staging mutations require a workflow_dispatch event');
  }
  if (
    runtime.stagingServiceAccountEmail === undefined
    || runtime.serviceAccountEmail !== runtime.stagingServiceAccountEmail
  ) {
    throw new Error('Staging mutations require the configured staging service account');
  }
  if (runtime.stagingFolderId === undefined || runtime.folderId !== runtime.stagingFolderId) {
    throw new Error('Staging mutations require the configured staging folder');
  }
  if (!runtime.branch.startsWith(runtime.smokeBranchPrefix)) {
    throw new Error('Staging mutations require the configured smoke branch prefix');
  }
}

/**
 * Validate controls before any Google or filesystem call is made.
 *
 * Identifiers are compared but deliberately omitted from every error message so
 * that a failed public workflow cannot disclose private Workspace configuration.
 */
export function assertRolloverRuntimeSafety(runtimeInput, { faultInjection } = {}) {
  const runtime = normalizeRuntime(runtimeInput);
  const hasSimulatedTime = hasValue(runtime.simulatedNow);
  const hasFault = hasValue(faultInjection);

  assertStagingOnlyControl({
    environment: runtime.environment,
    eventName: runtime.eventName,
    controlName: 'simulated time',
    enabled: hasSimulatedTime,
  });
  assertStagingOnlyControl({
    environment: runtime.environment,
    eventName: runtime.eventName,
    controlName: 'fault injection',
    enabled: hasFault,
  });

  if (hasFault && !Object.values(ROLLOVER_FAULTS).includes(faultInjection)) {
    throw new Error(`Unknown staging fault injection step ${JSON.stringify(faultInjection)}`);
  }

  if (runtime.environment === AUTOMATION_ENVIRONMENTS.PRODUCTION) {
    if (runtime.targetBranch !== runtime.defaultTargetBranch) {
      throw new Error('Production rollover cannot use an alternate target branch');
    }
    if (runtime.stagingFolderId !== undefined) {
      throw new Error('Production rollover cannot be supplied a staging folder');
    }
    if (
      runtime.stagingServiceAccountEmail !== undefined
      && runtime.serviceAccountEmail === runtime.stagingServiceAccountEmail
    ) {
      throw new Error('Production rollover cannot use the staging service account');
    }
  }

  if (hasSimulatedTime || hasFault) {
    assertStagingIdentity(runtime);
  }

  return runtime;
}

function assertMutationContext(runtime) {
  if (runtime.environment === AUTOMATION_ENVIRONMENTS.STAGING) {
    assertStagingIdentity(runtime);
  }
}

function assertDependencies({ google, page, clock, activationGate }) {
  if (google === null || typeof google !== 'object') {
    throw new TypeError('google must be an injected Google facade');
  }
  for (const method of REQUIRED_GOOGLE_METHODS) {
    if (typeof google[method] !== 'function') {
      throw new TypeError(`google.${method} must be a function`);
    }
  }
  if (page === null || typeof page !== 'object') {
    throw new TypeError('page must be an injected page facade');
  }
  for (const method of ['read', 'write']) {
    if (typeof page[method] !== 'function') {
      throw new TypeError(`page.${method} must be a function`);
    }
  }
  if (page.resolveResponderUri !== undefined && typeof page.resolveResponderUri !== 'function') {
    throw new TypeError('page.resolveResponderUri must be a function when supplied');
  }
  if (clock === null || typeof clock !== 'object' || typeof clock.now !== 'function') {
    throw new TypeError('clock.now must be an injected function');
  }
  if (activationGate !== undefined && typeof activationGate !== 'function') {
    throw new TypeError('activationGate must be a function when supplied');
  }
}

function managedProperties(runtime, role, cycle, state) {
  return {
    managedBy: ROLLOVER_MANAGED_BY,
    schemaVersion: '1',
    environment: runtime.environment,
    role,
    cycle,
    state,
  };
}

function managedQuery(runtime, role, cycle) {
  return {
    managedBy: ROLLOVER_MANAGED_BY,
    schemaVersion: '1',
    environment: runtime.environment,
    role,
    cycle,
  };
}

function cycleTitle(cycle) {
  const deadline = getCycleDeadline(cycle);
  return `${deadline.monthName} ${deadline.year} Progress Prizes`;
}

function smokeDate(runtime, clock) {
  return runtime.smokeDate ?? clock.now().toISOString().slice(0, 10);
}

function managedTitle(runtime, clock, role, cycle) {
  const title = cycleTitle(cycle);
  if (runtime.environment === AUTOMATION_ENVIRONMENTS.PRODUCTION) return title;
  return `[SMOKE ${role.toUpperCase()} ${smokeDate(runtime, clock)}] ${title}`;
}

function publishState(form) {
  const state = form?.publishSettings?.publishState;
  if (
    state === null
    || typeof state !== 'object'
    || typeof state.isPublished !== 'boolean'
    || typeof state.isAcceptingResponses !== 'boolean'
  ) {
    throw new Error('Form does not expose modern publishSettings; migrate it before rollover');
  }
  if (!state.isPublished && state.isAcceptingResponses) {
    throw new Error('Form reports an invalid publishing state');
  }
  return state;
}

function assertNoLinkedSheet(form) {
  if (hasValue(form?.linkedSheetId)) {
    throw new Error('Form has a linked response Sheet; automatic rollover is intentionally disabled');
  }
}

function canonicalize(value) {
  if (Array.isArray(value)) return value.map(canonicalize);
  if (value === null || typeof value !== 'object') return value;

  const ignoredKeys = new Set(['formId', 'itemId', 'questionId', 'revisionId']);
  return Object.fromEntries(
    Object.entries(value)
      .filter(([key]) => !ignoredKeys.has(key))
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, canonicalize(child)]),
  );
}

function formStructure(form) {
  const info = { ...(form?.info ?? {}) };
  delete info.title;
  delete info.documentTitle;
  return canonicalize({
    info,
    settings: form?.settings ?? {},
    items: form?.items ?? [],
  });
}

function assertSameStructure(source, target) {
  if (JSON.stringify(formStructure(source)) !== JSON.stringify(formStructure(target))) {
    throw new Error('Copied form structure or settings do not match the source form');
  }
}

function comparablePermissions(permissions) {
  return [
    ...filterDirectCollaboratorPermissions(permissions),
    ...filterPublishedResponderPermissions(permissions),
  ];
}

function effectiveCollaboratorPermissions(permissions) {
  return permissions.filter((permission) => (
    permission
    && permission.deleted !== true
    && permission.pendingOwner !== true
    && ['user', 'group'].includes(permission.type)
    && ['writer', 'commenter'].includes(permission.role)
    && typeof permission.emailAddress === 'string'
    && permission.emailAddress !== ''
  ));
}

function effectiveComparablePermissions(permissions) {
  return [
    ...effectiveCollaboratorPermissions(permissions),
    ...filterPublishedResponderPermissions(permissions),
  ];
}

function assertCollaboratorPermissions(permissions) {
  if (!Array.isArray(permissions)) {
    throw new TypeError('collaboratorPermissions must be an array');
  }
  for (const permission of permissions) {
    const matches = filterDirectCollaboratorPermissions([permission]);
    if (matches.length !== 1) {
      throw new Error('Configured collaborators must be direct user/group writers or commenters');
    }
    normalizePermissionForCreate(permission);
  }
}

function uniquePermissions(permissions) {
  const byKey = new Map();
  for (const permission of permissions) {
    byKey.set(permissionIdentityKey(permission), permission);
  }
  return [...byKey.values()];
}

function desiredPermissions(sourcePermissions, collaboratorPermissions, copySourceCollaborators) {
  assertCollaboratorPermissions(collaboratorPermissions);
  return uniquePermissions([
    ...(copySourceCollaborators
      ? filterDirectCollaboratorPermissions(sourcePermissions)
      : []),
    ...collaboratorPermissions,
    ...filterPublishedResponderPermissions(sourcePermissions),
  ]);
}

function assertPermissionsMatch(expected, actual) {
  const expectedKeys = new Set(expected.map(permissionIdentityKey));
  const effectiveKeys = new Set(effectiveComparablePermissions(actual).map(permissionIdentityKey));
  const unexpectedDirect = comparablePermissions(actual)
    .map(permissionIdentityKey)
    .some((key) => !expectedKeys.has(key));
  if (unexpectedDirect || [...expectedKeys].some((key) => !effectiveKeys.has(key))) {
    throw new Error('Target form collaborator or published-responder permissions do not match');
  }
}

function assertRequiredSourcePermissions(permissions, requiredCollaborators = []) {
  assertCollaboratorPermissions(requiredCollaborators);
  if (filterPublishedResponderPermissions(permissions).length === 0) {
    throw new Error('Source form is missing its published responder permission');
  }
  for (const required of requiredCollaborators) {
    const expected = normalizePermissionForCreate(required);
    const found = permissions.some((permission) => (
      permission?.deleted !== true
      && permission?.pendingOwner !== true
      && permission?.type === expected.type
      && permission?.role === expected.role
      && permission?.emailAddress?.toLowerCase() === expected.emailAddress?.toLowerCase()
    ));
    if (!found) {
      throw new Error('Source form is missing a configured internal collaborator');
    }
  }
}

async function ensurePermissions(google, fileId, expected) {
  let current = await google.getAllPermissions({ fileId });
  const currentKeys = new Set(effectiveComparablePermissions(current).map(permissionIdentityKey));
  for (const permission of expected) {
    const key = permissionIdentityKey(permission);
    if (!currentKeys.has(key)) {
      await google.createPermission({
        fileId,
        permission: normalizePermissionForCreate(permission),
        sendNotificationEmail: false,
      });
      currentKeys.add(key);
    }
  }
  current = await google.getAllPermissions({ fileId });
  assertPermissionsMatch(expected, current);
  return current;
}

function assertExpectedPublishing(form, expected) {
  const actual = publishState(form);
  if (
    actual.isPublished !== expected.isPublished
    || actual.isAcceptingResponses !== expected.isAcceptingResponses
  ) {
    throw new Error('Form publishing state does not match the rollover state');
  }
}

function publishingMatches(form, expected) {
  const actual = publishState(form);
  return actual.isPublished === expected.isPublished
    && actual.isAcceptingResponses === expected.isAcceptingResponses;
}

function activationTransition(sourceSnapshot, targetSnapshot) {
  const sourceState = sourceSnapshot.file.appProperties?.state;
  const targetState = targetSnapshot.file.appProperties?.state;
  const sourceOpen = publishingMatches(sourceSnapshot.form, {
    isPublished: true,
    isAcceptingResponses: true,
  });
  const sourceClosed = publishingMatches(sourceSnapshot.form, {
    isPublished: true,
    isAcceptingResponses: false,
  });
  const targetClosed = publishingMatches(targetSnapshot.form, {
    isPublished: true,
    isAcceptingResponses: false,
  });
  const targetOpen = publishingMatches(targetSnapshot.form, {
    isPublished: true,
    isAcceptingResponses: true,
  });

  if (
    sourceOpen
    && targetClosed
    && targetState === ROLLOVER_FILE_STATES.PREPARED
    && ![ROLLOVER_FILE_STATES.CLOSED, ROLLOVER_FILE_STATES.ARCHIVED].includes(sourceState)
  ) return 'prepared';
  if (
    sourceClosed
    && targetClosed
    && sourceState !== ROLLOVER_FILE_STATES.ARCHIVED
    && [ROLLOVER_FILE_STATES.PREPARED, ROLLOVER_FILE_STATES.ACTIVATING].includes(targetState)
  ) return 'recover-closed';
  if (
    sourceClosed
    && targetOpen
    && sourceState === ROLLOVER_FILE_STATES.CLOSED
    && targetState === ROLLOVER_FILE_STATES.ACTIVATING
  ) return 'recover-opened';
  if (
    sourceClosed
    && targetOpen
    && sourceState === ROLLOVER_FILE_STATES.CLOSED
    && targetState === ROLLOVER_FILE_STATES.ACTIVE
  ) return 'active';
  throw new Error('Forms are not in an allowed activation or recovery state');
}

async function ensurePublishState(google, form, expected) {
  const actual = publishState(form);
  if (
    actual.isPublished === expected.isPublished
    && actual.isAcceptingResponses === expected.isAcceptingResponses
  ) {
    return form;
  }
  await google.setPublishState({ formId: form.formId, ...expected });
  const updated = await google.getForm({ formId: form.formId });
  assertExpectedPublishing(updated, expected);
  return updated;
}

function assertTitle(form, file, expectedTitle) {
  if (form?.info?.title !== expectedTitle || file?.name !== expectedTitle) {
    throw new Error('Drive filename and visible form title do not match the expected cycle title');
  }
}

function assertSourceCapabilities(file, required = ['canCopy', 'canEdit', 'canShare']) {
  const capabilities = file?.capabilities ?? {};
  for (const capability of required) {
    if (capabilities[capability] !== true) {
      throw new Error(`Google identity lacks the required ${capability} capability`);
    }
  }
}

function assertSharedDriveLocation(file, runtime, { requireActiveFolder = true } = {}) {
  if (file?.driveId !== runtime.driveId) {
    throw new Error('Form is not in the configured Shared Drive');
  }
  if (requireActiveFolder && !file?.parents?.includes(runtime.folderId)) {
    throw new Error('Form is not in the configured active folder');
  }
}

function publicSummary({ cycle, title, form, permissions, file }) {
  const state = publishState(form);
  return Object.freeze({
    cycle,
    title,
    responderUri: assertPublicResponderUri(form.responderUri),
    isPublished: state.isPublished,
    isAcceptingResponses: state.isAcceptingResponses,
    collaboratorCount: effectiveCollaboratorPermissions(permissions).length,
    hasPublicResponderPermission: filterPublishedResponderPermissions(permissions).length > 0,
    canCopy: file?.capabilities?.canCopy === true,
    canEdit: file?.capabilities?.canEdit === true,
    canShare: file?.capabilities?.canShare === true,
  });
}

async function resolvePublicResponder(page, responderUri) {
  const publicUri = assertPublicResponderUri(responderUri);
  const resolved = page.resolveResponderUri === undefined
    ? publicUri
    : await page.resolveResponderUri(publicUri);
  return assertPublicResponderUri(resolved);
}

async function assertResponderUrisMatch(page, actualUri, expectedUri) {
  const [actual, expected] = await Promise.all([
    resolvePublicResponder(page, actualUri),
    resolvePublicResponder(page, expectedUri),
  ]);
  if (actual !== expected) {
    throw new Error('The managed website responder URL does not match the Google form');
  }
}

/**
 * Build the orchestration service. The facade owns no filesystem, GitHub, or
 * process-global state; callers inject those boundaries so workflow tests can
 * exercise the exact production transition code without real side effects.
 */
export function createRolloverService({
  google,
  page,
  clock,
  runtime: runtimeInput,
  activationGate,
} = {}) {
  assertDependencies({ google, page, clock, activationGate });
  const runtime = assertRolloverRuntimeSafety(runtimeInput);

  async function findManagedFile(role, cycle, { inActiveFolder = true } = {}) {
    const files = await google.listFilesByAppProperties({
      appProperties: managedQuery(runtime, role, cycle),
      driveId: runtime.driveId,
    });
    if (files.length > 1) {
      throw new Error(`Multiple managed ${role} forms exist for cycle ${cycle}; refusing to guess`);
    }
    if (files[0] !== undefined) {
      assertSharedDriveLocation(files[0], runtime, { requireActiveFolder: inActiveFolder });
    }
    return files[0];
  }

  async function requireManagedFile(role, cycle, options) {
    const file = await findManagedFile(role, cycle, options);
    if (file === undefined) {
      throw new Error(`Managed ${role} form is missing for cycle ${cycle}`);
    }
    return file;
  }

  async function resolveSourceFile({ sourceFormId, sourceCycle, inActiveFolder = true }) {
    const [managedTarget, managedSource] = await Promise.all([
      findManagedFile(ROLLOVER_FILE_ROLES.TARGET, sourceCycle, { inActiveFolder }),
      findManagedFile(ROLLOVER_FILE_ROLES.SOURCE, sourceCycle, { inActiveFolder }),
    ]);
    if (managedTarget !== undefined && managedSource !== undefined) {
      throw new Error(`Multiple managed source candidates exist for cycle ${sourceCycle}`);
    }
    if (managedTarget !== undefined || managedSource !== undefined) {
      return managedTarget ?? managedSource;
    }
    if (sourceFormId !== undefined) {
      assertNonEmptyString(sourceFormId, 'sourceFormId');
      const file = await google.getFile({ fileId: sourceFormId });
      assertSharedDriveLocation(file, runtime, { requireActiveFolder: inActiveFolder });
      return file;
    }
    throw new Error(`Managed source form is missing for cycle ${sourceCycle}`);
  }

  async function loadSnapshot(file) {
    const [form, currentFile, permissions] = await Promise.all([
      google.getForm({ formId: file.id }),
      google.getFile({ fileId: file.id }),
      google.getAllPermissions({ fileId: file.id }),
    ]);
    assertNoLinkedSheet(form);
    publishState(form);
    assertPublicResponderUri(form.responderUri);
    return { form, file: currentFile, permissions };
  }

  async function markFile(file, state, extra = {}) {
    const current = await google.getFile({ fileId: file.id });
    const appProperties = {
      ...(current.appProperties ?? {}),
      ...extra,
      state,
    };
    const alreadyMarked = Object.entries(appProperties).every(
      ([key, value]) => current.appProperties?.[key] === value,
    );
    if (alreadyMarked) return current;
    return google.updateFile({ fileId: file.id, appProperties });
  }

  async function updateTitles(file, form, title) {
    let currentFile = file;
    let currentForm = form;
    if (currentFile.name !== title) {
      currentFile = await google.updateFile({ fileId: currentFile.id, name: title });
    }
    if (currentForm.info?.title !== title) {
      await google.updateFormTitle({
        formId: currentForm.formId,
        title,
        requiredRevisionId: currentForm.revisionId,
      });
      currentForm = await google.getForm({ formId: currentForm.formId });
    }
    return { file: currentFile, form: currentForm };
  }

  async function assertPageState(cycle, responderUri) {
    const parsed = parseProgressPrizeMarkdown(await page.read());
    if (parsed.cycle !== cycle) {
      throw new Error(`Managed website cycle is ${parsed.cycle}; expected ${cycle}`);
    }
    await assertResponderUrisMatch(page, parsed.responderUri, responderUri);
    return parsed;
  }

  async function validate({
    sourceFormId,
    sourceCycle,
    collaboratorPermissions = [],
    expectedTitle = cycleTitle(sourceCycle),
    expectedResponderUri,
    verifyPage = true,
    expectAcceptingResponses = true,
  } = {}) {
    parseCycle(sourceCycle);
    const file = await resolveSourceFile({ sourceFormId, sourceCycle });
    const snapshot = await loadSnapshot(file);
    assertSourceCapabilities(snapshot.file);
    assertRequiredSourcePermissions(snapshot.permissions, collaboratorPermissions);
    assertTitle(snapshot.form, snapshot.file, expectedTitle);
    const state = publishState(snapshot.form);
    if (!state.isPublished || state.isAcceptingResponses !== expectAcceptingResponses) {
      throw new Error('Source form is not in the expected live publishing state');
    }
    if (expectedResponderUri !== undefined) {
      await assertResponderUrisMatch(page, snapshot.form.responderUri, expectedResponderUri);
    }
    if (verifyPage) {
      await assertPageState(sourceCycle, snapshot.form.responderUri);
    }
    return Object.freeze({
      action: 'validate',
      status: 'valid',
      ...publicSummary({ cycle: sourceCycle, title: expectedTitle, ...snapshot }),
      hasLinkedSheet: false,
    });
  }

  async function bootstrapStagingSource({
    sourceFormId,
    sourceCycle,
    collaboratorPermissions = [],
    dryRun = false,
    faultInjection,
  } = {}) {
    assertRolloverRuntimeSafety(runtime, { faultInjection });
    assertStagingIdentity(runtime);
    parseCycle(sourceCycle);
    assertNonEmptyString(sourceFormId, 'sourceFormId');

    const liveFile = await google.getFile({ fileId: sourceFormId });
    const liveSnapshot = await loadSnapshot(liveFile);
    // The staging identity deliberately has read/copy-only access to production.
    // It must never need edit or share capability on the active live form.
    assertSourceCapabilities(liveSnapshot.file, ['canCopy']);
    assertRequiredSourcePermissions(liveSnapshot.permissions);
    assertTitle(liveSnapshot.form, liveSnapshot.file, cycleTitle(sourceCycle));
    assertExpectedPublishing(liveSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: true,
    });
    await assertPageState(sourceCycle, liveSnapshot.form.responderUri);

    let stagedFile = await findManagedFile(ROLLOVER_FILE_ROLES.SOURCE, sourceCycle);
    if (dryRun && stagedFile === undefined) {
      return Object.freeze({
        action: 'bootstrap',
        status: 'planned',
        cycle: sourceCycle,
        title: managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.SOURCE, sourceCycle),
        created: false,
        resumed: false,
      });
    }

    let created = false;
    if (stagedFile === undefined) {
      stagedFile = await google.copyFile({
        fileId: sourceFormId,
        name: managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.SOURCE, sourceCycle),
        parentId: runtime.folderId,
        appProperties: managedProperties(
          runtime,
          ROLLOVER_FILE_ROLES.SOURCE,
          sourceCycle,
          ROLLOVER_FILE_STATES.COPIED,
        ),
      });
      created = true;
      if (faultInjection === ROLLOVER_FAULTS.AFTER_COPY) {
        throw new RolloverFaultError(ROLLOVER_FAULTS.AFTER_COPY);
      }
    }

    if (dryRun) {
      return Object.freeze({
        action: 'bootstrap',
        status: 'planned',
        cycle: sourceCycle,
        title: managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.SOURCE, sourceCycle),
        created: false,
        resumed: true,
      });
    }

    let stagedSnapshot = await loadSnapshot(stagedFile);
    const title = managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.SOURCE, sourceCycle);
    const titled = await updateTitles(stagedSnapshot.file, stagedSnapshot.form, title);
    stagedSnapshot = {
      ...stagedSnapshot,
      ...titled,
    };
    stagedSnapshot.form = await ensurePublishState(google, stagedSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: true,
    });
    const expectedPermissions = desiredPermissions(
      liveSnapshot.permissions,
      collaboratorPermissions,
      false,
    );
    stagedSnapshot.permissions = await ensurePermissions(
      google,
      stagedSnapshot.file.id,
      expectedPermissions,
    );
    assertSameStructure(liveSnapshot.form, stagedSnapshot.form);
    assertTitle(stagedSnapshot.form, stagedSnapshot.file, title);
    assertExpectedPublishing(stagedSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: true,
    });
    await markFile(stagedSnapshot.file, ROLLOVER_FILE_STATES.ACTIVE, {
      sourceCycle,
    });

    const liveAfter = await loadSnapshot(liveFile);
    if (
      JSON.stringify(formStructure(liveSnapshot.form)) !== JSON.stringify(formStructure(liveAfter.form))
      || liveSnapshot.form.info?.title !== liveAfter.form.info?.title
      || JSON.stringify(publishState(liveSnapshot.form)) !== JSON.stringify(publishState(liveAfter.form))
      || liveSnapshot.form.responderUri !== liveAfter.form.responderUri
    ) {
      throw new Error('Production source changed during staging bootstrap');
    }

    return Object.freeze({
      action: 'bootstrap',
      status: 'active',
      created,
      resumed: !created,
      ...publicSummary({ cycle: sourceCycle, title, ...stagedSnapshot }),
    });
  }

  async function prepare({
    targetCycle,
    sourceFormId,
    collaboratorPermissions = [],
    copySourceCollaborators = true,
    expectedCurrentResponderUri,
    preparationDays = 7,
    dryRun = false,
    faultInjection,
  } = {}) {
    assertRolloverRuntimeSafety(runtime, { faultInjection });
    assertMutationContext(runtime);
    parseCycle(targetCycle);
    const rollover = planRollover({ targetCycle, now: clock.now(), preparationDays });
    if (rollover.phase === ROLLOVER_PHASES.WAITING) {
      return Object.freeze({
        action: 'prepare',
        status: 'waiting',
        sourceCycle: rollover.sourceCycle,
        targetCycle,
        preparationOpensAt: rollover.preparationOpensAt.toISOString(),
      });
    }

    const sourceFile = await resolveSourceFile({
      sourceFormId,
      sourceCycle: rollover.sourceCycle,
    });
    const sourceSnapshot = await loadSnapshot(sourceFile);
    assertSourceCapabilities(sourceSnapshot.file);
    assertRequiredSourcePermissions(sourceSnapshot.permissions, collaboratorPermissions);
    assertExpectedPublishing(sourceSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: true,
    });
    const sourceTitle = managedTitle(
      runtime,
      clock,
      ROLLOVER_FILE_ROLES.SOURCE,
      rollover.sourceCycle,
    );
    assertTitle(sourceSnapshot.form, sourceSnapshot.file, sourceTitle);

    const currentPage = parseProgressPrizeMarkdown(await page.read());
    if (currentPage.cycle !== rollover.sourceCycle && currentPage.cycle !== targetCycle) {
      throw new Error('Managed page is not at the source or target rollover cycle');
    }
    if (currentPage.cycle === rollover.sourceCycle) {
      const expectedCurrent = expectedCurrentResponderUri
        ?? (runtime.environment === AUTOMATION_ENVIRONMENTS.PRODUCTION
          ? sourceSnapshot.form.responderUri
          : undefined);
      if (expectedCurrent !== undefined) {
        await assertResponderUrisMatch(page, currentPage.responderUri, expectedCurrent);
      }
    }

    let targetFile = await findManagedFile(ROLLOVER_FILE_ROLES.TARGET, targetCycle);
    if (dryRun && targetFile === undefined) {
      return Object.freeze({
        action: 'prepare',
        status: 'planned',
        sourceCycle: rollover.sourceCycle,
        targetCycle,
        title: managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.TARGET, targetCycle),
        created: false,
        resumed: false,
      });
    }

    let created = false;
    if (targetFile === undefined) {
      targetFile = await google.copyFile({
        fileId: sourceSnapshot.file.id,
        name: managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.TARGET, targetCycle),
        parentId: runtime.folderId,
        appProperties: managedProperties(
          runtime,
          ROLLOVER_FILE_ROLES.TARGET,
          targetCycle,
          ROLLOVER_FILE_STATES.COPIED,
        ),
      });
      created = true;
      if (faultInjection === ROLLOVER_FAULTS.AFTER_COPY) {
        const copiedSnapshot = await loadSnapshot(targetFile);
        const closedCopy = await ensurePublishState(google, copiedSnapshot.form, {
          isPublished: true,
          isAcceptingResponses: false,
        });
        assertExpectedPublishing(closedCopy, {
          isPublished: true,
          isAcceptingResponses: false,
        });
        throw new RolloverFaultError(ROLLOVER_FAULTS.AFTER_COPY);
      }
    }

    if (dryRun) {
      const existing = await loadSnapshot(targetFile);
      return Object.freeze({
        action: 'prepare',
        status: 'planned',
        sourceCycle: rollover.sourceCycle,
        targetCycle,
        title: managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.TARGET, targetCycle),
        created: false,
        resumed: true,
        responderUri: assertPublicResponderUri(existing.form.responderUri),
      });
    }

    let targetSnapshot = await loadSnapshot(targetFile);
    // A copied form may inherit an accepting state. Close it before title and
    // ACL work so an unprepared target is never left accepting responses.
    targetSnapshot.form = await ensurePublishState(google, targetSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: false,
    });
    const title = managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.TARGET, targetCycle);
    const titled = await updateTitles(targetSnapshot.file, targetSnapshot.form, title);
    targetSnapshot = { ...targetSnapshot, ...titled };
    const expectedPermissions = desiredPermissions(
      sourceSnapshot.permissions,
      collaboratorPermissions,
      copySourceCollaborators,
    );
    targetSnapshot.permissions = await ensurePermissions(
      google,
      targetSnapshot.file.id,
      expectedPermissions,
    );
    assertSameStructure(sourceSnapshot.form, targetSnapshot.form);
    assertTitle(targetSnapshot.form, targetSnapshot.file, title);
    assertExpectedPublishing(targetSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: false,
    });
    await markFile(targetSnapshot.file, ROLLOVER_FILE_STATES.PREPARED, {
      sourceCycle: rollover.sourceCycle,
    });

    const markdown = await page.read();
    const latestPage = parseProgressPrizeMarkdown(markdown);
    let update;
    if (latestPage.cycle === targetCycle) {
      await assertResponderUrisMatch(
        page,
        latestPage.responderUri,
        targetSnapshot.form.responderUri,
      );
      update = {
        content: markdown,
        changed: false,
        current: latestPage,
      };
    } else {
      update = updateProgressPrizeMarkdown(markdown, {
        targetCycle,
        responderUri: targetSnapshot.form.responderUri,
        expectedCurrentCycle: rollover.sourceCycle,
      });
    }
    if (update.changed) {
      await page.write(update.content);
    }
    await assertPageState(targetCycle, targetSnapshot.form.responderUri);

    return Object.freeze({
      action: 'prepare',
      status: 'prepared',
      sourceCycle: rollover.sourceCycle,
      targetCycle,
      created,
      resumed: !created,
      pageChanged: update.changed,
      ...publicSummary({ cycle: targetCycle, title, ...targetSnapshot }),
    });
  }

  async function activate({
    targetCycle,
    sourceFormId,
    collaboratorPermissions = [],
    copySourceCollaborators = true,
    preparationDays = 7,
    faultInjection,
    headSha,
  } = {}) {
    assertRolloverRuntimeSafety(runtime, { faultInjection });
    assertMutationContext(runtime);
    parseCycle(targetCycle);
    const rollover = planRollover({ targetCycle, now: clock.now(), preparationDays });
    if (rollover.phase !== ROLLOVER_PHASES.ACTIVATE) {
      throw new Error('Activation is forbidden before the source-cycle cutoff');
    }

    const sourceFile = await resolveSourceFile({
      sourceFormId,
      sourceCycle: rollover.sourceCycle,
    });
    const targetFile = await requireManagedFile(ROLLOVER_FILE_ROLES.TARGET, targetCycle);
    const [sourceSnapshot, targetSnapshotBefore] = await Promise.all([
      loadSnapshot(sourceFile),
      loadSnapshot(targetFile),
    ]);
    assertSameStructure(sourceSnapshot.form, targetSnapshotBefore.form);
    assertRequiredSourcePermissions(sourceSnapshot.permissions, collaboratorPermissions);
    assertPermissionsMatch(
      desiredPermissions(
        sourceSnapshot.permissions,
        collaboratorPermissions,
        copySourceCollaborators,
      ),
      targetSnapshotBefore.permissions,
    );
    assertTitle(
      sourceSnapshot.form,
      sourceSnapshot.file,
      managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.SOURCE, rollover.sourceCycle),
    );
    assertTitle(
      targetSnapshotBefore.form,
      targetSnapshotBefore.file,
      managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.TARGET, targetCycle),
    );
    await assertPageState(targetCycle, targetSnapshotBefore.form.responderUri);
    const transition = activationTransition(sourceSnapshot, targetSnapshotBefore);

    if (activationGate === undefined) {
      throw new Error('Activation requires an injected successful code and Vercel preview gate');
    }
    if (runtime.environment === AUTOMATION_ENVIRONMENTS.PRODUCTION) {
      assertNonEmptyString(headSha, 'headSha');
    }
    const gate = await activationGate({
      targetCycle,
      responderUri: targetSnapshotBefore.form.responderUri,
      branch: runtime.branch,
      targetBranch: runtime.targetBranch,
      headSha,
    });
    if (gate !== true && gate?.ok !== true) {
      throw new Error('Code checks and Vercel preview gate did not pass for activation');
    }

    if (transition === 'active') {
      return Object.freeze({
        action: 'activate',
        status: 'active',
        sourceCycle: rollover.sourceCycle,
        targetCycle,
        sourceAcceptingResponses: false,
        targetAcceptingResponses: true,
        responderUri: assertPublicResponderUri(targetSnapshotBefore.form.responderUri),
      });
    }

    const sourceForm = await ensurePublishState(google, sourceSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: false,
    });
    await markFile(sourceSnapshot.file, ROLLOVER_FILE_STATES.CLOSED, {
      targetCycle,
    });
    if (faultInjection === ROLLOVER_FAULTS.AFTER_CLOSE_SOURCE) {
      throw new RolloverFaultError(ROLLOVER_FAULTS.AFTER_CLOSE_SOURCE);
    }

    await markFile(targetSnapshotBefore.file, ROLLOVER_FILE_STATES.ACTIVATING, {
      sourceCycle: rollover.sourceCycle,
    });
    const targetForm = await ensurePublishState(google, targetSnapshotBefore.form, {
      isPublished: true,
      isAcceptingResponses: true,
    });
    await markFile(targetSnapshotBefore.file, ROLLOVER_FILE_STATES.ACTIVE, {
      sourceCycle: rollover.sourceCycle,
    });

    assertExpectedPublishing(sourceForm, {
      isPublished: true,
      isAcceptingResponses: false,
    });
    assertExpectedPublishing(targetForm, {
      isPublished: true,
      isAcceptingResponses: true,
    });

    return Object.freeze({
      action: 'activate',
      status: 'active',
      sourceCycle: rollover.sourceCycle,
      targetCycle,
      sourceAcceptingResponses: false,
      targetAcceptingResponses: true,
      responderUri: assertPublicResponderUri(targetForm.responderUri),
    });
  }

  async function verify({
    targetCycle,
    sourceFormId,
    mode = 'prepared',
    collaboratorPermissions = [],
    copySourceCollaborators = true,
  } = {}) {
    parseCycle(targetCycle);
    if (!ALLOWED_VERIFY_MODES.has(mode)) {
      throw new Error('verify mode must be prepared, active, or cleaned');
    }
    const sourceCycle = previousCycle(targetCycle);
    const inActiveFolder = mode !== 'cleaned';
    const sourceFile = mode === 'cleaned'
      ? await requireManagedFile(ROLLOVER_FILE_ROLES.SOURCE, sourceCycle, { inActiveFolder: false })
      : await resolveSourceFile({ sourceFormId, sourceCycle, inActiveFolder });
    const targetFile = await requireManagedFile(ROLLOVER_FILE_ROLES.TARGET, targetCycle, {
      inActiveFolder,
    });
    const [sourceSnapshot, targetSnapshot] = await Promise.all([
      loadSnapshot(sourceFile),
      loadSnapshot(targetFile),
    ]);
    assertSameStructure(sourceSnapshot.form, targetSnapshot.form);
    assertTitle(
      sourceSnapshot.form,
      sourceSnapshot.file,
      managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.SOURCE, sourceCycle),
    );
    assertTitle(
      targetSnapshot.form,
      targetSnapshot.file,
      managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.TARGET, targetCycle),
    );

    if (mode !== 'cleaned') {
      assertRequiredSourcePermissions(sourceSnapshot.permissions, collaboratorPermissions);
    }
    const expectedPermissions = desiredPermissions(
      sourceSnapshot.permissions,
      collaboratorPermissions,
      copySourceCollaborators,
    );
    assertPermissionsMatch(expectedPermissions, targetSnapshot.permissions);

    if (mode === 'prepared') {
      assertExpectedPublishing(sourceSnapshot.form, {
        isPublished: true,
        isAcceptingResponses: true,
      });
      assertExpectedPublishing(targetSnapshot.form, {
        isPublished: true,
        isAcceptingResponses: false,
      });
    } else if (mode === 'active') {
      assertExpectedPublishing(sourceSnapshot.form, {
        isPublished: true,
        isAcceptingResponses: false,
      });
      assertExpectedPublishing(targetSnapshot.form, {
        isPublished: true,
        isAcceptingResponses: true,
      });
    } else {
      assertStagingIdentity(runtime);
      assertNonEmptyString(runtime.archiveFolderId, 'archiveFolderId');
      assertExpectedPublishing(sourceSnapshot.form, {
        isPublished: false,
        isAcceptingResponses: false,
      });
      assertExpectedPublishing(targetSnapshot.form, {
        isPublished: false,
        isAcceptingResponses: false,
      });
      if (
        filterPublishedResponderPermissions(sourceSnapshot.permissions).length > 0
        || filterPublishedResponderPermissions(targetSnapshot.permissions).length > 0
      ) {
        throw new Error('Cleaned staging forms still have public responder permissions');
      }
      for (const file of [sourceSnapshot.file, targetSnapshot.file]) {
        if (
          file.appProperties?.state !== ROLLOVER_FILE_STATES.ARCHIVED
          || !file.parents?.includes(runtime.archiveFolderId)
          || file.parents?.includes(runtime.folderId)
        ) {
          throw new Error('Cleaned staging forms are not in the configured archive state');
        }
      }
    }

    if (mode !== 'cleaned') {
      await assertPageState(targetCycle, targetSnapshot.form.responderUri);
    }
    return Object.freeze({
      action: 'verify',
      status: 'valid',
      mode,
      sourceCycle,
      targetCycle,
      responderUri: assertPublicResponderUri(targetSnapshot.form.responderUri),
    });
  }

  async function cleanup({ targetCycle } = {}) {
    assertStagingIdentity(runtime);
    parseCycle(targetCycle);
    assertNonEmptyString(runtime.archiveFolderId, 'archiveFolderId');
    const sourceCycle = previousCycle(targetCycle);
    const files = [
      await requireManagedFile(ROLLOVER_FILE_ROLES.SOURCE, sourceCycle, { inActiveFolder: false }),
      await requireManagedFile(ROLLOVER_FILE_ROLES.TARGET, targetCycle, { inActiveFolder: false }),
    ];

    for (const file of files) {
      let form = await google.getForm({ formId: file.id });
      form = await ensurePublishState(google, form, {
        isPublished: false,
        isAcceptingResponses: false,
      });
      const permissions = await google.getAllPermissions({ fileId: file.id });
      for (const permission of filterPublishedResponderPermissions(permissions)) {
        await google.deletePermission({ fileId: file.id, permissionId: permission.id });
      }

      const current = await google.getFile({ fileId: file.id });
      const parents = current.parents ?? [];
      const needsParentMove = !parents.includes(runtime.archiveFolderId)
        || parents.includes(runtime.folderId);
      const appProperties = {
        ...(current.appProperties ?? {}),
        state: ROLLOVER_FILE_STATES.ARCHIVED,
      };
      const needsState = current.appProperties?.state !== ROLLOVER_FILE_STATES.ARCHIVED;
      if (needsParentMove || needsState) {
        await google.updateFile({
          fileId: file.id,
          appProperties,
          addParentIds: parents.includes(runtime.archiveFolderId)
            ? undefined
            : [runtime.archiveFolderId],
          removeParentIds: parents.includes(runtime.folderId)
            ? [runtime.folderId]
            : undefined,
        });
      }
      const archived = await google.getFile({ fileId: file.id });
      if (
        archived.appProperties?.state !== ROLLOVER_FILE_STATES.ARCHIVED
        || !archived.parents?.includes(runtime.archiveFolderId)
        || archived.parents?.includes(runtime.folderId)
      ) {
        throw new Error('Staging form could not be verified in the configured archive');
      }
      assertExpectedPublishing(form, {
        isPublished: false,
        isAcceptingResponses: false,
      });
    }

    return Object.freeze({
      action: 'cleanup',
      status: 'archived',
      sourceCycle,
      targetCycle,
      archivedCount: files.length,
    });
  }

  return Object.freeze({
    validate,
    bootstrapStagingSource,
    prepare,
    activate,
    verify,
    cleanup,
  });
}
