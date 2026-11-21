import { createHash } from 'node:crypto';

import {
  AUTOMATION_ENVIRONMENTS,
  ROLLOVER_PHASES,
  assertAutomationEnvironment,
  assertPublicResponderUri,
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
  isInheritedPermission,
  normalizePermissionForCreate,
  permissionIdentityKey,
} from './google-api.mjs';

export const ROLLOVER_MANAGED_BY = 'scrollprize-progress-prizes';

export const ROLLOVER_FILE_ROLES = Object.freeze({
  SOURCE: 'source',
  TARGET: 'target',
});

// Only preparation writes managed state. Forms from earlier cycles may still
// carry the retired activating/active/closed markers; nothing reads them.
export const ROLLOVER_FILE_STATES = Object.freeze({
  COPIED: 'copied',
  PREPARED: 'prepared',
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
const GOOGLE_FORM_MIME_TYPE = 'application/vnd.google-apps.form';
const GOOGLE_FOLDER_MIME_TYPE = 'application/vnd.google-apps.folder';
const INITIAL_EXPLICIT_SOURCE_CYCLE = '2026-07';
const ANONYMOUS_PUBLISHED_RESPONDER_PERMISSION = Object.freeze({
  type: 'anyone',
  role: 'reader',
  allowFileDiscovery: false,
  view: 'published',
});
const SOURCE_ORIGINS = Object.freeze({
  MANAGED: 'managed',
  EXPLICIT_SHARED_DRIVE: 'explicit-shared-drive',
  EXPLICIT_MY_DRIVE: 'explicit-my-drive',
});

function assertNonEmptyString(value, label) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new TypeError(`${label} must be a non-empty string`);
  }
  return value;
}

function normalizeEmail(value, label) {
  const normalized = assertNonEmptyString(value, label).trim().toLowerCase();
  if (!/^[^@\s]+@[^@\s]+$/.test(normalized)) {
    throw new TypeError(`${label} must be an email address`);
  }
  return normalized;
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
    serviceAccountEmail: normalizeEmail(runtime.serviceAccountEmail, 'serviceAccountEmail'),
    driveAdminEmail: normalizeEmail(runtime.driveAdminEmail, 'driveAdminEmail'),
    stagingServiceAccountEmail: hasValue(runtime.stagingServiceAccountEmail)
      ? normalizeEmail(runtime.stagingServiceAccountEmail, 'stagingServiceAccountEmail')
      : undefined,
    branch: assertNonEmptyString(runtime.branch, 'branch'),
    targetBranch: runtime.targetBranch ?? 'main',
    defaultTargetBranch: runtime.defaultTargetBranch ?? 'main',
  };

  assertNonEmptyString(normalized.targetBranch, 'targetBranch');
  assertNonEmptyString(normalized.defaultTargetBranch, 'defaultTargetBranch');
  if (
    normalized.driveAdminEmail === normalized.serviceAccountEmail
    || normalized.driveAdminEmail.endsWith('.gserviceaccount.com')
  ) {
    throw new Error('driveAdminEmail must identify a separate human break-glass administrator');
  }

  return Object.freeze(normalized);
}

/**
 * Validate controls before any Google or filesystem call is made.
 *
 * Identifiers are compared but deliberately omitted from every error message so
 * that a failed public workflow cannot disclose private Workspace configuration.
 */
export function assertRolloverRuntimeSafety(runtimeInput) {
  const runtime = normalizeRuntime(runtimeInput);
  if (runtime.environment !== AUTOMATION_ENVIRONMENTS.PRODUCTION) {
    throw new Error('Google rollover operations run only in production');
  }
  if (hasValue(runtime.simulatedNow)) {
    throw new Error('simulated time is forbidden in production');
  }
  if (runtime.targetBranch !== runtime.defaultTargetBranch) {
    throw new Error('Production rollover cannot use an alternate target branch');
  }
  if (
    runtime.stagingServiceAccountEmail !== undefined
    && runtime.serviceAccountEmail === runtime.stagingServiceAccountEmail
  ) {
    throw new Error('Production rollover cannot use the staging service account');
  }
  return runtime;
}

function assertDependencies({ google, page, clock }) {
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


function publishState(form) {
  const settings = form?.publishSettings;
  if (settings === undefined) {
    throw new Error('Form does not expose modern publishSettings; migrate it before rollover');
  }
  if (
    settings === null
    || typeof settings !== 'object'
    || Array.isArray(settings)
  ) {
    throw new Error('Form exposes malformed modern publishSettings');
  }
  // Forms uses protobuf JSON. A modern unpublished form can expose an empty
  // publishSettings message, and false scalar fields can be omitted. Only the
  // absence of publishSettings itself identifies a legacy form.
  const encodedState = settings.publishState;
  if (
    encodedState !== undefined
    && (
      encodedState === null
      || typeof encodedState !== 'object'
      || Array.isArray(encodedState)
    )
  ) {
    throw new Error('Form exposes malformed modern publishSettings');
  }
  for (const field of ['isPublished', 'isAcceptingResponses']) {
    if (encodedState?.[field] !== undefined && typeof encodedState[field] !== 'boolean') {
      throw new Error('Form exposes malformed modern publishSettings');
    }
  }
  const state = {
    isPublished: encodedState?.isPublished ?? false,
    isAcceptingResponses: encodedState?.isAcceptingResponses ?? false,
  };
  if (!state.isPublished && state.isAcceptingResponses) {
    throw new Error('Form reports an invalid publishing state');
  }
  return state;
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

function effectiveCollaboratorPermissions(permissions) {
  // Only writer/commenter entries are reproducible form collaborators. A
  // separate effective-access comparison also detects inherited domain grants
  // and Shared Drive organizer/fileOrganizer roles before copying.
  return permissions.filter((permission) => (
    permission
    && permission.deleted !== true
    && permission.pendingOwner !== true
    && ['writer', 'commenter'].includes(permission.role)
    && ['user', 'group'].includes(permission.type)
    && typeof permission.emailAddress === 'string'
    && permission.emailAddress !== ''
  ));
}

function effectiveEditablePermissions(permissions) {
  return permissions.filter((permission) => (
    permission
    && permission.deleted !== true
    && permission.pendingOwner !== true
    && ['writer', 'commenter', 'organizer', 'fileOrganizer', 'owner'].includes(permission.role)
    && (
      (
        ['user', 'group'].includes(permission.type)
        && typeof permission.emailAddress === 'string'
        && permission.emailAddress !== ''
      )
      || (
        permission.type === 'domain'
        && typeof permission.domain === 'string'
        && permission.domain !== ''
      )
    )
  ));
}

function effectiveComparablePermissions(permissions) {
  return [
    ...effectiveEditablePermissions(permissions),
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

function isAutomationIdentity(permission, serviceAccountEmail) {
  return (
    typeof permission?.emailAddress === 'string'
    && permission.emailAddress.toLowerCase() === serviceAccountEmail.toLowerCase()
  );
}

function isGoogleServiceAccountIdentity(permission) {
  if (typeof permission?.emailAddress !== 'string') return false;
  return permission.emailAddress.toLowerCase().endsWith('.gserviceaccount.com');
}

function isDriveAdminIdentity(permission, driveAdminEmail) {
  return (
    typeof permission?.emailAddress === 'string'
    && permission.emailAddress.toLowerCase() === driveAdminEmail.toLowerCase()
  );
}

function assertNoUnexpectedServiceAccountPermissions(permissions, serviceAccountEmail) {
  assertNonEmptyString(serviceAccountEmail, 'serviceAccountEmail');
  const hasUnexpectedServiceAccount = permissions.some((permission) => (
    permission?.deleted !== true
    && isGoogleServiceAccountIdentity(permission)
    && !isAutomationIdentity(permission, serviceAccountEmail)
  ));
  if (hasUnexpectedServiceAccount) {
    throw new Error('Google resource has an unexpected service-account permission');
  }
}

function assertExactInheritedOrganizer(permissions, identityEmail, driveId, { label } = {}) {
  const matches = permissions.filter((permission) => (
    permission?.deleted !== true
    && permission?.pendingOwner !== true
    && typeof permission?.emailAddress === 'string'
    && permission.emailAddress.toLowerCase() === identityEmail.toLowerCase()
  ));
  const permission = matches[0];
  const details = permission?.permissionDetails;
  const detail = details?.[0];
  // Drive uses one Permission resource per grantee and records every direct and
  // inherited role source in permissionDetails. Checking only the top-level
  // effective role would therefore hide a merged direct file/folder grant.
  const valid = matches.length === 1
    && permission.type === 'user'
    && permission.role === 'organizer'
    && !hasValue(permission.expirationTime)
    && Array.isArray(details)
    && details.length === 1
    && detail.permissionType === 'member'
    && detail.role === 'organizer'
    && detail.inherited === true
    && detail.inheritedFrom === driveId;
  if (!valid) {
    throw new Error(`Shared Drive ${label} access is not exactly one inherited Manager permission`);
  }
}

function assertSharedDriveManagerPermissions(permissions, runtime) {
  assertNoUnexpectedServiceAccountPermissions(permissions, runtime.serviceAccountEmail);
  assertExactInheritedOrganizer(permissions, runtime.serviceAccountEmail, runtime.driveId, {
    label: 'automation',
  });
  assertExactInheritedOrganizer(permissions, runtime.driveAdminEmail, runtime.driveId, {
    label: 'break-glass administrator',
  });
}

function assertNoInheritedOrAdministrativeFormCollaborators(permissions, runtime) {
  const hasForbiddenAccess = effectiveEditablePermissions(permissions).some((permission) => (
    !isAutomationIdentity(permission, runtime.serviceAccountEmail)
    && !isDriveAdminIdentity(permission, runtime.driveAdminEmail)
    && (
      isInheritedPermission(permission)
      || ['organizer', 'fileOrganizer', 'owner'].includes(permission.role)
      || permission.type === 'domain'
    )
  ));
  if (hasForbiddenAccess) {
    throw new Error('Managed form has unexpected inherited or administrative edit access');
  }
}

function assertCollaboratorConfiguration(
  collaboratorPermissions,
  serviceAccountEmail,
  driveAdminEmail,
) {
  assertCollaboratorPermissions(collaboratorPermissions);
  if (collaboratorPermissions.some(
    (permission) => isGoogleServiceAccountIdentity(permission)
      || isAutomationIdentity(permission, serviceAccountEmail),
  )) {
    throw new Error('An automation service account cannot be configured as a form collaborator');
  }
  if (
    driveAdminEmail !== undefined
    && collaboratorPermissions.some(
      (permission) => isDriveAdminIdentity(permission, driveAdminEmail),
    )
  ) {
    throw new Error('The break-glass administrator cannot be configured as a form collaborator');
  }
}

function publicResponderPermissions(permissions) {
  return filterPublishedResponderPermissions(permissions).filter(
    (permission) => permission.type === 'anyone',
  );
}

function desiredPermissions(
  sourcePermissions,
  collaboratorPermissions,
  copySourceCollaborators,
  {
    includeInheritedCollaborators = false,
    serviceAccountEmail,
    driveAdminEmail,
  } = {},
) {
  assertNonEmptyString(serviceAccountEmail, 'serviceAccountEmail');
  assertCollaboratorConfiguration(
    collaboratorPermissions,
    serviceAccountEmail,
    driveAdminEmail,
  );
  const sourceCollaborators = includeInheritedCollaborators
    ? effectiveCollaboratorPermissions(sourcePermissions)
    : filterDirectCollaboratorPermissions(sourcePermissions);
  if (
    copySourceCollaborators
    && sourceCollaborators.some(
      (permission) => isGoogleServiceAccountIdentity(permission)
        && !isAutomationIdentity(permission, serviceAccountEmail),
    )
  ) {
    throw new Error('Source form has an unexpected service-account collaborator');
  }
  if (
    copySourceCollaborators
    && effectiveEditablePermissions(sourcePermissions).some(
      (permission) => permission.type === 'domain',
    )
  ) {
    throw new Error('Source form has an unsupported domain collaborator');
  }
  return uniquePermissions([
    ...(copySourceCollaborators
      ? sourceCollaborators.filter(
        (permission) => !isAutomationIdentity(permission, serviceAccountEmail),
      )
      : []),
    ...collaboratorPermissions,
    ...publicResponderPermissions(sourcePermissions),
  ]);
}

function assertPermissionsMatch(expected, actual, { runtime } = {}) {
  assertSharedDriveManagerPermissions(actual, runtime);
  assertNoInheritedOrAdministrativeFormCollaborators(actual, runtime);
  const expectedKeys = new Set(expected.map(permissionIdentityKey));
  const effectivePermissions = effectiveComparablePermissions(actual).filter(
    (permission) => (
      !isAutomationIdentity(permission, runtime.serviceAccountEmail)
      && !isDriveAdminIdentity(permission, runtime.driveAdminEmail)
    ),
  );
  const effectiveKeys = new Set(effectivePermissions.map(permissionIdentityKey));
  const hasUnexpectedPermission = effectivePermissions
    .some((permission) => !expectedKeys.has(permissionIdentityKey(permission)));
  if (hasUnexpectedPermission || [...expectedKeys].some((key) => !effectiveKeys.has(key))) {
    throw new Error('Target form collaborator or published-responder permissions do not match');
  }
}

function assertRequiredSourcePermissions(permissions, requiredCollaborators = []) {
  assertCollaboratorPermissions(requiredCollaborators);
  const publishedPermissions = filterPublishedResponderPermissions(permissions);
  if (publicResponderPermissions(permissions).length === 0) {
    throw new Error('Source form is missing its anonymous published responder permission');
  }
  if (publishedPermissions.some((permission) => permission.type !== 'anyone')) {
    throw new Error('Source form has an unexpected non-public published responder permission');
  }
  for (const required of requiredCollaborators) {
    const expected = normalizePermissionForCreate(required);
    const found = permissions.some((permission) => (
      permission?.deleted !== true
      && permission?.pendingOwner !== true
      && !isInheritedPermission(permission)
      && permission?.type === expected.type
      && permission?.role === expected.role
      && permission?.emailAddress?.toLowerCase() === expected.emailAddress?.toLowerCase()
    ));
    if (!found) {
      throw new Error('Source form is missing a configured internal collaborator');
    }
  }
}

function assertDirectAutomationPermission(permissions, runtime, expectedRole) {
  const matches = permissions.filter((permission) => (
    permission?.deleted !== true
    && permission?.pendingOwner !== true
    && permission?.type === 'user'
    && permission?.role === expectedRole
    && !isInheritedPermission(permission)
    && !hasValue(permission.expirationTime)
    && isAutomationIdentity(permission, runtime.serviceAccountEmail)
  ));
  if (matches.length !== 1) {
    throw new Error('The bootstrap form lacks the required direct automation access');
  }
}

function assertSingleBootstrapReaderPermission(permissions, runtime) {
  if (runtime.stagingServiceAccountEmail === undefined) {
    throw new Error('Production validation requires the configured staging automation identity');
  }
  const nonPublishedReaders = permissions.filter((permission) => (
    permission?.deleted !== true
    && permission?.pendingOwner !== true
    && permission?.role === 'reader'
    && permission?.view !== 'published'
  ));
  const bootstrapReaders = nonPublishedReaders.filter((permission) => (
    permission.type === 'user'
    && !isInheritedPermission(permission)
    && !hasValue(permission.expirationTime)
    && isGoogleServiceAccountIdentity(permission)
    && isAutomationIdentity(permission, runtime.stagingServiceAccountEmail)
  ));
  if (nonPublishedReaders.length !== 1 || bootstrapReaders.length !== 1) {
    throw new Error('The production form must have exactly one direct copy-only automation reader');
  }
}

async function ensurePermissions(google, fileId, expected, options) {
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
  assertPermissionsMatch(expected, current, options);
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

// The source may already have been closed by hand at or before the deadline.
// Preparation only needs it to stay published; a human closes it at cutoff.
function assertSourcePublished(form) {
  if (publishState(form).isPublished !== true) {
    throw new Error('Form publishing state does not match the rollover state');
  }
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

function assertUsableFormFile(file) {
  if (file?.mimeType !== GOOGLE_FORM_MIME_TYPE) {
    throw new Error('The configured source is not a Google Form');
  }
  if (file.trashed === true) {
    throw new Error('The configured source form is trashed');
  }
}

function assertSharedDriveLocation(file, runtime) {
  if (file?.driveId !== runtime.driveId) {
    throw new Error('Form is not in the configured Shared Drive');
  }
  if (!file?.parents?.includes(runtime.folderId)) {
    throw new Error('Form is not in the configured active folder');
  }
}

function assertManagedFile(file, runtime, role, cycle, { state } = {}) {
  assertUsableFormFile(file);
  assertSharedDriveLocation(file, runtime);
  const expected = {
    ...managedQuery(runtime, role, cycle),
    ...(state === undefined ? {} : { state }),
  };
  if (Object.entries(expected).some(
    ([key, value]) => file?.appProperties?.[key] !== value,
  )) {
    throw new Error('Managed form metadata does not match the requested rollover cycle');
  }
}

function assertDestinationFolder(folder, runtime) {
  if (
    folder?.id !== runtime.folderId
    || folder?.mimeType !== GOOGLE_FOLDER_MIME_TYPE
    || folder?.trashed === true
    || folder?.driveId !== runtime.driveId
    || folder?.capabilities?.canAddChildren !== true
    || folder?.capabilities?.canShare !== true
  ) {
    throw new Error('The configured destination is not a writable Shared Drive folder');
  }
}

function sourceFingerprint(runtime, sourceCycle, sourceFileId) {
  return createHash('sha256')
    .update(`${runtime.environment}\0${sourceCycle}\0${sourceFileId}`)
    .digest('hex');
}

function assertSourceFingerprint(targetFile, sourceFile, runtime, sourceCycle) {
  const expected = sourceFingerprint(runtime, sourceCycle, sourceFile.id);
  if (targetFile?.appProperties?.sourceFingerprint !== expected) {
    throw new Error('Prepared target is not bound to the resolved source form');
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
    hasPublicResponderPermission: publicResponderPermissions(permissions).length > 0,
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
} = {}) {
  assertDependencies({ google, page, clock });
  const runtime = assertRolloverRuntimeSafety(runtimeInput);

  async function findManagedFile(role, cycle) {
    const files = await google.listFilesByAppProperties({
      appProperties: managedQuery(runtime, role, cycle),
      driveId: runtime.driveId,
    });
    if (files.length > 1) {
      throw new Error(`Multiple managed ${role} forms exist for cycle ${cycle}; refusing to guess`);
    }
    if (files[0] !== undefined) {
      assertManagedFile(files[0], runtime, role, cycle);
    }
    return files[0];
  }

  async function requireManagedFile(role, cycle) {
    const file = await findManagedFile(role, cycle);
    if (file === undefined) {
      throw new Error(`Managed ${role} form is missing for cycle ${cycle}`);
    }
    return file;
  }

  function assertResolvedSourceLocation(resolution, file) {
    if (file?.id !== resolution.file.id) {
      throw new Error('Resolved source identity changed during preflight');
    }
    if (resolution.origin === SOURCE_ORIGINS.MANAGED) {
      assertManagedFile(file, runtime, resolution.role, resolution.sourceCycle);
      return;
    }

    assertUsableFormFile(file);
    if (file.appProperties?.managedBy === ROLLOVER_MANAGED_BY) {
      throw new Error('An explicit fallback cannot claim managed rollover metadata');
    }
    if (resolution.origin === SOURCE_ORIGINS.EXPLICIT_MY_DRIVE) {
      if (file.driveId !== undefined && file.driveId !== null) {
        throw new Error('The My Drive fallback source changed location during preflight');
      }
      return;
    }
    assertSharedDriveLocation(file, runtime);
  }

  function sourceResolution(file, origin, { role, sourceCycle }) {
    const resolution = Object.freeze({
      file,
      origin,
      role,
      sourceCycle,
    });
    assertResolvedSourceLocation(resolution, file);
    return resolution;
  }

  async function resolveSource({ sourceFormId, sourceCycle }) {
    const [managedTarget, managedSource] = await Promise.all([
      findManagedFile(ROLLOVER_FILE_ROLES.TARGET, sourceCycle),
      findManagedFile(ROLLOVER_FILE_ROLES.SOURCE, sourceCycle),
    ]);
    if (managedTarget !== undefined && managedSource !== undefined) {
      throw new Error(`Multiple managed source candidates exist for cycle ${sourceCycle}`);
    }
    if (managedTarget !== undefined || managedSource !== undefined) {
      return sourceResolution(
        managedTarget ?? managedSource,
        SOURCE_ORIGINS.MANAGED,
        {
          role: managedTarget === undefined
            ? ROLLOVER_FILE_ROLES.SOURCE
            : ROLLOVER_FILE_ROLES.TARGET,
          sourceCycle,
        },
      );
    }
    if (sourceFormId !== undefined) {
      if (sourceCycle !== INITIAL_EXPLICIT_SOURCE_CYCLE) {
        throw new Error('The explicit fallback is restricted to the initial bootstrap cycle');
      }
      assertNonEmptyString(sourceFormId, 'sourceFormId');
      const file = await google.getFile({ fileId: sourceFormId });
      assertUsableFormFile(file);
      if (file.appProperties?.managedBy === ROLLOVER_MANAGED_BY) {
        throw new Error('An explicit fallback cannot claim managed rollover metadata');
      }
      if (file.driveId === undefined || file.driveId === null) {
        return sourceResolution(file, SOURCE_ORIGINS.EXPLICIT_MY_DRIVE, { sourceCycle });
      }
      assertSharedDriveLocation(file, runtime);
      return sourceResolution(file, SOURCE_ORIGINS.EXPLICIT_SHARED_DRIVE, { sourceCycle });
    }
    throw new Error(`Managed source form is missing for cycle ${sourceCycle}`);
  }

  async function requireDestinationFolder() {
    const [folder, permissions] = await Promise.all([
      google.getFile({ fileId: runtime.folderId }),
      google.getAllPermissions({ fileId: runtime.folderId }),
    ]);
    assertDestinationFolder(folder, runtime);
    assertSharedDriveManagerPermissions(permissions, runtime);
    const editablePermissions = effectiveEditablePermissions(permissions).filter(
      (permission) => (
        !isAutomationIdentity(permission, runtime.serviceAccountEmail)
        && !isDriveAdminIdentity(permission, runtime.driveAdminEmail)
      ),
    );
    if (editablePermissions.length !== 0) {
      throw new Error('The destination folder has unexpected inherited edit access');
    }
    return folder;
  }

  function assertResolvedSourceAccess(resolution, snapshot, {
    requireCopy = false,
    requireShare = false,
  } = {}) {
    assertResolvedSourceLocation(resolution, snapshot.file);
    // The explicit owner-My-Drive July source is the only exception: it carries
    // the production writer and the intentional direct staging reader needed
    // for the one-time bootstrap. Managed and Shared Drive sources may expose
    // only the configured inherited automation and break-glass Managers plus
    // the expected form collaborators for their own environment.
    if (resolution.origin !== SOURCE_ORIGINS.EXPLICIT_MY_DRIVE) {
      assertSharedDriveManagerPermissions(snapshot.permissions, runtime);
      assertNoInheritedOrAdministrativeFormCollaborators(snapshot.permissions, runtime);
    }
    const requiredCapabilities = [
      'canEdit',
      ...(requireCopy ? ['canCopy'] : []),
      ...(requireShare && resolution.origin !== SOURCE_ORIGINS.EXPLICIT_MY_DRIVE
        ? ['canShare']
        : []),
    ];
    assertSourceCapabilities(snapshot.file, requiredCapabilities);
    if (resolution.origin === SOURCE_ORIGINS.EXPLICIT_MY_DRIVE) {
      assertDirectAutomationPermission(snapshot.permissions, runtime, 'writer');
    }
  }

  function expectedTargetPermissions(
    sourcePermissions,
    collaboratorPermissions,
    copySourceCollaborators,
    sourceOrigin,
  ) {
    return desiredPermissions(
      sourcePermissions,
      collaboratorPermissions,
      copySourceCollaborators,
      {
        includeInheritedCollaborators: sourceOrigin === SOURCE_ORIGINS.EXPLICIT_MY_DRIVE,
        serviceAccountEmail: runtime.serviceAccountEmail,
        driveAdminEmail: runtime.driveAdminEmail,
      },
    );
  }

  async function loadReadableSnapshot(file) {
    const [form, currentFile] = await Promise.all([
      google.getForm({ formId: file.id }),
      google.getFile({ fileId: file.id }),
    ]);
    publishState(form);
    assertPublicResponderUri(form.responderUri);
    return { form, file: currentFile };
  }

  async function loadSnapshot(file) {
    const [readable, permissions] = await Promise.all([
      loadReadableSnapshot(file),
      google.getAllPermissions({ fileId: file.id }),
    ]);
    const { form, file: currentFile } = readable;
    return { form, file: currentFile, permissions };
  }

  async function markFile(file, state, extra = {}, { expectedAppProperties } = {}) {
    // The workflow's static concurrency group is the mutation lock. Refetching
    // here additionally rejects a state changed since the last full snapshot
    // before issuing the metadata patch.
    const current = await google.getFile({ fileId: file.id });
    if (expectedAppProperties !== undefined) {
      if (
        expectedAppProperties === null
        || typeof expectedAppProperties !== 'object'
        || Array.isArray(expectedAppProperties)
      ) {
        throw new TypeError('expectedAppProperties must be an object');
      }
      if (Object.entries(expectedAppProperties).some(
        ([key, value]) => current.appProperties?.[key] !== value,
      )) {
        throw new Error('Managed file changed before its rollover state transition');
      }
    }
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
    assertCollaboratorConfiguration(
      collaboratorPermissions,
      runtime.serviceAccountEmail,
      runtime.driveAdminEmail,
    );
    const source = await resolveSource({ sourceFormId, sourceCycle });
    const snapshot = await loadSnapshot(source.file);
    assertResolvedSourceAccess(source, snapshot, {
      requireCopy: true,
      requireShare: true,
    });
    assertRequiredSourcePermissions(snapshot.permissions, collaboratorPermissions);
    if (source.origin === SOURCE_ORIGINS.EXPLICIT_MY_DRIVE) {
      assertSingleBootstrapReaderPermission(snapshot.permissions, runtime);
    }
    expectedTargetPermissions(
      snapshot.permissions,
      collaboratorPermissions,
      true,
      source.origin,
    );
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
  } = {}) {
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
    assertCollaboratorConfiguration(
      collaboratorPermissions,
      runtime.serviceAccountEmail,
      runtime.driveAdminEmail,
    );

    const source = await resolveSource({
      sourceFormId,
      sourceCycle: rollover.sourceCycle,
    });
    const sourceSnapshot = await loadSnapshot(source.file);
    assertResolvedSourceAccess(source, sourceSnapshot, {
      requireCopy: true,
      requireShare: true,
    });
    assertRequiredSourcePermissions(sourceSnapshot.permissions, collaboratorPermissions);
    assertSourcePublished(sourceSnapshot.form);
    const sourceTitle = cycleTitle(rollover.sourceCycle);
    assertTitle(sourceSnapshot.form, sourceSnapshot.file, sourceTitle);
    const expectedPermissions = expectedTargetPermissions(
      sourceSnapshot.permissions,
      collaboratorPermissions,
      copySourceCollaborators,
      source.origin,
    );

    const currentPage = parseProgressPrizeMarkdown(await page.read());
    if (currentPage.cycle !== rollover.sourceCycle && currentPage.cycle !== targetCycle) {
      throw new Error('Managed page is not at the source or target rollover cycle');
    }
    if (currentPage.cycle === rollover.sourceCycle) {
      await assertResponderUrisMatch(
        page,
        currentPage.responderUri,
        expectedCurrentResponderUri ?? sourceSnapshot.form.responderUri,
      );
    }

    let targetFile = await findManagedFile(ROLLOVER_FILE_ROLES.TARGET, targetCycle);
    if (targetFile === undefined) {
      await requireDestinationFolder();
    } else {
      assertSourceFingerprint(
        targetFile,
        sourceSnapshot.file,
        runtime,
        rollover.sourceCycle,
      );
    }
    if (dryRun && targetFile === undefined) {
      return Object.freeze({
        action: 'prepare',
        status: 'planned',
        sourceCycle: rollover.sourceCycle,
        targetCycle,
        title: cycleTitle(targetCycle),
        created: false,
        resumed: false,
      });
    }

    let created = false;
    if (targetFile === undefined) {
      targetFile = await google.copyFile({
        fileId: sourceSnapshot.file.id,
        name: cycleTitle(targetCycle),
        parentId: runtime.folderId,
        appProperties: {
          ...managedProperties(
            runtime,
            ROLLOVER_FILE_ROLES.TARGET,
            targetCycle,
            ROLLOVER_FILE_STATES.COPIED,
          ),
          sourceFingerprint: sourceFingerprint(
            runtime,
            rollover.sourceCycle,
            sourceSnapshot.file.id,
          ),
        },
      });
      assertNonEmptyString(targetFile?.id, 'copied form ID');
      const copiedForm = await google.getForm({ formId: targetFile.id });
      const closedCopy = await ensurePublishState(google, copiedForm, {
        isPublished: true,
        isAcceptingResponses: false,
      });
      assertExpectedPublishing(closedCopy, {
        isPublished: true,
        isAcceptingResponses: false,
      });
      assertManagedFile(
        targetFile,
        runtime,
        ROLLOVER_FILE_ROLES.TARGET,
        targetCycle,
        { state: ROLLOVER_FILE_STATES.COPIED },
      );
      assertSourceFingerprint(
        targetFile,
        sourceSnapshot.file,
        runtime,
        rollover.sourceCycle,
      );
      created = true;
    }

    if (dryRun) {
      const existing = await loadSnapshot(targetFile);
      assertManagedFile(
        existing.file,
        runtime,
        ROLLOVER_FILE_ROLES.TARGET,
        targetCycle,
      );
      assertSourceCapabilities(existing.file);
      assertSourceFingerprint(
        existing.file,
        sourceSnapshot.file,
        runtime,
        rollover.sourceCycle,
      );
      assertPermissionsMatch(
        expectedPermissions,
        existing.permissions,
        { runtime },
      );
      return Object.freeze({
        action: 'prepare',
        status: 'planned',
        sourceCycle: rollover.sourceCycle,
        targetCycle,
        title: cycleTitle(targetCycle),
        created: false,
        resumed: true,
        responderUri: assertPublicResponderUri(existing.form.responderUri),
      });
    }

    let targetSnapshot = await loadSnapshot(targetFile);
    assertManagedFile(
      targetSnapshot.file,
      runtime,
      ROLLOVER_FILE_ROLES.TARGET,
      targetCycle,
    );
    assertSourceFingerprint(
      targetSnapshot.file,
      sourceSnapshot.file,
      runtime,
      rollover.sourceCycle,
    );
    assertSourceCapabilities(targetSnapshot.file);
    // A copied form may inherit an accepting state. Close it before title and
    // ACL work so an unprepared target is never left accepting responses.
    targetSnapshot.form = await ensurePublishState(google, targetSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: false,
    });
    const title = cycleTitle(targetCycle);
    const titled = await updateTitles(targetSnapshot.file, targetSnapshot.form, title);
    targetSnapshot = { ...targetSnapshot, ...titled };
    targetSnapshot.permissions = await ensurePermissions(
      google,
      targetSnapshot.file.id,
      expectedPermissions,
      { runtime },
    );
    assertSameStructure(sourceSnapshot.form, targetSnapshot.form);
    assertTitle(targetSnapshot.form, targetSnapshot.file, title);
    assertExpectedPublishing(targetSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: false,
    });
    targetSnapshot.file = await markFile(targetSnapshot.file, ROLLOVER_FILE_STATES.PREPARED, {
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

  async function verify({
    targetCycle,
    sourceFormId,
    collaboratorPermissions = [],
    copySourceCollaborators = true,
  } = {}) {
    parseCycle(targetCycle);
    assertCollaboratorConfiguration(
      collaboratorPermissions,
      runtime.serviceAccountEmail,
      runtime.driveAdminEmail,
    );
    const sourceCycle = previousCycle(targetCycle);
    const source = await resolveSource({ sourceFormId, sourceCycle });
    const targetFile = await requireManagedFile(ROLLOVER_FILE_ROLES.TARGET, targetCycle);
    const [sourceSnapshot, targetSnapshot] = await Promise.all([
      loadSnapshot(source.file),
      loadSnapshot(targetFile),
    ]);
    assertManagedFile(targetSnapshot.file, runtime, ROLLOVER_FILE_ROLES.TARGET, targetCycle);
    assertSourceFingerprint(targetSnapshot.file, sourceSnapshot.file, runtime, sourceCycle);
    assertResolvedSourceAccess(source, sourceSnapshot);
    assertSourceCapabilities(targetSnapshot.file);
    assertSameStructure(sourceSnapshot.form, targetSnapshot.form);
    assertTitle(sourceSnapshot.form, sourceSnapshot.file, cycleTitle(sourceCycle));
    assertTitle(targetSnapshot.form, targetSnapshot.file, cycleTitle(targetCycle));
    assertRequiredSourcePermissions(sourceSnapshot.permissions, collaboratorPermissions);
    assertPermissionsMatch(
      expectedTargetPermissions(
        sourceSnapshot.permissions,
        collaboratorPermissions,
        copySourceCollaborators,
        source.origin,
      ),
      targetSnapshot.permissions,
      { runtime },
    );
    assertSourcePublished(sourceSnapshot.form);
    assertExpectedPublishing(targetSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: false,
    });
    await assertPageState(targetCycle, targetSnapshot.form.responderUri);
    return Object.freeze({
      action: 'verify',
      status: 'valid',
      sourceCycle,
      targetCycle,
      responderUri: assertPublicResponderUri(targetSnapshot.form.responderUri),
    });
  }

  return Object.freeze({
    validate,
    prepare,
    verify,
  });
}
