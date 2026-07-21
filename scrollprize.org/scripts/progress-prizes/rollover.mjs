import { createHash } from 'node:crypto';

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
  isInheritedPermission,
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
const GOOGLE_FORM_MIME_TYPE = 'application/vnd.google-apps.form';
const GOOGLE_FOLDER_MIME_TYPE = 'application/vnd.google-apps.folder';
const INITIAL_EXPLICIT_SOURCE_CYCLE = '2026-07';
const SOURCE_ORIGINS = Object.freeze({
  MANAGED: 'managed',
  EXPLICIT_SHARED_DRIVE: 'explicit-shared-drive',
  EXPLICIT_MY_DRIVE: 'explicit-my-drive',
});

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
  if (
    normalized.driveAdminEmail === normalized.serviceAccountEmail
    || normalized.driveAdminEmail.endsWith('.gserviceaccount.com')
  ) {
    throw new Error('driveAdminEmail must identify a separate human break-glass administrator');
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

function effectiveCollaboratorPermissions(permissions) {
  // Only writer/commenter entries are reproducible form collaborators. A
  // separate effective-access comparison also detects inherited domain grants
  // and Shared Drive organizer/fileOrganizer roles before activation.
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
    sourceOpen
    && targetClosed
    && targetState === ROLLOVER_FILE_STATES.ACTIVATING
    && ![ROLLOVER_FILE_STATES.CLOSED, ROLLOVER_FILE_STATES.ARCHIVED].includes(sourceState)
  ) return 'recover-intent';
  if (
    sourceClosed
    && targetClosed
    && sourceState !== ROLLOVER_FILE_STATES.ARCHIVED
    && targetState === ROLLOVER_FILE_STATES.ACTIVATING
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

function assertReadCopyOnlyCapabilities(file) {
  assertSourceCapabilities(file, ['canCopy']);
  if (file?.capabilities?.canEdit === true || file?.capabilities?.canShare === true) {
    throw new Error('The staging identity has forbidden edit or share access to production');
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

function assertSharedDriveLocation(file, runtime, { requireActiveFolder = true } = {}) {
  if (file?.driveId !== runtime.driveId) {
    throw new Error('Form is not in the configured Shared Drive');
  }
  if (requireActiveFolder && !file?.parents?.includes(runtime.folderId)) {
    throw new Error('Form is not in the configured active folder');
  }
}

function assertManagedFile(file, runtime, role, cycle, {
  state,
  requireActiveFolder = true,
} = {}) {
  assertUsableFormFile(file);
  assertSharedDriveLocation(file, runtime, { requireActiveFolder });
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
      assertManagedFile(files[0], runtime, role, cycle, {
        requireActiveFolder: inActiveFolder,
      });
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

  function assertResolvedSourceLocation(resolution, file) {
    if (file?.id !== resolution.file.id) {
      throw new Error('Resolved source identity changed during preflight');
    }
    if (resolution.origin === SOURCE_ORIGINS.MANAGED) {
      assertManagedFile(file, runtime, resolution.role, resolution.sourceCycle, {
        requireActiveFolder: resolution.inActiveFolder,
      });
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
    assertSharedDriveLocation(file, runtime, {
      requireActiveFolder: resolution.inActiveFolder,
    });
  }

  function sourceResolution(file, origin, {
    role,
    sourceCycle,
    inActiveFolder,
  }) {
    const resolution = Object.freeze({
      file,
      origin,
      role,
      sourceCycle,
      inActiveFolder,
    });
    assertResolvedSourceLocation(resolution, file);
    return resolution;
  }

  async function resolveSource({ sourceFormId, sourceCycle, inActiveFolder = true }) {
    const [managedTarget, managedSource] = await Promise.all([
      findManagedFile(ROLLOVER_FILE_ROLES.TARGET, sourceCycle, { inActiveFolder }),
      findManagedFile(ROLLOVER_FILE_ROLES.SOURCE, sourceCycle, { inActiveFolder }),
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
          inActiveFolder,
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
        if (runtime.environment !== AUTOMATION_ENVIRONMENTS.PRODUCTION) {
          throw new Error('A My Drive fallback source is restricted to production');
        }
        return sourceResolution(file, SOURCE_ORIGINS.EXPLICIT_MY_DRIVE, {
          sourceCycle,
          inActiveFolder,
        });
      }
      assertSharedDriveLocation(file, runtime, { requireActiveFolder: inActiveFolder });
      return sourceResolution(file, SOURCE_ORIGINS.EXPLICIT_SHARED_DRIVE, {
        sourceCycle,
        inActiveFolder,
      });
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
    if (sourceCycle !== INITIAL_EXPLICIT_SOURCE_CYCLE) {
      throw new Error('Staging bootstrap is restricted to the initial explicit source cycle');
    }
    assertNonEmptyString(sourceFormId, 'sourceFormId');
    assertCollaboratorConfiguration(
      collaboratorPermissions,
      runtime.serviceAccountEmail,
      runtime.driveAdminEmail,
    );

    const liveFile = await google.getFile({ fileId: sourceFormId });
    assertUsableFormFile(liveFile);
    if (
      (liveFile.driveId !== undefined && liveFile.driveId !== null)
      || liveFile.appProperties?.managedBy === ROLLOVER_MANAGED_BY
    ) {
      throw new Error('Staging bootstrap requires the initial explicit My Drive source');
    }
    const liveSnapshot = await loadSnapshot(liveFile);
    // The staging identity deliberately has read/copy-only access to production.
    // It must never need edit or share capability on the active live form.
    assertReadCopyOnlyCapabilities(liveSnapshot.file);
    assertDirectAutomationPermission(liveSnapshot.permissions, runtime, 'reader');
    assertRequiredSourcePermissions(liveSnapshot.permissions);
    assertTitle(liveSnapshot.form, liveSnapshot.file, cycleTitle(sourceCycle));
    assertExpectedPublishing(liveSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: true,
    });
    await assertPageState(sourceCycle, liveSnapshot.form.responderUri);

    let stagedFile = await findManagedFile(ROLLOVER_FILE_ROLES.SOURCE, sourceCycle);
    if (stagedFile === undefined) {
      await requireDestinationFolder();
    }
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
      assertNonEmptyString(stagedFile?.id, 'copied form ID');
      // The source copy inherits the live form's accepting state. Close it
      // before trusting or reconciling any copied Drive metadata or ACLs.
      const copiedForm = await google.getForm({ formId: stagedFile.id });
      const closedCopy = await ensurePublishState(google, copiedForm, {
        isPublished: true,
        isAcceptingResponses: false,
      });
      assertExpectedPublishing(closedCopy, {
        isPublished: true,
        isAcceptingResponses: false,
      });
      assertManagedFile(
        stagedFile,
        runtime,
        ROLLOVER_FILE_ROLES.SOURCE,
        sourceCycle,
        { state: ROLLOVER_FILE_STATES.COPIED },
      );
      created = true;
      if (faultInjection === ROLLOVER_FAULTS.AFTER_COPY) {
        throw new RolloverFaultError(ROLLOVER_FAULTS.AFTER_COPY);
      }
    }

    if (dryRun) {
      if (stagedFile !== undefined) {
        const permissions = await google.getAllPermissions({ fileId: stagedFile.id });
        assertSharedDriveManagerPermissions(permissions, runtime);
        assertNoInheritedOrAdministrativeFormCollaborators(permissions, runtime);
      }
      return Object.freeze({
        action: 'bootstrap',
        status: 'planned',
        cycle: sourceCycle,
        title: managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.SOURCE, sourceCycle),
        created: false,
        resumed: true,
      });
    }

    if (stagedFile.appProperties?.state === ROLLOVER_FILE_STATES.COPIED) {
      const recoveryForm = await google.getForm({ formId: stagedFile.id });
      await ensurePublishState(google, recoveryForm, {
        isPublished: true,
        isAcceptingResponses: false,
      });
    }
    let stagedSnapshot = await loadSnapshot(stagedFile);
    assertManagedFile(
      stagedSnapshot.file,
      runtime,
      ROLLOVER_FILE_ROLES.SOURCE,
      sourceCycle,
    );
    assertSourceCapabilities(stagedSnapshot.file);
    const title = managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.SOURCE, sourceCycle);
    const titled = await updateTitles(stagedSnapshot.file, stagedSnapshot.form, title);
    stagedSnapshot = {
      ...stagedSnapshot,
      ...titled,
    };
    const expectedPermissions = expectedTargetPermissions(
      liveSnapshot.permissions,
      collaboratorPermissions,
      false,
      SOURCE_ORIGINS.EXPLICIT_MY_DRIVE,
    );
    stagedSnapshot.permissions = await ensurePermissions(
      google,
      stagedSnapshot.file.id,
      expectedPermissions,
      { runtime },
    );
    assertSameStructure(liveSnapshot.form, stagedSnapshot.form);
    assertTitle(stagedSnapshot.form, stagedSnapshot.file, title);
    const stagedState = stagedSnapshot.file.appProperties?.state;
    if (
      stagedState !== ROLLOVER_FILE_STATES.COPIED
      && stagedState !== ROLLOVER_FILE_STATES.ACTIVE
    ) {
      throw new Error('Managed staging source is not in a bootstrap recovery state');
    }
    assertExpectedPublishing(stagedSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: stagedState === ROLLOVER_FILE_STATES.ACTIVE,
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

    stagedSnapshot.form = await ensurePublishState(google, stagedSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: true,
    });
    assertExpectedPublishing(stagedSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: true,
    });
    await markFile(stagedSnapshot.file, ROLLOVER_FILE_STATES.ACTIVE, {
      sourceCycle,
    });

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
      const expectedCurrent = expectedCurrentResponderUri
        ?? (runtime.environment === AUTOMATION_ENVIRONMENTS.PRODUCTION
          ? sourceSnapshot.form.responderUri
          : undefined);
      if (expectedCurrent !== undefined) {
        await assertResponderUrisMatch(page, currentPage.responderUri, expectedCurrent);
      }
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
      if (faultInjection === ROLLOVER_FAULTS.AFTER_COPY) {
        throw new RolloverFaultError(ROLLOVER_FAULTS.AFTER_COPY);
      }
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
        title: managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.TARGET, targetCycle),
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
    const title = managedTitle(runtime, clock, ROLLOVER_FILE_ROLES.TARGET, targetCycle);
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

  async function loadActivationState({
    targetCycle,
    sourceFormId,
    collaboratorPermissions,
    copySourceCollaborators,
    sourceCycle,
  }) {
    const source = await resolveSource({ sourceFormId, sourceCycle });
    const targetFile = await requireManagedFile(ROLLOVER_FILE_ROLES.TARGET, targetCycle);
    const [sourceSnapshot, targetSnapshot] = await Promise.all([
      loadSnapshot(source.file),
      loadSnapshot(targetFile),
    ]);
    assertResolvedSourceAccess(source, sourceSnapshot);
    assertManagedFile(
      targetSnapshot.file,
      runtime,
      ROLLOVER_FILE_ROLES.TARGET,
      targetCycle,
    );
    assertSourceCapabilities(targetSnapshot.file);
    assertSourceFingerprint(targetSnapshot.file, sourceSnapshot.file, runtime, sourceCycle);
    assertSameStructure(sourceSnapshot.form, targetSnapshot.form);
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
    await assertPageState(targetCycle, targetSnapshot.form.responderUri);
    return Object.freeze({
      source,
      sourceSnapshot,
      targetSnapshot,
      transition: activationTransition(sourceSnapshot, targetSnapshot),
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
    assertCollaboratorConfiguration(
      collaboratorPermissions,
      runtime.serviceAccountEmail,
      runtime.driveAdminEmail,
    );

    const initialState = await loadActivationState({
      targetCycle,
      sourceFormId,
      collaboratorPermissions,
      copySourceCollaborators,
      sourceCycle: rollover.sourceCycle,
    });

    if (activationGate === undefined) {
      throw new Error('Activation requires an injected successful code and Vercel preview gate');
    }
    if (runtime.environment === AUTOMATION_ENVIRONMENTS.PRODUCTION) {
      assertNonEmptyString(headSha, 'headSha');
    }
    const gate = await activationGate({
      targetCycle,
      responderUri: initialState.targetSnapshot.form.responderUri,
      branch: runtime.branch,
      targetBranch: runtime.targetBranch,
      headSha,
    });
    if (gate !== true && gate?.ok !== true) {
      throw new Error('Code checks and Vercel preview gate did not pass for activation');
    }

    const currentState = await loadActivationState({
      targetCycle,
      sourceFormId,
      collaboratorPermissions,
      copySourceCollaborators,
      sourceCycle: rollover.sourceCycle,
    });
    await assertResponderUrisMatch(
      page,
      currentState.targetSnapshot.form.responderUri,
      initialState.targetSnapshot.form.responderUri,
    );

    if (currentState.transition === 'active') {
      return Object.freeze({
        action: 'activate',
        status: 'active',
        sourceCycle: rollover.sourceCycle,
        targetCycle,
        sourceAcceptingResponses: false,
        targetAcceptingResponses: true,
        responderUri: assertPublicResponderUri(currentState.targetSnapshot.form.responderUri),
      });
    }

    if (currentState.transition === 'prepared') {
      await markFile(currentState.targetSnapshot.file, ROLLOVER_FILE_STATES.ACTIVATING, {
        sourceCycle: rollover.sourceCycle,
      });
    }

    let sourceForm = currentState.sourceSnapshot.form;
    if (currentState.transition !== 'recover-opened') {
      sourceForm = await ensurePublishState(google, currentState.sourceSnapshot.form, {
        isPublished: true,
        isAcceptingResponses: false,
      });
      await markFile(currentState.sourceSnapshot.file, ROLLOVER_FILE_STATES.CLOSED, {
        targetCycle,
      });
      if (faultInjection === ROLLOVER_FAULTS.AFTER_CLOSE_SOURCE) {
        throw new RolloverFaultError(ROLLOVER_FAULTS.AFTER_CLOSE_SOURCE);
      }
    }

    const targetForm = await ensurePublishState(google, currentState.targetSnapshot.form, {
      isPublished: true,
      isAcceptingResponses: true,
    });
    await markFile(currentState.targetSnapshot.file, ROLLOVER_FILE_STATES.ACTIVE, {
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

    // Do not treat the mutation responses as proof of a completed rollover.
    // Reload both resources so a concurrent ACL, title, structure, metadata,
    // binding, marker, or publish-state change cannot escape the activation
    // boundary and allow the website PR to merge against an invalid form pair.
    const finalState = await loadActivationState({
      targetCycle,
      sourceFormId,
      collaboratorPermissions,
      copySourceCollaborators,
      sourceCycle: rollover.sourceCycle,
    });
    await assertResponderUrisMatch(
      page,
      finalState.targetSnapshot.form.responderUri,
      initialState.targetSnapshot.form.responderUri,
    );
    if (finalState.transition !== 'active') {
      throw new Error('Activation did not reach the fully verified active transition');
    }

    return Object.freeze({
      action: 'activate',
      status: 'active',
      sourceCycle: rollover.sourceCycle,
      targetCycle,
      sourceAcceptingResponses: false,
      targetAcceptingResponses: true,
      responderUri: assertPublicResponderUri(finalState.targetSnapshot.form.responderUri),
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
    assertCollaboratorConfiguration(
      collaboratorPermissions,
      runtime.serviceAccountEmail,
      runtime.driveAdminEmail,
    );
    const sourceCycle = previousCycle(targetCycle);
    const inActiveFolder = mode !== 'cleaned';
    const source = mode === 'cleaned'
      ? Object.freeze({
        file: await requireManagedFile(
          ROLLOVER_FILE_ROLES.SOURCE,
          sourceCycle,
          { inActiveFolder: false },
        ),
        origin: SOURCE_ORIGINS.MANAGED,
      })
      : await resolveSource({ sourceFormId, sourceCycle, inActiveFolder });
    const targetFile = await requireManagedFile(ROLLOVER_FILE_ROLES.TARGET, targetCycle, {
      inActiveFolder,
    });
    const [sourceSnapshot, targetSnapshot] = await Promise.all([
      loadSnapshot(source.file),
      loadSnapshot(targetFile),
    ]);
    if (mode === 'cleaned') {
      assertSharedDriveManagerPermissions(sourceSnapshot.permissions, runtime);
      assertNoInheritedOrAdministrativeFormCollaborators(sourceSnapshot.permissions, runtime);
    }
    assertManagedFile(
      targetSnapshot.file,
      runtime,
      ROLLOVER_FILE_ROLES.TARGET,
      targetCycle,
      { requireActiveFolder: inActiveFolder },
    );
    assertSourceFingerprint(targetSnapshot.file, sourceSnapshot.file, runtime, sourceCycle);
    if (mode !== 'cleaned') {
      assertResolvedSourceAccess(source, sourceSnapshot);
      assertSourceCapabilities(targetSnapshot.file);
    }
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
    const expectedPermissions = expectedTargetPermissions(
      sourceSnapshot.permissions,
      collaboratorPermissions,
      copySourceCollaborators,
      source.origin,
    );
    assertPermissionsMatch(
      expectedPermissions,
      targetSnapshot.permissions,
      { runtime },
    );

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

    const permissionsByFile = new Map(await Promise.all(files.map(async (file) => {
      const permissions = await google.getAllPermissions({ fileId: file.id });
      assertSharedDriveManagerPermissions(permissions, runtime);
      assertNoInheritedOrAdministrativeFormCollaborators(permissions, runtime);
      return [file.id, permissions];
    })));

    for (const file of files) {
      let form = await google.getForm({ formId: file.id });
      form = await ensurePublishState(google, form, {
        isPublished: false,
        isAcceptingResponses: false,
      });
      const permissions = permissionsByFile.get(file.id);
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
