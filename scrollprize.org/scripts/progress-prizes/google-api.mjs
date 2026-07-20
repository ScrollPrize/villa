const DRIVE_API_BASE = "https://www.googleapis.com/drive/v3";
const FORMS_API_BASE = "https://forms.googleapis.com/v1";

const DEFAULT_RETRY = Object.freeze({
  maxAttempts: 5,
  baseDelayMs: 500,
  maxDelayMs: 10_000,
  jitterRatio: 0.2,
});

const RETRYABLE_STATUS_CODES = new Set([429, 500, 502, 503, 504]);
const REDACTED = "[REDACTED]";

const DEFAULT_FILE_FIELDS =
  "id,name,mimeType,parents,appProperties,trashed,driveId,capabilities(canCopy,canEdit,canShare)";
const DEFAULT_PERMISSION_FIELDS =
  "id,type,role,emailAddress,domain,allowFileDiscovery,expirationTime,deleted,displayName,permissionDetails,view,pendingOwner,inheritedPermissionsDisabled";

/**
 * A deliberately small, log-safe error for Google REST failures.
 *
 * Raw response bodies and request URLs are not retained because they can contain
 * editor form IDs, ACL identities, or other private Workspace metadata.
 */
export class GoogleApiError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = "GoogleApiError";
    this.operation = options.operation;
    this.status = options.status;
    this.googleCode = options.googleCode;
    this.googleStatus = options.googleStatus;
    this.reasons = options.reasons ?? [];
    this.attempts = options.attempts;
    this.retryable = options.retryable ?? false;
  }
}

function assertNonEmptyString(value, label) {
  if (typeof value !== "string" || value.trim() === "") {
    throw new TypeError(`${label} must be a non-empty string`);
  }
}

function assertPlainObject(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
}

function normalizeRetryPolicy(retry = {}) {
  const policy = { ...DEFAULT_RETRY, ...retry };

  if (!Number.isInteger(policy.maxAttempts) || policy.maxAttempts < 1) {
    throw new TypeError("retry.maxAttempts must be a positive integer");
  }

  for (const key of ["baseDelayMs", "maxDelayMs", "jitterRatio"]) {
    if (typeof policy[key] !== "number" || !Number.isFinite(policy[key]) || policy[key] < 0) {
      throw new TypeError(`retry.${key} must be a non-negative finite number`);
    }
  }

  if (policy.maxDelayMs < policy.baseDelayMs) {
    throw new TypeError("retry.maxDelayMs must be greater than or equal to retry.baseDelayMs");
  }

  if (policy.jitterRatio > 1) {
    throw new TypeError("retry.jitterRatio must be no greater than 1");
  }

  return Object.freeze(policy);
}

function parseRetryAfter(value, now) {
  if (!value) return undefined;

  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return seconds * 1_000;
  }

  const timestamp = Date.parse(value);
  if (Number.isNaN(timestamp)) return undefined;
  return Math.max(0, timestamp - now());
}

function retryDelayMs(attempt, response, policy, random, now) {
  const retryAfter = parseRetryAfter(response?.headers?.get?.("retry-after"), now);
  if (retryAfter !== undefined) {
    return Math.min(policy.maxDelayMs, retryAfter);
  }

  const exponential = Math.min(
    policy.maxDelayMs,
    policy.baseDelayMs * 2 ** Math.max(0, attempt - 1),
  );
  const jitterMultiplier = 1 + (random() * 2 - 1) * policy.jitterRatio;
  return Math.max(0, Math.round(exponential * jitterMultiplier));
}

async function parseResponseBody(response) {
  if (response.status === 204) return undefined;

  const contentType = response.headers?.get?.("content-type") ?? "";
  if (contentType.includes("application/json")) {
    try {
      return await response.json();
    } catch {
      return undefined;
    }
  }

  try {
    const text = await response.text();
    return text === "" ? undefined : text;
  } catch {
    return undefined;
  }
}

function googleErrorMetadata(body) {
  if (body === null || typeof body !== "object" || Array.isArray(body)) return {};
  const error = body.error;
  if (error === null || typeof error !== "object" || Array.isArray(error)) return {};

  const reasons = Array.isArray(error.errors)
    ? error.errors
        .map((entry) => (entry && typeof entry.reason === "string" ? entry.reason : undefined))
        .filter(Boolean)
    : [];

  return {
    googleCode: typeof error.code === "number" ? error.code : undefined,
    googleStatus: typeof error.status === "string" ? error.status : undefined,
    reasons,
  };
}

function isNetworkError(error) {
  return error instanceof TypeError && error.name !== "AbortError";
}

function createTransport({
  accessToken,
  getAccessToken,
  fetchImpl,
  sleep,
  random,
  now,
  retry,
  driveApiBase,
  formsApiBase,
}) {
  if (accessToken !== undefined && getAccessToken !== undefined) {
    throw new TypeError("Provide either accessToken or getAccessToken, not both");
  }
  if (accessToken === undefined && getAccessToken === undefined) {
    throw new TypeError("accessToken or getAccessToken is required");
  }
  if (accessToken !== undefined) assertNonEmptyString(accessToken, "accessToken");
  if (getAccessToken !== undefined && typeof getAccessToken !== "function") {
    throw new TypeError("getAccessToken must be a function");
  }
  if (typeof fetchImpl !== "function") throw new TypeError("fetchImpl must be a function");
  if (typeof sleep !== "function") throw new TypeError("sleep must be a function");
  if (typeof random !== "function") throw new TypeError("random must be a function");
  if (typeof now !== "function") throw new TypeError("now must be a function");

  const defaultRetry = normalizeRetryPolicy(retry);
  const tokenProvider =
    getAccessToken ??
    (() => {
      return accessToken;
    });

  async function requestJson({
    operation,
    baseUrl,
    path,
    method = "GET",
    query,
    body,
    retry: operationRetry,
  }) {
    const policy = operationRetry
      ? normalizeRetryPolicy({ ...defaultRetry, ...operationRetry })
      : defaultRetry;
    const url = new URL(`${baseUrl}${path}`);
    for (const [key, value] of Object.entries(query ?? {})) {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }

    let lastNetworkError;
    for (let attempt = 1; attempt <= policy.maxAttempts; attempt += 1) {
      const token = await tokenProvider();
      assertNonEmptyString(token, "Google access token");

      let response;
      try {
        response = await fetchImpl(url, {
          method,
          headers: {
            accept: "application/json",
            authorization: `Bearer ${token}`,
            ...(body === undefined ? {} : { "content-type": "application/json" }),
          },
          ...(body === undefined ? {} : { body: JSON.stringify(body) }),
        });
      } catch (error) {
        if (!isNetworkError(error)) throw error;
        lastNetworkError = error;
        if (attempt === policy.maxAttempts) {
          throw new GoogleApiError(
            `Google API request failed (${operation}): network error after ${attempt} attempts`,
            { operation, attempts: attempt, retryable: true },
          );
        }
        await sleep(retryDelayMs(attempt, undefined, policy, random, now));
        continue;
      }

      const responseBody = await parseResponseBody(response);
      if (response.ok) return responseBody;

      const retryable = RETRYABLE_STATUS_CODES.has(response.status);
      if (retryable && attempt < policy.maxAttempts) {
        await sleep(retryDelayMs(attempt, response, policy, random, now));
        continue;
      }

      const metadata = googleErrorMetadata(responseBody);
      throw new GoogleApiError(
        `Google API request failed (${operation}): HTTP ${response.status}`,
        {
          operation,
          status: response.status,
          attempts: attempt,
          retryable,
          ...metadata,
        },
      );
    }

    // The loop always returns or throws. This guard keeps failure explicit if the
    // retry implementation is changed later.
    throw new GoogleApiError(`Google API request failed (${operation})`, {
      operation,
      retryable: Boolean(lastNetworkError),
    });
  }

  return Object.freeze({
    drive(path, options = {}) {
      return requestJson({
        operation: options.operation ?? "drive.request",
        baseUrl: driveApiBase,
        path,
        ...options,
      });
    },
    forms(path, options = {}) {
      return requestJson({
        operation: options.operation ?? "forms.request",
        baseUrl: formsApiBase,
        path,
        ...options,
      });
    },
  });
}

function encodePath(value, label) {
  assertNonEmptyString(value, label);
  return encodeURIComponent(value);
}

function escapeDriveQueryValue(value) {
  assertNonEmptyString(value, "Drive query value");
  return value.replaceAll("\\", "\\\\").replaceAll("'", "\\'");
}

/** Build the narrow Drive query used to resume one managed rollover cycle. */
export function buildAppPropertiesQuery(appProperties, options = {}) {
  assertPlainObject(appProperties, "appProperties");
  const entries = Object.entries(appProperties).sort(([left], [right]) => left.localeCompare(right));
  if (entries.length === 0) {
    throw new TypeError("appProperties must contain at least one key");
  }

  const clauses = entries.map(([key, value]) => {
    assertNonEmptyString(key, "appProperties key");
    assertNonEmptyString(value, `appProperties.${key}`);
    return `appProperties has { key='${escapeDriveQueryValue(key)}' and value='${escapeDriveQueryValue(value)}' }`;
  });

  if (options.parentId !== undefined) {
    clauses.push(`'${escapeDriveQueryValue(options.parentId)}' in parents`);
  }
  if (options.includeTrashed !== true) clauses.push("trashed = false");
  return clauses.join(" and ");
}

/** Return true when Drive reports that a permission comes from a parent. */
export function isInheritedPermission(permission) {
  if (permission === null || typeof permission !== "object") return false;
  if (permission.inherited === true) return true;
  return (
    Array.isArray(permission.permissionDetails) &&
    permission.permissionDetails.some((detail) => detail?.inherited === true)
  );
}

/**
 * Reduce a Drive Permission resource to fields accepted by permissions.create.
 * This intentionally excludes IDs, display names, inheritance metadata, and
 * ownership-transfer fields.
 */
export function normalizePermissionForCreate(permission) {
  assertPlainObject(permission, "permission");
  assertNonEmptyString(permission.type, "permission.type");
  assertNonEmptyString(permission.role, "permission.role");

  const normalized = {
    type: permission.type,
    role: permission.role,
  };

  if (permission.type === "user" || permission.type === "group") {
    assertNonEmptyString(permission.emailAddress, "permission.emailAddress");
    normalized.emailAddress = permission.emailAddress.toLowerCase();
  } else if (permission.type === "domain") {
    assertNonEmptyString(permission.domain, "permission.domain");
    normalized.domain = permission.domain.toLowerCase();
  }

  if (typeof permission.allowFileDiscovery === "boolean") {
    normalized.allowFileDiscovery = permission.allowFileDiscovery;
  }
  if (typeof permission.expirationTime === "string" && permission.expirationTime !== "") {
    normalized.expirationTime = permission.expirationTime;
  }
  if (permission.view !== undefined) {
    if (permission.view !== "published") {
      throw new TypeError("Only the published permission view can be recreated");
    }
    normalized.view = permission.view;
  }

  return normalized;
}

/**
 * Select only direct internal collaborators safe to reproduce on a new form.
 * Public/domain readers and ownership roles must be handled explicitly by the
 * orchestrator instead of being copied accidentally.
 */
export function filterDirectCollaboratorPermissions(
  permissions,
  { roles = ["writer", "commenter"], types = ["user", "group"] } = {},
) {
  if (!Array.isArray(permissions)) throw new TypeError("permissions must be an array");
  if (!Array.isArray(roles) || !Array.isArray(types)) {
    throw new TypeError("roles and types must be arrays");
  }
  const allowedRoles = new Set(roles);
  const allowedTypes = new Set(types);

  return permissions.filter(
    (permission) =>
      permission &&
      permission.deleted !== true &&
      permission.pendingOwner !== true &&
      !isInheritedPermission(permission) &&
      allowedRoles.has(permission.role) &&
      allowedTypes.has(permission.type),
  );
}

/** Select responder permissions from Drive's special published permission view. */
export function filterPublishedResponderPermissions(permissions) {
  if (!Array.isArray(permissions)) throw new TypeError("permissions must be an array");
  return permissions.filter(
    (permission) =>
      permission &&
      permission.deleted !== true &&
      permission.pendingOwner !== true &&
      permission.view === "published" &&
      permission.role === "reader" &&
      ["anyone", "domain", "group", "user"].includes(permission.type),
  );
}

/** Stable comparison key for normalized source/target ACL checks. */
export function permissionIdentityKey(permission) {
  const normalized = normalizePermissionForCreate(permission);
  const identity =
    normalized.emailAddress ?? normalized.domain ?? (normalized.type === "anyone" ? "*" : "");
  return [
    normalized.type,
    normalized.role,
    identity,
    normalized.allowFileDiscovery ?? "",
    normalized.expirationTime ?? "",
    normalized.view ?? "",
  ].join(":");
}

async function collectPages(fetchPage, itemKey) {
  const items = [];
  const seenTokens = new Set();
  let pageToken;

  do {
    const page = (await fetchPage(pageToken)) ?? {};
    if (Array.isArray(page[itemKey])) items.push(...page[itemKey]);
    pageToken = page.nextPageToken;
    if (pageToken !== undefined) {
      if (seenTokens.has(pageToken)) {
        throw new GoogleApiError("Google API returned a repeated pagination token", {
          operation: "pagination",
        });
      }
      seenTokens.add(pageToken);
    }
  } while (pageToken !== undefined && pageToken !== "");

  return items;
}

/**
 * Create a dependency-free Google Forms + Drive client.
 *
 * The caller normally supplies the short-lived token emitted by the Google WIF
 * authentication Action. `getAccessToken` is available for longer-running jobs
 * that refresh credentials externally.
 */
export function createGoogleApiClient({
  accessToken,
  getAccessToken,
  fetchImpl = globalThis.fetch,
  sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds)),
  random = Math.random,
  now = Date.now,
  retry,
  driveApiBase = DRIVE_API_BASE,
  formsApiBase = FORMS_API_BASE,
} = {}) {
  const transport = createTransport({
    accessToken,
    getAccessToken,
    fetchImpl,
    sleep,
    random,
    now,
    retry,
    driveApiBase,
    formsApiBase,
  });

  const files = Object.freeze({
    async get({ fileId, fields = DEFAULT_FILE_FIELDS }) {
      return transport.drive(`/files/${encodePath(fileId, "fileId")}`, {
        operation: "drive.files.get",
        query: { supportsAllDrives: true, fields },
      });
    },

    async copy({
      fileId,
      name,
      parentId,
      appProperties,
      fields = DEFAULT_FILE_FIELDS,
      retry: operationRetry,
    }) {
      assertNonEmptyString(name, "name");
      if (appProperties !== undefined) assertPlainObject(appProperties, "appProperties");
      const body = {
        name,
        ...(parentId === undefined ? {} : { parents: [parentId] }),
        ...(appProperties === undefined ? {} : { appProperties }),
      };

      return transport.drive(`/files/${encodePath(fileId, "fileId")}/copy`, {
        operation: "drive.files.copy",
        method: "POST",
        query: {
          supportsAllDrives: true,
          ignoreDefaultVisibility: true,
          fields,
        },
        body,
        // A lost response to files.copy is ambiguous: retrying here can create a
        // second form. The orchestrator resumes by searching appProperties instead.
        retry: { ...operationRetry, maxAttempts: 1 },
      });
    },

    async update({
      fileId,
      name,
      appProperties,
      addParentIds,
      removeParentIds,
      fields = DEFAULT_FILE_FIELDS,
      retry: operationRetry,
    }) {
      if (name !== undefined) assertNonEmptyString(name, "name");
      if (appProperties !== undefined) assertPlainObject(appProperties, "appProperties");
      for (const [label, values] of [
        ["addParentIds", addParentIds],
        ["removeParentIds", removeParentIds],
      ]) {
        if (values !== undefined && (!Array.isArray(values) || values.some((value) => typeof value !== "string"))) {
          throw new TypeError(`${label} must be an array of strings`);
        }
      }

      const body = {
        ...(name === undefined ? {} : { name }),
        ...(appProperties === undefined ? {} : { appProperties }),
      };
      if (Object.keys(body).length === 0 && !addParentIds?.length && !removeParentIds?.length) {
        throw new TypeError("files.update requires metadata or a parent change");
      }

      return transport.drive(`/files/${encodePath(fileId, "fileId")}`, {
        operation: "drive.files.update",
        method: "PATCH",
        query: {
          supportsAllDrives: true,
          addParents: addParentIds?.join(","),
          removeParents: removeParentIds?.join(","),
          fields,
        },
        body,
        retry: operationRetry,
      });
    },

    async listByAppProperties({
      appProperties,
      parentId,
      includeTrashed = false,
      driveId,
      pageSize = 100,
      fields = DEFAULT_FILE_FIELDS,
    }) {
      if (!Number.isInteger(pageSize) || pageSize < 1 || pageSize > 1_000) {
        throw new TypeError("pageSize must be an integer from 1 through 1000");
      }
      const q = buildAppPropertiesQuery(appProperties, { parentId, includeTrashed });

      return collectPages(
        (pageToken) =>
          transport.drive("/files", {
            operation: "drive.files.list",
            query: {
              q,
              pageSize,
              pageToken,
              spaces: "drive",
              orderBy: "createdTime desc",
              supportsAllDrives: true,
              includeItemsFromAllDrives: true,
              corpora: driveId === undefined ? undefined : "drive",
              driveId,
              fields: `nextPageToken,files(${fields})`,
            },
          }),
        "files",
      );
    },
  });

  const permissions = Object.freeze({
    async list({
      fileId,
      fields = DEFAULT_PERMISSION_FIELDS,
      pageSize = 100,
      includePublished = true,
    }) {
      if (!Number.isInteger(pageSize) || pageSize < 1 || pageSize > 100) {
        throw new TypeError("pageSize must be an integer from 1 through 100");
      }

      return collectPages(
        (pageToken) =>
          transport.drive(`/files/${encodePath(fileId, "fileId")}/permissions`, {
            operation: "drive.permissions.list",
            query: {
              supportsAllDrives: true,
              useDomainAdminAccess: false,
              includePermissionsForView: includePublished ? "published" : undefined,
              pageSize,
              pageToken,
              fields: `nextPageToken,permissions(${fields})`,
            },
          }),
        "permissions",
      );
    },

    async create({
      fileId,
      permission,
      sendNotificationEmail = false,
      emailMessage,
      fields = DEFAULT_PERMISSION_FIELDS,
      retry: operationRetry,
    }) {
      assertPlainObject(permission, "permission");
      assertNonEmptyString(permission.type, "permission.type");
      assertNonEmptyString(permission.role, "permission.role");
      if (permission.role === "owner" && sendNotificationEmail === false) {
        throw new TypeError("Ownership transfers require notification email");
      }

      return transport.drive(`/files/${encodePath(fileId, "fileId")}/permissions`, {
        operation: "drive.permissions.create",
        method: "POST",
        query: {
          supportsAllDrives: true,
          sendNotificationEmail,
          emailMessage,
          fields,
        },
        body: permission,
        // Permission creation is also recovered by relisting the target ACL.
        retry: { ...operationRetry, maxAttempts: 1 },
      });
    },

    async delete({ fileId, permissionId, retry: operationRetry }) {
      return transport.drive(
        `/files/${encodePath(fileId, "fileId")}/permissions/${encodePath(permissionId, "permissionId")}`,
        {
          operation: "drive.permissions.delete",
          method: "DELETE",
          query: { supportsAllDrives: true },
          retry: operationRetry,
        },
      );
    },
  });

  const forms = Object.freeze({
    async get({ formId }) {
      return transport.forms(`/forms/${encodePath(formId, "formId")}`, {
        operation: "forms.get",
      });
    },

    async batchUpdate({
      formId,
      requests,
      includeFormInResponse = true,
      requiredRevisionId,
      targetRevisionId,
      retry: operationRetry,
    }) {
      if (!Array.isArray(requests) || requests.length === 0) {
        throw new TypeError("requests must be a non-empty array");
      }
      if (requiredRevisionId !== undefined && targetRevisionId !== undefined) {
        throw new TypeError("Use either requiredRevisionId or targetRevisionId, not both");
      }
      if (requiredRevisionId !== undefined) {
        assertNonEmptyString(requiredRevisionId, "requiredRevisionId");
      }
      if (targetRevisionId !== undefined) {
        assertNonEmptyString(targetRevisionId, "targetRevisionId");
      }

      const writeControl =
        requiredRevisionId !== undefined
          ? { requiredRevisionId }
          : targetRevisionId !== undefined
            ? { targetRevisionId }
            : undefined;

      return transport.forms(`/forms/${encodePath(formId, "formId")}:batchUpdate`, {
        operation: "forms.batchUpdate",
        method: "POST",
        body: {
          includeFormInResponse,
          requests,
          ...(writeControl === undefined ? {} : { writeControl }),
        },
        retry: operationRetry,
      });
    },

    async setPublishSettings({
      formId,
      isPublished,
      isAcceptingResponses,
      retry: operationRetry,
    }) {
      if (typeof isPublished !== "boolean" || typeof isAcceptingResponses !== "boolean") {
        throw new TypeError("isPublished and isAcceptingResponses must be booleans");
      }
      if (!isPublished && isAcceptingResponses) {
        throw new TypeError("An unpublished form cannot accept responses");
      }

      return transport.forms(`/forms/${encodePath(formId, "formId")}:setPublishSettings`, {
        operation: "forms.setPublishSettings",
        method: "POST",
        body: {
          publishSettings: {
            publishState: { isPublished, isAcceptingResponses },
          },
          updateMask: "publishState",
        },
        retry: operationRetry,
      });
    },
  });

  return Object.freeze({
    drive: Object.freeze({ files, permissions }),
    forms,
    getForm: (options) => forms.get(options),
    getFile: (options) => files.get(options),
    listFilesByAppProperties: (options) => files.listByAppProperties(options),
    copyFile: (options) => files.copy(options),
    updateFile: (options) => files.update(options),
    getAllPermissions: (options) => permissions.list(options),
    createPermission: (options) => permissions.create(options),
    deletePermission: (options) => permissions.delete(options),
    updateFormTitle({ formId, title, requiredRevisionId, targetRevisionId, retry: operationRetry }) {
      assertNonEmptyString(title, "title");
      return forms.batchUpdate({
        formId,
        requests: [
          {
            updateFormInfo: {
              info: { title },
              updateMask: "title",
            },
          },
        ],
        includeFormInResponse: true,
        requiredRevisionId,
        targetRevisionId,
        retry: operationRetry,
      });
    },
    setPublishState: (options) => forms.setPublishSettings(options),
  });
}

function isSensitiveKey(key) {
  return /(?:authorization|access[_-]?token|refresh[_-]?token|credential|private[_-]?key|service[_-]?account|form[_-]?id|file[_-]?id|folder[_-]?id|permission[_-]?id|email(?:address)?|editor(?:uri|url)?|domain)/i.test(
    key,
  );
}

function redactString(value, secrets) {
  let redacted = value.replace(/Bearer\s+[A-Za-z0-9._~+\/-]+=*/gi, `Bearer ${REDACTED}`);
  redacted = redacted.replace(
    /https:\/\/docs\.google\.com\/forms\/d\/(?!e\/)[^/\s]+\/(?:edit|preview)(?:[^\s]*)?/gi,
    REDACTED,
  );
  redacted = redacted.replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, REDACTED);
  for (const secret of secrets) {
    if (typeof secret === "string" && secret !== "") {
      redacted = redacted.split(secret).join(REDACTED);
    }
  }
  return redacted;
}

/**
 * Return a redacted JSON-compatible clone suitable for workflow logs/summaries.
 * Public `responderUri` values remain visible by design; editor IDs and ACL
 * identities do not.
 */
export function redactForLog(value, { secrets = [] } = {}) {
  if (!Array.isArray(secrets)) throw new TypeError("secrets must be an array");

  const visit = (current, key = "") => {
    if (isSensitiveKey(key)) return REDACTED;
    if (typeof current === "string") return redactString(current, secrets);
    if (Array.isArray(current)) return current.map((item) => visit(item));
    if (current && typeof current === "object") {
      return Object.fromEntries(
        Object.entries(current).map(([childKey, childValue]) => [childKey, visit(childValue, childKey)]),
      );
    }
    return current;
  };

  return visit(value);
}

export const GOOGLE_API_SCOPES = Object.freeze([
  "https://www.googleapis.com/auth/drive.file",
  "https://www.googleapis.com/auth/forms.body",
]);
