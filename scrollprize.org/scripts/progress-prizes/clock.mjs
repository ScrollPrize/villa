export const AUTOMATION_ENVIRONMENTS = Object.freeze({
  STAGING: 'staging',
  PRODUCTION: 'production',
});

export function assertAutomationEnvironment(environment) {
  if (!Object.values(AUTOMATION_ENVIRONMENTS).includes(environment)) {
    throw new TypeError('environment must be either staging or production');
  }
  return environment;
}

export function parseInstant(value, name = 'instant') {
  if (value instanceof Date) {
    if (Number.isNaN(value.getTime())) {
      throw new TypeError(`${name} must be a valid instant`);
    }
    return new Date(value.getTime());
  }

  if (
    typeof value !== 'string'
    || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?(?:Z|[+-]\d{2}:\d{2})$/.test(value)
  ) {
    throw new TypeError(`${name} must be an ISO 8601 timestamp with an explicit offset`);
  }

  const instant = new Date(value);
  if (Number.isNaN(instant.getTime())) {
    throw new TypeError(`${name} must be a valid instant`);
  }
  return instant;
}

export class SystemClock {
  now() {
    return new Date();
  }
}

export class FixedClock {
  #instant;

  constructor(instant) {
    this.#instant = parseInstant(instant, 'simulatedNow');
  }

  now() {
    return new Date(this.#instant.getTime());
  }
}

export function assertStagingOnlyControl({ environment, eventName, controlName, enabled }) {
  assertAutomationEnvironment(environment);
  if (!enabled) {
    return;
  }
  if (environment !== AUTOMATION_ENVIRONMENTS.STAGING) {
    throw new Error(`${controlName} is forbidden in production`);
  }
  if (eventName !== 'workflow_dispatch') {
    throw new Error(`${controlName} is allowed only for a manually dispatched staging run`);
  }
}

export function createClock({ environment, eventName, simulatedNow } = {}) {
  assertAutomationEnvironment(environment);
  const hasSimulatedTime = simulatedNow !== undefined && simulatedNow !== null && simulatedNow !== '';
  assertStagingOnlyControl({
    environment,
    eventName,
    controlName: 'simulated time',
    enabled: hasSimulatedTime,
  });

  return hasSimulatedTime ? new FixedClock(simulatedNow) : new SystemClock();
}
