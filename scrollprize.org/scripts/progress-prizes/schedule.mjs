import { parseInstant } from './clock.mjs';
import {
  PACIFIC_TIME_ZONE,
  daysInMonth,
  formatCycle,
  getCycleDeadline,
  getPacificParts,
  nextCycle,
  pacificDateTimeToInstant,
  previousCycle,
  subtractPacificCalendarDays,
} from './dates.mjs';

export const SCHEDULE_CRONS = Object.freeze({
  PREPARE: '17 6 * * *',
  PRE_CUTOFF: '40 23 28-31 * *',
  EARLY_RECOVERY: '17 0 1 * *',
  LATE_RECOVERY: '47 6 1 * *',
});

// Workflow-facing alias kept explicit so the public control contract reads as
// Progress Prize-specific at the dispatch boundary.
export const PROGRESS_PRIZE_SCHEDULES = SCHEDULE_CRONS;

export const SCHEDULE_OPERATIONS = Object.freeze({
  NONE: 'none',
  DRY_RUN: 'dry-run',
  PREPARE: 'prepare',
  ACTIVATE: 'activate',
});

export const SCHEDULE_REASONS = Object.freeze({
  OUTSIDE_WINDOW: 'outside-window',
  WRONG_SLOT: 'wrong-schedule-slot',
  SLOT_NOT_DUE: 'schedule-slot-not-due',
  MANUAL_DRY_RUN: 'manual-dry-run',
  PREPARATION: 'preparation-window',
  PRE_CUTOFF: 'pre-cutoff',
  EARLY_RECOVERY: 'early-recovery',
  LATE_RECOVERY: 'late-recovery',
});

const ALLOWED_EVENTS = new Set(['schedule', 'workflow_dispatch']);
const ALLOWED_CRONS = new Set(Object.values(SCHEDULE_CRONS));
const ACTIVATION_CRONS = new Set([
  SCHEDULE_CRONS.PRE_CUTOFF,
  SCHEDULE_CRONS.EARLY_RECOVERY,
  SCHEDULE_CRONS.LATE_RECOVERY,
]);
const PREPARATION_DAYS = 7;

function calendarDate({ year, month, day }) {
  return Object.freeze({ year, month, day });
}

function previousCalendarDate(parts) {
  const date = new Date(0);
  date.setUTCFullYear(parts.year, parts.month - 1, parts.day);
  date.setUTCHours(0, 0, 0, 0);
  date.setUTCDate(date.getUTCDate() - 1);
  return calendarDate({
    year: date.getUTCFullYear(),
    month: date.getUTCMonth() + 1,
    day: date.getUTCDate(),
  });
}

function isFirstDay(parts) {
  return parts.day === 1;
}

function isLastDay(parts) {
  return parts.day === daysInMonth(parts.year, parts.month);
}

function cronWallTime(cron) {
  if (cron === SCHEDULE_CRONS.PREPARE) return Object.freeze({ hour: 6, minute: 17 });
  if (cron === SCHEDULE_CRONS.PRE_CUTOFF) return Object.freeze({ hour: 23, minute: 40 });
  if (cron === SCHEDULE_CRONS.EARLY_RECOVERY) return Object.freeze({ hour: 0, minute: 17 });
  if (cron === SCHEDULE_CRONS.LATE_RECOVERY) return Object.freeze({ hour: 6, minute: 47 });
  throw new TypeError('scheduledCron is outside the Progress Prize schedule allowlist');
}

function scheduledOriginDate(observedAt, local, cron) {
  const { hour, minute } = cronWallTime(cron);
  const scheduledToday = pacificDateTimeToInstant({
    year: local.year,
    month: local.month,
    day: local.day,
    hour,
    minute,
  });
  return observedAt >= scheduledToday
    ? calendarDate(local)
    : previousCalendarDate(local);
}

function assertTrigger({ eventName, scheduledCron }) {
  if (!ALLOWED_EVENTS.has(eventName)) {
    throw new TypeError('eventName must be schedule or workflow_dispatch');
  }
  if (eventName === 'schedule') {
    if (typeof scheduledCron !== 'string' || !ALLOWED_CRONS.has(scheduledCron)) {
      throw new TypeError('scheduledCron is outside the Progress Prize schedule allowlist');
    }
    return scheduledCron;
  }
  if (scheduledCron !== undefined && scheduledCron !== null && scheduledCron !== '') {
    throw new Error('workflow_dispatch must not claim a scheduled cron slot');
  }
  return '';
}

function noDispatch({
  eventName,
  scheduledCron,
  observedAt,
  reason,
  sourceCycle,
  targetCycle,
}) {
  return Object.freeze({
    dispatch: false,
    operation: SCHEDULE_OPERATIONS.NONE,
    reason,
    eventName,
    scheduledCron,
    timeZone: PACIFIC_TIME_ZONE,
    observedAt,
    sourceCycle,
    targetCycle,
    preparationOpensAt: null,
    activationAt: null,
  });
}

function dispatchPlan({
  operation,
  reason,
  eventName,
  scheduledCron,
  observedAt,
  sourceCycle,
  targetCycle,
  preparationOpensAt,
  activationAt,
}) {
  return Object.freeze({
    dispatch: true,
    operation,
    reason,
    eventName,
    scheduledCron,
    timeZone: PACIFIC_TIME_ZONE,
    observedAt,
    sourceCycle,
    targetCycle,
    preparationOpensAt,
    activationAt,
  });
}

function classifyObservedTime(observedAt, local) {
  const localCycle = formatCycle(local.year, local.month);

  // The whole first Pacific calendar day is reserved for bounded activation
  // recovery. A run delayed into day two must never select the previous cycle.
  if (isFirstDay(local)) {
    const sourceCycle = previousCycle(localCycle);
    const activationAt = getCycleDeadline(sourceCycle).cutoffAt;
    const lateRecoveryAt = pacificDateTimeToInstant({
      year: local.year,
      month: local.month,
      day: 1,
      hour: 6,
      minute: 47,
    });
    return {
      operation: SCHEDULE_OPERATIONS.ACTIVATE,
      reason: observedAt < lateRecoveryAt
        ? SCHEDULE_REASONS.EARLY_RECOVERY
        : SCHEDULE_REASONS.LATE_RECOVERY,
      sourceCycle,
      targetCycle: localCycle,
      preparationOpensAt: subtractPacificCalendarDays(activationAt, PREPARATION_DAYS),
      activationAt,
    };
  }

  const sourceCycle = localCycle;
  const targetCycle = nextCycle(sourceCycle);
  const sourceDeadline = getCycleDeadline(sourceCycle);
  const preparationOpensAt = subtractPacificCalendarDays(
    sourceDeadline.cutoffAt,
    PREPARATION_DAYS,
  );
  const preCutoffAt = pacificDateTimeToInstant({
    year: sourceDeadline.year,
    month: sourceDeadline.month,
    day: sourceDeadline.day,
    hour: 23,
    minute: 40,
  });

  if (observedAt >= preCutoffAt && observedAt < sourceDeadline.cutoffAt) {
    return {
      operation: SCHEDULE_OPERATIONS.ACTIVATE,
      reason: SCHEDULE_REASONS.PRE_CUTOFF,
      sourceCycle,
      targetCycle,
      preparationOpensAt,
      activationAt: sourceDeadline.cutoffAt,
    };
  }
  if (observedAt >= preparationOpensAt && observedAt < preCutoffAt) {
    return {
      operation: SCHEDULE_OPERATIONS.PREPARE,
      reason: SCHEDULE_REASONS.PREPARATION,
      sourceCycle,
      targetCycle,
      preparationOpensAt,
      activationAt: sourceDeadline.cutoffAt,
    };
  }
  return undefined;
}

function scheduledActivationIsDue({ scheduledCron, observedAt, local }) {
  if (scheduledCron === SCHEDULE_CRONS.PRE_CUTOFF) {
    // The pre-cutoff cron cannot originate on day one (its day-of-month field
    // is 28-31), so every first-day observation is a delayed final-day event,
    // even after the 23:40 wall-clock time has passed again.
    return isLastDay(local)
      || (isFirstDay(local) && isLastDay(previousCalendarDate(local)));
  }
  const origin = scheduledOriginDate(observedAt, local, scheduledCron);
  return isFirstDay(origin) && isFirstDay(local);
}

/**
 * Select at most one secret-free production dispatch from the real observed
 * instant and its trusted GitHub trigger context.
 *
 * Schedule slots are intentionally not interchangeable: only the daily 06:17
 * slot can prepare. The three activation slots may recover a delayed final-day
 * run during the first Pacific day, but never during day two.
 */
export function planScheduleDispatch({
  now = new Date(),
  eventName,
  scheduledCron,
} = {}) {
  const observedAt = parseInstant(now, 'now');
  const trustedCron = assertTrigger({ eventName, scheduledCron });
  const local = getPacificParts(observedAt);
  const currentPacificCycle = formatCycle(local.year, local.month);

  // A manual scheduler run is a smoke test of the dispatch chain, never an
  // authority to prepare or activate. It derives the current Pacific source
  // month and asks the production workflow for its independently guarded,
  // read-only dry-run.
  if (eventName === 'workflow_dispatch') {
    const sourceCycle = currentPacificCycle;
    const targetCycle = nextCycle(sourceCycle);
    const activationAt = getCycleDeadline(sourceCycle).cutoffAt;
    return dispatchPlan({
      operation: SCHEDULE_OPERATIONS.DRY_RUN,
      reason: SCHEDULE_REASONS.MANUAL_DRY_RUN,
      eventName,
      scheduledCron: trustedCron,
      observedAt,
      sourceCycle,
      targetCycle,
      preparationOpensAt: subtractPacificCalendarDays(activationAt, PREPARATION_DAYS),
      activationAt,
    });
  }

  const classified = classifyObservedTime(observedAt, local);

  if (classified === undefined) {
    return noDispatch({
      eventName,
      scheduledCron: trustedCron,
      observedAt,
      reason: SCHEDULE_REASONS.OUTSIDE_WINDOW,
      sourceCycle: currentPacificCycle,
      targetCycle: nextCycle(currentPacificCycle),
    });
  }

  if (eventName === 'schedule') {
    if (classified.operation === SCHEDULE_OPERATIONS.PREPARE) {
      if (trustedCron !== SCHEDULE_CRONS.PREPARE) {
        return noDispatch({
          eventName,
          scheduledCron: trustedCron,
          observedAt,
          reason: SCHEDULE_REASONS.WRONG_SLOT,
          sourceCycle: classified.sourceCycle,
          targetCycle: classified.targetCycle,
        });
      }
    } else if (
      !ACTIVATION_CRONS.has(trustedCron)
      || !scheduledActivationIsDue({ scheduledCron: trustedCron, observedAt, local })
    ) {
      return noDispatch({
        eventName,
        scheduledCron: trustedCron,
        observedAt,
        reason: ACTIVATION_CRONS.has(trustedCron)
          ? SCHEDULE_REASONS.SLOT_NOT_DUE
          : SCHEDULE_REASONS.WRONG_SLOT,
        sourceCycle: classified.sourceCycle,
        targetCycle: classified.targetCycle,
      });
    }
  }

  return dispatchPlan({
    ...classified,
    eventName,
    scheduledCron: trustedCron,
    observedAt,
  });
}

export function serializeScheduleDispatch(plan) {
  if (plan === null || typeof plan !== 'object') {
    throw new TypeError('plan must be a schedule dispatch plan');
  }
  return Object.freeze({
    dispatch: plan.dispatch,
    operation: plan.operation,
    reason: plan.reason,
    eventName: plan.eventName,
    scheduledCron: plan.scheduledCron,
    timeZone: plan.timeZone,
    observedAt: plan.observedAt.toISOString(),
    sourceCycle: plan.sourceCycle,
    targetCycle: plan.targetCycle,
    preparationOpensAt: plan.preparationOpensAt?.toISOString() ?? '',
    activationAt: plan.activationAt?.toISOString() ?? '',
  });
}
