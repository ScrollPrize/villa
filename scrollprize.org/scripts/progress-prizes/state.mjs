import {
  getCycleDeadline,
  previousCycle,
  subtractPacificCalendarDays,
} from './dates.mjs';
import { parseInstant } from './clock.mjs';

export const ROLLOVER_PHASES = Object.freeze({
  WAITING: 'waiting',
  PREPARE: 'prepare',
  ACTIVATE: 'activate',
});

export function planRollover({ targetCycle, now, preparationDays = 7 }) {
  if (!Number.isInteger(preparationDays) || preparationDays < 1) {
    throw new TypeError('preparationDays must be a positive integer');
  }

  const observedAt = parseInstant(now, 'now');
  const sourceCycle = previousCycle(targetCycle);
  const sourceDeadline = getCycleDeadline(sourceCycle);
  const targetDeadline = getCycleDeadline(targetCycle);
  const preparationOpensAt = subtractPacificCalendarDays(sourceDeadline.cutoffAt, preparationDays);

  let phase = ROLLOVER_PHASES.ACTIVATE;
  if (observedAt < preparationOpensAt) {
    phase = ROLLOVER_PHASES.WAITING;
  } else if (observedAt < sourceDeadline.cutoffAt) {
    phase = ROLLOVER_PHASES.PREPARE;
  }

  return Object.freeze({
    phase,
    observedAt,
    sourceCycle,
    targetCycle,
    preparationDays,
    preparationOpensAt,
    activationAt: sourceDeadline.cutoffAt,
    sourceDeadline,
    targetDeadline,
  });
}

export function serializeRolloverPlan(plan) {
  return {
    phase: plan.phase,
    observedAt: plan.observedAt.toISOString(),
    sourceCycle: plan.sourceCycle,
    targetCycle: plan.targetCycle,
    preparationDays: plan.preparationDays,
    preparationOpensAt: plan.preparationOpensAt.toISOString(),
    activationAt: plan.activationAt.toISOString(),
    sourceDeadline: {
      cycle: plan.sourceDeadline.cycle,
      label: plan.sourceDeadline.label,
      cutoffAt: plan.sourceDeadline.cutoffAt.toISOString(),
    },
    targetDeadline: {
      cycle: plan.targetDeadline.cycle,
      label: plan.targetDeadline.label,
      cutoffAt: plan.targetDeadline.cutoffAt.toISOString(),
    },
  };
}
