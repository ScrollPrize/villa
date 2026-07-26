import test from 'node:test';
import assert from 'node:assert/strict';

import {
  FixedClock,
  ROLLOVER_PHASES,
  assertStagingOnlyControl,
  createClock,
  parseInstant,
  planRollover,
  serializeRolloverPlan,
} from '../index.mjs';

test('fixed clock returns defensive Date copies', () => {
  const clock = new FixedClock('2026-07-25T07:00:00Z');
  const first = clock.now();
  first.setUTCFullYear(2030);
  assert.equal(clock.now().toISOString(), '2026-07-25T07:00:00.000Z');
});

test('simulated time is staging-only and manual-dispatch-only', () => {
  assert.throws(
    () => createClock({ environment: 'production', eventName: 'workflow_dispatch', simulatedNow: '2026-07-25T07:00:00Z' }),
    /forbidden in production/,
  );
  assert.throws(
    () => createClock({ environment: 'staging', eventName: 'schedule', simulatedNow: '2026-07-25T07:00:00Z' }),
    /manually dispatched/,
  );
  const clock = createClock({
    environment: 'staging',
    eventName: 'workflow_dispatch',
    simulatedNow: '2026-07-25T07:00:00-07:00',
  });
  assert.equal(clock.now().toISOString(), '2026-07-25T14:00:00.000Z');
  assert.throws(() => createClock({ environment: 'preview' }), /staging or production/);
});

test('generic staging-only guard is reusable for later fault injection', () => {
  assert.doesNotThrow(() => assertStagingOnlyControl({
    environment: 'staging',
    eventName: 'workflow_dispatch',
    controlName: 'fault injection',
    enabled: true,
  }));
  assert.throws(() => assertStagingOnlyControl({
    environment: 'production',
    eventName: 'workflow_dispatch',
    controlName: 'fault injection',
    enabled: true,
  }), /forbidden/);
});

test('instant parser requires an explicit timezone', () => {
  assert.equal(parseInstant('2026-07-20T12:00:00+02:00').toISOString(), '2026-07-20T10:00:00.000Z');
  assert.throws(() => parseInstant('2026-07-20T12:00:00'), /explicit offset/);
  assert.throws(() => parseInstant('not-a-date'), /explicit offset/);
});

test('rollover plan transitions at the seven-day window and Pacific cutoff', () => {
  const waiting = planRollover({ targetCycle: '2026-08', now: '2026-07-25T06:59:59Z' });
  assert.equal(waiting.phase, ROLLOVER_PHASES.WAITING);
  assert.equal(waiting.preparationOpensAt.toISOString(), '2026-07-25T07:00:00.000Z');
  assert.equal(waiting.activationAt.toISOString(), '2026-08-01T07:00:00.000Z');

  assert.equal(
    planRollover({ targetCycle: '2026-08', now: '2026-07-25T07:00:00Z' }).phase,
    ROLLOVER_PHASES.PREPARE,
  );
  assert.equal(
    planRollover({ targetCycle: '2026-08', now: '2026-08-01T06:59:59Z' }).phase,
    ROLLOVER_PHASES.PREPARE,
  );
  const activation = planRollover({ targetCycle: '2026-08', now: '2026-08-01T07:00:00Z' });
  assert.equal(activation.phase, ROLLOVER_PHASES.ACTIVATE);
  assert.deepEqual(serializeRolloverPlan(activation), {
    phase: 'activate',
    observedAt: '2026-08-01T07:00:00.000Z',
    sourceCycle: '2026-07',
    targetCycle: '2026-08',
    preparationDays: 7,
    preparationOpensAt: '2026-07-25T07:00:00.000Z',
    activationAt: '2026-08-01T07:00:00.000Z',
    sourceDeadline: {
      cycle: '2026-07',
      label: '11:59pm Pacific, July 31st, 2026',
      cutoffAt: '2026-08-01T07:00:00.000Z',
    },
    targetDeadline: {
      cycle: '2026-08',
      label: '11:59pm Pacific, August 31st, 2026',
      cutoffAt: '2026-09-01T07:00:00.000Z',
    },
  });
});
