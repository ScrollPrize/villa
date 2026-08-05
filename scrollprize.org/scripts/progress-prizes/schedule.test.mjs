import assert from 'node:assert/strict';
import test from 'node:test';

import {
  PROGRESS_PRIZE_SCHEDULES,
  SCHEDULE_CRONS,
  SCHEDULE_OPERATIONS,
  SCHEDULE_REASONS,
  planScheduleDispatch,
  serializeScheduleDispatch,
} from './schedule.mjs';

function plan(now, scheduledCron, eventName = 'schedule') {
  return planScheduleDispatch({ now, eventName, scheduledCron });
}

test('schedule constants use the four exact Pacific cron slots', () => {
  assert.deepEqual(SCHEDULE_CRONS, {
    PREPARE: '17 6 * * *',
    PRE_CUTOFF: '40 23 28-31 * *',
    EARLY_RECOVERY: '17 0 1 * *',
    LATE_RECOVERY: '47 6 1 * *',
  });
  assert.equal(PROGRESS_PRIZE_SCHEDULES, SCHEDULE_CRONS);
});

test('only the preparation slot dispatches during the seven-day window', () => {
  const preparation = plan('2026-07-28T13:17:00Z', SCHEDULE_CRONS.PREPARE);
  assert.equal(preparation.dispatch, true);
  assert.equal(preparation.operation, SCHEDULE_OPERATIONS.PREPARE);
  assert.equal(preparation.reason, SCHEDULE_REASONS.PREPARATION);
  assert.equal(preparation.sourceCycle, '2026-07');
  assert.equal(preparation.targetCycle, '2026-08');

  const candidateActivation = plan(
    '2026-07-29T06:40:00Z',
    SCHEDULE_CRONS.PRE_CUTOFF,
  );
  assert.equal(candidateActivation.dispatch, false);
  assert.equal(candidateActivation.operation, SCHEDULE_OPERATIONS.NONE);
  assert.equal(candidateActivation.reason, SCHEDULE_REASONS.WRONG_SLOT);

  for (const cron of [SCHEDULE_CRONS.EARLY_RECOVERY, SCHEDULE_CRONS.LATE_RECOVERY]) {
    const wrongSlot = plan('2026-07-28T13:17:00Z', cron);
    assert.equal(wrongSlot.dispatch, false);
    assert.equal(wrongSlot.reason, SCHEDULE_REASONS.WRONG_SLOT);
  }
});

test('scheduled planning has exact seven-day preparation boundaries', () => {
  const before = planScheduleDispatch({
    now: '2026-07-25T06:59:59.999Z',
    eventName: 'schedule',
    scheduledCron: SCHEDULE_CRONS.PREPARE,
  });
  assert.equal(before.dispatch, false);
  assert.equal(before.reason, SCHEDULE_REASONS.OUTSIDE_WINDOW);

  const opens = planScheduleDispatch({
    now: '2026-07-25T07:00:00Z',
    eventName: 'schedule',
    scheduledCron: SCHEDULE_CRONS.PREPARE,
  });
  assert.equal(opens.operation, SCHEDULE_OPERATIONS.PREPARE);
  assert.equal(opens.preparationOpensAt.toISOString(), '2026-07-25T07:00:00.000Z');
  assert.equal(opens.activationAt.toISOString(), '2026-08-01T07:00:00.000Z');

  const lastPreparationInstant = planScheduleDispatch({
    now: '2026-08-01T06:39:59.999Z',
    eventName: 'schedule',
    scheduledCron: SCHEDULE_CRONS.PREPARE,
  });
  assert.equal(lastPreparationInstant.operation, SCHEDULE_OPERATIONS.PREPARE);
});

test('preparation boundaries follow Pacific calendar days in PST and leap February', () => {
  const november = plan('2026-11-24T14:17:00Z', SCHEDULE_CRONS.PREPARE);
  assert.equal(november.operation, SCHEDULE_OPERATIONS.PREPARE);
  assert.equal(november.sourceCycle, '2026-11');
  assert.equal(november.targetCycle, '2026-12');
  assert.equal(november.preparationOpensAt.toISOString(), '2026-11-24T08:00:00.000Z');

  const ordinaryFebruary = plan('2027-02-22T14:17:00Z', SCHEDULE_CRONS.PREPARE);
  assert.equal(ordinaryFebruary.operation, SCHEDULE_OPERATIONS.PREPARE);
  assert.equal(ordinaryFebruary.preparationOpensAt.toISOString(), '2027-02-22T08:00:00.000Z');

  const leapBefore = plan('2028-02-22T14:17:00Z', SCHEDULE_CRONS.PREPARE);
  assert.equal(leapBefore.operation, SCHEDULE_OPERATIONS.NONE);
  const leapOpens = plan('2028-02-23T14:17:00Z', SCHEDULE_CRONS.PREPARE);
  assert.equal(leapOpens.operation, SCHEDULE_OPERATIONS.PREPARE);
  assert.equal(leapOpens.preparationOpensAt.toISOString(), '2028-02-23T08:00:00.000Z');
});

test('the final-day 23:40 slot dispatches before both PDT and PST cutoffs', () => {
  const pdt = plan('2026-08-01T06:40:00Z', SCHEDULE_CRONS.PRE_CUTOFF);
  assert.equal(pdt.operation, SCHEDULE_OPERATIONS.ACTIVATE);
  assert.equal(pdt.reason, SCHEDULE_REASONS.PRE_CUTOFF);
  assert.equal(pdt.sourceCycle, '2026-07');
  assert.equal(pdt.targetCycle, '2026-08');
  assert.equal(pdt.activationAt.toISOString(), '2026-08-01T07:00:00.000Z');

  const pst = plan('2027-01-01T07:40:00Z', SCHEDULE_CRONS.PRE_CUTOFF);
  assert.equal(pst.operation, SCHEDULE_OPERATIONS.ACTIVATE);
  assert.equal(pst.reason, SCHEDULE_REASONS.PRE_CUTOFF);
  assert.equal(pst.sourceCycle, '2026-12');
  assert.equal(pst.targetCycle, '2027-01');
  assert.equal(pst.activationAt.toISOString(), '2027-01-01T08:00:00.000Z');
});

test('candidate pre-cutoff dates validate the actual last Pacific day', () => {
  const july28 = plan('2026-07-29T06:40:00Z', SCHEDULE_CRONS.PRE_CUTOFF);
  assert.equal(july28.dispatch, false);

  const april30 = plan('2027-05-01T06:40:00Z', SCHEDULE_CRONS.PRE_CUTOFF);
  assert.equal(april30.dispatch, true);
  assert.equal(april30.sourceCycle, '2027-04');
  assert.equal(april30.targetCycle, '2027-05');
});

test('leap February uses February 29 while non-leap February uses February 28', () => {
  const leapCandidate = plan('2028-02-29T07:40:00Z', SCHEDULE_CRONS.PRE_CUTOFF);
  assert.equal(leapCandidate.dispatch, false);

  const leapLastDay = plan('2028-03-01T07:40:00Z', SCHEDULE_CRONS.PRE_CUTOFF);
  assert.equal(leapLastDay.dispatch, true);
  assert.equal(leapLastDay.sourceCycle, '2028-02');
  assert.equal(leapLastDay.targetCycle, '2028-03');
  assert.equal(leapLastDay.activationAt.toISOString(), '2028-03-01T08:00:00.000Z');

  const ordinaryLastDay = plan('2027-03-01T07:40:00Z', SCHEDULE_CRONS.PRE_CUTOFF);
  assert.equal(ordinaryLastDay.dispatch, true);
  assert.equal(ordinaryLastDay.sourceCycle, '2027-02');
  assert.equal(ordinaryLastDay.targetCycle, '2027-03');
});

test('first-day recovery slots dispatch at 00:17 and 06:47 Pacific', () => {
  const early = plan('2026-08-01T07:17:00Z', SCHEDULE_CRONS.EARLY_RECOVERY);
  assert.equal(early.operation, SCHEDULE_OPERATIONS.ACTIVATE);
  assert.equal(early.reason, SCHEDULE_REASONS.EARLY_RECOVERY);
  assert.equal(early.sourceCycle, '2026-07');
  assert.equal(early.targetCycle, '2026-08');

  const late = plan('2026-08-01T13:47:00Z', SCHEDULE_CRONS.LATE_RECOVERY);
  assert.equal(late.operation, SCHEDULE_OPERATIONS.ACTIVATE);
  assert.equal(late.reason, SCHEDULE_REASONS.LATE_RECOVERY);
  assert.equal(late.sourceCycle, '2026-07');
  assert.equal(late.targetCycle, '2026-08');
});

test('delayed activation events recover by their actual first-day Pacific window', () => {
  const delayedPreCutoff = plan('2026-08-01T07:43:00Z', SCHEDULE_CRONS.PRE_CUTOFF);
  assert.equal(delayedPreCutoff.operation, SCHEDULE_OPERATIONS.ACTIVATE);
  assert.equal(delayedPreCutoff.reason, SCHEDULE_REASONS.EARLY_RECOVERY);

  const earlyDelayedPastLateSlot = plan(
    '2026-08-01T16:05:00Z',
    SCHEDULE_CRONS.EARLY_RECOVERY,
  );
  assert.equal(earlyDelayedPastLateSlot.operation, SCHEDULE_OPERATIONS.ACTIVATE);
  assert.equal(earlyDelayedPastLateSlot.reason, SCHEDULE_REASONS.LATE_RECOVERY);

  const lateDelayedToEvening = plan(
    '2026-08-02T05:59:59Z',
    SCHEDULE_CRONS.LATE_RECOVERY,
  );
  assert.equal(lateDelayedToEvening.operation, SCHEDULE_OPERATIONS.ACTIVATE);
  assert.equal(lateDelayedToEvening.reason, SCHEDULE_REASONS.LATE_RECOVERY);

  const preCutoffDelayedToEndOfFirstDay = plan(
    '2026-08-02T06:59:59Z',
    SCHEDULE_CRONS.PRE_CUTOFF,
  );
  assert.equal(preCutoffDelayedToEndOfFirstDay.operation, SCHEDULE_OPERATIONS.ACTIVATE);
  assert.equal(preCutoffDelayedToEndOfFirstDay.sourceCycle, '2026-07');
});

test('day-two delays always no-op instead of selecting a stale cycle', () => {
  for (const cron of [
    SCHEDULE_CRONS.PRE_CUTOFF,
    SCHEDULE_CRONS.EARLY_RECOVERY,
    SCHEDULE_CRONS.LATE_RECOVERY,
  ]) {
    const dayTwo = plan('2026-08-02T07:17:00Z', cron);
    assert.equal(dayTwo.dispatch, false, cron);
    assert.equal(dayTwo.operation, SCHEDULE_OPERATIONS.NONE, cron);
    assert.equal(dayTwo.reason, SCHEDULE_REASONS.OUTSIDE_WINDOW, cron);
    assert.equal(dayTwo.sourceCycle, '2026-08');
    assert.equal(dayTwo.targetCycle, '2026-09');
  }
});

test('manual dispatch is always a real-clock dry-run with consecutive Pacific cycles', () => {
  const preparation = planScheduleDispatch({
    now: '2026-07-28T13:17:00Z',
    eventName: 'workflow_dispatch',
  });
  assert.equal(preparation.dispatch, true);
  assert.equal(preparation.operation, SCHEDULE_OPERATIONS.DRY_RUN);
  assert.equal(preparation.reason, SCHEDULE_REASONS.MANUAL_DRY_RUN);
  assert.equal(preparation.sourceCycle, '2026-07');
  assert.equal(preparation.targetCycle, '2026-08');

  const activation = planScheduleDispatch({
    now: '2026-08-01T20:00:00Z',
    eventName: 'workflow_dispatch',
  });
  assert.equal(activation.operation, SCHEDULE_OPERATIONS.DRY_RUN);
  assert.equal(activation.reason, SCHEDULE_REASONS.MANUAL_DRY_RUN);
  assert.equal(activation.sourceCycle, '2026-08');
  assert.equal(activation.targetCycle, '2026-09');

  const noOp = planScheduleDispatch({
    now: '2026-08-02T20:00:00Z',
    eventName: 'workflow_dispatch',
  });
  assert.equal(noOp.operation, SCHEDULE_OPERATIONS.DRY_RUN);
  assert.equal(noOp.sourceCycle, '2026-08');
  assert.equal(noOp.targetCycle, '2026-09');
});

test('unknown trigger contexts and cron claims fail closed', () => {
  assert.throws(
    () => planScheduleDispatch({ now: '2026-07-28T13:17:00Z', eventName: 'pull_request' }),
    /eventName/,
  );
  assert.throws(
    () => plan('2026-07-28T13:17:00Z', '0 0 * * *'),
    /allowlist/,
  );
  assert.throws(
    () => planScheduleDispatch({
      now: '2026-07-28T13:17:00Z',
      eventName: 'workflow_dispatch',
      scheduledCron: SCHEDULE_CRONS.PREPARE,
    }),
    /must not claim/,
  );
  assert.throws(
    () => planScheduleDispatch({ now: '2026-07-28T13:17:00', eventName: 'workflow_dispatch' }),
    /explicit offset/,
  );
});

test('serialized plans expose only stable public control fields', () => {
  const serialized = serializeScheduleDispatch(plan(
    '2026-08-01T06:40:00Z',
    SCHEDULE_CRONS.PRE_CUTOFF,
  ));
  assert.deepEqual(serialized, {
    dispatch: true,
    operation: 'activate',
    reason: 'pre-cutoff',
    eventName: 'schedule',
    scheduledCron: '40 23 28-31 * *',
    timeZone: 'America/Los_Angeles',
    observedAt: '2026-08-01T06:40:00.000Z',
    sourceCycle: '2026-07',
    targetCycle: '2026-08',
    preparationOpensAt: '2026-07-25T07:00:00.000Z',
    activationAt: '2026-08-01T07:00:00.000Z',
  });
});
