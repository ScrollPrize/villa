import test from 'node:test';
import assert from 'node:assert/strict';

import {
  addMonths,
  daysInMonth,
  getCycleDeadline,
  nextCycle,
  ordinalDay,
  pacificDateTimeToInstant,
  parseCycle,
  previousCycle,
} from '../index.mjs';

test('cycle helpers are strict and cross year boundaries', () => {
  assert.deepEqual(parseCycle('2026-07'), { cycle: '2026-07', year: 2026, month: 7 });
  assert.equal(nextCycle('2026-12'), '2027-01');
  assert.equal(previousCycle('2026-01'), '2025-12');
  assert.equal(addMonths('2026-01', 14), '2027-03');
  for (const invalid of ['2026-7', '26-07', '2026-00', '2026-13', '0000-01', ' 2026-07']) {
    assert.throws(() => parseCycle(invalid), /Invalid cycle/);
  }
  assert.throws(() => addMonths('2026-01', 1.5), /integer/);
});

test('month lengths and ordinal suffixes cover leap years and teens', () => {
  assert.equal(daysInMonth(2024, 2), 29);
  assert.equal(daysInMonth(2026, 2), 28);
  assert.equal(daysInMonth(2026, 7), 31);
  assert.deepEqual(
    [1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 31].map(ordinalDay),
    ['1st', '2nd', '3rd', '4th', '11th', '12th', '13th', '21st', '22nd', '23rd', '31st'],
  );
});

test('cycle deadline displays 11:59pm but activates at Pacific midnight', () => {
  const july = getCycleDeadline('2026-07');
  assert.equal(july.label, '11:59pm Pacific, July 31st, 2026');
  assert.equal(july.displayAt.toISOString(), '2026-08-01T06:59:00.000Z');
  assert.equal(july.cutoffAt.toISOString(), '2026-08-01T07:00:00.000Z');

  const january = getCycleDeadline('2026-01');
  assert.equal(january.cutoffAt.toISOString(), '2026-02-01T08:00:00.000Z');

  const leapFebruary = getCycleDeadline('2024-02');
  assert.equal(leapFebruary.label, '11:59pm Pacific, February 29th, 2024');
  assert.equal(leapFebruary.cutoffAt.toISOString(), '2024-03-01T08:00:00.000Z');
});

test('Pacific conversion rejects a nonexistent DST wall time', () => {
  assert.throws(
    () => pacificDateTimeToInstant({ year: 2026, month: 3, day: 8, hour: 2, minute: 30 }),
    /does not exist/,
  );
});
