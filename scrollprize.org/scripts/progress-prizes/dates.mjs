const MONTH_NAMES = Object.freeze([
  'January',
  'February',
  'March',
  'April',
  'May',
  'June',
  'July',
  'August',
  'September',
  'October',
  'November',
  'December',
]);

export const PACIFIC_TIME_ZONE = 'America/Los_Angeles';

const pacificFormatter = new Intl.DateTimeFormat('en-US', {
  timeZone: PACIFIC_TIME_ZONE,
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
  hourCycle: 'h23',
});

function utcEpoch({ year, month, day, hour = 0, minute = 0, second = 0 }) {
  const date = new Date(0);
  date.setUTCFullYear(year, month - 1, day);
  date.setUTCHours(hour, minute, second, 0);
  return date.getTime();
}

function assertIntegerInRange(value, name, minimum, maximum) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new TypeError(`${name} must be an integer from ${minimum} through ${maximum}`);
  }
}

export function parseCycle(cycle) {
  if (typeof cycle !== 'string') {
    throw new TypeError('cycle must be a string in YYYY-MM format');
  }

  const match = /^(\d{4})-(0[1-9]|1[0-2])$/.exec(cycle);
  if (!match || match[1] === '0000') {
    throw new TypeError(`Invalid cycle ${JSON.stringify(cycle)}; expected YYYY-MM`);
  }

  return Object.freeze({
    cycle,
    year: Number(match[1]),
    month: Number(match[2]),
  });
}

export function formatCycle(year, month) {
  assertIntegerInRange(year, 'year', 1, 9999);
  assertIntegerInRange(month, 'month', 1, 12);
  return `${String(year).padStart(4, '0')}-${String(month).padStart(2, '0')}`;
}

export function addMonths(cycle, amount) {
  const { year, month } = parseCycle(cycle);
  if (!Number.isInteger(amount)) {
    throw new TypeError('month offset must be an integer');
  }

  const zeroBased = year * 12 + month - 1 + amount;
  const nextYear = Math.floor(zeroBased / 12);
  const nextMonth = ((zeroBased % 12) + 12) % 12 + 1;
  return formatCycle(nextYear, nextMonth);
}

export function previousCycle(cycle) {
  return addMonths(cycle, -1);
}

export function nextCycle(cycle) {
  return addMonths(cycle, 1);
}

export function daysInMonth(year, month) {
  assertIntegerInRange(year, 'year', 1, 9999);
  assertIntegerInRange(month, 'month', 1, 12);
  const date = new Date(0);
  date.setUTCFullYear(year, month, 0);
  return date.getUTCDate();
}

export function ordinalDay(day) {
  assertIntegerInRange(day, 'day', 1, 31);
  const finalTwoDigits = day % 100;
  if (finalTwoDigits >= 11 && finalTwoDigits <= 13) {
    return `${day}th`;
  }

  return `${day}${({ 1: 'st', 2: 'nd', 3: 'rd' })[day % 10] ?? 'th'}`;
}

export function getPacificParts(instant) {
  const date = instant instanceof Date ? instant : new Date(instant);
  if (Number.isNaN(date.getTime())) {
    throw new TypeError('instant must be a valid Date or timestamp');
  }

  const parts = Object.fromEntries(
    pacificFormatter
      .formatToParts(date)
      .filter(({ type }) => type !== 'literal')
      .map(({ type, value }) => [type, Number(value)]),
  );

  return Object.freeze({
    year: parts.year,
    month: parts.month,
    day: parts.day,
    hour: parts.hour,
    minute: parts.minute,
    second: parts.second,
  });
}

export function pacificDateTimeToInstant(parts) {
  const { year, month, day, hour = 0, minute = 0, second = 0 } = parts ?? {};
  assertIntegerInRange(year, 'year', 1, 9999);
  assertIntegerInRange(month, 'month', 1, 12);
  assertIntegerInRange(day, 'day', 1, daysInMonth(year, month));
  assertIntegerInRange(hour, 'hour', 0, 23);
  assertIntegerInRange(minute, 'minute', 0, 59);
  assertIntegerInRange(second, 'second', 0, 59);

  const desiredWallTime = utcEpoch({ year, month, day, hour, minute, second });
  let candidate = desiredWallTime;

  // Converting a wall clock into an instant requires discovering the zone offset at
  // that instant. Iterating on the formatted delta handles both PST and PDT without
  // hard-coding either offset.
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const actual = getPacificParts(candidate);
    const delta = desiredWallTime - utcEpoch(actual);
    candidate += delta;
    if (delta === 0) {
      return new Date(candidate);
    }
  }

  const actual = getPacificParts(candidate);
  const wanted = { year, month, day, hour, minute, second };
  if (Object.keys(wanted).every((key) => wanted[key] === actual[key])) {
    return new Date(candidate);
  }

  throw new RangeError(`Pacific wall time does not exist: ${JSON.stringify(wanted)}`);
}

export function getCycleDeadline(cycle) {
  const { year, month } = parseCycle(cycle);
  const day = daysInMonth(year, month);
  const followingCycle = parseCycle(nextCycle(cycle));
  const displayAt = pacificDateTimeToInstant({ year, month, day, hour: 23, minute: 59 });
  const cutoffAt = pacificDateTimeToInstant({
    year: followingCycle.year,
    month: followingCycle.month,
    day: 1,
  });

  return Object.freeze({
    cycle,
    year,
    month,
    monthName: MONTH_NAMES[month - 1],
    day,
    ordinalDay: ordinalDay(day),
    label: `11:59pm Pacific, ${MONTH_NAMES[month - 1]} ${ordinalDay(day)}, ${year}`,
    displayAt,
    cutoffAt,
  });
}

export function cycleFromDeadlineParts({ year, monthName, ordinal }) {
  const month = MONTH_NAMES.indexOf(monthName) + 1;
  if (month === 0 || !Number.isInteger(year)) {
    throw new TypeError('Deadline contains an invalid month or year');
  }

  const cycle = formatCycle(year, month);
  const deadline = getCycleDeadline(cycle);
  if (deadline.ordinalDay !== ordinal) {
    throw new TypeError(`Deadline must be the last day of ${monthName} ${year}`);
  }
  return cycle;
}

export function subtractPacificCalendarDays(instant, dayCount) {
  if (!Number.isInteger(dayCount) || dayCount < 0) {
    throw new TypeError('dayCount must be a non-negative integer');
  }
  const local = getPacificParts(instant);
  const calendarDate = new Date(utcEpoch(local));
  calendarDate.setUTCDate(calendarDate.getUTCDate() - dayCount);
  return pacificDateTimeToInstant({
    year: calendarDate.getUTCFullYear(),
    month: calendarDate.getUTCMonth() + 1,
    day: calendarDate.getUTCDate(),
    hour: local.hour,
    minute: local.minute,
    second: local.second,
  });
}
