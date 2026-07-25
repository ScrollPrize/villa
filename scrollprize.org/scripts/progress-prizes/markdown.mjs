import {
  cycleFromDeadlineParts,
  getCycleDeadline,
  previousCycle,
} from './dates.mjs';

export const PROGRESS_PRIZE_MARKERS = Object.freeze({
  deadlineStart: '{/* progress-prizes:deadline:start */}',
  deadlineEnd: '{/* progress-prizes:deadline:end */}',
  formStart: '{/* progress-prizes:form:start */}',
  formEnd: '{/* progress-prizes:form:end */}',
});

const DEADLINE_PREFIX = 'Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is ';

function findAll(haystack, needle) {
  const offsets = [];
  let offset = 0;
  while ((offset = haystack.indexOf(needle, offset)) !== -1) {
    offsets.push(offset);
    offset += needle.length;
  }
  return offsets;
}

function uniqueOffset(markdown, marker) {
  const offsets = findAll(markdown, marker);
  if (offsets.length !== 1) {
    throw new Error(`Expected exactly one ${marker} marker, found ${offsets.length}`);
  }
  return offsets[0];
}

function locateProgressPrizeSection(markdown) {
  const headings = [...markdown.matchAll(/^## Progress Prizes\r?$/gm)];
  if (headings.length !== 1) {
    throw new Error(`Expected exactly one Progress Prizes section, found ${headings.length}`);
  }
  const start = headings[0].index;
  const remainderStart = start + headings[0][0].length;
  const nextHeading = /^## [^#].*\r?$/gm;
  nextHeading.lastIndex = remainderStart;
  const match = nextHeading.exec(markdown);
  return Object.freeze({ start, end: match?.index ?? markdown.length });
}

function detectNewline(markdown) {
  const withoutCrlf = markdown.replaceAll('\r\n', '');
  if (withoutCrlf.includes('\r')) {
    throw new Error('Markdown contains unsupported bare carriage returns');
  }
  return markdown.includes('\r\n') ? '\r\n' : '\n';
}

function readManagedLine(markdown, startMarker, endMarker, section, newline) {
  const start = uniqueOffset(markdown, startMarker);
  const end = uniqueOffset(markdown, endMarker);
  if (start < section.start || end > section.end || end <= start) {
    throw new Error(`${startMarker} and ${endMarker} must be ordered inside the Progress Prizes section`);
  }

  const contentStart = start + startMarker.length;
  if (!markdown.startsWith(newline, contentStart) || !markdown.startsWith(newline, end - newline.length)) {
    throw new Error(`Managed content between ${startMarker} and ${endMarker} must be exactly one line`);
  }
  const lineStart = contentStart + newline.length;
  const lineEnd = end - newline.length;
  const line = markdown.slice(lineStart, lineEnd);
  if (!line || line.includes('\n') || line.includes('\r')) {
    throw new Error(`Managed content between ${startMarker} and ${endMarker} must be exactly one non-empty line`);
  }

  return Object.freeze({ start, end: end + endMarker.length, lineStart, lineEnd, line });
}

function parseDeadlineLine(line) {
  const match = new RegExp(
    `^${DEADLINE_PREFIX.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}11:59pm Pacific, ([A-Z][a-z]+) (\\d{1,2}(?:st|nd|rd|th)), (\\d{4})!$`,
  ).exec(line);
  if (!match) {
    throw new Error('Managed deadline line has an unexpected format');
  }

  const cycle = cycleFromDeadlineParts({
    monthName: match[1],
    ordinal: match[2],
    year: Number(match[3]),
  });
  const deadline = getCycleDeadline(cycle);
  if (line !== `${DEADLINE_PREFIX}${deadline.label}!`) {
    throw new Error('Managed deadline line is not canonical');
  }
  return deadline;
}

export function assertPublicResponderUri(responderUri) {
  if (typeof responderUri !== 'string') {
    throw new TypeError('responderUri must be a string');
  }

  let url;
  try {
    url = new URL(responderUri);
  } catch {
    throw new TypeError('responderUri must be a valid URL');
  }

  if (url.protocol !== 'https:' || url.username || url.password || url.hash) {
    throw new TypeError('responderUri must be a public HTTPS Google Forms URL without credentials or a fragment');
  }

  const isShortUrl = url.hostname === 'forms.gle' && /^\/[A-Za-z0-9_-]+\/?$/.test(url.pathname);
  const isCanonicalUrl = url.hostname === 'docs.google.com'
    && /^\/forms\/d\/e\/[A-Za-z0-9_-]+\/viewform\/?$/.test(url.pathname);
  if (!isShortUrl && !isCanonicalUrl) {
    throw new TypeError('responderUri must use forms.gle or the public docs.google.com Forms view URL');
  }

  return url.toString();
}

export function assertCanonicalPublicResponderUri(responderUri) {
  const normalized = assertPublicResponderUri(responderUri);
  const url = new URL(normalized);
  if (url.search !== '') {
    throw new TypeError('responderUri must not contain query parameters');
  }
  return normalized;
}

function parseFormLine(line) {
  const match = /^\[Submission Form\]\((https:\/\/[^\s)]+)\)$/.exec(line);
  if (!match) {
    throw new Error('Managed form line has an unexpected format');
  }
  return assertCanonicalPublicResponderUri(match[1]);
}

export function parseProgressPrizeMarkdown(markdown) {
  if (typeof markdown !== 'string') {
    throw new TypeError('markdown must be a string');
  }

  const newline = detectNewline(markdown);
  const section = locateProgressPrizeSection(markdown);
  const deadlineRegion = readManagedLine(
    markdown,
    PROGRESS_PRIZE_MARKERS.deadlineStart,
    PROGRESS_PRIZE_MARKERS.deadlineEnd,
    section,
    newline,
  );
  const formRegion = readManagedLine(
    markdown,
    PROGRESS_PRIZE_MARKERS.formStart,
    PROGRESS_PRIZE_MARKERS.formEnd,
    section,
    newline,
  );
  if (deadlineRegion.end >= formRegion.start) {
    throw new Error('Managed deadline must appear before the managed form link');
  }

  const deadline = parseDeadlineLine(deadlineRegion.line);
  const responderUri = parseFormLine(formRegion.line);
  return Object.freeze({
    cycle: deadline.cycle,
    deadline,
    responderUri,
    newline,
    section,
    regions: Object.freeze({ deadline: deadlineRegion, form: formRegion }),
  });
}

function replaceLine(markdown, region, line) {
  return `${markdown.slice(0, region.lineStart)}${line}${markdown.slice(region.lineEnd)}`;
}

export function updateProgressPrizeMarkdown(markdown, {
  targetCycle,
  responderUri,
  expectedCurrentCycle = previousCycle(targetCycle),
  expectedCurrentResponderUri,
} = {}) {
  const current = parseProgressPrizeMarkdown(markdown);
  const targetDeadline = getCycleDeadline(targetCycle);
  const normalizedResponderUri = assertCanonicalPublicResponderUri(responderUri);
  const next = Object.freeze({
    cycle: targetCycle,
    deadline: targetDeadline,
    responderUri: normalizedResponderUri,
  });

  if (current.cycle === targetCycle) {
    if (current.responderUri !== normalizedResponderUri) {
      throw new Error(`Managed page already has cycle ${targetCycle} with a different responder URI`);
    }
    return Object.freeze({ content: markdown, changed: false, current, next });
  }

  if (current.cycle !== expectedCurrentCycle) {
    throw new Error(`Expected current cycle ${expectedCurrentCycle}, found ${current.cycle}`);
  }
  if (
    expectedCurrentResponderUri !== undefined
    && current.responderUri !== assertPublicResponderUri(expectedCurrentResponderUri)
  ) {
    throw new Error('Current responder URI does not match the expected source form');
  }

  const deadlineLine = `${DEADLINE_PREFIX}${targetDeadline.label}!`;
  const formLine = `[Submission Form](${normalizedResponderUri})`;
  let content = replaceLine(markdown, current.regions.form, formLine);
  content = replaceLine(content, current.regions.deadline, deadlineLine);
  const parsed = parseProgressPrizeMarkdown(content);
  if (parsed.cycle !== targetCycle || parsed.responderUri !== normalizedResponderUri) {
    throw new Error('Updated Markdown failed post-write validation');
  }

  return Object.freeze({ content, changed: content !== markdown, current, next });
}
