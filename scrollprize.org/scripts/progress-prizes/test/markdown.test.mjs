import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

import {
  PROGRESS_PRIZE_MARKERS,
  assertPublicResponderUri,
  parseProgressPrizeMarkdown,
  updateProgressPrizeMarkdown,
} from '../index.mjs';

const oldUri = 'https://forms.gle/OldForm_123';
const newUri = 'https://docs.google.com/forms/d/e/NewForm_456/viewform';

function fixture({ newline = '\n', deadline = '11:59pm Pacific, July 31st, 2026', uri = oldUri } = {}) {
  return [
    '# Prizes',
    '',
    '[Submission Form](https://forms.gle/UnrelatedForm)',
    '',
    '## Progress Prizes',
    '',
    PROGRESS_PRIZE_MARKERS.deadlineStart,
    `Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is ${deadline}!`,
    PROGRESS_PRIZE_MARKERS.deadlineEnd,
    '',
    '<details>criteria</details>',
    '',
    PROGRESS_PRIZE_MARKERS.formStart,
    `[Submission Form](${uri})`,
    PROGRESS_PRIZE_MARKERS.formEnd,
    '',
    '## Terms and Conditions',
    '',
    '[Submission Form](https://forms.gle/AnotherUnrelatedForm)',
    '',
  ].join(newline);
}

test('repository prize page has one valid managed Progress Prize pair', async () => {
  const here = dirname(fileURLToPath(import.meta.url));
  const page = await readFile(resolve(here, '../../../docs/34_prizes.md'), 'utf8');
  const parsed = parseProgressPrizeMarkdown(page);
  assert.match(parsed.cycle, /^\d{4}-(?:0[1-9]|1[0-2])$/);
  assert.equal(assertPublicResponderUri(parsed.responderUri), parsed.responderUri);
});

test('strict update changes only managed content and preserves CRLF', () => {
  const source = fixture({ newline: '\r\n' });
  const result = updateProgressPrizeMarkdown(source, {
    targetCycle: '2026-08',
    responderUri: newUri,
    expectedCurrentResponderUri: oldUri,
  });
  assert.equal(result.changed, true);
  assert.equal(parseProgressPrizeMarkdown(result.content).cycle, '2026-08');
  assert.match(result.content, /August 31st, 2026/);
  assert.match(result.content, /UnrelatedForm/);
  assert.match(result.content, /AnotherUnrelatedForm/);
  assert.equal(result.content.replaceAll('\r\n', '').includes('\n'), false);

  const changedLines = source.split('\r\n').filter((line, index) => line !== result.content.split('\r\n')[index]);
  assert.deepEqual(changedLines, [
    'Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is 11:59pm Pacific, July 31st, 2026!',
    `[Submission Form](${oldUri})`,
  ]);
});

test('updating an already-current page is idempotent', () => {
  const first = updateProgressPrizeMarkdown(fixture(), {
    targetCycle: '2026-08',
    responderUri: newUri,
    expectedCurrentResponderUri: oldUri,
  });
  const second = updateProgressPrizeMarkdown(first.content, {
    targetCycle: '2026-08',
    responderUri: newUri,
    expectedCurrentResponderUri: oldUri,
  });
  assert.equal(second.changed, false);
  assert.equal(second.content, first.content);
});

test('updater rejects stale state, skipped cycles, and inconsistent idempotency', () => {
  assert.throws(() => updateProgressPrizeMarkdown(fixture(), {
    targetCycle: '2026-08',
    responderUri: newUri,
    expectedCurrentResponderUri: 'https://forms.gle/DifferentSource',
  }), /does not match/);
  assert.throws(() => updateProgressPrizeMarkdown(fixture(), {
    targetCycle: '2026-09',
    responderUri: newUri,
    expectedCurrentResponderUri: oldUri,
  }), /Expected current cycle 2026-08/);
  assert.throws(() => updateProgressPrizeMarkdown(fixture({
    deadline: '11:59pm Pacific, August 31st, 2026',
  }), {
    targetCycle: '2026-08',
    responderUri: newUri,
  }), /different responder URI/);
});

test('parser rejects missing, duplicate, misplaced, and multiline markers', () => {
  assert.throws(
    () => parseProgressPrizeMarkdown(fixture().replace(PROGRESS_PRIZE_MARKERS.formEnd, '')),
    /found 0/,
  );
  assert.throws(
    () => parseProgressPrizeMarkdown(`${fixture()}${PROGRESS_PRIZE_MARKERS.deadlineStart}`),
    /found 2/,
  );
  assert.throws(
    () => parseProgressPrizeMarkdown(fixture().replace(
      PROGRESS_PRIZE_MARKERS.deadlineStart,
      `${PROGRESS_PRIZE_MARKERS.deadlineStart}\nextra line`,
    )),
    /exactly one/,
  );
  assert.throws(
    () => parseProgressPrizeMarkdown(fixture().replace('## Progress Prizes', '## Other Prizes')),
    /found 0/,
  );
});

test('parser rejects noncanonical or non-month-end deadlines', () => {
  assert.throws(
    () => parseProgressPrizeMarkdown(fixture({ deadline: '11:59pm Pacific, July 30th, 2026' })),
    /last day/,
  );
  assert.throws(
    () => parseProgressPrizeMarkdown(fixture({ deadline: '11:59 PM Pacific, July 31st, 2026' })),
    /unexpected format/,
  );
  assert.throws(
    () => parseProgressPrizeMarkdown(fixture({ deadline: '11:59pm Pacific, July 31th, 2026' })),
    /last day/,
  );
});

test('only public responder URLs are accepted', () => {
  assert.equal(assertPublicResponderUri(oldUri), oldUri);
  assert.equal(assertPublicResponderUri(`${newUri}?usp=sf_link`), `${newUri}?usp=sf_link`);
  for (const uri of [
    'http://forms.gle/OldForm',
    'https://docs.google.com/forms/d/e/EditorForm/edit',
    'https://docs.google.com/forms/d/private-editor-id/viewform',
    'https://evil.example/forms/viewform',
    'https://forms.gle/OldForm#token',
  ]) {
    assert.throws(() => assertPublicResponderUri(uri), /responderUri/);
  }
});
