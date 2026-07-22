import test from 'node:test';
import assert from 'node:assert/strict';

import { parseCliArgs } from '../cli-args.mjs';
import { runCoreCli } from '../cli.mjs';
import { PROGRESS_PRIZE_MARKERS, parseProgressPrizeMarkdown } from '../index.mjs';

const sourceUri = 'https://forms.gle/SourceForm';
const targetUri = 'https://docs.google.com/forms/d/e/TargetForm/viewform';
const source = [
  '## Progress Prizes',
  PROGRESS_PRIZE_MARKERS.deadlineStart,
  'Submissions are evaluated monthly, and multiple submissions/awards per month are permitted. The next deadline is 11:59pm Pacific, July 31st, 2026!',
  PROGRESS_PRIZE_MARKERS.deadlineEnd,
  PROGRESS_PRIZE_MARKERS.formStart,
  `[Submission Form](${sourceUri})`,
  PROGRESS_PRIZE_MARKERS.formEnd,
  '## Terms and Conditions',
].join('\n');

test('CLI parser rejects unknown, repeated, and valueless options', () => {
  assert.deepEqual(parseCliArgs(['validate', '--file', 'page.md']), {
    command: 'validate',
    options: { file: 'page.md' },
  });
  assert.throws(() => parseCliArgs(['unknown']), /Command/);
  assert.throws(() => parseCliArgs(['validate', '--wat']), /Unknown option/);
  assert.throws(() => parseCliArgs(['validate', '--file']), /requires a value/);
  assert.throws(() => parseCliArgs(['validate', '--file=a', '--file=b']), /only once/);
});

test('validate emits a machine-readable page summary', async () => {
  const outputs = [];
  const result = await runCoreCli(['validate', '--file', 'page.md'], {
    read: async () => source,
    output: (value) => outputs.push(value),
  });
  assert.equal(result.page.cycle, '2026-07');
  assert.deepEqual(JSON.parse(outputs[0]), result);
});

test('prepare-page dry run validates simulated clock without writing', async () => {
  const outputs = [];
  let writes = 0;
  const result = await runCoreCli([
    'prepare-page',
    '--file', 'page.md',
    '--target-cycle', '2026-08',
    '--responder-uri', targetUri,
    '--expected-current-responder-uri', sourceUri,
    '--environment', 'staging',
    '--event-name', 'workflow_dispatch',
    '--simulated-now', '2026-07-25T07:00:00Z',
    '--dry-run',
  ], {
    read: async () => source,
    write: async () => { writes += 1; },
    output: (value) => outputs.push(value),
  });
  assert.equal(result.changed, true);
  assert.equal(result.wroteFile, false);
  assert.equal(result.rollover.phase, 'prepare');
  assert.equal(writes, 0);
  assert.deepEqual(JSON.parse(outputs[0]), result);
});

test('prepare-page writes validated content outside dry-run', async () => {
  let written;
  const result = await runCoreCli([
    'prepare-page',
    '--file=page.md',
    '--target-cycle=2026-08',
    `--responder-uri=${targetUri}`,
    `--expected-current-responder-uri=${sourceUri}`,
  ], {
    read: async () => source,
    write: async (_file, content) => { written = content; },
    output: () => {},
  });
  assert.equal(result.wroteFile, true);
  assert.equal(parseProgressPrizeMarkdown(written).cycle, '2026-08');
});

test('prepare-page rejects a production simulated clock before writing', async () => {
  let writes = 0;
  await assert.rejects(() => runCoreCli([
    'prepare-page',
    '--target-cycle', '2026-08',
    '--responder-uri', targetUri,
    '--expected-current-responder-uri', sourceUri,
    '--environment', 'production',
    '--event-name', 'workflow_dispatch',
    '--simulated-now', '2026-07-25T07:00:00Z',
  ], {
    read: async () => source,
    write: async () => { writes += 1; },
    output: () => {},
  }), /forbidden in production/);
  assert.equal(writes, 0);
});
