import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, open, rename, rm, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { AUTOMATION_ERROR_FALLBACK } from './automation-cli.mjs';
import { readBoundedDiagnostic } from './error-diagnostic-cli.mjs';

const cli = fileURLToPath(new URL('./error-diagnostic-cli.mjs', import.meta.url));

function runDiagnostic(file, runnerTemp) {
  return spawnSync(process.execPath, [cli, file], {
    encoding: 'utf8',
    env: { ...process.env, RUNNER_TEMP: runnerTemp },
  });
}

test('diagnostic reader emits one validated line and rejects every unsafe file shape', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'progress-prize-diagnostic-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const file = join(directory, 'progress-prizes-error.txt');
  const safe = 'progress-prizes: Google API request failed (drive.files.get): HTTP 403';

  await writeFile(file, `${safe}\n`, 'utf8');
  let child = runDiagnostic(file, directory);
  assert.equal(child.status, 0);
  assert.equal(child.stdout, '');
  assert.equal(child.stderr, `${safe}\n`);

  for (const unsafe of [
    `progress-prizes: ${'private'.repeat(1_000)}\n`,
    `${safe}\nprivate-second-line\n`,
    `${safe}\r\n`,
    'not-progress-prizes: private\n',
  ]) {
    await writeFile(file, unsafe, 'utf8');
    child = runDiagnostic(file, directory);
    assert.equal(child.status, 0);
    assert.equal(child.stdout, '');
    assert.equal(child.stderr, `${AUTOMATION_ERROR_FALLBACK}\n`);
    assert.equal(child.stderr.includes('private-second-line'), false);
  }

  const outside = join(directory, 'outside.txt');
  await writeFile(outside, `${safe}\n`, 'utf8');
  child = runDiagnostic(outside, directory);
  assert.equal(child.stderr, `${AUTOMATION_ERROR_FALLBACK}\n`);
});

test('diagnostic reader rejects same-size replacement and symlink races', async (t) => {
  const directory = await mkdtemp(join(tmpdir(), 'progress-prize-diagnostic-race-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const file = join(directory, 'progress-prizes-error.txt');
  const replacement = join(directory, 'replacement.txt');
  const outside = join(directory, 'outside.txt');
  const originalLine = `progress-prizes: ${'A'.repeat(32)}\n`;
  const replacementLine = `progress-prizes: ${'B'.repeat(32)}\n`;

  await writeFile(file, originalLine, 'utf8');
  await writeFile(replacement, replacementLine, 'utf8');
  let replaced = false;
  const replacedDiagnostic = await readBoundedDiagnostic(file, directory, {
    openFile: async (path, flags) => {
      if (!replaced) {
        replaced = true;
        await rename(replacement, path);
      }
      return open(path, flags);
    },
  });
  assert.equal(replacedDiagnostic, AUTOMATION_ERROR_FALLBACK);

  await writeFile(file, originalLine, 'utf8');
  await writeFile(outside, replacementLine, 'utf8');
  let linked = false;
  const linkedDiagnostic = await readBoundedDiagnostic(file, directory, {
    openFile: async (path, flags) => {
      if (!linked) {
        linked = true;
        await rm(path);
        await symlink(outside, path);
      }
      return open(path, flags);
    },
  });
  assert.equal(linkedDiagnostic, AUTOMATION_ERROR_FALLBACK);
});
