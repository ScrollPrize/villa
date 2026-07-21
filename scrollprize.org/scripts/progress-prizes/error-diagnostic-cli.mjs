#!/usr/bin/env node

import { Buffer } from 'node:buffer';
import { constants } from 'node:fs';
import { lstat, open } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  AUTOMATION_ERROR_FALLBACK,
  safeAutomationDiagnostic,
} from './automation-cli.mjs';

function sameFileIdentity(left, right) {
  return left.dev === right.dev && left.ino === right.ino;
}

function sameFileVersion(left, right) {
  return sameFileIdentity(left, right)
    && left.size === right.size
    && left.mtimeNs === right.mtimeNs
    && left.ctimeNs === right.ctimeNs;
}

export async function readBoundedDiagnostic(
  file,
  runnerTemp,
  { openFile = open } = {},
) {
  try {
    if (typeof file !== 'string' || typeof runnerTemp !== 'string' || runnerTemp === '') {
      return AUTOMATION_ERROR_FALLBACK;
    }
    const expected = resolve(runnerTemp, 'progress-prizes-error.txt');
    if (resolve(file) !== expected) return AUTOMATION_ERROR_FALLBACK;
    const pathStat = await lstat(expected, { bigint: true });
    if (!pathStat.isFile() || pathStat.size < 2n || pathStat.size > 601n) {
      return AUTOMATION_ERROR_FALLBACK;
    }

    const handle = await openFile(expected, constants.O_RDONLY | constants.O_NOFOLLOW);
    try {
      const openedStat = await handle.stat({ bigint: true });
      if (!openedStat.isFile() || !sameFileVersion(pathStat, openedStat)) {
        return AUTOMATION_ERROR_FALLBACK;
      }
      const bytes = Buffer.alloc(Number(openedStat.size));
      let offset = 0;
      while (offset < bytes.length) {
        const { bytesRead } = await handle.read(bytes, offset, bytes.length - offset, offset);
        if (bytesRead === 0) break;
        offset += bytesRead;
      }
      const finalStat = await handle.stat({ bigint: true });
      if (offset !== bytes.length || !sameFileVersion(openedStat, finalStat)) {
        return AUTOMATION_ERROR_FALLBACK;
      }
      return safeAutomationDiagnostic(bytes);
    } finally {
      await handle.close();
    }
  } catch {
    return AUTOMATION_ERROR_FALLBACK;
  }
}

async function main() {
  const diagnostic = await readBoundedDiagnostic(process.argv[2], process.env.RUNNER_TEMP);
  process.stderr.write(`${diagnostic}\n`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
