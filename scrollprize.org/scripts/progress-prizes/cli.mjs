#!/usr/bin/env node

import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve } from 'node:path';

import { createClock } from './clock.mjs';
import { parseProgressPrizeMarkdown, updateProgressPrizeMarkdown } from './markdown.mjs';
import { planRollover, serializeRolloverPlan } from './state.mjs';
import {
  CORE_CLI_USAGE,
  GOOGLE_COMMANDS,
  parseCliArgs,
  parsePositiveIntegerOption,
  requireOption,
} from './cli-args.mjs';
import { runAutomationCli } from './automation-cli.mjs';

const defaultPagePath = resolve(dirname(fileURLToPath(import.meta.url)), '../../docs/34_prizes.md');

function pageSummary(parsed) {
  return {
    cycle: parsed.cycle,
    deadline: parsed.deadline.label,
    cutoffAt: parsed.deadline.cutoffAt.toISOString(),
    responderUri: parsed.responderUri,
  };
}

function runtimeFromOptions(options) {
  const environment = options.environment ?? 'production';
  const clock = createClock({
    environment,
    eventName: options['event-name'],
    simulatedNow: options['simulated-now'],
  });
  return { environment, clock };
}

export async function runCoreCli(argv, {
  read = readFile,
  write = writeFile,
  output = (value) => process.stdout.write(`${value}\n`),
} = {}) {
  const { command, options } = parseCliArgs(argv);
  if (options.help) {
    output(CORE_CLI_USAGE);
    return { help: true };
  }

  const file = resolve(options.file ?? defaultPagePath);
  if (command === 'validate') {
    const markdown = await read(file, 'utf8');
    const result = { command, file, page: pageSummary(parseProgressPrizeMarkdown(markdown)) };
    output(JSON.stringify(result));
    return result;
  }

  const targetCycle = requireOption(options, 'target-cycle');
  const preparationDays = parsePositiveIntegerOption(options, 'preparation-days', 7);
  const { environment, clock } = runtimeFromOptions(options);
  const rollover = planRollover({ targetCycle, now: clock.now(), preparationDays });

  if (command === 'plan') {
    const result = {
      command,
      environment,
      rollover: serializeRolloverPlan(rollover),
    };
    output(JSON.stringify(result));
    return result;
  }

  const markdown = await read(file, 'utf8');
  const update = updateProgressPrizeMarkdown(markdown, {
    targetCycle,
    responderUri: requireOption(options, 'responder-uri'),
    expectedCurrentCycle: options['expected-current-cycle'],
    expectedCurrentResponderUri: requireOption(options, 'expected-current-responder-uri'),
  });
  const dryRun = options['dry-run'] === true;
  if (update.changed && !dryRun) {
    await write(file, update.content, 'utf8');
  }
  const result = {
    command,
    file,
    environment,
    dryRun,
    changed: update.changed,
    wroteFile: update.changed && !dryRun,
    previous: pageSummary(update.current),
    next: pageSummary({
      cycle: update.next.cycle,
      deadline: update.next.deadline,
      responderUri: update.next.responderUri,
    }),
    rollover: serializeRolloverPlan(rollover),
  };
  output(JSON.stringify(result));
  return result;
}

export async function runCli(argv, dependencies = {}) {
  const { command, options } = parseCliArgs(argv);
  const googleCommand = command !== 'validate'
    ? GOOGLE_COMMANDS.includes(command)
    : options.google === true || options['source-cycle'] !== undefined;
  return googleCommand
    ? runAutomationCli(argv, dependencies)
    : runCoreCli(argv, dependencies);
}

async function main() {
  try {
    await runCli(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`progress-prizes: ${error.message}\n`);
    process.exitCode = 1;
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  await main();
}
