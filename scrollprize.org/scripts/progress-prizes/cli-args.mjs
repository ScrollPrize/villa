const BOOLEAN_OPTIONS = new Set(['allow-activation-rewind', 'dry-run', 'google', 'help']);
const VALUE_OPTIONS = new Set([
  'environment',
  'event-name',
  'expected-current-cycle',
  'expected-current-responder-uri',
  'fault',
  'file',
  'head-sha',
  'mode',
  'page-path',
  'preparation-days',
  'responder-uri',
  'simulated-now',
  'source-cycle',
  'target-cycle',
  'verified-sha',
]);

export const CORE_COMMANDS = Object.freeze(['validate', 'plan', 'prepare-page']);
export const GOOGLE_COMMANDS = Object.freeze([
  'validate',
  'bootstrap',
  'prepare',
  'activate',
  'verify',
  'cleanup',
]);
export const CLI_COMMANDS = Object.freeze([
  ...new Set([...CORE_COMMANDS, ...GOOGLE_COMMANDS]),
]);

export function parseCliArgs(argv) {
  if (!Array.isArray(argv)) {
    throw new TypeError('argv must be an array');
  }
  const [command, ...tokens] = argv;
  if (!CLI_COMMANDS.includes(command)) {
    throw new Error(`Command must be one of: ${CLI_COMMANDS.join(', ')}`);
  }

  const options = {};
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (!token.startsWith('--')) {
      throw new Error(`Unexpected positional argument ${JSON.stringify(token)}`);
    }

    const equalsAt = token.indexOf('=');
    const name = token.slice(2, equalsAt === -1 ? undefined : equalsAt);
    if (!BOOLEAN_OPTIONS.has(name) && !VALUE_OPTIONS.has(name)) {
      throw new Error(`Unknown option --${name}`);
    }
    if (Object.hasOwn(options, name)) {
      throw new Error(`Option --${name} may be supplied only once`);
    }

    if (BOOLEAN_OPTIONS.has(name)) {
      if (equalsAt !== -1) {
        throw new Error(`Boolean option --${name} does not accept a value`);
      }
      options[name] = true;
      continue;
    }

    const value = equalsAt === -1 ? tokens[++index] : token.slice(equalsAt + 1);
    if (value === undefined || value.startsWith('--') || value === '') {
      throw new Error(`Option --${name} requires a value`);
    }
    options[name] = value;
  }

  return Object.freeze({ command, options: Object.freeze(options) });
}

export function requireOption(options, name) {
  const value = options[name];
  if (value === undefined) {
    throw new Error(`Missing required option --${name}`);
  }
  return value;
}

export function parsePositiveIntegerOption(options, name, fallback) {
  if (options[name] === undefined) {
    return fallback;
  }
  if (!/^[1-9]\d*$/.test(options[name])) {
    throw new Error(`Option --${name} must be a positive integer`);
  }
  return Number(options[name]);
}

export const CORE_CLI_USAGE = `Usage:
  node cli.mjs validate [--file PATH]
  node cli.mjs plan --target-cycle YYYY-MM [clock options]
  node cli.mjs prepare-page --target-cycle YYYY-MM --responder-uri URL \\
    --expected-current-responder-uri URL [--dry-run] [--file PATH] [clock options]

Clock options:
  --environment staging|production   Defaults to production
  --event-name NAME                  GitHub event name (required for simulated time)
  --simulated-now ISO_TIMESTAMP      Staging workflow_dispatch only
  --preparation-days N               Defaults to 7
`;

export const GOOGLE_CLI_USAGE = `Google automation (private values are environment variables only):
  node automation-cli.mjs validate --source-cycle YYYY-MM [--page-path PATH]
  node automation-cli.mjs bootstrap --source-cycle YYYY-MM [--dry-run] [staging controls]
    [--allow-activation-rewind --target-cycle YYYY-MM]
  node automation-cli.mjs prepare --target-cycle YYYY-MM [--source-cycle YYYY-MM] [--dry-run]
  node automation-cli.mjs activate --target-cycle YYYY-MM [--source-cycle YYYY-MM] [--fault STEP]
  node automation-cli.mjs verify --target-cycle YYYY-MM [--source-cycle YYYY-MM] \\
    [--mode prepared|active|cleaned]
  node automation-cli.mjs cleanup --target-cycle YYYY-MM [--source-cycle YYYY-MM]

Public controls:
  --environment staging|production   Defaults to PROGRESS_PRIZE_ENVIRONMENT
  --event-name NAME                  Defaults to GITHUB_EVENT_NAME
  --page-path PATH                   Defaults to scrollprize.org/docs/34_prizes.md
  --preparation-days N               Defaults to 7
  --simulated-now ISO_TIMESTAMP      Staging workflow_dispatch only
  --fault after-copy|after-close-source
  --allow-activation-rewind          Explicit staging bootstrap recovery only
  --head-sha SHA --verified-sha SHA  Exact preview-verified activation commit
  --dry-run                          Bootstrap and prepare only
`;
