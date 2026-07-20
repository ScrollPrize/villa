const BOOLEAN_OPTIONS = new Set(['dry-run', 'help']);
const VALUE_OPTIONS = new Set([
  'environment',
  'event-name',
  'expected-current-cycle',
  'expected-current-responder-uri',
  'file',
  'preparation-days',
  'responder-uri',
  'simulated-now',
  'target-cycle',
]);

export const CORE_COMMANDS = Object.freeze(['validate', 'plan', 'prepare-page']);

export function parseCliArgs(argv) {
  if (!Array.isArray(argv)) {
    throw new TypeError('argv must be an array');
  }
  const [command, ...tokens] = argv;
  if (!CORE_COMMANDS.includes(command)) {
    throw new Error(`Command must be one of: ${CORE_COMMANDS.join(', ')}`);
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
